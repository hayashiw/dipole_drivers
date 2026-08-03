import argparse
import numpy as np
import time

from collections.abc import Callable
from datetime import timedelta, datetime
from mpi4py import MPI

from simsopt._core import Optimizable
from simsopt.util import comm_world, proc0_print

from ..defaults import *
from ..drivers.logging import DriverLog

EPSILONS = (1e-4, 1e-5, 1e-6)

__all__ = [
    "run_taylor_test",
    "build_taylortest_parser",
]

def run_taylor_test(
    objective: Optimizable,
    log: DriverLog | Callable = proc0_print,
    ignore_failure: bool = False,
    collective_raise: bool = False,
) -> None:

    log("Running central-difference Taylor test...")
    start_time = time.monotonic()
    def fun(x):
        objective.x = x
        J = objective.J()
        dJ = objective.dJ()
        return J, dJ

    x0 = np.asarray(objective.x).copy()
    J, dJ = fun(x0)
    h = np.random.uniform(size=x0.shape)
    errors = []
    for eps in EPSILONS:
        J_plus, _ = fun(x0 + eps * h)
        J_minus, _ = fun(x0 - eps * h)
        central_diff = (J_plus - J_minus) / (2 * eps)
        error = np.abs(central_diff - dJ @ h)
        errors.append(error)
        elapsed_time = str(timedelta(seconds=time.monotonic() - start_time))
        log(f"    [{elapsed_time}] eps={eps:.0e}: error={error:.3e}, J={J:.3e}, dJ@h={dJ @ h:.3e}")
    log("")
    objective.x = x0  # Reset the objective's x to its original value

    ierr = 0
    ipass = 0
    log("Checking errors...")
    for i in range(1, len(errors)):
        eps = EPSILONS[i]
        ratio = errors[i-1] / errors[i]
        expected = (EPSILONS[i-1] / EPSILONS[i])**2
        floor = np.finfo(float).eps * abs(J) / eps
        if errors[i] < 10*floor:
            log(f"    eps={eps:.0e}: below the finite-difference noise floor, stopping.")
            break
        if not (0.5*expected < ratio < 2.0*expected):
            log(f"    eps={eps:.0e}: error ratio={ratio:.2f} (expected ~{expected:.0f}).")
            ierr += 1
        ipass += 1

    if ipass == 0:
        ierr = 1
        log(f"All error below noise floor. Test inconclusive.")
    elif ierr > 0:
        log(f"Taylor test failed for {ierr} out of {len(errors)-1} steps. "
            f"Check the implementation of the objective function and its gradient.")
    else:
        log("Taylor test passed.")

    if (comm_world is not None) and collective_raise:
        ierr = comm_world.allreduce(ierr, op=MPI.MAX)

    if ierr > 0 and (not ignore_failure):
        if (not collective_raise) and (comm_world is not None) and comm_world.size > 1:
            comm_world.Abort(1)
        raise RuntimeError("Taylor test failed. Check the implementation of "
                           "the objective function and its gradient.")
    elif ierr > 0 and ignore_failure:
        log("Ignoring Taylor test failure.")
    end_time = time.monotonic()
    elapsed_time = str(timedelta(seconds=end_time - start_time))
    log(f"Taylor test completed. Run time: {elapsed_time}.")
    log("")

def build_taylortest_parser(**kwargs) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--taylor-test-ignore", action="store_true", required=False,
        help="Ignore the Taylor test failure. Default: False."
    )
    parser.add_argument(
        "--taylor-test-collective", action="store_true", required=False,
        help="Collectively raise an exception if the Taylor test fails across all MPI ranks. Default: False."
    )
    return parser
