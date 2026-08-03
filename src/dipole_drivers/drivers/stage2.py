import argparse
import inspect
import numpy as np
import os
import time

from datetime import timedelta, datetime
from scipy.optimize import minimize

from ..defaults import *
from ..objectives.build import (
    build_stage2_objective_function,
    build_stage2_objective_parser,
    check_objective_kwargs
)
from ..utils.biotsavart import build_biotsavart_parser
from ..utils.taylortest import *

OBJECTIVE_KWARGS = list(inspect.signature(build_stage2_objective_function).parameters.keys())

__all__ = [
    "build_stage2_parser",
    "stage2",
]

def build_stage2_parser(
    *,
    maxiter_required: bool = False,
    ftol_required: bool = False,
    gtol_required: bool = False,
    maxcor_required: bool = False,
    **kwargs
):
    parser = argparse.ArgumentParser(add_help=False, parents=[
        build_biotsavart_parser(**kwargs),
        build_stage2_objective_parser(**kwargs),
        build_taylortest_parser(**kwargs),
    ])
    parser.add_argument(
        "--maxiter", type=int, default=DEFAULT_MAXITER, required=maxiter_required,
        help=f"Maximum number of iterations for the optimizer. Default: {DEFAULT_MAXITER}."
    )
    parser.add_argument(
        "--ftol", type=float, default=DEFAULT_FTOL, required=ftol_required,
        help=f"Function tolerance for the optimizer. Default: {DEFAULT_FTOL}."
    )
    parser.add_argument(
        "--gtol", type=float, default=DEFAULT_GTOL, required=gtol_required,
        help=f"Gradient tolerance for the optimizer. Default: {DEFAULT_GTOL}."
    )
    parser.add_argument(
        "--maxcor", type=int, default=DEFAULT_MAXCOR, required=maxcor_required,
        help=f"Maximum number of variable metric corrections used in the L-BFGS-B algorithm. Default: {DEFAULT_MAXCOR}."
    )
    parser.add_argument(
        "--save-iter-dir", type=str, default=None, required=False,
        help="Directory to save iteration data. Default: None."
    )
    parser.add_argument(
        "--save-iter-freq", type=int, default=DEFAULT_ITER_FREQ, required=False,
        help=f"Frequency to save iteration data. Default: {DEFAULT_ITER_FREQ}."
    )
    parser.add_argument(
        "--skip-optimization", action="store_true", required=False,
        help="Skips the optimization. Useful for testing the objective function and gradient. Default: False."
    )
    return parser

def stage2(
    biotsavart,
    surfaces,
    log,
    **kwargs,
):
    save_iter_dir = kwargs.get("save_iter_dir", None)
    save_iter_freq = kwargs.get("save_iter_freq", DEFAULT_ITER_FREQ)
    save_iters = (save_iter_dir is not None) and (save_iter_freq is not None) and (save_iter_freq > 0)
    if save_iters:
        os.makedirs(save_iter_dir, exist_ok=True) # pyright: ignore[reportArgumentType]
        log(f"Saving iteration data to {save_iter_dir}")
        log("")

    save_iter_tag = kwargs.get("tag", "biotsavart")

    check_objective_kwargs(log=log, **kwargs)
    JF, objectives = build_stage2_objective_function(biotsavart, surfaces, **kwargs)
    log(f"Built stage 2 objective function with {np.asarray(JF.x).size} dofs:")
    for key in objectives:
        if key.startswith("J_"):
            log(f"    {key}")
    log("")

    ignore_failure = kwargs.get("taylor_test_ignore", False)
    collective_raise = kwargs.get("taylor_test_collective", False)
    skip_optimization = kwargs.get("skip_optimization", False)
    if skip_optimization:
        ignore_failure = True
        collective_raise = False
    run_taylor_test(JF, log=log,
                    ignore_failure=ignore_failure,
                    collective_raise=collective_raise)
    if skip_optimization:
        log("Skipping optimization.")

        class DummyResult:
            def __init__(self):
                self.success = True
                self.message = "Optimization skipped."
                self.nit = 0
                self.nfev = 0
        return DummyResult()

    start_time = time.monotonic()
    log(f"[{datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}] Starting stage 2 optimization...")
    tracker = dict(iters=0, evals=0)
    def log_row(J=None, dJ=None):
        delta_t = time.monotonic() - start_time
        if J is None:
            J = JF.J()
        if dJ is None:
            dJ = JF.dJ()
        J_inf_norm = np.linalg.norm(dJ, ord=np.inf)
        J_2_norm = np.linalg.norm(dJ, ord=2)
        vals = [delta_t, tracker["iters"], tracker["evals"], J, J_inf_norm, J_2_norm, *(f.J() for f in objectives.values())]
        log(",".join(f"{v}" for v in vals), data=True)
    log(",".join(["time", "iters", "evals", "J", "dJ_inf_norm", "dJ_2_norm", *objectives.keys()]), data=True)  # header
    log_row()  # initial values

    last = {}
    def fun(x):
        tracker["evals"] += 1
        JF.x = x
        J = float(JF.J())
        dJ = np.asarray(JF.dJ(), dtype=float)
        last.update(x=np.array(x), J=J, dJ=dJ)
        log_row(J, dJ)
        return J, dJ

    def callback(xk, state=None):
        tracker["iters"] += 1
        tracker["evals"] = 0
        if last and np.array_equal(np.asarray(xk), last["x"]):
            log_row(last["J"], last["dJ"])
        else:
            log_row()
        if save_iters and (tracker["iters"] % int(save_iter_freq) == 0): # pyright: ignore[reportArgumentType]
            iter_file = os.path.join(str(save_iter_dir), f"{save_iter_tag}.stage2.iter{tracker['iters']}.json")
            biotsavart.save(iter_file)
            log(f"Saved iteration {tracker['iters']} to {iter_file}")

    x0 = np.asarray(JF.x, dtype=float)
    bounds = list(zip(np.asarray(JF.lower_bounds), np.asarray(JF.upper_bounds)))
    res = minimize(
        fun,
        x0,
        callback=callback,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": kwargs.get("maxiter", DEFAULT_MAXITER),
            "ftol": kwargs.get("ftol", DEFAULT_FTOL),
            "gtol": kwargs.get("gtol", DEFAULT_GTOL),
            "maxcor": kwargs.get("maxcor", DEFAULT_MAXCOR),
        }
    )
    end_time = time.monotonic()
    opt_msg = "Optimization " + ("SUCCESS" if res.success else "FAILED")
    nit = res.nit
    nfev = res.nfev
    log(opt_msg)
    log(res.message)
    log(f"    {nit = }")
    log(f"    {nfev = }")
    log(f"    Run time: {str(timedelta(seconds=end_time - start_time))}")
    log(f"[{datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}] Completed stage 2 optimization.")
    log("")
    
    return res

