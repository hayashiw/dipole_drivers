from numpy import number as npnum

from simsopt.geo import SurfaceRZFourier
from simsopt.util import comm_world, proc0_print

__all__ = [
    "dummy_surface",
    "check_required_inputs",
    "is_positive_scalar_or_not_empty_sequence",
    "is_real_surfacerzfourier",
    "is_positive_scalar",
    "is_not_empty_sequence",
]

class _DummySurfaceRZFourier(SurfaceRZFourier):
    pass

dummy_surface = _DummySurfaceRZFourier()

def check_required_inputs(
    *,
    log = None,
    **kwargs
):
    if log is None:
        log = proc0_print

    ierr = 0
    if len(kwargs) == 0:
        msg = "No inputs provided to check_required_inputs."
        log(msg)
        if (comm_world is not None) and (comm_world.size > 1):
            comm_world.Abort(1)
        raise Exception(msg)

    msg = "Missing required inputs:\n"
    for key, val in kwargs.items():
        if not hasattr(val, "__len__") or len(val) != 2:
            msg = f"Improper use of check_required_inputs. Expected a dictionary of key: (func, value) pairs, but got {key}: {val}."
            log(msg)
            if (comm_world is not None) and (comm_world.size > 1):
                comm_world.Abort(1)
            raise Exception(msg)
        else:
            func, value = val
            if not _is_not_none(value):
                msg += f"    {key}: Expected a non-None value for {key}, but got None.\n"
                ierr += 1
            else:
                check, emsg = func(value)
                if not check:
                    msg += f"    {key}: {emsg}\n"
                    ierr += 1
    if ierr > 0:
        log(msg)
        if (comm_world is not None) and (comm_world.size > 1):
            comm_world.Abort(1)
        raise Exception(msg)

def is_positive_scalar_or_not_empty_sequence(val):
    if hasattr(val, "__len__"):
        return is_not_empty_sequence(val)
    else:
        return is_positive_scalar(val)

def is_real_surfacerzfourier(val):
    msg = f"Expected a SurfaceRZFourier, but got {val} of type <{type(val).__name__}>."
    return isinstance(val, SurfaceRZFourier), msg

def is_positive_scalar(val):
    msg = f"Expected a positive scalar, but got {val} of type <{type(val).__name__}>."
    return (isinstance(val, (int, float, npnum)) and val > 0), msg

def is_not_empty_sequence(val):
    msg = f"Expected a non-empty sequence, but got {val} of type <{type(val).__name__}>."
    return (hasattr(val, "__len__") and len(val) > 0), msg

def _is_not_none(val):
    return val is not None
