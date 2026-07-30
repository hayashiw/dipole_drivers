from typing import overload
from numpy import sum as npsum
from numpy import float64
from numpy.typing import NDArray
from numpy.linalg import norm

from simsopt.field import BiotSavart
from simsopt.geo import BoozerSurface, Surface

__all__ = ["calculate_bdotn_norm"]

@overload
def calculate_bdotn_norm(
    arg: BoozerSurface,
/) -> NDArray[float64]: ...

@overload
def calculate_bdotn_norm(
    arg: BiotSavart,
    surface: Surface,
/) -> NDArray[float64]: ...

def calculate_bdotn_norm(arg, surface=None, /) -> NDArray[float64]:
    if isinstance(arg, BoozerSurface) and (surface is None):
        biotsavart, surface = arg.biotsavart, arg.surface
    elif isinstance(arg, BiotSavart) and isinstance(surface, Surface):
        biotsavart = arg
    else:
        raise TypeError(
            f"Expected (BoozerSurface, None) or (BiotSavart, Surface) "
            f"but got {type(arg)} and {type(surface)}")
    biotsavart.set_points(surface.gamma().reshape(-1, 3))
    B = biotsavart.B().reshape(surface.gamma().shape)
    modB = norm(B, axis=-1)
    Bdotn_norm = npsum(B * surface.unitnormal(), axis=-1) / modB
    return Bdotn_norm
