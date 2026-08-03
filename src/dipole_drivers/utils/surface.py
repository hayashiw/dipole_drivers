import argparse
import numpy as np

from typing import Sequence

from simsopt.geo import SurfaceRZFourier, SurfaceXYZTensorFourier

from ..defaults import *

__all__ = [
    "build_surface_parameters_parser",
    "build_surface_wout_parser",
    "change_surface_range",
    "change_surface_resolution",
    "convert_rzfourier_to_xyztensorfourier",
    "convert_xyztensorfourier_to_rzfourier",
    "copy_surface",
    "generate_surfaces_from_wout",
    "get_surface_range",
    "rebuild_surface",
]

def build_surface_parameters_parser(
    *,
    surface_mpol_required: bool = False,
    surface_ntor_required: bool = False,
    surface_range_required: bool = False,
    surface_ntheta_required: bool = False,
    surface_nphi_required: bool = False,
    **kwargs,
):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--surface-mpol", type=int, default=None, required=surface_mpol_required,
        help="Poloidal mode number for the surface."
    )
    parser.add_argument(
        "--surface-ntor", type=int, default=None, required=surface_ntor_required,
        help="Toroidal mode number for the surface."
    )
    parser.add_argument(
        "--surface-range", type=str, default=None, required=surface_range_required,
        choices=["full-torus", "field-period", "half-period"],
        help="Range of the surface. Must be one of ['full-torus', 'field-period', 'half-period']."
    )
    parser.add_argument(
        "--surface-ntheta", type=int, default=None, required=surface_ntheta_required,
        help="Number of poloidal quadrature points for the surface."
    )
    parser.add_argument(
        "--surface-nphi", type=int, default=None, required=surface_nphi_required,
        help="Number of toroidal quadrature points for the surface."
    )
    return parser

def build_surface_wout_parser(
    *,
    surface_wout_file_required: bool = False,
    surface_wout_s_required: bool = False,
    **kwargs,
):
    parser = argparse.ArgumentParser(add_help=False, parents=[build_surface_parameters_parser(**kwargs)])
    parser.add_argument(
        "--surface-wout-file", type=str, default=None, required=surface_wout_file_required,
        help="Path to the wout file to read the surface from."
    )
    parser.add_argument(
        f"--surface-wout-s", type=float, nargs="+", default=[DEFAULT_SURFACE_WOUT_S], required=surface_wout_s_required,
        help=f"s value(s) for the surface(s) to read from the wout file. Default: [{DEFAULT_SURFACE_WOUT_S}]."
    )
    return parser

def change_surface_range(
    surface: SurfaceXYZTensorFourier | SurfaceRZFourier,
    *,
    surface_range: str | None = None,
    surface_ntheta: int | None = None,
    surface_nphi: int | None = None,
) -> SurfaceXYZTensorFourier | SurfaceRZFourier:
    if not isinstance(surface, (SurfaceXYZTensorFourier, SurfaceRZFourier)):
        raise TypeError(f"Expected surface of type <SurfaceXYZTensorFourier> or <SurfaceRZFourier>, got <{type(surface).__name__}> instead.")
    if surface_range is not None:
        assert surface_range in ["full-torus", "field-period", "half-period"], f"Invalid range: {surface_range}. Must be one of ['full-torus', 'field-period', 'half-period']."
    
    if (surface_range is None) and (surface_ntheta is None) and (surface_nphi is None):
        return copy_surface(surface)

    if surface_range is None:
        surface_range = get_surface_range(surface)
        
    if surface_range == "full-torus":
        phimax: float = 1.0
    elif surface_range == "field-period":
        phimax: float = 1 / surface.nfp
    elif surface_range == "half-period":
        phimax: float = 1 / 2 / surface.nfp
    else:
        raise ValueError(f"Invalid range: {surface_range}. Must be one of ['full-torus', 'field-period', 'half-period'].")

    use_ntheta: int = surface_ntheta if surface_ntheta else surface.quadpoints_theta.size
    use_nphi: int = surface_nphi if surface_nphi else surface.quadpoints_phi.size

    quadpoints_phi = np.linspace(0, phimax, use_nphi, endpoint=False)
    quadpoints_theta = np.linspace(0, 1, use_ntheta, endpoint=False)
    constructor = SurfaceXYZTensorFourier if isinstance(surface, SurfaceXYZTensorFourier) else SurfaceRZFourier
    new_surface = constructor(
        mpol=surface.mpol,
        ntor=surface.ntor,
        nfp=surface.nfp,
        stellsym=surface.stellsym,
        quadpoints_theta=quadpoints_theta,
        quadpoints_phi=quadpoints_phi,
    )
    new_surface.x = np.asarray(surface.x).copy()
    return new_surface

def change_surface_resolution(
    surface: SurfaceXYZTensorFourier | SurfaceRZFourier,
    *,
    surface_mpol: int | None = None,
    surface_ntor: int | None = None,
) -> SurfaceXYZTensorFourier | SurfaceRZFourier:
    if not isinstance(surface, (SurfaceXYZTensorFourier, SurfaceRZFourier)):
        raise TypeError(f"Expected surface of type <SurfaceXYZTensorFourier> or <SurfaceRZFourier>, got <{type(surface).__name__}> instead.")
    
    if (surface_mpol is None) and (surface_ntor is None):
        return copy_surface(surface)

    use_mpol: int = surface_mpol if surface_mpol else surface.mpol
    use_ntor: int = surface_ntor if surface_ntor else surface.ntor

    constructor = SurfaceXYZTensorFourier if isinstance(surface, SurfaceXYZTensorFourier) else SurfaceRZFourier
    new_surface = constructor(
        mpol=use_mpol,
        ntor=use_ntor,
        nfp=surface.nfp,
        stellsym=surface.stellsym,
        quadpoints_theta=surface.quadpoints_theta,
        quadpoints_phi=surface.quadpoints_phi,
    )
    if use_mpol >= surface.mpol and use_ntor >= surface.ntor:
        m_map = lambda i: i if i <= surface.mpol else use_mpol + (i - surface.mpol)
        n_map = lambda j: j if j <= surface.ntor else use_ntor + (j - surface.ntor)
        new_surface.x = np.zeros_like(np.asarray(new_surface.x))
        for coord in "xyz":
            for i in range(2*surface.mpol + 1):
                for j in range(2*surface.ntor + 1):
                    if surface.skip(coord, i, j): continue
                    new_surface.set(f"{coord}({m_map(i)},{n_map(j)})",
                                    surface.get(f"{coord}({i},{j})"))
    else:
        new_surface.least_squares_fit(surface.gamma())
    return new_surface

def convert_rzfourier_to_xyztensorfourier(
    surface: SurfaceRZFourier,
) -> SurfaceXYZTensorFourier:
    if not isinstance(surface, SurfaceRZFourier):
        raise TypeError(f"Expected surface of type <SurfaceRZFourier>, got <{type(surface).__name__}> instead.")

    new_surface = SurfaceXYZTensorFourier(
        mpol=surface.mpol,
        ntor=surface.ntor,
        nfp=surface.nfp,
        stellsym=surface.stellsym,
        quadpoints_theta=surface.quadpoints_theta.copy(),
        quadpoints_phi=surface.quadpoints_phi.copy(),
    )
    new_surface.least_squares_fit(surface.gamma())
    return new_surface

def convert_xyztensorfourier_to_rzfourier(
    surface: SurfaceXYZTensorFourier,
) -> SurfaceRZFourier:
    if not isinstance(surface, SurfaceXYZTensorFourier):
        raise TypeError(f"Expected surface of type <SurfaceXYZTensorFourier>, got <{type(surface).__name__}> instead.")

    new_surface = SurfaceRZFourier(
        mpol=surface.mpol,
        ntor=surface.ntor,
        nfp=surface.nfp,
        stellsym=surface.stellsym,
        quadpoints_theta=surface.quadpoints_theta.copy(),
        quadpoints_phi=surface.quadpoints_phi.copy(),
    )
    new_surface.least_squares_fit(surface.gamma())
    return new_surface

def copy_surface(surface: SurfaceXYZTensorFourier | SurfaceRZFourier) -> SurfaceXYZTensorFourier | SurfaceRZFourier:
    if not isinstance(surface, (SurfaceXYZTensorFourier, SurfaceRZFourier)):
        raise TypeError(f"Expected surface of type <SurfaceXYZTensorFourier> or <SurfaceRZFourier>, got <{type(surface).__name__}> instead.")

    constructor = SurfaceXYZTensorFourier if isinstance(surface, SurfaceXYZTensorFourier) else SurfaceRZFourier
    new_surface = constructor(
        mpol=surface.mpol,
        ntor=surface.ntor,
        nfp=surface.nfp,
        stellsym=surface.stellsym,
        quadpoints_theta=surface.quadpoints_theta.copy(),
        quadpoints_phi=surface.quadpoints_phi.copy(),
    )
    new_surface.x = np.asarray(surface.x).copy()
    return new_surface

def generate_surfaces_from_wout(
    surface_wout_file: str,
    *,
    surface_wout_s: float | Sequence[float] = DEFAULT_SURFACE_WOUT_S,
    surface_mpol: int | None = None,
    surface_ntor: int | None = None,
    surface_range: str | None = None,
    surface_ntheta: int | None = None,
    surface_nphi: int | None = None,
    **kwargs
) -> list[SurfaceRZFourier]:
    if not isinstance(surface_wout_s, Sequence):
        surface_wout_s = [surface_wout_s]

    surfaces = []
    for s_i in sorted(surface_wout_s): # boundary_surface = surfaces[-1]
        surface = SurfaceRZFourier.from_wout(surface_wout_file, s=s_i, **kwargs)
        surface = rebuild_surface(
            surface,
            surface_mpol=surface_mpol,
            surface_ntor=surface_ntor,
            surface_range=surface_range,
            surface_ntheta=surface_ntheta,
            surface_nphi=surface_nphi,
        )

        surfaces.append(surface)

    return surfaces

def get_surface_range(surface: SurfaceXYZTensorFourier | SurfaceRZFourier) -> str:
    phimax = surface.quadpoints_phi.max()
    idx = np.abs(np.array([1/2/surface.nfp, 1/surface.nfp, 1]) - phimax).argmin()
    surface_range = ["half-period", "field-period", "full-torus"][idx]
    return surface_range

def rebuild_surface(
    surface: SurfaceXYZTensorFourier | SurfaceRZFourier,
    *,
    surface_mpol: int | None = None,
    surface_ntor: int | None = None,
    surface_range: str | None = None,
    surface_ntheta: int | None = None,
    surface_nphi: int | None = None,
) -> SurfaceXYZTensorFourier | SurfaceRZFourier:
    if not isinstance(surface, (SurfaceXYZTensorFourier, SurfaceRZFourier)):
        raise TypeError(f"Expected surface of type <SurfaceXYZTensorFourier> or <SurfaceRZFourier>, got <{type(surface).__name__}> instead.")

    new_surface = change_surface_resolution(surface, surface_mpol=surface_mpol, surface_ntor=surface_ntor)
    new_surface = change_surface_range(new_surface, surface_range=surface_range, surface_ntheta=surface_ntheta, surface_nphi=surface_nphi)
    return new_surface
