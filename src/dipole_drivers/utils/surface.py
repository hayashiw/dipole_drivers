import numpy as np

from typing import Sequence

from simsopt.geo import SurfaceRZFourier, SurfaceXYZTensorFourier

__all__ = [
    "change_surface_range",
    "change_surface_resolution",
    "convert_rzfourier_to_xyztensorfourier",
    "convert_xyztensorfourier_to_rzfourier",
    "copy_surface",
    "create_surfaces_from_wout",
    "get_surface_range",
    "rebuild_surface",
]

def change_surface_range(
    surface: SurfaceXYZTensorFourier | SurfaceRZFourier,
    *,
    surface_range: str | None = None,
    ntheta: int | None = None,
    nphi: int | None = None,
) -> SurfaceXYZTensorFourier | SurfaceRZFourier:
    if not isinstance(surface, (SurfaceXYZTensorFourier, SurfaceRZFourier)):
        raise TypeError(f"Expected surface of type <SurfaceXYZTensorFourier> or <SurfaceRZFourier>, got <{type(surface).__name__}> instead.")
    if surface_range is not None:
        assert surface_range in ["full-torus", "field-period", "half-period"], f"Invalid range: {surface_range}. Must be one of ['full-torus', 'field-period', 'half-period']."
    
    if (surface_range is None) and (ntheta is None) and (nphi is None):
        return copy_surface(surface)

    if surface_range is None:
        surface_range = get_surface_range(surface)
        
    if surface_range == "full-torus":
        phimax: float = 1.0
    elif surface_range == "field-period":
        phimax: float = 1 / surface.nfp
    elif surface_range == "half-period":
        phimax: float = 1 / 2 / surface.nfp

    use_ntheta: int = ntheta if ntheta else surface.quadpoints_theta.size
    use_nphi: int = nphi if nphi else surface.quadpoints_phi.size

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
    mpol: int | None = None,
    ntor: int | None = None,
) -> SurfaceXYZTensorFourier | SurfaceRZFourier:
    if not isinstance(surface, (SurfaceXYZTensorFourier, SurfaceRZFourier)):
        raise TypeError(f"Expected surface of type <SurfaceXYZTensorFourier> or <SurfaceRZFourier>, got <{type(surface).__name__}> instead.")
    
    if (mpol is None) and (ntor is None):
        return copy_surface(surface)

    use_mpol: int = mpol if mpol else surface.mpol
    use_ntor: int = ntor if ntor else surface.ntor

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

def create_surfaces_from_wout(
    wout_file: str,
    *,
    s: float | Sequence[float] = 1.0,
    **kwargs
) -> list[SurfaceRZFourier]:
    if not isinstance(s, Sequence):
        s = [s]

    surfaces = []
    for s_i in sorted(s):
        surface = SurfaceRZFourier.from_wout(wout_file, s=s_i, **kwargs)
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
    mpol: int | None = None,
    ntor: int | None = None,
    surface_range: str | None = None,
    ntheta: int | None = None,
    nphi: int | None = None,
) -> SurfaceXYZTensorFourier | SurfaceRZFourier:
    if not isinstance(surface, (SurfaceXYZTensorFourier, SurfaceRZFourier)):
        raise TypeError(f"Expected surface of type <SurfaceXYZTensorFourier> or <SurfaceRZFourier>, got <{type(surface).__name__}> instead.")

    new_surface = change_surface_resolution(surface, mpol=mpol, ntor=ntor)
    new_surface = change_surface_range(new_surface, surface_range=surface_range, ntheta=ntheta, nphi=nphi)
    return new_surface
