import numpy as np

from simsopt.geo import SurfaceRZFourier

from .check import *
from .ellipse import calculate_even_ellipse_spacing

__all__ = [
    "copy_windingsurface",
    "generate_dipole_windingsurface",
    "get_windingsurface_geometry",
]

def copy_windingsurface(windingsurface: SurfaceRZFourier) -> SurfaceRZFourier:
    windingsurface_major_radius, windingsurface_minor_radius_r, windingsurface_minor_radius_z = \
        get_windingsurface_geometry(windingsurface)
    ndipoles_pol_total = windingsurface.quadpoints_theta.size
    ndipoles_tor_total = windingsurface.quadpoints_phi.size
    return generate_dipole_windingsurface(
        winding_surface_major_radius=windingsurface_major_radius,
        winding_surface_minor_radius_r=windingsurface_minor_radius_r,
        winding_surface_minor_radius_z=windingsurface_minor_radius_z,
        ndipoles_pol_total=ndipoles_pol_total,
        ndipoles_tor_total=ndipoles_tor_total
    )

def generate_dipole_windingsurface(
    *,
    winding_surface_major_radius: float = 0.0,
    winding_surface_minor_radius_r: float = 0.0,
    winding_surface_minor_radius_z: float = 0.0,
    ndipoles_pol_total: int = 0,
    ndipoles_tor_total: int = 0,
    **kwargs,
) -> SurfaceRZFourier:
    check_required_inputs(
        winding_surface_major_radius=(is_positive_scalar, winding_surface_major_radius),
        winding_surface_minor_radius_r=(is_positive_scalar, winding_surface_minor_radius_r),
        winding_surface_minor_radius_z=(is_positive_scalar, winding_surface_minor_radius_z),
        ndipoles_pol_total=(is_positive_scalar, ndipoles_pol_total),
        ndipoles_tor_total=(is_positive_scalar, ndipoles_tor_total),
    )
    phis = np.linspace(0, 1, ndipoles_tor_total, endpoint=False)
    phis += (phis[1] - phis[0])/2
    
    thetas = calculate_even_ellipse_spacing(
        winding_surface_minor_radius_r,
        winding_surface_minor_radius_z,
        ndipoles_pol_total
    ) / (2*np.pi)

    windingsurface = SurfaceRZFourier(
        mpol=1,
        ntor=0,
        nfp=1,
        stellsym=True,
        quadpoints_theta=thetas,
        quadpoints_phi=phis,
    )
    windingsurface.set_rc(0, 0, winding_surface_major_radius)
    windingsurface.set_rc(1, 0, winding_surface_minor_radius_r)
    windingsurface.set_zs(1, 0, winding_surface_minor_radius_z)
    return windingsurface

def get_windingsurface_geometry(
    windingsurface: SurfaceRZFourier
) -> tuple[float, float, float]:
    windingsurface_major_radius = windingsurface.get_rc(0, 0)
    windingsurface_minor_radius_r = windingsurface.get_rc(1, 0)
    windingsurface_minor_radius_z = windingsurface.get_zs(1, 0)
    return (
        windingsurface_major_radius,
        windingsurface_minor_radius_r,
        windingsurface_minor_radius_z
    )
    