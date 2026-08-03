import argparse
import numpy as np

from typing import Sequence

from simsopt.field import BiotSavart
from simsopt.geo import SurfaceRZFourier, SurfaceXYZTensorFourier
from simsopt.geo.curvesuperellipse import SHAPE_DOF_NAMES

from ..defaults import *
from .check import *
from .circle import fit_minimum_enclosing_circle
from .coil import copy_coils, generate_coils, get_windingsurface_from_coils
from .ellipse import is_enclosed_by_ellipse
from .surface import rebuild_surface
from .windingsurface import generate_dipole_windingsurface

__all__ = [
    "build_biotsavart_parser",
    "copy_biotsavart",
    "generate_biotsavart",
    "get_windingsurface_from_biotsavart",
]

def build_biotsavart_parser(
    *,
    windingsurface_major_radius_required: bool = False,
    windingsurface_minor_radius_r_required: bool = False,
    windingsurface_minor_radius_z_required: bool = False,
    ndipoles_pol_total_required: bool = False,
    ndipoles_tor_total_required: bool = False,
    tf_major_radius_required: bool = False,
    tf_minor_radius_r_required: bool = False,
    tf_minor_radius_z_required: bool = False,
    ntfs_total_required: bool = False,
    curve_curve_distance_required: bool = False,
    dipole_base_currents_required: bool = False,
    tf_base_currents_required: bool = False,
    superellipse_n_required: bool = False,
    nquadpoints_required: bool = False,
    **kwargs,
):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--windingsurface-major-radius", type=float, default=None, required=windingsurface_major_radius_required,
        help="Major radius of the dipole winding surface."
    )
    parser.add_argument(
        "--windingsurface-minor-radius-r", type=float, default=None, required=windingsurface_minor_radius_r_required,
        help="Minor radius (r) of the dipole winding surface."
    )
    parser.add_argument(
        "--windingsurface-minor-radius-z", type=float, default=None, required=windingsurface_minor_radius_z_required,
        help="Minor radius (z) of the dipole winding surface."
    )
    parser.add_argument(
        "--ndipoles-pol-total", type=int, default=DEFAULT_NDIPOLES_POL_TOTAL, required=ndipoles_pol_total_required,
        help=f"Total number of poloidal dipoles. Default: {DEFAULT_NDIPOLES_POL_TOTAL}."
    )
    parser.add_argument(
        "--ndipoles-tor-total", type=int, default=DEFAULT_NDIPOLES_TOR_TOTAL, required=ndipoles_tor_total_required,
        help=f"Total number of toroidal dipoles. Default: {DEFAULT_NDIPOLES_TOR_TOTAL}."
    )
    parser.add_argument(
        "--tf-major-radius", type=float, default=None, required=tf_major_radius_required,
        help="Major radius of the toroidal field coils."
    )
    parser.add_argument(
        "--tf-minor-radius-r", type=float, default=None, required=tf_minor_radius_r_required,
        help="Minor radius (r) of the elliptic toroidal field coils."
    )
    parser.add_argument(
        "--tf-minor-radius-z", type=float, default=None, required=tf_minor_radius_z_required,
        help="Minor radius (z) of the elliptic toroidal field coils."
    )
    parser.add_argument(
        "--ntfs-total", type=int, default=DEFAULT_NTFS_TOTAL, required=ntfs_total_required,
        help=f"Total number of toroidal field coils. Default: {DEFAULT_NTFS_TOTAL}."
    )
    parser.add_argument(
        "--curve-curve-distance", type=float, default=None, required=curve_curve_distance_required,
        help="Distance between the dipole and toroidal field coil curves."
    )
    parser.add_argument(
        "--dipole-base-currents", type=float, nargs="+", default=DEFAULT_DIPOLE_BASE_CURRENTS, required=dipole_base_currents_required,
        help=f"Base currents for the dipole coils. Default: {DEFAULT_DIPOLE_BASE_CURRENTS}. Can be a single value or a list of values."
    )
    parser.add_argument(
        "--tf-base-currents", type=float, nargs="+", default=DEFAULT_TF_BASE_CURRENTS, required=tf_base_currents_required,
        help=f"Base currents for the toroidal field coils. Default: {DEFAULT_TF_BASE_CURRENTS}. Can be a single value or a list of values."
    )
    parser.add_argument(
        "--skip-outboard-pol", action="store_true",
        help="If set, skip the outboard poloidal dipoles."
    )
    parser.add_argument(
        "--superellipse-n", type=float, default=DEFAULT_SUPERELLIPSE_N, required=superellipse_n_required,
        help=f"Exponent for the superellipse shape of the coils. Default: {DEFAULT_SUPERELLIPSE_N}."
    )
    parser.add_argument(
        "--nquadpoints", type=int, default=DEFAULT_NQUADPOINTS, required=nquadpoints_required,
        help=f"Number of quadrature points for the coil curves. Default: {DEFAULT_NQUADPOINTS}."
    )
    parser.add_argument(
        "--dipole-shared-dofs", type=str, nargs="*", choices=SHAPE_DOF_NAMES, default=(), required=False,
        help=f"Dipole shape dofs shared across all dipole curves. Default: ()."
    )
    return parser

def copy_biotsavart(
    biotsavart: BiotSavart,
) -> BiotSavart:
    coils = biotsavart.coils
    copied_coils = copy_coils(coils)
    return BiotSavart(copied_coils)

def generate_biotsavart(
    *,
    windingsurface_major_radius: float = 0.0,
    windingsurface_minor_radius_r: float = 0.0,
    windingsurface_minor_radius_z: float = 0.0,
    ndipoles_pol_total: int = DEFAULT_NDIPOLES_POL_TOTAL,
    ndipoles_tor_total: int = DEFAULT_NDIPOLES_TOR_TOTAL,
    tf_major_radius: float = 0.0,
    tf_minor_radius_r: float = 0.0,
    tf_minor_radius_z: float = 0.0,
    ntfs_total: int = DEFAULT_NTFS_TOTAL,
    curve_curve_distance: float = DEFAULT_CURVE_CURVE_DISTANCE,
    nfp: int = 0,
    dipole_base_currents: float | Sequence[float] = DEFAULT_DIPOLE_BASE_CURRENTS,
    tf_base_currents: float | Sequence[float] = DEFAULT_TF_BASE_CURRENTS,
    skip_outboard_pol: bool = DEFAULT_SKIP_OUTBOARD_POL,
    superellipse_n: float = DEFAULT_SUPERELLIPSE_N,
    nquadpoints: int = DEFAULT_NQUADPOINTS,
    **kwargs,
) -> BiotSavart:
    check_required_inputs(
        windingsurface_major_radius=(is_positive_scalar, windingsurface_major_radius),
        windingsurface_minor_radius_r=(is_positive_scalar, windingsurface_minor_radius_r),
        windingsurface_minor_radius_z=(is_positive_scalar, windingsurface_minor_radius_z),
        ndipoles_pol_total=(is_positive_scalar, ndipoles_pol_total),
        ndipoles_tor_total=(is_positive_scalar, ndipoles_tor_total),
        tf_major_radius=(is_positive_scalar, tf_major_radius),
        tf_minor_radius_r=(is_positive_scalar, tf_minor_radius_r),
        tf_minor_radius_z=(is_positive_scalar, tf_minor_radius_z),
        ntfs_total=(is_positive_scalar, ntfs_total),
        curve_curve_distance=(is_positive_scalar, curve_curve_distance),
        dipole_base_currents=(is_positive_scalar_or_not_empty_sequence, dipole_base_currents),
        tf_base_currents=(is_positive_scalar_or_not_empty_sequence, tf_base_currents),
    )
    assert ntfs_total % nfp == 0, "ntfs_total must be divisible by nfp"
    assert ndipoles_tor_total % nfp == 0, "ndipoles_tor_total must be divisible by nfp"
    assert ndipoles_tor_total % ntfs_total == 0, "ndipoles_tor_total must be divisible by ntfs_total"
    windingsurface = generate_dipole_windingsurface(
        winding_surface_major_radius=windingsurface_major_radius,
        winding_surface_minor_radius_r=windingsurface_minor_radius_r,
        winding_surface_minor_radius_z=windingsurface_minor_radius_z,
        ndipoles_pol_total=ndipoles_pol_total,
        ndipoles_tor_total=ndipoles_tor_total,
        **kwargs,
    )
    coils = generate_coils(
        windingsurface=windingsurface,
        tf_major_radius=tf_major_radius,
        tf_minor_radius_r=tf_minor_radius_r,
        tf_minor_radius_z=tf_minor_radius_z,
        ntfs_total=ntfs_total,
        curve_curve_distance=curve_curve_distance,
        nfp=nfp,
        dipole_base_currents=dipole_base_currents,
        tf_base_currents=tf_base_currents,
        skip_outboard_pol=skip_outboard_pol,
        superellipse_n=superellipse_n,
        nquadpoints=nquadpoints,
        **kwargs,
    )
    return BiotSavart(coils)

def generate_biotsavart_from_surface(
    surface: SurfaceRZFourier | SurfaceXYZTensorFourier,
    *,
    windingsurface_major_radius: float = 0.0,
    windingsurface_minor_radius_r: float = 0.0,
    windingsurface_minor_radius_z: float = 0.0,
    ndipoles_pol_total: int = DEFAULT_NDIPOLES_POL_TOTAL,
    ndipoles_tor_total: int = DEFAULT_NDIPOLES_TOR_TOTAL,
    tf_major_radius: float = 0.0,
    tf_minor_radius_r: float = 0.0,
    tf_minor_radius_z: float = 0.0,
    ntfs_total: int = DEFAULT_NTFS_TOTAL,
    curve_curve_distance: float = DEFAULT_CURVE_CURVE_DISTANCE,
    dipole_base_currents: float | Sequence[float] = DEFAULT_DIPOLE_BASE_CURRENTS,
    tf_base_currents: float | Sequence[float] = DEFAULT_TF_BASE_CURRENTS,
    skip_outboard_pol: bool = DEFAULT_SKIP_OUTBOARD_POL,
    superellipse_n: float = DEFAULT_SUPERELLIPSE_N,
    nquadpoints: int = DEFAULT_NQUADPOINTS,
    **kwargs,
) -> BiotSavart:
    nfp = surface.nfp
    if ntfs_total > 0:
        assert ntfs_total % nfp == 0, "ntfs_total must be divisible by nfp"
    if ndipoles_tor_total > 0:
        assert ndipoles_tor_total % nfp == 0, "ndipoles_tor_total must be divisible by nfp"

    highres_surface = rebuild_surface(
        surface, surface_range="half-period",
        surface_nphi=129, # Use nphi != ntheta to avoid misaligned shapes
        surface_ntheta=128,
    )

    x, y, z = highres_surface.gamma().reshape(-1, 3).T
    r = np.sqrt(x**2 + y**2)
    points = np.column_stack([r, z])
    (circle_major_radius, _), circle_minor_radius =  fit_minimum_enclosing_circle(points) # pyright: ignore[reportArgumentType]
    if windingsurface_major_radius <= 0.0:
        windingsurface_major_radius = circle_major_radius

    z_to_axis = windingsurface_major_radius - circle_minor_radius
    windingsurface_minor_radius = windingsurface_major_radius - (2*z_to_axis/3)
    if windingsurface_minor_radius_r <= 0.0:
        windingsurface_minor_radius_r = float(windingsurface_minor_radius)
    if windingsurface_minor_radius_z <= 0.0:
        windingsurface_minor_radius_z = float(windingsurface_minor_radius)
    assert 0 < windingsurface_minor_radius_r < windingsurface_major_radius, \
        f"windingsurface_minor_radius_r ({windingsurface_minor_radius_r}) must be positive and less than windingsurface_major_radius ({windingsurface_major_radius})."

    tf_minor_radius = windingsurface_major_radius - z_to_axis/3
    if tf_major_radius <= 0.0:
        tf_major_radius = windingsurface_major_radius
    if tf_minor_radius_r <= 0.0:
        tf_minor_radius_r = float(tf_minor_radius)
    if tf_minor_radius_z <= 0.0:
        tf_minor_radius_z = float(tf_minor_radius)
    assert 0 < tf_minor_radius_r < tf_major_radius, \
        f"tf_minor_radius_r ({tf_minor_radius_r}) must be positive and less than tf_major_radius ({tf_major_radius})."

    # Need to make sure that:
    # (1) winding surface encloses surface (can just check against the minimum enclosing circle)
    # (2) the toroidal field coils enclose the winding surface
    assert is_enclosed_by_ellipse(
        circle_major_radius, float(circle_minor_radius), float(circle_minor_radius),
        windingsurface_major_radius, windingsurface_minor_radius_r, windingsurface_minor_radius_z,
    ), "Winding surface does not enclose the plasma surface."
    assert is_enclosed_by_ellipse(
        windingsurface_major_radius, windingsurface_minor_radius_r, windingsurface_minor_radius_z,
        tf_major_radius, tf_minor_radius_r, tf_minor_radius_z,
    ), "Toroidal field coils do not enclose the winding surface."
    
    biotsavart = generate_biotsavart(
        windingsurface_major_radius=windingsurface_major_radius,
        windingsurface_minor_radius_r=windingsurface_minor_radius_r,
        windingsurface_minor_radius_z=windingsurface_minor_radius_z,
        ndipoles_pol_total=ndipoles_pol_total,
        ndipoles_tor_total=ndipoles_tor_total,
        tf_major_radius=tf_major_radius,
        tf_minor_radius_r=tf_minor_radius_r,
        tf_minor_radius_z=tf_minor_radius_z,
        ntfs_total=ntfs_total,
        curve_curve_distance=curve_curve_distance,
        nfp=nfp,
        dipole_base_currents=dipole_base_currents,
        tf_base_currents=tf_base_currents,
        skip_outboard_pol=skip_outboard_pol,
        superellipse_n=superellipse_n,
        nquadpoints=nquadpoints,
        **kwargs,
    )

    return biotsavart

def get_windingsurface_from_biotsavart(
    biotsavart: BiotSavart,
) -> SurfaceRZFourier:
    return get_windingsurface_from_coils(biotsavart.coils)

