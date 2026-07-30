from typing import Sequence

from simsopt.field import BiotSavart
from simsopt.geo import SurfaceRZFourier

from .coil import copy_coils, generate_coils

__all__ = [
    "copy_biotsavart",
    "generate_biotsavart",
]

def copy_biotsavart(
    windingsurface: SurfaceRZFourier,
    biotsavart: BiotSavart,
    curve_curve_distance: float | None = None,
    superellipse_n: int | None = None,
) -> BiotSavart:
    coils = biotsavart.coils
    copied_coils = copy_coils(
        windingsurface,
        coils,
        curve_curve_distance=curve_curve_distance,
        superellipse_n=superellipse_n,
    )
    return BiotSavart(copied_coils)

def generate_biotsavart(
    windingsurface: SurfaceRZFourier,
    tf_major_radius: float,
    tf_minor_radius_r: float,
    tf_minor_radius_z: float,
    ntfs_total: int,
    curve_curve_distance: float,
    nfp: int,
    dipole_base_currents: float | Sequence[float],
    tf_base_currents: float | Sequence[float],
    *,
    dipole_curve_order: int = 12,
    superellipse_n: int = 4,
    nquadpoints: int = 128,
) -> BiotSavart:
    coils = generate_coils(
        windingsurface,
        tf_major_radius,
        tf_minor_radius_r,
        tf_minor_radius_z,
        ntfs_total,
        curve_curve_distance,
        nfp,
        dipole_base_currents,
        tf_base_currents,
        dipole_curve_order=dipole_curve_order,
        superellipse_n=superellipse_n,
        nquadpoints=nquadpoints,
    )
    return BiotSavart(coils)
