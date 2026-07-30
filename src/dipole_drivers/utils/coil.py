import numpy as np

from numbers import Real
from typing import Sequence

from simsopt.field import Coil, Current, coils_via_symmetries
from simsopt.field.coil import ScaledCurrent
from simsopt.geo import (
    CurvePlanarFourier,
    CurvePlanarEllipticalCylindrical,
    RotatedCurve,
    SurfaceRZFourier,
    create_equally_spaced_cylindrical_curves
)

from .curve import *

__all__ = [
    "copy_coil",
    "copy_coils",
    "copy_current",
    "copy_scaledcurrent",
    "copy_tf_coils",
    "copy_dipole_coils",
    "generate_coils",
    "generate_dipole_coils",
    "generate_tf_coils",
    "get_dipole_base_coils",
    "get_dipole_base_currents",
    "get_dipole_coils",
    "get_tf_base_coils",
    "get_tf_base_currents",
    "get_tf_coils",
]

def copy_coil(coil: Coil) -> Coil:
    curve = copy_curve(coil.curve)
    current = copy_current(coil.current)
    copied_coil = Coil(curve, current)
    return copied_coil

def copy_coils(
    windingsurface: SurfaceRZFourier,
    coils: Sequence[Coil],
    curve_curve_distance: float | None = None,
    superellipse_n: int | None = None,
) -> Sequence[Coil]:

    tf_coils = get_tf_coils(coils)
    if len(tf_coils):
        copied_tf_coils = copy_tf_coils(tf_coils)
    else:
        copied_tf_coils = []

    dipole_coils = get_dipole_coils(coils)
    if len(dipole_coils):
        if (curve_curve_distance is None) or (superellipse_n is None):
            raise ValueError(
                "curve_curve_distance and superellipse_n must be provided "
                "when copying dipole coils."
            )
        copied_dipole_coils = copy_dipole_coils(
            windingsurface,
            dipole_coils,
            curve_curve_distance=curve_curve_distance,
            superellipse_n=superellipse_n,
        )
    else:
        copied_dipole_coils = []

    return list(copied_tf_coils) + list(copied_dipole_coils)

def copy_current(current: Current | ScaledCurrent) -> Current | ScaledCurrent:
    if isinstance(current, Current):
        copied_current = Current(current.get_value())
    elif isinstance(current, ScaledCurrent):
        copied_current = copy_scaledcurrent(current)
    else:
        raise ValueError(
            "current must be an instance of Current or ScaledCurrent, "
            f"got {type(current).__name__} instead."
        )
    return copied_current

def copy_scaledcurrent(scaledcurrent: ScaledCurrent) -> ScaledCurrent:
    copied_scaledcurrent = ScaledCurrent(
        scaledcurrent.current_to_scale,
        scaledcurrent.scale
    )
    return copied_scaledcurrent

def copy_tf_coils(coils: Sequence[Coil]) -> Sequence[Coil]:
    tf_coils = get_tf_coils(coils)
    nfp = get_nfp_from_curves([coil.curve for coil in tf_coils])
    ntfs_total = len(tf_coils)
    tf_base_currents = [c.get_value() for c in get_tf_base_currents(coils)]
    nquadpoints = tf_coils[0].curve.quadpoints.size

    tf_major_radius, tf_minor_radius_r, tf_minor_radius_z = \
        get_tf_geometry(tf_coils[0].curve)

    return generate_tf_coils(
        tf_major_radius,
        tf_minor_radius_r,
        tf_minor_radius_z,
        nfp,
        ntfs_total,
        tf_base_currents,
        nquadpoints
    )

def copy_dipole_coils(
    windingsurface: SurfaceRZFourier,
    coils: Sequence[Coil],
    curve_curve_distance: float,
    superellipse_n: int,
) -> Sequence[Coil]:
    dipole_coils = get_dipole_coils(coils)
    dipole_curves = [coil.curve for coil in dipole_coils]
    nfp = get_nfp_from_curves(dipole_curves)

    dipole_base_currents = [c.get_value() for c in get_dipole_base_currents(coils)]
    dipole_curve_order = dipole_curves[0].order
    nquadpoints = dipole_curves[0].quadpoints.size

    return generate_dipole_coils(
        windingsurface,
        curve_curve_distance,
        nfp,
        dipole_base_currents,
        dipole_curve_order=dipole_curve_order,
        superellipse_n=superellipse_n,
        nquadpoints=nquadpoints,
    )

def generate_coils(
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
) -> Sequence[Coil]:
    ndipoles_tor_total = windingsurface.quadpoints_phi.size
    assert ndipoles_tor_total % ntfs_total == 0, \
        "The number of dipoles in the toroidal direction must be " \
        "divisible by the number of TF coils. Got " \
        f"{ndipoles_tor_total} dipoles and {ntfs_total} TF coils."

    tf_coils = generate_tf_coils(
        tf_major_radius,
        tf_minor_radius_r,
        tf_minor_radius_z,
        nfp,
        ntfs_total,
        tf_base_currents,
        nquadpoints
    )

    dipole_coils = generate_dipole_coils(
        windingsurface,
        curve_curve_distance,
        nfp,
        dipole_base_currents,
        dipole_curve_order=dipole_curve_order,
        superellipse_n=superellipse_n,
        nquadpoints=nquadpoints,
    )

    return list(tf_coils) + list(dipole_coils)

def generate_dipole_coils(
    windingsurface: SurfaceRZFourier,
    curve_curve_distance: float,
    nfp: int,
    dipole_base_currents: float | Sequence[float],
    *,
    dipole_curve_order: int = 12,
    superellipse_n: int = 4,
    nquadpoints: int = 128,
) -> Sequence[Coil]:
    dipole_base_curves = generate_dipole_base_curves(
        windingsurface,
        curve_curve_distance,
        nfp,
        dipole_curve_order=dipole_curve_order,
        superellipse_n=superellipse_n,
        nquadpoints=nquadpoints
    )

    if isinstance(dipole_base_currents, Real):
        dipole_base_currents = [float(dipole_base_currents)]*len(dipole_base_curves)
    elif isinstance(dipole_base_currents, Sequence):
        assert len(dipole_base_currents) == len(dipole_base_curves), \
            "Number of dipole currents must match the number of " \
            "dipole coils in a half period."
    else:
        raise ValueError(
            "dipole_base_currents must be a float or a list of floats, "
            f"got {type(dipole_base_currents).__name__} instead."
        )

    dipole_base_currents = [ScaledCurrent(Current(1.0), current) for current in dipole_base_currents]
    dipole_coils = coils_via_symmetries(
        dipole_base_curves, dipole_base_currents, nfp, True
    )

    return dipole_coils

def generate_tf_coils(
    tf_major_radius: float,
    tf_minor_radius_r: float,
    tf_minor_radius_z: float,
    nfp: int,
    ntfs_total: int,
    tf_base_currents: float | Sequence[float],
    nquadpoints: int = 128,
) -> Sequence[Coil]:
    assert ntfs_total % (2*nfp) == 0, \
        "The number of TF coils must be divisible by 2*nfp. Got "\
        f"{ntfs_total} TF coils and {nfp} field periods."
    ntfs_per_half = ntfs_total // (2*nfp)
    if isinstance(tf_base_currents, Real):
        tf_base_currents = [float(tf_base_currents)]*ntfs_per_half
    elif isinstance(tf_base_currents, Sequence):
        assert len(tf_base_currents) == ntfs_per_half, \
            "Number of TF currents must match the number of TF coils " \
            "in one half period."
    else:
        raise ValueError(
            "tf_base_currents must be a float or a list of floats, "
            f"got {type(tf_base_currents).__name__} instead."
        )
        
    base_tf_curves = create_equally_spaced_cylindrical_curves(
        ntfs_per_half,
        nfp,
        True,
        tf_major_radius,
        tf_minor_radius_r,
        tf_minor_radius_z,
        nquadpoints
    )
    for curve in base_tf_curves:
        for key in ["phi", "Z0", "phi_rotation", "z_rotation"]:
            curve.fix(key)
        for key in ["R0", "r_rotation"]:
            curve.unfix(key)

    tf_base_currents = [ScaledCurrent(Current(1.0), current) for current in tf_base_currents]
    tf_coils = coils_via_symmetries(
        base_tf_curves, tf_base_currents, nfp, True)
    
    return tf_coils

def get_dipole_base_coils(coils: Sequence[Coil]) -> Sequence[Coil]:
    dipole_base_coils = []
    for coil in coils:
        if isinstance(coil.curve, CurvePlanarFourier):
            dipole_base_coils.append(coil)
    return dipole_base_coils

def get_dipole_base_currents(coils: Sequence[Coil]) -> list[Current | ScaledCurrent]:
    dipole_base_coils = get_dipole_base_coils(coils)
    dipole_base_currents = [coil.current for coil in dipole_base_coils]
    return dipole_base_currents

def get_dipole_coils(coils: Sequence[Coil]) -> Sequence[Coil]:
    dipole_coils = []
    for coil in coils:
        curve = coil.curve
        while isinstance(curve, RotatedCurve):
            curve = curve.curve
        if isinstance(curve, CurvePlanarFourier):
            dipole_coils.append(coil)
    return dipole_coils

def get_tf_base_coils(coils: Sequence[Coil]) -> Sequence[Coil]:
    tf_base_coils = []
    for coil in coils:
        if isinstance(coil.curve, CurvePlanarEllipticalCylindrical):
            tf_base_coils.append(coil)
    return tf_base_coils

def get_tf_base_currents(coils: Sequence[Coil]) -> list[Current | ScaledCurrent]:
    tf_base_coils = get_tf_base_coils(coils)
    tf_base_currents = [coil.current for coil in tf_base_coils]
    return tf_base_currents

def get_tf_coils(coils: Sequence[Coil]) -> Sequence[Coil]:
    tf_coils = []
    for coil in coils:
        curve = coil.curve
        while isinstance(curve, RotatedCurve):
            curve = curve.curve
        if isinstance(curve, CurvePlanarEllipticalCylindrical):
            tf_coils.append(coil)
    return tf_coils
