import numpy as np

from numbers import Real
from typing import Sequence

from simsopt.field import (
    Coil,
    Current,
    apply_symmetries_to_currents
)
from simsopt.field.coil import ScaledCurrent
from simsopt.geo import (
    CurvePlanarEllipticalCylindrical,
    CurvePlanarOnSurface,
    RotatedCurve,
    SurfaceRZFourier,
)

from ..defaults import *
from .curve import *
from .check import *
from .windingsurface import copy_windingsurface

__all__ = [
    "copy_coils",
    "copy_dipole_coils",
    "copy_dipole_currents_from_coils",
    "copy_tf_coils",
    "generate_coils",
    "generate_dipole_coils",
    "generate_tf_coils",
    "generate_dipole_currents",
    "get_current_grid",
    "get_current_index_table",
    "get_dipole_base_coils",
    "get_dipole_base_currents",
    "get_dipole_coils",
    "get_index_table",
    "get_nfp_from_coils",
    "get_nfp_from_currents",
    "get_nfp_from_tf_coils",
    "get_tf_coils",
    "get_tf_base_coils",
    "get_tf_base_currents",
    "get_windingsurface_from_coils",
]

def copy_coils(
    coils: Sequence[Coil],
    **kwargs,
) -> Sequence[Coil]:
    dipole_coils = get_dipole_coils(coils)
    tf_coils = get_tf_coils(coils)

    copied_dipole_coils = copy_dipole_coils(dipole_coils, **kwargs)
    copied_tf_coils = copy_tf_coils(tf_coils, **kwargs)

    return list(copied_tf_coils) + list(copied_dipole_coils)

def copy_dipole_coils(
    coils: Sequence[Coil],
    **kwargs,
) -> Sequence[Coil]:
    dipole_coils = get_dipole_coils(coils)
    if not dipole_coils:
        return []
    
    dipole_curves = [coil.curve for coil in dipole_coils]
    first_curve = dipole_curves[0] # CurvePlanarOnSurface
    shape = first_curve.curve # ScaledCurveSuperEllipse
    curve = shape.curve_to_scale # CurveSuperEllipse
    windingsurface = copy_windingsurface(first_curve.surface)

    curve_curve_distance = get_dipole_curve_curve_distance(dipole_curves)
    nfp = get_nfp_from_coils(dipole_coils)
    dipole_base_currents = get_dipole_base_currents(dipole_coils)
    skip_outboard_pol = is_missing_outboard(dipole_curves)
    superellipse_n = float(curve.get("n"))
    nquadpoints = curve.quadpoints.size

    copied_dipole_coils = generate_dipole_coils(
        windingsurface=windingsurface,
        curve_curve_distance=curve_curve_distance,
        nfp=nfp,
        dipole_base_currents=dipole_base_currents,
        skip_outboard_pol=skip_outboard_pol,
        superellipse_n=superellipse_n,
        nquadpoints=nquadpoints,
        **kwargs,
    )
    return copied_dipole_coils

def copy_dipole_currents_from_coils(
    coils: Sequence[Coil],
    **kwargs,
) -> Sequence[Current | ScaledCurrent]:
    dipole_coils = get_dipole_coils(coils)
    if not dipole_coils:
        return []

    dipole_base_currents = get_dipole_base_currents(dipole_coils)

    grid = get_current_grid(dipole_coils)
    ndipoles_tor_total, ndipoles_pol_total = grid.shape

    nfp = get_nfp_from_coils(dipole_coils)

    copied_currents = generate_dipole_currents(
        dipole_base_currents=dipole_base_currents,
        ndipoles_tor_total=ndipoles_tor_total,
        ndipoles_pol_total=ndipoles_pol_total,
        nfp=nfp,
        **kwargs,
    )
    return copied_currents

def copy_tf_coils(
    coils: Sequence[Coil],
    **kwargs,
) -> Sequence[Coil]:
    coils = get_tf_coils(coils)
    if not coils:
        return []
    curve = coils[0].curve

    tf_major_radius = float(curve.get("R0"))
    tf_minor_radius_r = curve.a
    tf_minor_radius_z = curve.b
    nfp = get_nfp_from_tf_coils(coils)
    ntfs_total = len(coils)
    tf_base_currents = get_tf_base_currents(coils)
    nquadpoints = curve.quadpoints.size

    copied_tf_coils = generate_tf_coils(
        tf_major_radius=tf_major_radius,
        tf_minor_radius_r=tf_minor_radius_r,
        tf_minor_radius_z=tf_minor_radius_z,
        nfp=nfp,
        ntfs_total=ntfs_total,
        tf_base_currents=tf_base_currents,
        nquadpoints=nquadpoints,
        **kwargs,
    )
    return copied_tf_coils

def generate_coils(
    *,
    windingsurface: SurfaceRZFourier = dummy_surface,
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
) -> Sequence[Coil]:
    check_required_inputs(
        windingsurface=(is_real_surfacerzfourier, windingsurface),
        tf_major_radius=(is_positive_scalar, tf_major_radius),
        tf_minor_radius_r=(is_positive_scalar, tf_minor_radius_r),
        tf_minor_radius_z=(is_positive_scalar, tf_minor_radius_z),
        ntfs_total=(is_positive_scalar, ntfs_total),
        curve_curve_distance=(is_positive_scalar, curve_curve_distance),
        nfp=(is_positive_scalar, nfp),
        dipole_base_currents=(is_positive_scalar_or_not_empty_sequence, dipole_base_currents),
        tf_base_currents=(is_positive_scalar_or_not_empty_sequence, tf_base_currents),
    )
    ndipoles_tor_total = windingsurface.quadpoints_phi.size
    assert ndipoles_tor_total % ntfs_total == 0, \
        "The number of dipoles in the toroidal direction must be " \
        "divisible by the number of TF coils. Got " \
        f"{ndipoles_tor_total} dipoles and {ntfs_total} TF coils."

    tf_coils = generate_tf_coils(
        tf_major_radius=tf_major_radius,
        tf_minor_radius_r=tf_minor_radius_r,
        tf_minor_radius_z=tf_minor_radius_z,
        nfp=nfp,
        ntfs_total=ntfs_total,
        tf_base_currents=tf_base_currents,
        nquadpoints=nquadpoints,
        **kwargs,
    )

    dipole_coils = generate_dipole_coils(
        windingsurface=windingsurface,
        curve_curve_distance=curve_curve_distance,
        nfp=nfp,
        dipole_base_currents=dipole_base_currents,
        skip_outboard_pol=skip_outboard_pol,
        superellipse_n=superellipse_n,
        nquadpoints=nquadpoints,
        **kwargs,
    )

    return list(tf_coils) + list(dipole_coils)

def generate_dipole_coils(
    *,
    windingsurface: SurfaceRZFourier = dummy_surface,
    curve_curve_distance: float = DEFAULT_CURVE_CURVE_DISTANCE,
    nfp: int = 0,
    dipole_base_currents: float | Sequence[float] = DEFAULT_DIPOLE_BASE_CURRENTS,
    skip_outboard_pol: bool = DEFAULT_SKIP_OUTBOARD_POL,
    superellipse_n: float = DEFAULT_SUPERELLIPSE_N,
    nquadpoints: int = DEFAULT_NQUADPOINTS,
    **kwargs,
) -> Sequence[Coil]:
    dipole_curves = generate_dipole_curves(
        windingsurface=windingsurface,
        curve_curve_distance=curve_curve_distance,
        skip_outboard_pol=skip_outboard_pol,
        superellipse_n=superellipse_n,
        nquadpoints=nquadpoints,
        **kwargs,
    )

    ndipoles_tor_total = windingsurface.quadpoints_phi.size
    ndipoles_pol_total = windingsurface.quadpoints_theta.size
    dipole_currents = generate_dipole_currents(
        dipole_base_currents=dipole_base_currents,
        ndipoles_tor_total=ndipoles_tor_total,
        ndipoles_pol_total=ndipoles_pol_total,
        nfp=nfp,
        **kwargs,
    )

    curve_table = get_dipole_index_table(dipole_curves)
    current_table = get_current_index_table(
        ndipoles_tor_total, ndipoles_pol_total, nfp)
    dipole_coils = [None]*len(dipole_curves)
    for j in range(ndipoles_tor_total):
        for i in range(ndipoles_pol_total):
            c = curve_table[j, i]
            if c >= 0:
                dipole_coils[c] = Coil(dipole_curves[c],
                                       dipole_currents[current_table[j, i]])

                
    assert all(coil is not None for coil in dipole_coils), \
        "Some dipole coils were not assigned a current."
    
    return dipole_coils # pyright: ignore[reportReturnType]

def generate_dipole_currents(
    *,
    dipole_base_currents: float | Sequence[float] = DEFAULT_DIPOLE_BASE_CURRENTS,
    ndipoles_tor_total: int = DEFAULT_NDIPOLES_TOR_TOTAL,
    ndipoles_pol_total: int = DEFAULT_NDIPOLES_POL_TOTAL,
    nfp: int = 0,
    **kwargs,
):
    check_required_inputs(
        dipole_base_currents=(is_positive_scalar_or_not_empty_sequence, dipole_base_currents),
        ndipoles_tor_total=(is_positive_scalar, ndipoles_tor_total),
        ndipoles_pol_total=(is_positive_scalar, ndipoles_pol_total),
        nfp=(is_positive_scalar, nfp),
    )
    assert ndipoles_tor_total % (2*nfp) == 0, (
        "The number of dipoles in the toroidal direction must be "
        "divisible by 2*nfp. Got "
        f"{ndipoles_tor_total} dipoles and {nfp} field periods."
    )
    ndipoles_tor_per_half = ndipoles_tor_total // (2*nfp)

    base_index = [(j, i) for j in range(ndipoles_tor_per_half)
                  for i in range(ndipoles_pol_total)]
    if isinstance(dipole_base_currents, Real):
        base_dipole_currents = \
            [ScaledCurrent(Current(1.0), dipole_base_currents)
             for j, i in base_index]
    elif isinstance(dipole_base_currents, Sequence):
        if len(dipole_base_currents) == 1:
            dipole_base_currents = [float(dipole_base_currents[0])]*len(base_index)
        assert len(dipole_base_currents) == len(base_index), \
            "Number of dipole currents must match the number of dipoles " \
            "in one half period."
        base_dipole_currents = \
            [ScaledCurrent(Current(1.0), current)
             for current in dipole_base_currents]
    else:
        raise ValueError(
            "dipole_base_currents must be a float or a list of floats, "
            f"got {type(dipole_base_currents).__name__} instead."
        )
    
    dipole_currents = \
        apply_symmetries_to_currents(base_dipole_currents, nfp, True)
    return dipole_currents

def generate_tf_coils(
    *,
    tf_major_radius: float = 0.0,
    tf_minor_radius_r: float = 0.0,
    tf_minor_radius_z: float = 0.0,
    nfp: int = 0,
    ntfs_total: int = DEFAULT_NTFS_TOTAL,
    tf_base_currents: float | Sequence[float] = DEFAULT_TF_BASE_CURRENTS,
    nquadpoints: int = DEFAULT_NQUADPOINTS,
    **kwargs,
) -> Sequence[Coil]:
    check_required_inputs( # redundant if being called by generate_coils, but useful if called directly
        tf_major_radius=(is_positive_scalar, tf_major_radius),
        tf_minor_radius_r=(is_positive_scalar, tf_minor_radius_r),
        tf_minor_radius_z=(is_positive_scalar, tf_minor_radius_z),
        nfp=(is_positive_scalar, nfp),
        ntfs_total=(is_positive_scalar, ntfs_total),
        tf_base_currents=(is_positive_scalar_or_not_empty_sequence, tf_base_currents),
    )

    assert ntfs_total % (2*nfp) == 0, (
        "The number of TF coils must be divisible by 2*nfp. Got "
        f"{ntfs_total} TF coils and {nfp} field periods."
    )
    ntfs_per_half = ntfs_total // (2*nfp)
    if isinstance(tf_base_currents, Real):
        tf_base_currents = [float(tf_base_currents)]*ntfs_per_half
    elif isinstance(tf_base_currents, Sequence):
        if len(tf_base_currents) == 1:
            tf_base_currents = [float(tf_base_currents[0])]*ntfs_per_half
        assert len(tf_base_currents) == ntfs_per_half, \
            "Number of TF currents must match the number of TF coils " \
            "in one half period."
    else:
        raise ValueError(
            "tf_base_currents must be a float or a list of floats, "
            f"got {type(tf_base_currents).__name__} instead."
        )
        
    tf_curves = generate_tf_curves(
        tf_major_radius=tf_major_radius,
        tf_minor_radius_r=tf_minor_radius_r,
        tf_minor_radius_z=tf_minor_radius_z,
        ntfs_per_half=ntfs_per_half,
        nfp=nfp,
        nquadpoints=nquadpoints,
        **kwargs,
    )

    tf_base_currents = \
        [ScaledCurrent(Current(1.0), current) for current in tf_base_currents]
    tf_currents = apply_symmetries_to_currents(tf_base_currents, nfp, True)
    
    tf_coils = [
        Coil(curve, current) for curve, current
        in zip(tf_curves, tf_currents)]
    return tf_coils

def get_current_grid(
    coils: Sequence[Coil]
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    dipole_coils = get_dipole_coils(coils)
    table = get_index_table(dipole_coils)
    grid = np.full(table.shape, np.nan)
    filled = table >= 0
    grid[filled] = [dipole_coils[c].current.get_value() for c in table[filled]] # pyright: ignore[reportAttributeAccessIssue]
    return grid

def get_current_index_table(ndipoles_tor_total, ndipoles_pol_total, nfp):
    nphi, ntheta = ndipoles_tor_total, ndipoles_pol_total
    dphi, m = nphi//nfp, 0
    table = np.full((nphi, ntheta), -1, dtype=int)
    for k in range(nfp):
        for flip in (False, True):
            for j in range(nphi//(2*nfp)):
                for i in range(ntheta):
                    table[((nphi-1-j-k*dphi) % nphi, (-i) % ntheta) if flip
                          else ((j+k*dphi) % nphi, i)] = m
                    m += 1
    return table

def get_dipole_base_coils(
    coils: Sequence[Coil],
) -> Sequence[Coil]:
    dipole_coils = get_dipole_coils(coils)
    table = get_index_table(dipole_coils)
    nfp = get_nfp_from_coils(dipole_coils)
    ntor_half = table.shape[0] // (2*nfp)
    return [dipole_coils[table[j, i]] if table[j, i] >= 0 else None
            for j in range(ntor_half) for i in range(table.shape[1])] # pyright: ignore[reportReturnType]

def get_dipole_base_currents(coils):
    dipole_coils = get_dipole_coils(coils)
    table = get_index_table(dipole_coils)
    nfp = get_nfp_from_coils(dipole_coils)
    ntor_half = table.shape[0]//(2*nfp)
    return [dipole_coils[table[j, i]].current.get_value() if table[j, i] >= 0 else 0.0 # pyright: ignore[reportAttributeAccessIssue]
            for j in range(ntor_half) for i in range(table.shape[1])]

def get_dipole_coils(
    coils: Sequence[Coil]
) -> Sequence[Coil]:
    dipole_coils = []
    for coil in coils:
        curve = coil.curve
        while isinstance(curve, RotatedCurve):
            curve = curve.curve
        if isinstance(curve, CurvePlanarOnSurface):
            dipole_coils.append(coil)
    return dipole_coils

def get_index_table(
    coils: Sequence[Coil]
) -> np.ndarray[tuple[int, int], np.dtype[np.int64]]:
    return get_dipole_index_table([coil.curve for coil in get_dipole_coils(coils)])

def get_nfp_from_coils(
    coils: Sequence[Coil]
) -> int:
    dipole_coils = get_dipole_coils(coils)
    if len(dipole_coils):
        dipole_currents = [coil.current for coil in dipole_coils]
        nfp = get_nfp_from_currents(dipole_currents)
    else:
        nfp = get_nfp_from_tf_coils(coils)
    return nfp

def get_nfp_from_currents(
    currents: Sequence[Current | ScaledCurrent]
) -> int:
    n_currents = len(currents)
    n_base = len({n for c in currents for n in c.full_dof_names})
    assert n_currents % (2*n_base) == 0, \
        "The number of currents must be divisible by 2*n_base. Got " \
        f"{n_currents} currents and {n_base} base currents."
    nfp = n_currents // (2*n_base)
    return nfp

def get_nfp_from_tf_coils(
    coils: Sequence[Coil]
) -> int:
    tf_coils = get_tf_coils(coils)
    tf_curves = [coil.curve for coil in tf_coils]
    nfp = get_nfp_from_tf_curves(tf_curves)
    return nfp

def get_tf_coils(
    coils: Sequence[Coil]
) -> Sequence[Coil]:
    tf_coils = []
    for coil in coils:
        curve = coil.curve
        while isinstance(curve, RotatedCurve):
            curve = curve.curve
        if isinstance(curve, CurvePlanarEllipticalCylindrical):
            tf_coils.append(coil)
    return tf_coils

def get_tf_base_coils(
    coils: Sequence[Coil],
) -> Sequence[Coil]:
    tf_coils = get_tf_coils(coils)
    tf_base_coils = []
    for coil in tf_coils:
        curve = coil.curve
        if isinstance(curve, CurvePlanarEllipticalCylindrical):
            tf_base_coils.append(coil)
    return tf_base_coils

def get_tf_base_currents(
    coils: Sequence[Coil]
) -> Sequence[Current]:
    tf_coils = get_tf_coils(coils)
    tf_base_currents = [
        coil.current.get_value() for coil in tf_coils
        if isinstance(coil.curve, CurvePlanarEllipticalCylindrical)
    ]
    return tf_base_currents

def get_windingsurface_from_coils(
    coils: Sequence[Coil]
) -> SurfaceRZFourier:
    curves = [coil.curve for coil in get_dipole_coils(coils)]
    return get_windingsurface_from_curves(curves)
