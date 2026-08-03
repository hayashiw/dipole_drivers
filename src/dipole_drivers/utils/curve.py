import numpy as np

from typing import Sequence

from simsopt.field import apply_symmetries_to_curves
from simsopt.geo import (
    CurveSuperEllipse,
    CurvePlanarEllipticalCylindrical,
    CurvePlanarOnSurface,
    RotatedCurve,
    ScaledCurveSuperEllipse,
    SurfaceRZFourier,
    create_equally_spaced_cylindrical_curves,
)

from ..defaults import *
from .check import *
from .ellipse import calculate_total_arc_length
from .windingsurface import copy_windingsurface, get_windingsurface_geometry

__all__ = [
    "copy_dipole_curves",
    "copy_tf_curves",
    "generate_dipole_base_curves",
    "generate_dipole_curves",
    "generate_tf_curves",
    "get_base_curves",
    "get_dipole_index_table",
    "get_dipole_base_curves",
    "get_dipole_curves",
    "get_dipole_curve_curve_distance",
    "get_tf_base_curves",
    "get_tf_curves",
    "get_nfp_from_tf_curves",
    "get_windingsurface_from_curves",
    "is_missing_outboard",
]

def copy_dipole_curves(
    dipole_curves: Sequence[CurvePlanarOnSurface],
    **kwargs,
) -> Sequence[CurvePlanarOnSurface | RotatedCurve]:
    first_curve = dipole_curves[0] # CurvePlanarOnSurface
    shape = first_curve.curve # ScaledCurveSuperEllipse
    curve = shape.curve_to_scale # CurveSuperEllipse
    superellipse_n = float(curve.get("n"))
    nquadpoints = curve.quadpoints.size

    windingsurface = copy_windingsurface(first_curve.surface)
    skip_outboard_pol = is_missing_outboard(dipole_curves)
    curve_curve_distance = get_dipole_curve_curve_distance(dipole_curves)

    copied_dipole_curves = generate_dipole_curves(
        windingsurface=windingsurface,
        curve_curve_distance=curve_curve_distance,
        skip_outboard_pol=skip_outboard_pol,
        superellipse_n=superellipse_n,
        nquadpoints=nquadpoints,
        **kwargs,
    )
    return copied_dipole_curves

def copy_tf_curves(
    tf_curves: Sequence[CurvePlanarEllipticalCylindrical],
    **kwargs,
) -> Sequence[CurvePlanarEllipticalCylindrical | RotatedCurve]:
    curve = tf_curves[0]
    tf_major_radius = float(curve.get("R0"))
    tf_minor_radius_r = curve.a
    tf_minor_radius_z = curve.b
    nfp = get_nfp_from_tf_curves(tf_curves)
    ntfs_total = len(tf_curves)
    ntfs_per_half = ntfs_total // (2 * nfp)
    nquadpoints = curve.quadpoints.size
    
    copied_tf_curves = generate_tf_curves(
        tf_major_radius=tf_major_radius,
        tf_minor_radius_r=tf_minor_radius_r,
        tf_minor_radius_z=tf_minor_radius_z,
        ntfs_per_half=ntfs_per_half,
        nfp=nfp,
        nquadpoints=nquadpoints,
        **kwargs,
    )
    return copied_tf_curves
        
def generate_dipole_base_curves(
    *,
    windingsurface: SurfaceRZFourier = dummy_surface,
    curve_curve_distance: float = DEFAULT_CURVE_CURVE_DISTANCE,
    skip_outboard_pol: bool = DEFAULT_SKIP_OUTBOARD_POL,
    superellipse_n: float = DEFAULT_SUPERELLIPSE_N,
    nquadpoints: int = DEFAULT_NQUADPOINTS,
    **kwargs,
) -> Sequence[CurvePlanarOnSurface | RotatedCurve]:
    check_required_inputs(
        windingsurface=(is_real_surfacerzfourier, windingsurface),
        curve_curve_distance=(is_positive_scalar, curve_curve_distance),
    )
    windingsurface_major_radius, \
    windingsurface_minor_radius_r, \
    windingsurface_minor_radius_z = \
        get_windingsurface_geometry(windingsurface)

    ndipoles_tor_total, ndipoles_pol_total, _ = \
        windingsurface.gamma().shape
    
    is_even_npol = (ndipoles_pol_total % 2 == 0)

    thetas = windingsurface.quadpoints_theta*2*np.pi
    assert np.all(np.diff(thetas) > 0), \
        "Thetas must be in ascending order."
    theta_pi_idx = \
        ndipoles_pol_total//2 if is_even_npol else (ndipoles_pol_total-1)//2
    use_thetas = np.array([i for i, _ in enumerate(thetas[:theta_pi_idx+1])])
    symm_thetas = use_thetas.copy()[1:]
    if skip_outboard_pol:
        use_thetas = use_thetas[1:]
    if is_even_npol:
        symm_thetas = symm_thetas[:-1]

    poloidal_circumference = calculate_total_arc_length(
        windingsurface_minor_radius_r,
        windingsurface_minor_radius_z
    )
    poloidal_arc_length_per_coil = poloidal_circumference / ndipoles_pol_total
    dipole_minor_radius_theta = \
        (poloidal_arc_length_per_coil - curve_curve_distance)/2

    dipole_base_curves = []
    reference_curve = None
    for ith in use_thetas:
        major_radius = \
            windingsurface_major_radius + \
            windingsurface_minor_radius_r*np.cos(thetas[ith])
        toroidal_circumference = 2*np.pi*major_radius
        toroidal_circumference_per_coil = \
            toroidal_circumference / ndipoles_tor_total
        dipole_minor_radius_phi = \
            (toroidal_circumference_per_coil - curve_curve_distance)/2

        if reference_curve is None:
            reference_curve = CurveSuperEllipse(
                nquadpoints,
                dipole_minor_radius_theta,
                dipole_minor_radius_phi,
                superellipse_n,
            )

        shape = ScaledCurveSuperEllipse(
            reference_curve,
            shared_dofs=kwargs.get("dipole_shared_dofs", ()),
            a=dipole_minor_radius_theta,
            b=dipole_minor_radius_phi,
            n=superellipse_n)
        curve = CurvePlanarOnSurface(
            shape, windingsurface, 0, ith, align="theta")
        dipole_base_curves.append(curve)

    mirror_thetas = {int(ith) for ith in symm_thetas}
    symm_curves = [curve for curve in dipole_base_curves
                   if curve.itheta in mirror_thetas]
    dipole_base_curves += apply_symmetries_to_curves(
        symm_curves, 1, True)[len(symm_curves):]

    return dipole_base_curves

def generate_dipole_curves(
    *,
    windingsurface: SurfaceRZFourier = dummy_surface,
    curve_curve_distance: float = DEFAULT_CURVE_CURVE_DISTANCE,
    skip_outboard_pol: bool = DEFAULT_SKIP_OUTBOARD_POL,
    superellipse_n: float = DEFAULT_SUPERELLIPSE_N,
    nquadpoints: int = DEFAULT_NQUADPOINTS,
    **kwargs,
) -> Sequence[CurvePlanarOnSurface | RotatedCurve]:
    dipole_base_curves = generate_dipole_base_curves(
        windingsurface=windingsurface,
        curve_curve_distance=curve_curve_distance,
        skip_outboard_pol=skip_outboard_pol,
        superellipse_n=superellipse_n,
        nquadpoints=nquadpoints,
        **kwargs,
    )
    ndipoles_tor_total = windingsurface.quadpoints_phi.size
    dipole_curves = apply_symmetries_to_curves(
        dipole_base_curves, ndipoles_tor_total, False)

    return dipole_curves

def generate_tf_curves(
    *,
    tf_major_radius: float = 0.0,
    tf_minor_radius_r: float = 0.0,
    tf_minor_radius_z: float = 0.0,
    ntfs_per_half: int = 0,
    nfp: int = 0,
    nquadpoints: int = DEFAULT_NQUADPOINTS,
    **kwargs,
) -> Sequence[CurvePlanarEllipticalCylindrical | RotatedCurve]:
    if nfp > 0 and ntfs_per_half <= 0: # If nfp is given and ntfs_per_half is not
        ntfs_per_half = DEFAULT_NTFS_TOTAL // (2 * nfp)
    check_required_inputs(
        tf_major_radius=(is_positive_scalar, tf_major_radius),
        tf_minor_radius_r=(is_positive_scalar, tf_minor_radius_r),
        tf_minor_radius_z=(is_positive_scalar, tf_minor_radius_z),
        ntfs_per_half=(is_positive_scalar, ntfs_per_half),
        nfp=(is_positive_scalar, nfp),
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

    tf_curves = apply_symmetries_to_curves(base_tf_curves, nfp, True)
    
    return tf_curves

def get_base_curves(
    curves: Sequence[CurvePlanarOnSurface | CurvePlanarEllipticalCylindrical | RotatedCurve]
) -> Sequence[CurveSuperEllipse | CurvePlanarEllipticalCylindrical]:
    tf_base_curves = get_tf_base_curves(curves)
    dipole_base_curves = get_dipole_base_curves(curves)
    base_curves = list(tf_base_curves) + list(dipole_base_curves)
    return base_curves

def _quadpoint_lookup(quadpoints, tol=1e-9):
    q = np.mod(np.asarray(quadpoints, dtype=float), 1.0)
    def find(value):
        d = np.abs(np.mod(value, 1.0) - q)
        k = int(np.argmin(np.minimum(d, 1.0 - d)))
        if min(d[k], 1.0 - d[k]) > tol:
            raise ValueError(f"symmetry image {value} is not a quadpoint of the grid.")
        return k
    return find

def _placement(curve, find_phi, find_theta, phis, thetas):
    rotations = []
    while isinstance(curve, RotatedCurve):
        rotations.append(curve)
        curve = curve.curve
    if not isinstance(curve, CurvePlanarOnSurface):
        raise TypeError(f"expected a CurvePlanarOnSurface under the RotatedCurve "
                        f"wrappers, found {type(curve).__name__}.")
    iphi, itheta = curve.iphi, curve.itheta
    for rot in reversed(rotations):
        dphi = rot._phi/(2*np.pi)
        if rot.rotmat[2, 2] < 0:
            iphi = find_phi(-(phis[iphi] + dphi))
            itheta = find_theta(-thetas[itheta])
        else:
            iphi = find_phi(phis[iphi] + dphi)
    return iphi, itheta

def get_dipole_index_table(
    dipole_curves: Sequence[CurvePlanarOnSurface | RotatedCurve]
) -> np.ndarray[tuple[int, int], np.dtype[np.int64]]:
    base = dipole_curves[0]
    while isinstance(base, RotatedCurve):
        base = base.curve
    surface = base.surface
    phis, thetas = surface.quadpoints_phi, surface.quadpoints_theta
    find_phi, find_theta = _quadpoint_lookup(phis), _quadpoint_lookup(thetas)
    table = np.full((phis.size, thetas.size), -1, dtype=int)
    for c, curve in enumerate(dipole_curves):
        i, j = _placement(curve, find_phi, find_theta, phis, thetas)
        if table[i, j] >= 0:
            raise ValueError(
                f"curves {table[i, j]} and {c} share quadpoint ({i}, {j}).")
        table[i, j] = c
    return table

def get_dipole_base_curves(
    curves: Sequence[CurvePlanarOnSurface | CurvePlanarEllipticalCylindrical | RotatedCurve]
) -> Sequence[CurvePlanarOnSurface]:
    dipole_curves = get_dipole_curves(curves)
    seen, dipole_base_curves = set(), []
    for curve in dipole_curves:
        while isinstance(curve, RotatedCurve):
            curve = curve.curve
        if not isinstance(curve, CurvePlanarOnSurface):
            raise TypeError(f"expected a CurvePlanarOnSurface under the "
                            f"RotatedCurve wrappers, found "
                            f"{type(curve).__name__}.")
        if id(curve) not in seen:
            seen.add(id(curve))
            dipole_base_curves.append(curve)
    return dipole_base_curves

def get_dipole_curves(
    curves: Sequence[CurvePlanarOnSurface | CurvePlanarEllipticalCylindrical | RotatedCurve]
) -> Sequence[CurvePlanarOnSurface | RotatedCurve]:
    dipole_curves = []
    for curve in curves:
        check_curve = curve
        while isinstance(check_curve, RotatedCurve):
            check_curve = check_curve.curve
        if isinstance(check_curve, CurvePlanarOnSurface):
            dipole_curves.append(curve)
    return dipole_curves

def get_dipole_curve_curve_distance(
    dipole_curves: Sequence[CurvePlanarOnSurface | RotatedCurve]
) -> float:
    first_curve = dipole_curves[0] # CurvePlanarOnSurface
    shape = first_curve.curve # ScaledCurveSuperEllipse
    curve = shape.curve_to_scale # CurveSuperEllipse
    dipole_minor_radius_theta = float(curve.get("a"))

    windingsurface = first_curve.surface
    _, windingsurface_minor_radius_r, windingsurface_minor_radius_z = \
        get_windingsurface_geometry(windingsurface)
    thetas = windingsurface.quadpoints_theta*2*np.pi
    
    poloidal_circumference = calculate_total_arc_length(
        windingsurface_minor_radius_r,
        windingsurface_minor_radius_z
    )
    ndipoles_pol_total = thetas.size
    poloidal_arc_length_per_coil = \
        poloidal_circumference / ndipoles_pol_total
    curve_curve_distance = \
        poloidal_arc_length_per_coil - 2*dipole_minor_radius_theta
    return curve_curve_distance

def get_tf_base_curves(
    curves: Sequence[CurvePlanarOnSurface | CurvePlanarEllipticalCylindrical | RotatedCurve]
) -> Sequence[CurvePlanarEllipticalCylindrical]:
    tf_curves = get_tf_curves(curves)
    seen, tf_base_curves = set(), []
    for curve in tf_curves:
        while isinstance(curve, RotatedCurve):
            curve = curve.curve
        if not isinstance(curve, CurvePlanarEllipticalCylindrical):
            raise TypeError(f"expected a CurvePlanarEllipticalCylindrical "
                            f"under the RotatedCurve wrappers, found "
                            f"{type(curve).__name__}.")
        if id(curve) not in seen:
            seen.add(id(curve))
            tf_base_curves.append(curve)
    return tf_base_curves

def get_tf_curves(
    curves: Sequence[CurvePlanarOnSurface | CurvePlanarEllipticalCylindrical | RotatedCurve]
) -> Sequence[CurvePlanarEllipticalCylindrical | RotatedCurve]:
    tf_curves = []
    for curve in curves:
        check_curve = curve
        while isinstance(check_curve, RotatedCurve):
            check_curve = check_curve.curve
        if isinstance(check_curve, CurvePlanarEllipticalCylindrical):
            tf_curves.append(curve)
    return tf_curves

def get_nfp_from_tf_curves(
    tf_curves: Sequence[CurvePlanarEllipticalCylindrical | RotatedCurve]
) -> int:
    n_tf_curves = len(tf_curves)
    tf_base_curves = get_tf_base_curves(tf_curves)
    n_base_curves = len(tf_base_curves)

    assert n_tf_curves % (2*n_base_curves) == 0, \
        "Number of TF curves is not divisible by 2 times the number " \
        "of base curves."
    nfp = n_tf_curves // (2*n_base_curves)
    return nfp

def get_windingsurface_from_curves(
    curves: Sequence,
) -> SurfaceRZFourier:
    dipole_base_curves = get_dipole_base_curves(curves)
    assert len(dipole_base_curves) > 0, \
        "No dipole base curves found in the input curves."
    first_curve = dipole_base_curves[0]
    windingsurface = first_curve.surface
    return windingsurface

def is_missing_outboard(
    dipole_curves: Sequence[CurvePlanarOnSurface | RotatedCurve]
) -> bool:
    curve = dipole_curves[0]
    windingsurface = curve.surface
    n_surface_quadpoints = windingsurface.gamma().reshape(-1, 3).shape[0]
    n_dipole_curves = len(dipole_curves)
    return n_dipole_curves < n_surface_quadpoints
