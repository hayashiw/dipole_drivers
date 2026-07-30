import numpy as np

from typing import Sequence

from simsopt.field import apply_symmetries_to_curves
from simsopt.geo import (
    CurvePlanarFourier,
    CurvePlanarEllipticalCylindrical,
    RotatedCurve,
    SurfaceRZFourier,
)

from .ellipse import (
    calculate_total_arc_length,
)
from .quaternion import calculate_quaternion_alignment
from .superellipse import calculate_superellipse_fourier_coeffs
from .windingsurface import get_windingsurface_geometry

__all__ = [
    "copy_curve",
    "copy_dipole_curve",
    "copy_tf_curve",
    "copy_base_curves",
    "copy_curves",
    "generate_dipole_base_curves",
    "get_base_curves",
    "get_dipole_base_curves",
    "get_dipole_curves",
    "get_tf_base_curves",
    "get_tf_curves",
    "get_tf_geometry",
    "get_nfp_from_curves",
]

def copy_curve(
    curve: CurvePlanarFourier | CurvePlanarEllipticalCylindrical | RotatedCurve
) -> CurvePlanarFourier | CurvePlanarEllipticalCylindrical | RotatedCurve:
    if isinstance(curve, CurvePlanarFourier):
        return copy_dipole_curve(curve)
    elif isinstance(curve, CurvePlanarEllipticalCylindrical):
        return copy_tf_curve(curve)
    else:
        raise ValueError(
            "Input curve must be either CurvePlanarFourier, "
            "CurvePlanarEllipticalCylindrical, or RotatedCurve, "
            f"got {type(curve).__name__} instead."
        )

def copy_dipole_curve(dipole_curve: CurvePlanarFourier) -> CurvePlanarFourier:
    assert isinstance(dipole_curve, CurvePlanarFourier), \
        "Input curve must be a CurvePlanarFourier"

    copied_curve = CurvePlanarFourier(
        dipole_curve.quadpoints,
        dipole_curve.order,
    )
    copied_curve.x = np.asarray(dipole_curve.x).copy()

    return copied_curve

def copy_tf_curve(
    tf_curve: CurvePlanarEllipticalCylindrical
) -> CurvePlanarEllipticalCylindrical:
    assert isinstance(tf_curve, CurvePlanarEllipticalCylindrical), \
        "Input curve must be a CurvePlanarEllipticalCylindrical"

    copied_curve = CurvePlanarEllipticalCylindrical(
        tf_curve.quadpoints,
        tf_curve.a,
        tf_curve.b,
    )
    copied_curve.x = np.asarray(tf_curve.x).copy()

    return copied_curve

def copy_base_curves(
    base_curves: Sequence[CurvePlanarFourier | CurvePlanarEllipticalCylindrical]
) -> list[CurvePlanarFourier | CurvePlanarEllipticalCylindrical]:
    # Base curve means not rotated (i.e. not generate from nfp symmetry
    # or stellarator symmetry).
    # This can be simplified to assert that this is applied separately
    # to TF and dipole curves.
    ctypes = []
    for curve in base_curves:
        if isinstance(curve, CurvePlanarFourier):
            ctypes.append("dipole")
        elif isinstance(curve, CurvePlanarEllipticalCylindrical):
            ctypes.append("tf")
        else:
            raise ValueError(
                "Input base curves must be either CurvePlanarFourier or "
                "CurvePlanarEllipticalCylindrical only, got "
                f"{type(curve).__name__} instead."
            )
    assert len(set(ctypes)) == 1, \
        "Input base curves must be either all CurvePlanarFourier or " \
        "all CurvePlanarEllipticalCylindrical, got a mix of both."

    if ctypes[0] == "dipole":
        copied_base_curves = [
            copy_dipole_curve(curve) for curve in base_curves]
    else:
        copied_base_curves = [
            copy_tf_curve(curve) for curve in base_curves]

    return copied_base_curves
    
def copy_curves(
    curves: Sequence[CurvePlanarFourier | CurvePlanarEllipticalCylindrical | RotatedCurve],
) -> list[CurvePlanarFourier | CurvePlanarEllipticalCylindrical | RotatedCurve]:
    dipole_curves = get_dipole_curves(curves)
    tf_curves = get_tf_curves(curves)
    nfp = get_nfp_from_curves(curves)

    dipole_base_curves = get_base_curves(dipole_curves)
    if any([isinstance(curve, RotatedCurve) for curve in dipole_curves]):
        copied_dipole_curves = apply_symmetries_to_curves(
            dipole_base_curves, nfp, True)
    else:
        copied_dipole_curves = copy_base_curves(dipole_base_curves)

    tf_base_curves = get_base_curves(tf_curves)
    if any([isinstance(curve, RotatedCurve) for curve in tf_curves]):
        copied_tf_curves = apply_symmetries_to_curves(
            tf_base_curves, nfp, True)
    else:
        copied_tf_curves = copy_base_curves(tf_base_curves)

    # It's best if the order of curves is consistent i.e. all TF curves
    # first, then all the dipole curves. Or vice versa. We'll enforce
    # the curve order by first asserting that the input curves has a
    # consistent order, then we'll return the copied curves in the same
    # order.
    number_of_switches = 0
    ctype_is_dipole = isinstance(curves[0], CurvePlanarFourier)
    for curve in curves[1:]:
        while isinstance(curve, RotatedCurve):
            curve = curve.curve
        current_ctype = isinstance(curve, CurvePlanarFourier)
        if current_ctype == ctype_is_dipole:
            continue
        elif current_ctype != ctype_is_dipole:
            number_of_switches += 1
            ctype_is_dipole = current_ctype
    assert number_of_switches <= 1, \
        "Input curves must be ordered such that all TF curves are " \
        "together and all dipole curves are together. The order of " \
        "the curves in the output will be the same as the order of " \
        "the curves in the input. The input curves had " \
        f"{number_of_switches} switches between TF and dipole curves."

    dipoles_first = isinstance(curves[0], CurvePlanarFourier)
    if dipoles_first:
        copied_curves = copied_dipole_curves + copied_tf_curves
    else:
        copied_curves = copied_tf_curves + copied_dipole_curves
    assert len(copied_curves) == len(curves), \
        "Number of copied curves does not match the number " \
        "of input curves"
    return copied_curves
        
def generate_dipole_base_curves(
    windingsurface: SurfaceRZFourier,
    curve_curve_distance: float,
    nfp: int,
    *,
    dipole_curve_order: int = 12,
    superellipse_n: int = 4,
    nquadpoints: int = 128,
) -> Sequence[CurvePlanarFourier]:
    windingsurface_major_radius, windingsurface_minor_radius_r, windingsurface_minor_radius_z = \
        get_windingsurface_geometry(windingsurface)

    phis = windingsurface.quadpoints_phi * 2 * np.pi
    ndipoles_tor_total = phis.size

    thetas = windingsurface.quadpoints_theta * 2 * np.pi
    ndipoles_pol_total = thetas.size

    poloidal_circumference = calculate_total_arc_length(
        windingsurface_minor_radius_r,
        windingsurface_minor_radius_z
    )
    poloidal_arc_length_per_coil = poloidal_circumference / ndipoles_pol_total
    coil_radius_pol = (poloidal_arc_length_per_coil - curve_curve_distance)/2

    dipole_base_curves = []
    for i, theta in enumerate(thetas):
        major_radius = \
            windingsurface_major_radius + \
            windingsurface_minor_radius_r*np.cos(theta)
        toroidal_circumference = 2*np.pi*major_radius
        toroidal_circumference_per_coil = \
            toroidal_circumference / ndipoles_tor_total
        coil_radius_tor = \
            (toroidal_circumference_per_coil - curve_curve_distance)/2

        coeffs = calculate_superellipse_fourier_coeffs(
            coil_radius_pol,
            coil_radius_tor,
            superellipse_n,
            dipole_curve_order
        )

        for j, phi in enumerate(phis):
            if phi > 2*np.pi/nfp/2: continue

            curve = CurvePlanarFourier(nquadpoints, dipole_curve_order)
            for m, val in coeffs.items():
                curve.set(f"rc({m})", val)  # pyright: ignore[reportArgumentType]

            nhat  = windingsurface.unitnormal()[j, i]
            dg_dtheta = windingsurface.gammadash2()[j, i]
            quat = calculate_quaternion_alignment(nhat, dg_dtheta)
            for ikey, key in enumerate(["q0", "qi", "qj", "qk"]):
                curve.set(key, quat[ikey])

            gamma = windingsurface.gamma()[j, i]
            for ikey, key in enumerate(["X", "Y", "Z"]):
                curve.set(key, gamma[ikey])

            curve.fix_all()
            dipole_base_curves.append(curve)

    return dipole_base_curves

def get_base_curves(
    curves: Sequence[CurvePlanarFourier | CurvePlanarEllipticalCylindrical]
) -> list[CurvePlanarFourier | CurvePlanarEllipticalCylindrical]:
    base_curves = []
    ctypes = []
    for curve in curves:
        if isinstance(curve, RotatedCurve): continue
        base_curves.append(curve)
        ctypes.append(type(curve).__name__)
    assert len(set(ctypes)) == 1, \
        "Input curves must be either all CurvePlanarFourier or " \
        "all CurvePlanarEllipticalCylindrical, got a mix of both."
    return base_curves

def get_dipole_base_curves(
    dipole_curves: Sequence[CurvePlanarFourier | CurvePlanarEllipticalCylindrical]
) -> list[CurvePlanarFourier]:
    base_curves = [
        curve for curve in dipole_curves
        if isinstance(curve, CurvePlanarFourier)]
    return base_curves

def get_dipole_curves(
    curves: Sequence[CurvePlanarFourier | CurvePlanarEllipticalCylindrical]
) -> list[CurvePlanarFourier]:
    dipole_curves = []
    for curve in curves:
        check_curve = curve
        while isinstance(check_curve, RotatedCurve):
            check_curve = check_curve.curve
        if isinstance(check_curve, CurvePlanarFourier):
            dipole_curves.append(curve)
    return dipole_curves

def get_tf_base_curves(
    tf_curves: Sequence[CurvePlanarEllipticalCylindrical | CurvePlanarFourier]
) -> list[CurvePlanarEllipticalCylindrical]:
    base_curves = [
        curve for curve in tf_curves
        if isinstance(curve, CurvePlanarEllipticalCylindrical)]
    return base_curves

def get_tf_curves(
    curves: Sequence[CurvePlanarFourier | CurvePlanarEllipticalCylindrical]
) -> list[CurvePlanarEllipticalCylindrical]:
    tf_curves = []
    for curve in curves:
        check_curve = curve
        while isinstance(check_curve, RotatedCurve):
            check_curve = check_curve.curve
        if isinstance(check_curve, CurvePlanarEllipticalCylindrical):
            tf_curves.append(curve)
    return tf_curves

def get_tf_geometry(
    curve: CurvePlanarEllipticalCylindrical,
):
    assert isinstance(curve, CurvePlanarEllipticalCylindrical), \
        "Input curve must be a CurvePlanarEllipticalCylindrical."

    tf_major_radius = float(curve.get("R0"))
    tf_minor_radius_r = curve.a
    tf_minor_radius_z = curve.b
    return tf_major_radius, tf_minor_radius_r, tf_minor_radius_z

def get_nfp_from_curves(
    curves: Sequence[CurvePlanarFourier | CurvePlanarEllipticalCylindrical | RotatedCurve]
) -> int:
    tf_curves = get_tf_curves(curves)
    dipole_curves = get_dipole_curves(curves)

    if len(tf_curves) and len(dipole_curves):
        tf_nfp = get_nfp_from_curves(tf_curves)
        dipole_nfp = get_nfp_from_curves(dipole_curves)
        assert tf_nfp == dipole_nfp, \
            "Number of field periods (nfp) must be the same for both " \
            "the TF and dipole curves. The TF curves have nfp = " \
            f"{tf_nfp}, while the dipole curves have nfp = {dipole_nfp}."
        return tf_nfp
    elif (
        (len(tf_curves) == len(curves)) or
        (len(dipole_curves) == len(curves))
    ):
        ncurves = len(curves)
        base_curves = get_base_curves(curves)
        nbase_curves = len(base_curves)
        assert ncurves % (nbase_curves*2) == 0, \
            "Number of curves must be a multiple of twice the number " \
            "of base curves. This is because the curves are assumed to " \
            "be generated from the base curves via nfp symmetry and " \
            "stellarator symmetry. The number of base curves is " \
            f"{nbase_curves}, the number of curves is {ncurves}."
        nfp = ncurves // (nbase_curves*2)
        return nfp
    else:
        raise ValueError(
            "Input curves must be either all TF curves, all dipole "
            "curves, or sets of both with the same nfp."
        )

def get_ntf_total(
    curves: Sequence[CurvePlanarFourier | CurvePlanarEllipticalCylindrical | RotatedCurve]
) -> int:
    tf_curves = get_tf_curves(curves)
    ntf_total = len(tf_curves)
    return ntf_total
