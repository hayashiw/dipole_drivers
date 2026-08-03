import argparse
import numpy as np

from scipy.interpolate import RegularGridInterpolator
from typing import Sequence

from simsopt._core import OptimizableSum, ScaledOptimizable, Optimizable
from simsopt.field import BiotSavart
from simsopt.geo import (
    AspectRatio,
    BoozerResidual,
    CurveCurveDistance,
    CurveSurfaceDistance,
    CurveLength,
    Iotas,
    LpCurveCurvature,
    MeanSquaredCurvature,
    SurfaceRZFourier,
    SurfaceXYZTensorFourier,
)
from simsopt.mhd import VirtualCasing
from simsopt.objectives import QuadraticPenalty
from simsopt.util import proc0_print, comm_world

from ..defaults import *
from ..utils.biotsavart import get_windingsurface_from_biotsavart
from ..utils.coil import *
from ..utils.curve import *
from .bdotn_norm import calculate_bdotn_norm
from .currentpenalty import *
from .multisurfacesquaredflux import MultiSurfaceSquaredFlux

__all__ = [
    "build_stage2_objective_function",
    "build_stage2_objective_parser",
    "build_squaredflux_objectives",
    "build_squaredflux_objectives_parser",
    "build_curve_objectives",
    "build_curve_objectives_parser",
    "build_current_objectives",
    "build_current_objectives_parser",
    "check_objective_kwargs",
    "get_virtualcasing_target",
]

REQUIRED_OBJECTIVE_KWARGS = dict(
    squaredflux=["boundary_surface", "weight_squaredflux"],
    dipole_curvelength=["max_dipole_curvelength_threshold", "weight_max_dipole_curvelength"],
    dipole_curvecurvedistance=["min_dipole_curvecurvedistance_threshold", "weight_min_dipole_curvecurvedistance"],
    dipole_curvesurfacedistance=["min_dipole_curvesurfacedistance_threshold", "weight_min_dipole_curvesurfacedistance"],
    dipole_meansquaredcurvature=["max_dipole_meansquaredcurvature_threshold", "weight_max_dipole_meansquaredcurvature"],
    dipole_lpcurvecurvature=["dipole_lpcurvecurvature_p_value", "max_dipole_lpcurvecurvature_threshold", "weight_max_dipole_lpcurvecurvature"],
    tf_curvelength=["max_tf_curvelength_threshold", "weight_max_tf_curvelength"],
    tf_curvecurvedistance=["min_tf_curvecurvedistance_threshold", "weight_min_tf_curvecurvedistance"],
    tf_meansquaredcurvature=["max_tf_meansquaredcurvature_threshold", "weight_max_tf_meansquaredcurvature"],
    tf_lpcurvecurvature=["tf_lpcurvecurvature_p_value", "max_tf_lpcurvecurvature_threshold", "weight_max_tf_lpcurvecurvature"],
    tf_dipole_curvecurvedistance=["min_tf_dipole_curvecurvedistance_threshold", "weight_min_tf_dipole_curvecurvedistance"],
    current=["current_p_value", "max_current_threshold", "weight_max_current"],
    peak_current=["current_p_value", "weight_peak_current"],
    windingsurface_aspectratio=["max_windingsurface_aspectratio_threshold", "weight_max_windingsurface_aspectratio"],
)

def build_stage2_objective_function(
    biotsavart: BiotSavart,
    surfaces: Sequence[SurfaceRZFourier | SurfaceXYZTensorFourier],
    *,
    virtualcasings: Sequence[VirtualCasing | None] | None = None,
    squaredflux_definition: str = DEFAULT_SQUAREDFLUX_DEFINITION,
    boundary_surface: SurfaceRZFourier | SurfaceXYZTensorFourier | None = None,
    max_dipole_curvelength_threshold: float | None = None,
    min_dipole_curvecurvedistance_threshold: float | None = None,
    min_dipole_curvesurfacedistance_threshold: float | None = None,
    max_dipole_meansquaredcurvature_threshold: float | None = None,
    dipole_lpcurvecurvature_p_value: float | None = None,
    max_dipole_lpcurvecurvature_threshold: float | None = None,
    max_tf_curvelength_threshold: float | None = None,
    min_tf_curvecurvedistance_threshold: float | None = None,
    max_tf_meansquaredcurvature_threshold: float | None = None,
    tf_lpcurvecurvature_p_value: float | None = None,
    max_tf_lpcurvecurvature_threshold: float | None = None,
    min_tf_dipole_curvecurvedistance_threshold: float | None = None,
    current_p_value: float | None = None,
    max_current_threshold: float | None = None,
    max_windingsurface_aspectratio_threshold: float | None = None,
    weight_squaredflux: float | None = DEFAULT_WEIGHT_SQUAREDFLUX,
    weight_max_dipole_curvelength: float | None = None,
    weight_min_dipole_curvecurvedistance: float | None = None,
    weight_min_dipole_curvesurfacedistance: float | None = None,
    weight_max_dipole_meansquaredcurvature: float | None = None,
    weight_max_dipole_lpcurvecurvature: float | None = None,
    weight_max_tf_curvelength: float | None = None,
    weight_min_tf_curvecurvedistance: float | None = None,
    weight_max_tf_meansquaredcurvature: float | None = None,
    weight_max_tf_lpcurvecurvature: float | None = None,
    weight_min_tf_dipole_curvecurvedistance: float | None = None,
    weight_max_current: float | None = None,
    weight_peak_current: float | None = None,
    weight_max_windingsurface_aspectratio: float | None = None,
    skip_metrics: bool = False,
    **kwargs,
):
    all_objectives = {}
    all_objectives.update(build_squaredflux_objectives(
        biotsavart,
        surfaces,
        virtualcasings=virtualcasings,
        squaredflux_definition=squaredflux_definition,
    ))
    all_objectives.update(build_curve_objectives(
        biotsavart,
        boundary_surface=boundary_surface,
        max_dipole_curvelength_threshold=max_dipole_curvelength_threshold,
        min_dipole_curvecurvedistance_threshold=min_dipole_curvecurvedistance_threshold,
        min_dipole_curvesurfacedistance_threshold=min_dipole_curvesurfacedistance_threshold,
        max_dipole_meansquaredcurvature_threshold=max_dipole_meansquaredcurvature_threshold,
        dipole_lpcurvecurvature_p_value=dipole_lpcurvecurvature_p_value,
        max_dipole_lpcurvecurvature_threshold=max_dipole_lpcurvecurvature_threshold,
        max_tf_curvelength_threshold=max_tf_curvelength_threshold,
        min_tf_curvecurvedistance_threshold=min_tf_curvecurvedistance_threshold,
        max_tf_meansquaredcurvature_threshold=max_tf_meansquaredcurvature_threshold,
        tf_lpcurvecurvature_p_value=tf_lpcurvecurvature_p_value,
        max_tf_lpcurvecurvature_threshold=max_tf_lpcurvecurvature_threshold,
        min_tf_dipole_curvecurvedistance_threshold=min_tf_dipole_curvecurvedistance_threshold,
    ))
    all_objectives.update(build_current_objectives(
        biotsavart,
        max_current_threshold=max_current_threshold,
        current_p_value=current_p_value,
    ))
    windingsurface = get_windingsurface_from_biotsavart(biotsavart)
    all_objectives.update(build_windingsurface_objectives(
        windingsurface,
        max_windingsurface_aspectratio_threshold=max_windingsurface_aspectratio_threshold,
    ))

    weights = dict(
        weight_squaredflux=weight_squaredflux,
        weight_max_dipole_curvelength=weight_max_dipole_curvelength,
        weight_min_dipole_curvecurvedistance=weight_min_dipole_curvecurvedistance,
        weight_min_dipole_curvesurfacedistance=weight_min_dipole_curvesurfacedistance,
        weight_max_dipole_meansquaredcurvature=weight_max_dipole_meansquaredcurvature,
        weight_max_dipole_lpcurvecurvature=weight_max_dipole_lpcurvecurvature,
        weight_max_tf_curvelength=weight_max_tf_curvelength,
        weight_min_tf_curvecurvedistance=weight_min_tf_curvecurvedistance,
        weight_max_tf_meansquaredcurvature=weight_max_tf_meansquaredcurvature,
        weight_max_tf_lpcurvecurvature=weight_max_tf_lpcurvecurvature,
        weight_max_current=weight_max_current,
        weight_peak_current=weight_peak_current,
        weight_min_tf_dipole_curvecurvedistance=weight_min_tf_dipole_curvecurvedistance,
        weight_max_windingsurface_aspectratio=weight_max_windingsurface_aspectratio,
    )

    objectives = {}
    J_list = []
    for name, objective in all_objectives.items():
        if (not name.startswith("J_")) or (objective is None): continue
        if len(objective.x) == 0: continue # Drop if underlying objects are fixed
        weight_name = "weight_" + name.replace("J_", "")
        weight = weights.get(weight_name, None)
        if weight:
            J_list.append(ScaledOptimizable(weight, objective))
            objectives[name] = objective
            sub_key = name.replace("J_", "")
            if (sub_key in all_objectives) and (not skip_metrics):
                objectives[sub_key] = all_objectives[sub_key]
    assert len(J_list) > 0, \
        "No objectives were selected. Check inputs."

    JF = OptimizableSum(J_list)

    return JF, objectives

def build_stage2_objective_parser(
    **kwargs
):
    parser = argparse.ArgumentParser(add_help=False, parents=[
        build_squaredflux_objectives_parser(**kwargs),
        build_curve_objectives_parser(**kwargs),
        build_current_objectives_parser(**kwargs),
        build_windingsurface_objectives_parser(**kwargs),
    ])
    parser.add_argument(
        "--skip-metrics", action="store_true", required=False,
        help="Skip computing the physical metrics for the log; only the penalty "
             "values are recorded. Default: False."
    )
    return parser

def build_squaredflux_objectives(
    biotsavart: BiotSavart,
    surfaces: Sequence[SurfaceRZFourier | SurfaceXYZTensorFourier],
    *,
    virtualcasings: Sequence[VirtualCasing | None] | None = None,
    squaredflux_definition: str = DEFAULT_SQUAREDFLUX_DEFINITION,
    **kwargs,
) -> dict:
    if virtualcasings is None:
        virtualcasings = [None] * len(surfaces)
    else:
        assert len(surfaces) == len(virtualcasings), \
            "The number of surfaces must match the number of virtual casings."

    targets = []
    for surface, virtualcasing in zip(surfaces, virtualcasings):
        target = get_virtualcasing_target(surface, virtualcasing=virtualcasing)
        targets.append(target)
    J_squaredflux = MultiSurfaceSquaredFlux(surfaces, BiotSavart(biotsavart.coils), targets=targets, definition=squaredflux_definition)
    class wrapper_squaredflux():
        def __init__(self, _surfaces, _biotsavart):
            self._surfaces = _surfaces
            self._biotsavart = _biotsavart

        def J(self):
            # Return the maximum of the surface-averaged field errors
            bdotn_norms = [calculate_bdotn_norm(self._biotsavart, s) for s in self._surfaces]
            max_bdotn_norm = 0
            for bdotn_norm in bdotn_norms:
                avg_bdotn_norm = np.abs(bdotn_norm).mean()
                max_bdotn_norm = max(max_bdotn_norm, avg_bdotn_norm)
            return max_bdotn_norm
    squaredflux = wrapper_squaredflux(surfaces, biotsavart)

    objectives = dict(
        J_squaredflux=J_squaredflux,
        squaredflux=squaredflux,
    )
    
    return objectives

def build_squaredflux_objectives_parser(
    *,
    squaredflux_definition_required: bool = False,
    weight_squaredflux_required: bool = False,
    **kwargs,
):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--squaredflux-definition", type=str, default=DEFAULT_SQUAREDFLUX_DEFINITION, required=squaredflux_definition_required,
        help=f"Definition of the squared flux objective. Default: '{DEFAULT_SQUAREDFLUX_DEFINITION}'."
    )
    parser.add_argument(
        "--weight-squaredflux", type=float, default=DEFAULT_WEIGHT_SQUAREDFLUX, required=weight_squaredflux_required,
        help=f"Weight for the squared flux penalty. Default: {DEFAULT_WEIGHT_SQUAREDFLUX}."
    )
    return parser

def _arclength(curve):
    return float(np.mean(np.linalg.norm(curve.gammadash(), axis=1)))

def _unwrap(objective):
    while isinstance(objective, ScaledOptimizable):
        objective = objective.opt
    return objective

def build_curve_objectives(
    biotsavart: BiotSavart,
    *,
    boundary_surface: SurfaceRZFourier | SurfaceXYZTensorFourier | None = None,
    max_dipole_curvelength_threshold: float | None = None,
    min_dipole_curvecurvedistance_threshold: float | None = None,
    min_dipole_curvesurfacedistance_threshold: float | None = None,
    max_dipole_meansquaredcurvature_threshold: float | None = None,
    dipole_lpcurvecurvature_p_value: float | None = None,
    max_dipole_lpcurvecurvature_threshold: float | None = None,
    max_tf_curvelength_threshold: float | None = None,
    min_tf_curvecurvedistance_threshold: float | None = None,
    max_tf_meansquaredcurvature_threshold: float | None = None,
    tf_lpcurvecurvature_p_value: float | None = None,
    max_tf_lpcurvecurvature_threshold: float | None = None,
    min_tf_dipole_curvecurvedistance_threshold: float | None = None,
    use_native_curve_units: bool = False,
    **kwargs,
) -> dict:
    coils = biotsavart.coils
    curves = [coil.curve for coil in coils]

    normalize_units = not use_native_curve_units

    dipole_curves = get_dipole_curves(curves)
    dipole_base_curves = get_dipole_base_curves(dipole_curves)
    tf_curves = get_tf_curves(curves)
    tf_base_curves = get_tf_base_curves(tf_curves)

    table = get_dipole_index_table(dipole_curves)
    nphi, ntheta = table.shape
    keep = set()
    for c in dipole_base_curves:
        i0, j0 = c.iphi, c.itheta
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                k = int(table[(i0+di) % nphi, (j0+dj) % ntheta])
                if k >= 0:
                    keep.add(k)
    neighbor_dipole_curves = [dipole_curves[k] for k in sorted(keep)]

    nfp = get_nfp_from_coils(coils)
    halfperiod_dipole_curves = [
        dipole_curves[int(table[i, j])]
        for i in range(nphi // (2*nfp))
        for j in range(ntheta)
        if table[i, j] >= 0
    ]

    if max_dipole_curvelength_threshold is None:
        dipole_curvelengths = []
        J_max_dipole_curvelength = None
        max_dipole_curvelength = None
    else:
        dipole_curvelengths = [CurveLength(curve) for curve in dipole_base_curves]
        J_max_dipole_curvelength = OptimizableSum(
            [QuadraticPenalty(cl, max_dipole_curvelength_threshold, "max") for cl in dipole_curvelengths])
        if normalize_units:
            J_max_dipole_curvelength = ScaledOptimizable(
                1.0 / (len(dipole_base_curves) * 0.5 * max_dipole_curvelength_threshold**2),
                J_max_dipole_curvelength)
        class wrapper_dipole_curvelength():
            def __init__(self, _curvelengths):
                self._curvelengths = _curvelengths

            def J(self):
                return np.max([cl.J() for cl in self._curvelengths])
        max_dipole_curvelength = wrapper_dipole_curvelength(dipole_curvelengths)

    if min_dipole_curvecurvedistance_threshold is None:
        J_min_dipole_curvecurvedistance = None
        min_dipole_curvecurvedistance = None
    else:
        J_min_dipole_curvecurvedistance = CurveCurveDistance(neighbor_dipole_curves, min_dipole_curvecurvedistance_threshold)
        if normalize_units:
            scale_candidates = CurveCurveDistance(
                neighbor_dipole_curves, 2*min_dipole_curvecurvedistance_threshold)
            scale_candidates.compute_candidates()
            pair_scale = sum(_arclength(neighbor_dipole_curves[i]) * _arclength(neighbor_dipole_curves[j])
                             for i, j in scale_candidates.candidates) or 1.0 # pyright: ignore[reportOptionalIterable]
            J_min_dipole_curvecurvedistance = ScaledOptimizable(
                1.0 / (pair_scale * min_dipole_curvecurvedistance_threshold**2),
                J_min_dipole_curvecurvedistance)
            
        class wrapper_dipole_curvecurvedistance():
            def __init__(self, _J_min_dipole_curvecurvedistance):
                self._J_min_dipole_curvecurvedistance = _unwrap(_J_min_dipole_curvecurvedistance)

            def J(self):
                return self._J_min_dipole_curvecurvedistance.shortest_distance()
        min_dipole_curvecurvedistance = wrapper_dipole_curvecurvedistance(J_min_dipole_curvecurvedistance)

    if (min_dipole_curvesurfacedistance_threshold is None) or (boundary_surface is None):
        J_min_dipole_curvesurfacedistance = None
        min_dipole_curvesurfacedistance = None
    else:
        J_min_dipole_curvesurfacedistance = CurveSurfaceDistance(
            halfperiod_dipole_curves, boundary_surface, min_dipole_curvesurfacedistance_threshold)
        if normalize_units:
            surface_area = float(np.mean(np.linalg.norm(boundary_surface.normal(), axis=2)))
            J_min_dipole_curvesurfacedistance = ScaledOptimizable(
                1.0 / (sum(_arclength(c) for c in halfperiod_dipole_curves) * surface_area
                    * min_dipole_curvesurfacedistance_threshold**2),
                J_min_dipole_curvesurfacedistance)
            
        class wrapper_dipole_curvesurfacedistance():
            def __init__(self, _J_min_dipole_curvesurfacedistance):
                self._J_min_dipole_curvesurfacedistance = _unwrap(_J_min_dipole_curvesurfacedistance)

            def J(self):
                return self._J_min_dipole_curvesurfacedistance.shortest_distance()
        min_dipole_curvesurfacedistance = wrapper_dipole_curvesurfacedistance(J_min_dipole_curvesurfacedistance)

    if max_dipole_lpcurvecurvature_threshold is None:
        dipole_lpcurvecurvatures = []
        J_max_dipole_lpcurvecurvature = None
        max_dipole_lpcurvecurvature = None
    else:
        dipole_lpcurvecurvatures = \
            [LpCurveCurvature(curve, p=dipole_lpcurvecurvature_p_value, threshold=max_dipole_lpcurvecurvature_threshold)
             for curve in dipole_base_curves]
        J_max_dipole_lpcurvecurvature = OptimizableSum(dipole_lpcurvecurvatures)
        if normalize_units:
            J_max_dipole_lpcurvecurvature = ScaledOptimizable(
                dipole_lpcurvecurvature_p_value
                / (sum(_arclength(c) for c in dipole_base_curves)
                * max_dipole_lpcurvecurvature_threshold**dipole_lpcurvecurvature_p_value), # pyright: ignore[reportOperatorIssue]
                J_max_dipole_lpcurvecurvature)
        class wrapper_dipole_lpcurvecurvature():
            def __init__(self, _dipole_lpcurvecurvatures):
                self._dipole_lpcurvecurvatures = _dipole_lpcurvecurvatures

            def J(self): # Return the actual max curvature, not the Lp norm penalty
                return np.max([c.kappa().max() for c in dipole_base_curves])
        max_dipole_lpcurvecurvature = wrapper_dipole_lpcurvecurvature(dipole_lpcurvecurvatures)

    if max_dipole_meansquaredcurvature_threshold is None:
        dipole_meansquaredcurvatures = []
        J_max_dipole_meansquaredcurvature = None
        max_dipole_meansquaredcurvature = None
    else:
        dipole_meansquaredcurvatures = [MeanSquaredCurvature(curve) for curve in dipole_base_curves]
        J_max_dipole_meansquaredcurvature = OptimizableSum(
            [QuadraticPenalty(msc, max_dipole_meansquaredcurvature_threshold, "max") for msc in dipole_meansquaredcurvatures])
        if normalize_units:
            J_max_dipole_meansquaredcurvature = ScaledOptimizable(
                1.0 / (len(dipole_base_curves) * 0.5 * max_dipole_meansquaredcurvature_threshold**2),
                J_max_dipole_meansquaredcurvature)
        class wrapper_dipole_meansquaredcurvature():
            def __init__(self, _dipole_meansquaredcurvatures):
                self._dipole_meansquaredcurvatures = _dipole_meansquaredcurvatures

            def J(self):
                return np.max([msc.J() for msc in self._dipole_meansquaredcurvatures])
        max_dipole_meansquaredcurvature = wrapper_dipole_meansquaredcurvature(dipole_meansquaredcurvatures)

    if max_tf_curvelength_threshold is None:
        tf_curvelengths = []
        J_max_tf_curvelength = None
        max_tf_curvelength = None
    else:
        tf_curvelengths = [CurveLength(curve) for curve in tf_base_curves]
        J_max_tf_curvelength = OptimizableSum(
            [QuadraticPenalty(cl, max_tf_curvelength_threshold, "max") for cl in tf_curvelengths])
        if normalize_units:
            J_max_tf_curvelength = ScaledOptimizable(
                1.0 / (len(tf_base_curves) * 0.5 * max_tf_curvelength_threshold**2),
                J_max_tf_curvelength)
        class wrapper_tf_curvelength():
            def __init__(self, _curvelengths):
                self._curvelengths = _curvelengths

            def J(self):
                return np.max([cl.J() for cl in self._curvelengths])
        max_tf_curvelength = wrapper_tf_curvelength(tf_curvelengths)

    if min_tf_curvecurvedistance_threshold is None:
        J_min_tf_curvecurvedistance = None
        min_tf_curvecurvedistance = None
    else:
        J_min_tf_curvecurvedistance = CurveCurveDistance(tf_curves, min_tf_curvecurvedistance_threshold)
        if normalize_units:
            scale_candidates = CurveCurveDistance(
                tf_curves, 2*min_tf_curvecurvedistance_threshold)
            scale_candidates.compute_candidates()
            pair_scale = sum(_arclength(tf_curves[i]) * _arclength(tf_curves[j])
                             for i, j in scale_candidates.candidates) or 1.0 # pyright: ignore[reportOptionalIterable]
            J_min_tf_curvecurvedistance = ScaledOptimizable(
                1.0 / (pair_scale * min_tf_curvecurvedistance_threshold**2),
                J_min_tf_curvecurvedistance)
        class wrapper_tf_curvecurvedistance():
            def __init__(self, _J_min_tf_curvecurvedistance):
                self._J_min_tf_curvecurvedistance = _unwrap(_J_min_tf_curvecurvedistance)

            def J(self):
                return self._J_min_tf_curvecurvedistance.shortest_distance()
            
        _metric = CurveCurveDistance(tf_curves, 4*min_tf_curvecurvedistance_threshold) # pyright: ignore[reportOperatorIssue]
        min_tf_curvecurvedistance = wrapper_tf_curvecurvedistance(_metric)

    if max_tf_lpcurvecurvature_threshold is None:
        tf_lpcurvecurvatures = []
        J_max_tf_lpcurvecurvature = None
        max_tf_lpcurvecurvature = None
    else:
        tf_lpcurvecurvatures = \
            [LpCurveCurvature(curve, p=tf_lpcurvecurvature_p_value, threshold=max_tf_lpcurvecurvature_threshold)
             for curve in tf_base_curves]
        J_max_tf_lpcurvecurvature = OptimizableSum(tf_lpcurvecurvatures)
        if normalize_units:
            J_max_tf_lpcurvecurvature = ScaledOptimizable(
                tf_lpcurvecurvature_p_value
                / (sum(_arclength(c) for c in tf_base_curves)
                * max_tf_lpcurvecurvature_threshold**tf_lpcurvecurvature_p_value), # pyright: ignore[reportOperatorIssue]
                J_max_tf_lpcurvecurvature)
        class wrapper_tf_lpcurvecurvature():
            def __init__(self, _tf_lpcurvecurvatures):
                self._tf_lpcurvecurvatures = _tf_lpcurvecurvatures

            def J(self): # Return the actual max curvature, not the Lp norm penalty
                return np.max([c.kappa().max() for c in tf_base_curves])
        max_tf_lpcurvecurvature = wrapper_tf_lpcurvecurvature(tf_lpcurvecurvatures)

    if max_tf_meansquaredcurvature_threshold is None:
        tf_meansquaredcurvatures = []
        J_max_tf_meansquaredcurvature = None
        max_tf_meansquaredcurvature = None
    else:
        tf_meansquaredcurvatures = [MeanSquaredCurvature(curve) for curve in tf_base_curves]
        J_max_tf_meansquaredcurvature = OptimizableSum(
            [QuadraticPenalty(msc, max_tf_meansquaredcurvature_threshold, "max") for msc in tf_meansquaredcurvatures])
        if normalize_units:
            J_max_tf_meansquaredcurvature = ScaledOptimizable(
                1.0 / (len(tf_base_curves) * 0.5 * max_tf_meansquaredcurvature_threshold**2),
                J_max_tf_meansquaredcurvature)
        class wrapper_tf_meansquaredcurvature():
            def __init__(self, _tf_meansquaredcurvatures):
                self._tf_meansquaredcurvatures = _tf_meansquaredcurvatures

            def J(self):
                return np.max([msc.J() for msc in self._tf_meansquaredcurvatures])
        max_tf_meansquaredcurvature = wrapper_tf_meansquaredcurvature(tf_meansquaredcurvatures)

    if min_tf_dipole_curvecurvedistance_threshold is None:
        J_min_tf_dipole_curvecurvedistance = None
        min_tf_dipole_curvecurvedistance = None
    else:
        # Technically a TF-to-dipole-windingsurface penalty.
        windingsurface = dipole_base_curves[0].surface
        windingsurface_eval = SurfaceRZFourier(
            mpol=windingsurface.mpol, ntor=windingsurface.ntor, nfp=windingsurface.nfp,
            stellsym=windingsurface.stellsym,
            quadpoints_phi=np.linspace(0, 1, 128, endpoint=False),
            quadpoints_theta=np.linspace(0, 1, 128, endpoint=False),
            dofs=windingsurface.dofs,
        )
        J_min_tf_dipole_curvecurvedistance = CurveSurfaceDistance(
            tf_curves, windingsurface_eval, min_tf_dipole_curvecurvedistance_threshold)
        
        if normalize_units:
            surface_area = float(np.mean(np.linalg.norm(windingsurface_eval.normal(), axis=2)))
            J_min_tf_dipole_curvecurvedistance = ScaledOptimizable(
                1.0 / (sum(_arclength(c) for c in tf_curves) * surface_area
                    * min_tf_dipole_curvecurvedistance_threshold**2),
                J_min_tf_dipole_curvecurvedistance)
        class wrapper_tf_dipole_curvecurvedistance():
            def __init__(self, _J_min_tf_dipole_curvecurvedistance):
                self._J_min_tf_dipole_curvecurvedistance = _unwrap(_J_min_tf_dipole_curvecurvedistance)

            def J(self):
                return self._J_min_tf_dipole_curvecurvedistance.shortest_distance()

        _metric = CurveSurfaceDistance(tf_curves, windingsurface_eval, 4*min_tf_dipole_curvecurvedistance_threshold) # pyright: ignore[reportOperatorIssue]
        min_tf_dipole_curvecurvedistance = wrapper_tf_dipole_curvecurvedistance(_metric)

    objectives = dict(
        J_max_dipole_curvelength=J_max_dipole_curvelength,
        J_min_dipole_curvecurvedistance=J_min_dipole_curvecurvedistance,
        J_min_dipole_curvesurfacedistance=J_min_dipole_curvesurfacedistance,
        J_max_dipole_meansquaredcurvature=J_max_dipole_meansquaredcurvature,
        J_max_dipole_lpcurvecurvature=J_max_dipole_lpcurvecurvature,
        J_max_tf_curvelength=J_max_tf_curvelength,
        J_min_tf_curvecurvedistance=J_min_tf_curvecurvedistance,
        J_max_tf_meansquaredcurvature=J_max_tf_meansquaredcurvature,
        J_max_tf_lpcurvecurvature=J_max_tf_lpcurvecurvature,
        J_min_tf_dipole_curvecurvedistance=J_min_tf_dipole_curvecurvedistance,
        max_dipole_curvelength=max_dipole_curvelength,
        min_dipole_curvecurvedistance=min_dipole_curvecurvedistance,
        min_dipole_curvesurfacedistance=min_dipole_curvesurfacedistance,
        max_dipole_lpcurvecurvature=max_dipole_lpcurvecurvature,
        max_dipole_meansquaredcurvature=max_dipole_meansquaredcurvature,
        max_tf_curvelength=max_tf_curvelength,
        min_tf_curvecurvedistance=min_tf_curvecurvedistance,
        max_tf_lpcurvecurvature=max_tf_lpcurvecurvature,
        max_tf_meansquaredcurvature=max_tf_meansquaredcurvature,
        min_tf_dipole_curvecurvedistance=min_tf_dipole_curvecurvedistance,
    )
    return objectives

def build_curve_objectives_parser(
    *,
    max_dipole_curvelength_required: bool = False,
    min_dipole_curvecurvedistance_required: bool = False,
    min_dipole_curvesurfacedistance_required: bool = False,
    max_dipole_meansquaredcurvature_required: bool = False,
    dipole_lpcurvecurvature_p_value_required: bool = False,
    max_dipole_lpcurvecurvature_required: bool = False,
    max_tf_curvelength_required: bool = False,
    min_tf_curvecurvedistance_required: bool = False,
    max_tf_meansquaredcurvature_required: bool = False,
    tf_lpcurvecurvature_p_value_required: bool = False,
    max_tf_lpcurvecurvature_required: bool = False,
    min_tf_dipole_curvecurvedistance_required: bool = False,
    max_windingsurface_aspectratio_required: bool = False,
    weight_max_dipole_curvelength_required: bool = False,
    weight_min_dipole_curvecurvedistance_required: bool = False,
    weight_min_dipole_curvesurfacedistance_required: bool = False,
    weight_max_dipole_meansquaredcurvature_required: bool = False,
    weight_max_dipole_lpcurvecurvature_required: bool = False,
    weight_max_tf_curvelength_required: bool = False,
    weight_min_tf_curvecurvedistance_required: bool = False,
    weight_max_tf_meansquaredcurvature_required: bool = False,
    weight_max_tf_lpcurvecurvature_required: bool = False,
    weight_min_tf_dipole_curvecurvedistance_required: bool = False,
    weight_max_windingsurface_aspectratio_required: bool = False,
    **kwargs,
):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--max-dipole-curvelength-threshold", type=float, default=None, required=max_dipole_curvelength_required,
        help="Maximum allowed dipole curve length. Default: None (no penalty)."
    )
    parser.add_argument(
        "--min-dipole-curvecurvedistance-threshold", type=float, default=None, required=min_dipole_curvecurvedistance_required,
        help="Minimum allowed distance between dipole curves. Default: None (no penalty)."
    )
    parser.add_argument(
        "--min-dipole-curvesurfacedistance-threshold", type=float, default=None, required=min_dipole_curvesurfacedistance_required,
        help="Minimum allowed distance between dipole curves and a boundary surface. Default: None (no penalty)."
    )
    parser.add_argument(
        "--max-dipole-meansquaredcurvature-threshold", type=float, default=None, required=max_dipole_meansquaredcurvature_required,
        help="Maximum allowed dipole mean squared curvature. Default: None (no penalty)."
    )
    parser.add_argument(
        "--dipole-lpcurvecurvature-p-value", type=float, default=None, required=dipole_lpcurvecurvature_p_value_required,
        help="Lp norm exponent to penalize peak dipole curve curvature. Default: None (no penalty)."
    )
    parser.add_argument(
        "--max-dipole-lpcurvecurvature-threshold", type=float, default=None, required=max_dipole_lpcurvecurvature_required,
        help="Maximum allowed Lp norm of dipole curve curvature. Default: None (no penalty)."
    )
    parser.add_argument(
        "--max-tf-curvelength-threshold", type=float, default=None, required=max_tf_curvelength_required,
        help="Maximum allowed TF curve length. Default: None (no penalty)."
    )
    parser.add_argument(
        "--min-tf-curvecurvedistance-threshold", type=float, default=None, required=min_tf_curvecurvedistance_required,
        help="Minimum allowed distance between TF curves. Default: None (no penalty)."
    )
    parser.add_argument(
        "--max-tf-meansquaredcurvature-threshold", type=float, default=None, required=max_tf_meansquaredcurvature_required,
        help="Maximum allowed TF mean squared curvature. Default: None (no penalty)."
    )
    parser.add_argument(
        "--tf-lpcurvecurvature-p-value", type=float, default=None, required=tf_lpcurvecurvature_p_value_required,
        help="Lp norm exponent to penalize peak TF curve curvature. Default: None (no penalty)."
    )
    parser.add_argument(
        "--max-tf-lpcurvecurvature-threshold", type=float, default=None, required=max_tf_lpcurvecurvature_required,
        help="Maximum allowed Lp norm of TF curve curvature. Default: None (no penalty)."
    )
    parser.add_argument(
        "--min-tf-dipole-curvecurvedistance-threshold", type=float, default=None, required=min_tf_dipole_curvecurvedistance_required,
        help="Minimum allowed distance between TF curves and dipole curves. Default: None (no penalty)."
    )
    parser.add_argument(
        "--weight-max-dipole-curvelength", type=float, default=None, required=weight_max_dipole_curvelength_required,
        help="Weight for the dipole curve length penalty. Default: None (no penalty)."
    )
    parser.add_argument(
        "--weight-min-dipole-curvecurvedistance", type=float, default=None, required=weight_min_dipole_curvecurvedistance_required,
        help="Weight for the dipole curve-curve distance penalty. Default: None (no penalty)."
    )
    parser.add_argument(
        "--weight-min-dipole-curvesurfacedistance", type=float, default=None, required=weight_min_dipole_curvesurfacedistance_required,
        help="Weight for the dipole curve-surface distance penalty. Default: None (no penalty)."
    )
    parser.add_argument(
        "--weight-max-dipole-meansquaredcurvature", type=float, default=None, required=weight_max_dipole_meansquaredcurvature_required,
        help="Weight for the dipole mean squared curvature penalty. Default: None (no penalty)."
    )
    parser.add_argument(
        "--weight-max-dipole-lpcurvecurvature", type=float, default=None, required=weight_max_dipole_lpcurvecurvature_required,
        help="Weight for the dipole Lp curve curvature penalty. Default: None (no penalty)."
    )
    parser.add_argument(
        "--weight-max-tf-curvelength", type=float, default=None, required=weight_max_tf_curvelength_required,
        help="Weight for the TF curve length penalty. Default: None (no penalty)."
    )
    parser.add_argument(
        "--weight-min-tf-curvecurvedistance", type=float, default=None, required=weight_min_tf_curvecurvedistance_required,
        help="Weight for the TF curve-curve distance penalty. Default: None (no penalty)."
    )
    parser.add_argument(
        "--weight-max-tf-meansquaredcurvature", type=float, default=None, required=weight_max_tf_meansquaredcurvature_required,
        help="Weight for the TF mean squared curvature penalty. Default: None (no penalty)."
    )
    parser.add_argument(
        "--weight-max-tf-lpcurvecurvature", type=float, default=None, required=weight_max_tf_lpcurvecurvature_required,
        help="Weight for the TF Lp curve curvature penalty. Default: None (no penalty)."
    )
    parser.add_argument(
        "--weight-min-tf-dipole-curvecurvedistance", type=float, default=None, required=weight_min_tf_dipole_curvecurvedistance_required,
        help="Weight for the TF-dipole curve-curve distance penalty. Default: None (no penalty)."
    )
    parser.add_argument(
        "--use-native-curve-units", action="store_true", required=False,
        help=(
            "If set, curve penalties will contain units related to meters which needs to be accounted for in weights. "
            "Default: False (curve penalties are normalized to their respective thresholds)."
        )
    )
    return parser

def build_current_objectives(
    biotsavart: BiotSavart,
    *,
    max_current_threshold: float | None = None,
    current_p_value: float | None = None,
    use_native_current_units: bool = False,
    **kwargs,
) -> dict:
    coils = biotsavart.coils
    dipole_base_coils = get_dipole_base_coils(coils)
    dipole_base_currents = [coil.current for coil in dipole_base_coils if coil is not None]

    normalize_units = not use_native_current_units

    if (max_current_threshold is None) or (current_p_value is None):
        J_max_current = None
        max_current = None
    else:
        J_max_current = LpCurrentPenalty(
            dipole_base_currents, p=current_p_value, threshold=max_current_threshold)
        if normalize_units:
            J_max_current = ScaledOptimizable(
                1.0 / (len(dipole_base_currents) * max_current_threshold**current_p_value),
                J_max_current)
        class wrapper_current():
            def __init__(self, _dipole_base_currents):
                self._dipole_base_currents = _dipole_base_currents

            def J(self): # Return the actual max current, not the Lp norm penalty
                return np.max([abs(c.get_value()) for c in self._dipole_base_currents])
        max_current = wrapper_current(dipole_base_currents)

    if current_p_value is None:
        J_peak_current = None
    else:
        J_peak_current = CurrentPenalty(dipole_base_currents, p=current_p_value)
        if normalize_units and (max_current_threshold is not None):
            J_peak_current = ScaledOptimizable(
                1.0 / (len(dipole_base_currents)**(1.0/current_p_value) * max_current_threshold),
                J_peak_current)

    objectives = dict(
        J_max_current=J_max_current,
        J_peak_current=J_peak_current,
        max_current=max_current,
    )
    return objectives

def build_current_objectives_parser(
    *,
    max_current_required: bool = False,
    current_p_value_required: bool = False,
    weight_max_current_required: bool = False,
    weight_peak_current_required: bool = False,
    **kwargs,
):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--max-current-threshold", type=float, default=None, required=max_current_required,
        help="Maximum allowed current in the dipole coils. Default: None (no penalty)."
    )
    parser.add_argument(
        "--current-p-value", type=float, default=None, required=current_p_value_required,
        help="Lp norm exponent to penalize current maxes. Applies to both max current and peak current. Default: None (no penalty)."
    )
    parser.add_argument(
        "--weight-max-current", type=float, default=None, required=weight_max_current_required,
        help="Weight for the maximum current penalty. This applies the max current threshold. Default: None (no penalty)."
    )
    parser.add_argument(
        "--weight-peak-current", type=float, default=None, required=weight_peak_current_required,
        help="Weight for the peak current penalty. This penalizes the max current regardless of magnitude. Default: None (no penalty)."
    )
    parser.add_argument(
        "--use-native-current-units", action="store_true", required=False,
        help=(
            "If set, current penalties will contain units related to Amperes which needs to be accounted for in weights. "
            "Default: False (current penalties are normalized to their respective thresholds). "
            "Note that peak current penalties are normalized to --max-current-threshold."
        )
    )
    return parser

def build_toroidalflux_objective(
):
    # ToroidalFlux and QfmResidual
    # Can eventually combine with the MultiSurfaceSquaredFlux objective
    # since they're all biotsavart-surface objectives
    pass

def build_surface_objectives(
    surface: SurfaceRZFourier | SurfaceXYZTensorFourier,
    *,
    max_aspectratio_threshold: float | None = None,
    **kwargs,
):
    if max_aspectratio_threshold is None:
        J_max_aspectratio = None
        max_aspectratio = None
    else:
        max_aspectratio = AspectRatio(surface, nphi=64, ntheta=64) # The metric itself isn't a "max" per se, this is just for bookkeeping
        J_max_aspectratio = QuadraticPenalty(max_aspectratio, max_aspectratio_threshold, "max")
    
    # Area
    # Volume
    # PrincipalCurvature

    objectives = dict(
        J_max_aspectratio=J_max_aspectratio,
        max_aspectratio=max_aspectratio,
    )
    return objectives

def build_windingsurface_objectives(
    windingsurface: SurfaceRZFourier | SurfaceXYZTensorFourier,
    *,
    max_windingsurface_aspectratio_threshold: float | None = None,
    **kwargs,
):
    # Just a wrapper to separate input names
    objectives = build_surface_objectives(
        windingsurface,
        max_aspectratio_threshold=max_windingsurface_aspectratio_threshold,
        **kwargs
    )
    return {
        key.replace("aspectratio", "windingsurface_aspectratio"): val
        for key, val in objectives.items()
    }

def build_windingsurface_objectives_parser(
    *,
    max_windingsurface_aspectratio_required: bool = False,
    weight_max_windingsurface_aspectratio_required: bool = False,
    **kwargs,
):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--max-windingsurface-aspectratio-threshold", type=float, default=None, required=max_windingsurface_aspectratio_required,
        help="Maximum allowed aspect ratio of the dipole winding surface. Default: None (no penalty)."
    )
    parser.add_argument(
        "--weight-max-windingsurface-aspectratio", type=float, default=None, required=weight_max_windingsurface_aspectratio_required,
        help="Weight for the winding surface aspect ratio penalty. Default: None (no penalty)."
    )
    return parser

def check_objective_kwargs(
    *,
    log = None,
    **kwargs
):
    if log is None:
        log = proc0_print
    
    input_keys = list(kwargs.keys())

    required = dict(REQUIRED_OBJECTIVE_KWARGS)
    if not kwargs.get("use_native_current_units", False):
        required["peak_current"] = required["peak_current"] + ["max_current_threshold"]

    ierr = 0
    log("Checking missing input(s) for constructing objective...")
    for key, required_kwargs in required.items():
        missing = []
        for kwarg in required_kwargs:
            if kwarg not in input_keys:
                missing.append(kwarg)
        if (
            (len(missing) == len(required_kwargs)) or # missing all = no objective
            (len(missing) == 0)
        ):
            continue
        else:
            ierr += 1
            log(f"    [{key}] Missing required input(s): {', '.join(['--' + m.replace('_', '-') for m in missing])}")

    if ierr > 0:
        msg = f"{ierr} objective(s) are missing required inputs."
        log(msg)
        if (comm_world is not None) and (comm_world.size > 1):
            comm_world.Abort(1)
        raise Exception(msg)
    else:
        log("All required inputs for constructing objectives are present.")

def get_virtualcasing_target(
    surface: SurfaceRZFourier | SurfaceXYZTensorFourier,
    *,
    virtualcasing: VirtualCasing | None,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]] | None:
    if virtualcasing is None:
        return None

    target_in = virtualcasing.B_external_normal_extended # pyright: ignore[reportAttributeAccessIssue]

    surf_qpts_phi = surface.quadpoints_phi
    surf_qpts_theta = surface.quadpoints_theta

    vc_nphi_in, vc_ntheta_in, _ = target_in.shape
    vc_qpts_phi_in = np.linspace(0, 1, vc_nphi_in, endpoint=False)
    vc_qpts_theta_in = np.linspace(0, 1, vc_ntheta_in, endpoint=False)

    dbl_vc_qpts_phi_in = np.concatenate([vc_qpts_phi_in, vc_qpts_phi_in + 1.0])
    dbl_vc_qpts_theta_in = np.concatenate([vc_qpts_theta_in, vc_qpts_theta_in])
    quad_target_in = np.tile(target_in, (2, 2) + (1,) * (target_in.ndim - 2))

    interpolator = RegularGridInterpolator(
        (dbl_vc_qpts_phi_in, dbl_vc_qpts_theta_in),
        quad_target_in,
        bounds_error=False,
        fill_value=None,
    )
    phi_mesh, theta_mesh = np.meshgrid(surf_qpts_phi, surf_qpts_theta, indexing="ij")
    points = np.stack((phi_mesh, theta_mesh), axis=-1)
    target = interpolator(points)

    return target
