import numpy as np

from mpi4py import MPI
from scipy.interpolate import RegularGridInterpolator
from typing import Sequence

from simsopt.field import BiotSavart
from simsopt.geo import (
    AspectRatio,
    BoozerResidual,
    CurveCurveDistance,
    CurveSurfaceDistance,
    Iotas,
    MeanSquaredCurvature,
    SurfaceRZFourier,
    SurfaceXYZTensorFourier,
)
from simsopt.mhd import VirtualCasing
from simsopt.objectives import SquaredFlux, QuadraticPenalty

def build_objective_function():
    pass

def build_squared_flux_objectives(
    biotsavart: BiotSavart,
    surfaces: Sequence[SurfaceRZFourier | SurfaceXYZTensorFourier],
    virtualcasings: Sequence[VirtualCasing | None] | None = None,
    **kwargs,
) -> list[SquaredFlux]:
    if virtualcasings is None:
        virtualcasings = [None] * len(surfaces)
    else:
        assert len(surfaces) == len(virtualcasings), \
            "The number of surfaces must match the number of virtual casings."

    objectives = []
    for surface, virtualcasing in zip(surfaces, virtualcasings):
        target = get_virtualcasing_target(surface, virtualcasing)
        objective = SquaredFlux(biotsavart, surface, target=target, **kwargs)
        objectives.append(objective)
    return objectives

def build_coil_objectives(
    biotsavart: BiotSavart,
    boundary_surface: SurfaceRZFourier | SurfaceXYZTensorFourier,
    min_curvecurvedistance: float,
    min_curvesurfacedistance: float,
    max_meansquaredcurvature: float,
):
    coils = biotsavart.coils
    curves = [coil.curve for coil in coils]

    CurveCurveDistance(curves, min_curvecurvedistance)

def get_virtualcasing_target(
    surface: SurfaceRZFourier | SurfaceXYZTensorFourier,
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
