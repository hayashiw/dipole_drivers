import argparse

from simsopt.util import proc0_print

from .coil import get_dipole_base_coils, get_tf_base_coils
from ..defaults import *

__all__ = [
    "apply_dof_selection",
    "build_dof_selection_parser",
    "log_dof_options",
]

_DOF_ALIASES = {
    "curves": [
        ("CurveSuperEllipse", None),
        ("ScaledCurveSuperEllipse", None),
        ("CurvePlanarEllipticalCylindrical", None),
        ("SurfaceRZFourier", None)], # windingsurface
    "currents": [
        ("Current", None),
    ],
    "dipole-currents": [
        ("@dipole-current", None)],
    "dipole-a": [
        ("CurveSuperEllipse", "a"),
        ("ScaledCurveSuperEllipse", "a")],
    "dipole-b": [
        ("CurveSuperEllipse", "b"),
        ("ScaledCurveSuperEllipse", "b")],
    "dipole-n": [
        ("CurveSuperEllipse", "n"),
        ("ScaledCurveSuperEllipse", "n")],
    "dipole-curves": [
        ("CurveSuperEllipse", None),
        ("ScaledCurveSuperEllipse", None)],
    "tf-currents": [
        ("@tf-current", None)],
    "tf-major-radius": [
        ("CurvePlanarEllipticalCylindrical", "R0")],
    # "tf-minor-radius-r": ...
    # "tf-minor-radius-z": ...1
    "tf-rotation": [
        ("CurvePlanarEllipticalCylindrical", "r_rotation")],
    "tf-curves": [
        ("CurvePlanarEllipticalCylindrical", None)],
    "windingsurface": [
        ("SurfaceRZFourier", None)],
    "windingsurface-major-radius": [
        ("SurfaceRZFourier", "rc(0,0)")],
    "windingsurface-minor-radius-r": [
        ("SurfaceRZFourier", "rc(1,0)")],
    "windingsurface-minor-radius-z": [
        ("SurfaceRZFourier", "zs(1,0)")],
}

def apply_dof_selection(optimizable, fix=(), unfix=()):
    coils = getattr(optimizable, "coils", [])
    owner_sets = {
        "@dipole-current": {id(o) for c in get_dipole_base_coils(coils)
                            for o in c.current.unique_dof_lineage},
        "@tf-current": {id(o) for c in get_tf_base_coils(coils)
                        for o in c.current.unique_dof_lineage},
    }

    structural = {
        (id(owner), name)
        for owner in optimizable.unique_dof_lineage
        for name, free in zip(owner.local_full_dof_names, owner.local_dofs_free_status)
        if not free
    }

    targets = {}
    for aliases, should_fix in ((fix, True), (unfix, False)):
        for alias in aliases:
            if alias not in _DOF_ALIASES:
                raise ValueError(
                    f"unknown dof alias {alias!r}; expected one of "
                    f"{sorted(_DOF_ALIASES)}.")
            for pair in _DOF_ALIASES[alias]:
                targets[pair] = should_fix

    sorted_targets = sorted(
        targets.items(),
        key=lambda item: (item[0][1] is None, not item[0][0].startswith("@")))
    for owner in optimizable.unique_dof_lineage:
        names = list(owner.local_full_dof_names)
        if not names:
            continue
        owner_type = type(owner).__name__
        matched = {}
        for index, name in enumerate(names):
            for (want_type, want_dof), should_fix in sorted_targets:
                if want_dof not in (None, name):
                    continue
                if want_type.startswith("@"):
                    if id(owner) not in owner_sets[want_type]:
                        continue
                elif owner_type != want_type:
                    continue
                matched[name] = should_fix
                break

        if not matched:
            continue
        # One call per owner re-indexes the graph once instead of once per dof,
        # which matters for the winding surface: every dipole curve depends on it.
        if len(matched) == len(names) and len(set(matched.values())) == 1:
            if next(iter(matched.values())):
                owner.local_fix_all()
                continue
            if not any((id(owner), name) in structural for name in names):
                owner.local_unfix_all()
                continue

        free_status = list(owner.local_dofs_free_status)
        for index, name in enumerate(names):
            should_fix = matched.get(name)
            if should_fix is None:
                continue
            if should_fix and free_status[index]:
                owner.fix(name)
            elif ((not should_fix) and (not free_status[index])
                  and (id(owner), name) not in structural):
                owner.unfix(name)

def build_dof_selection_parser(**kwargs):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--fix-dofs", type=str, nargs="+", default=DEFAULT_FIX_DOFS, required=False,
        help=f"Dof aliases to fix. Use --list-dofs to see the available names. Default: {DEFAULT_FIX_DOFS}."
    )
    parser.add_argument(
        "--unfix-dofs", type=str, nargs="+", default=DEFAULT_UNFIX_DOFS, required=False,
        help=f"Dof aliases to unfix, applied after --fix-dofs. Prioritizes specificity. Default: {DEFAULT_UNFIX_DOFS}."
    )
    parser.add_argument(
        "--list-dofs", action="store_true", required=False,
        help="Log the available dof aliases and exit without optimizing. Default: False."
    )
    return parser

def log_dof_options(log=proc0_print):
    aliases = sorted(_DOF_ALIASES)
    log("Available dof aliases for --fix-dofs and --unfix-dofs:")
    for alias in aliases:
        pairs = ", ".join(
            (f"{owner[1:]} objects" if owner.startswith("@") else f"{owner}:{dof or '*'}")
            for owner, dof in _DOF_ALIASES[alias])
        log(f"    {alias:<32}{pairs}")
    log("")
    log("Values are whitespace-separated:")
    log(f"    --fix-dofs {' '.join(aliases[:2])}")
    log(f"    --unfix-dofs {aliases[0]}")
