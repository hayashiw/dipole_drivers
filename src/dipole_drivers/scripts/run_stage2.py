import argparse
import inspect
import os
import shutil
import time

from datetime import timedelta, datetime

from ..defaults import *
from ..drivers.logging import DriverLog
from ..drivers.stage2 import build_stage2_parser, stage2
from ..utils.biotsavart import generate_biotsavart, generate_biotsavart_from_surface
from ..utils.surface import build_surface_wout_parser, generate_surfaces_from_wout
from ..utils.dofs import *

SURFACES_KWARGS = [
    name for name, param in inspect.signature(generate_surfaces_from_wout).parameters.items()
    if param.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD)
]
BIOTSAVART_KWARGS = [
    name for name, param in inspect.signature(generate_biotsavart).parameters.items()
    if param.kind is inspect.Parameter.KEYWORD_ONLY
    and name not in ("nfp", "skip_outboard_pol", "superellipse_n", "nquadpoints")
]

required_kwargs = {
    key+"_required": True for key in [
        "surface_wout_file",
    ]
}

def build_parser():
    parser = argparse.ArgumentParser(
        description="Run stage 2 optimization for dipole coil design.",
        parents=[
            build_stage2_parser(**required_kwargs),
            build_surface_wout_parser(**required_kwargs),
            build_dof_selection_parser(**required_kwargs),
        ],
    )
    parser.add_argument(
        "--save-tag", type=str, default=None, required=False,
        help="Tag for naming save files. Default: None (derived from wout file name)."
    )
    parser.add_argument(
        "--out-dir", type=str, default=".", required=False,
        help="Directory to save output files. Default: current directory."
    )
    return parser

def main(argv=None):
    if build_dof_selection_parser().parse_known_args(argv)[0].list_dofs:
        log_dof_options()
        return 0

    args = build_parser().parse_args(argv)
    assert os.path.exists(args.surface_wout_file), f"Wout file {args.surface_wout_file} does not exist."

    wout_file = os.path.abspath(args.surface_wout_file)
    base_file = os.path.basename(wout_file)

    if hasattr(args, "save_tag") and args.save_tag is not None:
        tag = args.save_tag
    else:
        tag = base_file.replace("wout_", "").replace(".nc", "")

    if hasattr(args, "out_dir") and args.out_dir is not None:
        out_dir = os.path.abspath(args.out_dir)
    else:
        out_dir = os.path.abspath(".")
    os.makedirs(out_dir, exist_ok=True)

    log_file = os.path.join(out_dir, f"{tag}.stage2.log")
    log = DriverLog(log_file)
    start_time = time.monotonic()
    log(f" ----- {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')} ----- ")
    log("log file: " + log_file)

    kwargs = {}
    log("Input parameters:")
    for key, val in vars(args).items():
        if (val is not None) and (not (hasattr(val, "__len__") and len(val) == 0)):
            suffix = " (default)" if ( (key in DEFAULTS_INPUTS) and (val == DEFAULTS_INPUTS[key]) ) else ""
            log(f"    {key}: {val}{suffix}")
            kwargs[key] = val
    kwargs["tag"] = tag
    log("")

    surface_kwargs = {k: v for k, v in kwargs.items() if k in SURFACES_KWARGS}
    surfaces = generate_surfaces_from_wout(args.surface_wout_file, **surface_kwargs)
    for surface in surfaces:
        assert surface.stellsym, \
            f"Surface not stellarator-symmetric. Check wout file {args.surface_wout_file}."
    boundary_surface = surfaces[-1]
    kwargs["boundary_surface"] = boundary_surface
    log("Boundary surface parameters:")
    log(f"    mpol   = {boundary_surface.mpol}")
    log(f"    ntor   = {boundary_surface.ntor}")
    log(f"    nfp    = {boundary_surface.nfp}")
    log(f"    nphi   = {boundary_surface.quadpoints_phi.size}")
    log(f"    ntheta = {boundary_surface.quadpoints_theta.size}")
    log("")

    missing_any_biotsavart_kwargs = any(k not in kwargs for k in BIOTSAVART_KWARGS)
    if missing_any_biotsavart_kwargs:
        log("Missing some biotsavart parameters. Generating coil parameters from boundary surface.")
        biotsavart = generate_biotsavart_from_surface(boundary_surface, **kwargs)
    else:
        kwargs["nfp"] = int(boundary_surface.nfp)
        biotsavart = generate_biotsavart(**kwargs)
    apply_dof_selection(
        biotsavart,
        fix=kwargs.get("fix_dofs", []),
        unfix=kwargs.get("unfix_dofs", []),
    )
    biotsavart_preopt_file = os.path.join(out_dir, f"{tag}.biotsavart.stage2.preopt.json")
    biotsavart.save(biotsavart_preopt_file)
    log(f"Saved pre-optimization biotsavart to {biotsavart_preopt_file}")
    for s_i, surface in zip(surface_kwargs.get("surface_wout_s", [DEFAULT_SURFACE_WOUT_S]), surfaces):
        s_i_str = str(round(s_i, 9)).replace(".", "d")
        surface_file = os.path.join(out_dir, f"{tag}.surface_s{s_i_str}.json")
        surface.save(surface_file)
        log(f"Saved surface s={s_i} to {surface_file}")
    log("")

    res = stage2(biotsavart, surfaces, log, **kwargs)

    skip_optimization = kwargs.get("skip_optimization", False)
    if not skip_optimization:
        if res.success:
            suffix = ""
        else:
            suffix = ".failed" # failed can include maxiter, abnormal, etc.

        biotsavart_postopt_file = os.path.join(out_dir, f"{tag}.biotsavart.stage2.postopt{suffix}.json")
        biotsavart.save(biotsavart_postopt_file)
        log(f"Saved post-optimization biotsavart to {biotsavart_postopt_file}")
        log("")

    end_time = time.monotonic()
    elapsed_time = str(timedelta(seconds=end_time - start_time))
    log(f"Run time: {elapsed_time}")
    log(f" ----- {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')} ----- ")

    if (not res.success) and (not skip_optimization):
        # copy log file to failed log file
        failed_log_file = log_file.replace(".log", ".failed.log")
        shutil.copy(log_file, failed_log_file)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
