from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from spiropyran_dr.config_utils import load_config
from spiropyran_dr.stages import crest_stage, mm, prep, xtb_stage

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_ROOT / "config" / "default.yaml"
DEFAULT_SMARTS = PACKAGE_ROOT / "config" / "smarts.yaml"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spiropyran-dr",
        description="Spiropyran d.r. pipeline CLI (early development).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prep", help="Run stage 1 (prep) on a single SMILES.")
    p_prep.add_argument("smiles", help="Input SMILES string for the closed spiropyran.")
    p_prep.add_argument(
        "--workspace",
        type=Path,
        default=Path("./run_scratch"),
        help="Workspace directory; prep/stereocentres.json is written here.",
    )
    p_prep.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to pipeline config YAML.",
    )
    p_prep.add_argument(
        "--smarts",
        type=Path,
        default=DEFAULT_SMARTS,
        help="Path to atom-role SMARTS YAML.",
    )
    p_prep.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit the full submit() return dict as JSON on stdout.",
    )

    p_mm = sub.add_parser(
        "mm",
        help="Run stages 1-2 (prep + MM) on a single SMILES.",
    )
    p_mm.add_argument("smiles", help="Input SMILES string for the closed spiropyran.")
    p_mm.add_argument(
        "--workspace",
        type=Path,
        default=Path("./run_scratch"),
        help="Workspace directory; mm/{anti,syn}/conf_*.xyz are written here.",
    )
    p_mm.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to pipeline config YAML.",
    )
    p_mm.add_argument(
        "--smarts",
        type=Path,
        default=DEFAULT_SMARTS,
        help="Path to atom-role SMARTS YAML.",
    )
    p_mm.add_argument(
        "--n-embed",
        type=int,
        default=None,
        help="Override mm.n_embed from config (e.g. for fast smoke runs).",
    )
    p_mm.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override mm.random_seed from config.",
    )
    p_mm.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit the full submit() return dict as JSON on stdout.",
    )

    p_xtb = sub.add_parser(
        "xtb_constr",
        help="Run stages 1-3 (prep + MM + constrained xTB submit). Submits 2 PBS jobs "
        "(one per diastereomer) and exits.",
    )
    p_xtb.add_argument("smiles", help="Input SMILES string for the closed spiropyran.")
    p_xtb.add_argument(
        "--workspace",
        type=Path,
        default=Path("./run_scratch"),
        help="Workspace directory; xtb_constr/{anti,syn}/ are created here.",
    )
    p_xtb.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to pipeline config YAML.",
    )
    p_xtb.add_argument(
        "--smarts",
        type=Path,
        default=DEFAULT_SMARTS,
        help="Path to atom-role SMARTS YAML.",
    )
    p_xtb.add_argument(
        "--n-embed",
        type=int,
        default=None,
        help="Override mm.n_embed from config.",
    )
    p_xtb.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override mm.random_seed from config.",
    )
    p_xtb.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit the full submit() return dict as JSON on stdout.",
    )

    p_crest = sub.add_parser(
        "crest",
        help="Submit CREST (stage 4) assuming xtb_constr is already done. "
        "Loads manifest.json from the workspace and submits 4 PBS jobs.",
    )
    p_crest.add_argument(
        "--workspace",
        type=Path,
        default=Path("./run_scratch"),
        help="Workspace directory containing manifest.json.",
    )
    p_crest.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to pipeline config YAML.",
    )
    p_crest.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit the full submit() return dict as JSON on stdout.",
    )

    return parser


def _run_prep(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    config.setdefault("paths", {})["smarts"] = str(args.smarts)
    manifest = {"smiles_input": args.smiles}
    args.workspace.mkdir(parents=True, exist_ok=True)
    result = prep.submit(manifest, args.workspace, config)

    if args.as_json:
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        if result["status"] == "done":
            out = result["outputs"]
            print("status: done")
            print(f"smiles_canonical:   {out['smiles_canonical']}")
            print(f"spiro_carbon_idx:   {out['spiro_carbon_idx']}")
            print(f"chromene_oxygen_idx:{out['chromene_oxygen_idx']}")
            print(f"spiro_cip:          {out['spiro_cip']}")
            print(f"sidecar:            {args.workspace / out['stereocentres_path']}")
        else:
            print("status: failed", file=sys.stderr)
            print(
                f"reason: {result.get('failure_reason', '<unknown>')}", file=sys.stderr
            )

    return 0 if result["status"] == "done" else 1


def _run_mm(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    config.setdefault("paths", {})["smarts"] = str(args.smarts)
    if args.n_embed is not None:
        config["mm"]["n_embed"] = args.n_embed
    if args.seed is not None:
        config["mm"]["random_seed"] = args.seed

    args.workspace.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"smiles_input": args.smiles, "stages": {}}
    prep_result = prep.submit(manifest, args.workspace, config)
    manifest["stages"]["prep"] = prep_result
    if prep_result["status"] != "done":
        if args.as_json:
            json.dump(prep_result, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
        else:
            print("status: failed (prep)", file=sys.stderr)
            print(
                f"reason: {prep_result.get('failure_reason', '<unknown>')}",
                file=sys.stderr,
            )
        return 1

    result = mm.submit(manifest, args.workspace, config)

    if args.as_json:
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        if result["status"] == "done":
            out = result["outputs"]
            print("status: done")
            print(f"n_conformers_anti: {out['n_conformers_anti']}")
            print(f"n_conformers_syn:  {out['n_conformers_syn']}")
            print(f"anti_xyz_dir:      {args.workspace / out['anti_xyz_dir']}")
            print(f"syn_xyz_dir:       {args.workspace / out['syn_xyz_dir']}")
        else:
            print("status: failed", file=sys.stderr)
            print(
                f"reason: {result.get('failure_reason', '<unknown>')}", file=sys.stderr
            )

    return 0 if result["status"] == "done" else 1


def _run_xtb_constr(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    config.setdefault("paths", {})["smarts"] = str(args.smarts)
    if args.n_embed is not None:
        config["mm"]["n_embed"] = args.n_embed
    if args.seed is not None:
        config["mm"]["random_seed"] = args.seed

    args.workspace.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"smiles_input": args.smiles, "stages": {}}

    prep_result = prep.submit(manifest, args.workspace, config)
    manifest["stages"]["prep"] = prep_result
    if prep_result["status"] != "done":
        if args.as_json:
            json.dump(prep_result, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
        else:
            print("status: failed (prep)", file=sys.stderr)
            print(
                f"reason: {prep_result.get('failure_reason', '<unknown>')}",
                file=sys.stderr,
            )
        return 1

    mm_result = mm.submit(manifest, args.workspace, config)
    manifest["stages"]["mm"] = mm_result
    if mm_result["status"] != "done":
        if args.as_json:
            json.dump(mm_result, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
        else:
            print("status: failed (mm)", file=sys.stderr)
            print(
                f"reason: {mm_result.get('failure_reason', '<unknown>')}",
                file=sys.stderr,
            )
        return 1

    result = xtb_stage.submit(manifest, args.workspace, config)

    if args.as_json:
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        if result["status"] == "submitted":
            print("status: submitted")
            for label, jobid in result["pbs_job_ids"].items():
                print(f"{label} jobid: {jobid}")
            print(f"work_dirs: {args.workspace / 'xtb_constr'}")
        else:
            print("status: failed", file=sys.stderr)
            print(
                f"reason: {result.get('failure_reason', '<unknown>')}", file=sys.stderr
            )

    return 0 if result["status"] == "submitted" else 1


def _run_crest_resume(args: argparse.Namespace) -> int:
    manifest_path = args.workspace / "manifest.json"
    if not manifest_path.is_file():
        print("status: failed", file=sys.stderr)
        print(
            f"reason: manifest.json not found in {args.workspace}; "
            "run xtb_constr first",
            file=sys.stderr,
        )
        return 1

    with manifest_path.open(encoding="utf-8") as fh:
        manifest = json.load(fh)

    xtb_status = (manifest.get("stages") or {}).get("xtb_constr", {}).get("status")
    if xtb_status != "done":
        print("status: failed", file=sys.stderr)
        print(
            f"reason: xtb_constr stage is {xtb_status!r}, must be 'done' before CREST",
            file=sys.stderr,
        )
        return 1

    config = load_config(args.config)
    result = crest_stage.submit(manifest, args.workspace, config)

    if args.as_json:
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        if result["status"] == "submitted":
            print("status: submitted")
            for label, jobid in result["pbs_job_ids"].items():
                print(f"{label} jobid: {jobid}")
            print(f"work_dirs: {args.workspace / 'crest'}")
        else:
            print("status: failed", file=sys.stderr)
            print(
                f"reason: {result.get('failure_reason', '<unknown>')}", file=sys.stderr
            )

    return 0 if result["status"] == "submitted" else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "prep":
        return _run_prep(args)
    if args.command == "mm":
        return _run_mm(args)
    if args.command == "xtb_constr":
        return _run_xtb_constr(args)
    if args.command == "crest":
        return _run_crest_resume(args)
    parser.error(f"unknown command: {args.command}")
    return 2
