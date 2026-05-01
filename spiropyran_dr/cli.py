from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from spiropyran_dr.config_utils import load_config
from spiropyran_dr.stages import prep

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
            print(f"status: done")
            print(f"smiles_canonical:   {out['smiles_canonical']}")
            print(f"spiro_carbon_idx:   {out['spiro_carbon_idx']}")
            print(f"chromene_oxygen_idx:{out['chromene_oxygen_idx']}")
            print(f"spiro_cip:          {out['spiro_cip']}")
            print(f"sidecar:            {args.workspace / out['stereocentres_path']}")
        else:
            print(f"status: failed", file=sys.stderr)
            print(f"reason: {result.get('failure_reason', '<unknown>')}", file=sys.stderr)

    return 0 if result["status"] == "done" else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "prep":
        return _run_prep(args)
    parser.error(f"unknown command: {args.command}")
    return 2
