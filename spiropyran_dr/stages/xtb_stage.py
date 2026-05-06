"""Stage 3: Pre-CREST constrained xTB seed optimisation (xtb_constr).

Two jobs (one per base diastereomer, anti and syn), producing MECP-mimic
seed geometries for the constrained CREST branch (_mecp labels). The
optimisation is a GFN2-xTB run with a $constrain block fixing the C-O
distance at the MECP value from config (mecp.c_o_distance_angstrom).

See project.md section 10.3.
"""

from __future__ import annotations

import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spiropyran_dr.io_utils import (
    read_xyz,
    write_xcontrol_distance_constraint,
)
from spiropyran_dr.pbs_utils import (
    PBSSubmitError,
    submit_via_script,
    write_jobid,
)

LABELS: tuple[str, str] = ("anti", "syn")

_METHOD_FLAG: dict[str, list[str]] = {
    "gfn2": ["--gfn", "2"],
}


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _label_dir(workspace: Path, label: str) -> Path:
    return workspace / "xtb_constr" / label


def _lowest_energy_mm_xyz(mm_outputs: dict[str, Any], label: str) -> str | None:
    """Return the relative xyz path of the lowest-energy MM conformer for a label.

    Index 0 is lowest-energy because the mm stage writes conformers in
    ascending energy order.
    """
    entries = mm_outputs.get(label) or []
    if not entries:
        return None
    return entries[0]["xyz"]


def _constraint_atoms_0based(prep_outputs: dict[str, Any]) -> tuple[int, int]:
    return (
        int(prep_outputs["spiro_carbon_idx"]),
        int(prep_outputs["chromene_oxygen_idx"]),
    )


def _xtb_method_flag(method: str) -> list[str]:
    if method not in _METHOD_FLAG:
        raise ValueError(
            f"unsupported xtb method {method!r}; supported: {list(_METHOD_FLAG)}"
        )
    return _METHOD_FLAG[method]


def _parse_xtb_total_energy(out_path: Path) -> float:
    """Parse the final TOTAL ENERGY from an xtb output file.

    xtb writes lines like:
        | TOTAL ENERGY     -22.12345678 Eh |
    We take the last such line (final converged geometry).
    """
    value = None
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if "TOTAL ENERGY" in line:
            parts = line.split()
            for i, tok in enumerate(parts):
                if tok == "ENERGY" and i + 1 < len(parts):
                    try:
                        value = float(parts[i + 1])
                    except ValueError:
                        pass
    if value is None:
        raise ValueError(f"no TOTAL ENERGY line found in {out_path}")
    return value


def _measure_co_distance(xyz_path: Path, idx_a_0: int, idx_b_0: int) -> float:
    """Return Euclidean distance (Å) between two atoms in a single-frame XYZ."""
    _, coords, _ = read_xyz(xyz_path)
    ax, ay, az = coords[idx_a_0]
    bx, by, bz = coords[idx_b_0]
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


# -- stage interface -------------------------------------------------------


def is_ready(manifest: dict[str, Any], workspace: Path) -> bool:
    stages = manifest.get("stages") or {}
    mm_stage = stages.get("mm") or {}
    if mm_stage.get("status") != "done":
        return False
    mm_outputs = mm_stage.get("outputs") or {}
    if mm_outputs.get("n_conformers_anti", 0) < 1:
        return False
    if mm_outputs.get("n_conformers_syn", 0) < 1:
        return False
    prep_outputs = (stages.get("prep") or {}).get("outputs") or {}
    return "spiro_carbon_idx" in prep_outputs and "chromene_oxygen_idx" in prep_outputs


def submit(
    manifest: dict[str, Any], workspace: Path, config: dict[str, Any]
) -> dict[str, Any]:
    started = _now_iso()
    mm_outputs = manifest["stages"]["mm"]["outputs"]
    prep_outputs = manifest["stages"]["prep"]["outputs"]
    xtb_cfg = config.get("xtb_constr") or {}
    mecp_cfg = config.get("mecp") or {}
    walltime_hours = int(xtb_cfg["walltime_hours"])
    script_path = Path(str(xtb_cfg["script_path"])).expanduser()
    method = str(xtb_cfg.get("method", "gfn2"))
    method_flags = _xtb_method_flag(method)

    distance_ang = float(mecp_cfg["c_o_distance_angstrom"])
    force_constant = float(mecp_cfg["constraint_force_constant"])
    idx_a, idx_b = _constraint_atoms_0based(prep_outputs)

    pbs_job_ids: dict[str, str] = {}

    for label in LABELS:
        src_rel = _lowest_energy_mm_xyz(mm_outputs, label)
        if src_rel is None:
            return {
                "status": "failed",
                "started_at": started,
                "finished_at": _now_iso(),
                "failure_reason": f"mm stage produced no conformers for label {label!r}",
            }

        src_abs = workspace / src_rel
        if not src_abs.is_file():
            return {
                "status": "failed",
                "started_at": started,
                "finished_at": _now_iso(),
                "failure_reason": f"missing MM input xyz: {src_abs}",
            }

        work_dir = _label_dir(workspace, label)
        work_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_abs, work_dir / "input.xyz")
        write_xcontrol_distance_constraint(
            work_dir / "xtb.inp",
            idx_a,
            idx_b,
            distance_ang,
            force_constant,
        )

        args = [
            str(walltime_hours),
            "input.xyz",
            "--opt",
            *method_flags,
            "--input",
            "xtb.inp",
        ]
        try:
            jobid, _ = submit_via_script(script_path, args, cwd=work_dir)
        except PBSSubmitError as exc:
            return {
                "status": "failed",
                "started_at": started,
                "finished_at": _now_iso(),
                "failure_reason": f"sub_xtb.sh failed for {label}: {exc}",
            }

        write_jobid(work_dir / "jobid", jobid)
        pbs_job_ids[label] = jobid

    return {
        "status": "submitted",
        "started_at": started,
        "submitted_at": started,
        "pbs_job_ids": pbs_job_ids,
    }


def collect(
    manifest: dict[str, Any], workspace: Path, config: dict[str, Any]
) -> dict[str, Any]:
    mecp_cfg = config.get("mecp") or {}
    xtb_cfg = config.get("xtb_constr") or {}
    target_dist = float(mecp_cfg["c_o_distance_angstrom"])
    tolerance = float(xtb_cfg["co_distance_tolerance_angstrom"])
    prep_outputs = ((manifest.get("stages") or {}).get("prep") or {}).get(
        "outputs"
    ) or {}
    idx_a, idx_b = _constraint_atoms_0based(prep_outputs)

    outputs: dict[str, list[dict[str, Any]]] = {}

    # sub_xtb.sh runs `xtb --namespace input ...`, so xtb's own outputs
    # (xtbopt.xyz, xtbopt.log, ...) and the wrapper's stdout redirection
    # (xtb.log) are all prefixed with the basename of the input geometry.
    for label in LABELS:
        work_dir = _label_dir(workspace, label)
        xtbopt_path = work_dir / "input.xtbopt.xyz"
        xtblog_path = work_dir / "input.xtb.log"

        if not xtbopt_path.is_file():
            return {
                "status": "failed",
                "finished_at": _now_iso(),
                "failure_reason": f"input.xtbopt.xyz missing for {label}",
            }
        if not xtblog_path.is_file():
            return {
                "status": "failed",
                "finished_at": _now_iso(),
                "failure_reason": f"input.xtb.log missing for {label}",
            }

        try:
            energy = _parse_xtb_total_energy(xtblog_path)
        except ValueError as exc:
            return {
                "status": "failed",
                "finished_at": _now_iso(),
                "failure_reason": f"input.xtb.log parse failed for {label}: {exc}",
            }

        co_dist = _measure_co_distance(xtbopt_path, idx_a, idx_b)
        if abs(co_dist - target_dist) > tolerance:
            return {
                "status": "failed",
                "finished_at": _now_iso(),
                "failure_reason": (
                    f"constraint violation for {label}: C-O = {co_dist:.4f} Ang, "
                    f"target = {target_dist} Ang, tolerance = {tolerance} Ang"
                ),
            }

        xyz_rel = f"xtb_constr/{label}/input.xtbopt.xyz"
        outputs[label] = [
            {
                "conf_id": 0,
                "xyz": xyz_rel,
                "energy_hartree": energy,
                "co_distance_final_ang": co_dist,
                "label": label,
            }
        ]

    return {
        "status": "done",
        "finished_at": _now_iso(),
        "outputs": outputs,
    }
