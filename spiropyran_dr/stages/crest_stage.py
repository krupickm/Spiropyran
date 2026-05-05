"""Stage 4: CREST conformational sampling — 4 parallel jobs.

Labels: anti_min, syn_min, anti_mecp, syn_mecp.

_min labels run unconstrained CREST, seeded from the lowest MM conformer.
_mecp labels run under a C-O distance constraint (written as .xcontrol),
seeded from the xtb_constr output for that base diastereomer.

See project.md section 10.4.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spiropyran_dr.io_utils import (
    read_crest_energies,
    read_xyz_multiframe,
    write_xcontrol_distance_constraint,
    write_xyz_from_arrays,
)
from spiropyran_dr.pbs_utils import (
    PBSSubmitError,
    submit_via_script,
    write_jobid,
)

LABELS: tuple[str, str, str, str] = ("anti_min", "syn_min", "anti_mecp", "syn_mecp")


def _is_mecp(label: str) -> bool:
    return label.endswith("_mecp")


def _base_label(label: str) -> str:
    return label.removesuffix("_min").removesuffix("_mecp")


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _label_dir(workspace: Path, label: str) -> Path:
    return workspace / "crest" / label


def _lowest_energy_mm_xyz(mm_outputs: dict[str, Any], base: str) -> str | None:
    """Return the relative xyz path of the lowest-energy MM conformer for a base label."""
    entries = mm_outputs.get(base) or []
    if not entries:
        return None
    return entries[0]["xyz"]


# -- stage interface ------------------------------------------------------


def is_ready(manifest: dict[str, Any], workspace: Path) -> bool:
    stages = manifest.get("stages") or {}

    xtb_stage = stages.get("xtb_constr") or {}
    if xtb_stage.get("status") != "done":
        return False
    xtb_outputs = xtb_stage.get("outputs") or {}
    if not xtb_outputs.get("anti") or not xtb_outputs.get("syn"):
        return False

    mm_stage = stages.get("mm") or {}
    if mm_stage.get("status") != "done":
        return False
    mm_outputs = mm_stage.get("outputs") or {}
    return (
        mm_outputs.get("n_conformers_anti", 0) >= 1
        and mm_outputs.get("n_conformers_syn", 0) >= 1
    )


def submit(
    manifest: dict[str, Any], workspace: Path, config: dict[str, Any]
) -> dict[str, Any]:
    started = _now_iso()
    mm_outputs = manifest["stages"]["mm"]["outputs"]
    xtb_outputs = manifest["stages"]["xtb_constr"]["outputs"]
    prep_outputs = (manifest.get("stages") or {}).get("prep", {}).get("outputs") or {}
    crest_cfg = config.get("crest") or {}
    mecp_cfg = config.get("mecp") or {}
    walltime_hours = int(crest_cfg["walltime_hours"])
    script_path = Path(str(crest_cfg["script_path"])).expanduser()

    # Atom indices are required for writing .xcontrol on _mecp labels. Both
    # come from SMARTS detection in prep; their absence is a pipeline stopper.
    if (
        "spiro_carbon_idx" not in prep_outputs
        or "chromene_oxygen_idx" not in prep_outputs
    ):
        return {
            "status": "failed",
            "started_at": started,
            "finished_at": _now_iso(),
            "failure_reason": (
                "prep outputs missing spiro_carbon_idx or chromene_oxygen_idx; "
                "re-run prep before submitting CREST"
            ),
        }
    idx_a = int(prep_outputs["spiro_carbon_idx"])
    idx_b = int(prep_outputs["chromene_oxygen_idx"])

    pbs_job_ids: dict[str, str] = {}

    for label in LABELS:
        base = _base_label(label)

        if _is_mecp(label):
            entries = xtb_outputs.get(base) or []
            if not entries:
                return {
                    "status": "failed",
                    "started_at": started,
                    "finished_at": _now_iso(),
                    "failure_reason": f"xtb_constr stage has no output for base label {base!r}",
                }
            src_rel = entries[0]["xyz"]
        else:
            src_rel = _lowest_energy_mm_xyz(mm_outputs, base)
            if src_rel is None:
                return {
                    "status": "failed",
                    "started_at": started,
                    "finished_at": _now_iso(),
                    "failure_reason": f"mm stage produced no conformers for label {base!r}",
                }

        src_abs = workspace / src_rel
        if not src_abs.is_file():
            return {
                "status": "failed",
                "started_at": started,
                "finished_at": _now_iso(),
                "failure_reason": f"missing seed xyz: {src_abs}",
            }

        work_dir = _label_dir(workspace, label)
        work_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_abs, work_dir / "input.xyz")

        wrapper_args = [str(walltime_hours), "input.xyz"]

        if _is_mecp(label):
            write_xcontrol_distance_constraint(
                work_dir / ".xcontrol",
                idx_a,
                idx_b,
                float(mecp_cfg.get("c_o_distance_angstrom", 3.4)),
                float(mecp_cfg.get("constraint_force_constant", 1.0)),
            )
            wrapper_args += ["--cinp", ".xcontrol"]

        try:
            jobid, _ = submit_via_script(script_path, wrapper_args, cwd=work_dir)
        except PBSSubmitError as exc:
            return {
                "status": "failed",
                "started_at": started,
                "finished_at": _now_iso(),
                "failure_reason": f"sub_crest.sh failed for {label}: {exc}",
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
    max_per = int(
        (config.get("ensemble") or {}).get("max_conformers_per_diastereomer", 20)
    )

    outputs: dict[str, Any] = {}

    for label in LABELS:
        work_dir = _label_dir(workspace, label)
        confs_path = work_dir / "crest_conformers.xyz"
        energies_path = work_dir / "crest.energies"
        if not confs_path.is_file() or not energies_path.is_file():
            return {
                "status": "failed",
                "finished_at": _now_iso(),
                "failure_reason": (
                    f"CREST output missing for {label}: "
                    f"have {confs_path.exists()}/{energies_path.exists()} "
                    f"for (conformers.xyz / energies)"
                ),
            }

        frames = read_xyz_multiframe(confs_path)
        energies = read_crest_energies(energies_path)
        if len(frames) != len(energies):
            return {
                "status": "failed",
                "finished_at": _now_iso(),
                "failure_reason": (
                    f"frame/energy count mismatch for {label}: "
                    f"{len(frames)} frames vs {len(energies)} energies"
                ),
            }
        if not frames:
            return {
                "status": "failed",
                "finished_at": _now_iso(),
                "failure_reason": f"CREST produced no conformers for {label}",
            }

        order = sorted(range(len(energies)), key=lambda k: energies[k])
        kept = order[:max_per]

        filtered_dir = work_dir / "filtered"
        filtered_dir.mkdir(parents=True, exist_ok=True)
        e_min = energies[kept[0]]
        label_entries: list[dict[str, Any]] = []
        for new_id, frame_idx in enumerate(kept):
            symbols, coords, _ = frames[frame_idx]
            xyz_rel = Path("crest") / label / "filtered" / f"conf_{new_id}.xyz"
            comment = (
                f"label={label} conf_id={new_id} "
                f"crest_energy_hartree={energies[frame_idx]:.8f}"
            )
            write_xyz_from_arrays(workspace / xyz_rel, symbols, coords, comment)
            label_entries.append(
                {
                    "conf_id": new_id,
                    "source_frame": frame_idx,
                    "xyz": str(xyz_rel).replace("\\", "/"),
                    "energy_hartree": energies[frame_idx],
                    # 627.5094740631 kcal/mol per Hartree (CODATA)
                    "relative_energy_kcal_mol": (
                        (energies[frame_idx] - e_min) * 627.5094740631
                    ),
                    "label": label,
                }
            )

        outputs[f"n_conformers_{label}"] = len(label_entries)
        outputs[f"{label}_xyz_dir"] = f"crest/{label}/filtered"
        outputs[label] = label_entries

    return {
        "status": "done",
        "finished_at": _now_iso(),
        "outputs": outputs,
    }
