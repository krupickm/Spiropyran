"""Stage 3: CREST conformational sampling per diastereomer.

See project.md section 10.3. Submission is delegated to the user-maintained
`sub_crest.sh` wrapper (config: ``crest.script_path``), which writes its own
PBS file and qsub's it. The wrapper hardcodes NPROC=7 (experimentally
optimal on MetaCentrum) and the orchestrator passes no CREST flags --
GFN2 search, ewin = 6 kcal/mol, and other defaults all come from the
wrapper plus CREST itself. Method choice is intentionally not configurable
here; if it ever needs to change, edit ``sub_crest.sh`` upstream rather
than threading a flag through Python.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from spiropyran_dr.io_utils import (
    read_crest_energies,
    read_xyz_multiframe,
    write_xyz_from_arrays,
)
from spiropyran_dr.pbs_utils import (
    PBSSubmitError,
    submit_via_script,
    write_jobid,
)

Label = Literal["anti", "syn"]
LABELS: tuple[Label, Label] = ("anti", "syn")


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _label_dir(workspace: Path, label: Label) -> Path:
    return workspace / "crest" / label


def _lowest_energy_mm_xyz(mm_outputs: dict[str, Any], label: Label) -> str | None:
    """Return the relative xyz path of the lowest-energy MM conformer for a label.

    The mm stage already writes its conformers in ascending energy order
    (see stages/mm.py: `cluster_by_rmsd` walks an energy-sorted list).
    Index 0 is therefore the lowest-energy survivor for that label.
    """
    entries = mm_outputs.get(label) or []
    if not entries:
        return None
    return entries[0]["xyz"]


# -- stage interface ------------------------------------------------------


def is_ready(manifest: dict[str, Any], workspace: Path) -> bool:
    mm_stage = (manifest.get("stages") or {}).get("mm") or {}
    if mm_stage.get("status") != "done":
        return False
    outputs = mm_stage.get("outputs") or {}
    return (outputs.get("n_conformers_anti", 0) >= 1
            and outputs.get("n_conformers_syn", 0) >= 1)


def submit(
    manifest: dict[str, Any], workspace: Path, config: dict[str, Any]
) -> dict[str, Any]:
    started = _now_iso()
    mm_outputs = manifest["stages"]["mm"]["outputs"]
    crest_cfg = config.get("crest") or {}
    walltime_hours = int(crest_cfg["walltime_hours"])
    script_path = Path(str(crest_cfg["script_path"])).expanduser()

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
        dest = work_dir / "input.xyz"
        shutil.copyfile(src_abs, dest)

        try:
            jobid, _ = submit_via_script(
                script_path,
                [str(walltime_hours), "input.xyz"],
                cwd=work_dir,
            )
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
    crest_cfg = config.get("crest") or {}
    _ = crest_cfg  # currently no collect-time knobs; kept for future ewin overrides
    max_per = int(
        (config.get("ensemble") or {}).get("max_conformers_per_diastereomer", 20)
    )

    outputs_per_label: dict[str, list[dict[str, Any]]] = {"anti": [], "syn": []}

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

        # CREST writes its ensemble already sorted by energy; sort defensively
        # in case a future version stops doing so.
        order = sorted(range(len(energies)), key=lambda k: energies[k])
        kept = order[:max_per]

        filtered_dir = work_dir / "filtered"
        filtered_dir.mkdir(parents=True, exist_ok=True)
        e_min = energies[kept[0]]
        for new_id, frame_idx in enumerate(kept):
            symbols, coords, _ = frames[frame_idx]
            xyz_rel = Path("crest") / label / "filtered" / f"conf_{new_id}.xyz"
            comment = (
                f"label={label} conf_id={new_id} "
                f"crest_energy_hartree={energies[frame_idx]:.8f}"
            )
            write_xyz_from_arrays(workspace / xyz_rel, symbols, coords, comment)
            outputs_per_label[label].append(
                {
                    "conf_id": new_id,
                    "source_frame": frame_idx,
                    "xyz": str(xyz_rel).replace("\\", "/"),
                    "energy_hartree": energies[frame_idx],
                    # 627.5094740631 kcal/mol per Hartree (CODATA, exact in
                    # the unit conversion sense; CREST itself uses this).
                    "relative_energy_kcal_mol": (
                        (energies[frame_idx] - e_min) * 627.5094740631
                    ),
                    "label": label,
                }
            )

    return {
        "status": "done",
        "finished_at": _now_iso(),
        "outputs": {
            "n_conformers_anti": len(outputs_per_label["anti"]),
            "n_conformers_syn": len(outputs_per_label["syn"]),
            "anti_xyz_dir": "crest/anti/filtered",
            "syn_xyz_dir": "crest/syn/filtered",
            "anti": outputs_per_label["anti"],
            "syn": outputs_per_label["syn"],
        },
    }
