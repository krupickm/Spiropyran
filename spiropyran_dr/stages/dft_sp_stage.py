"""Stage 5: ORCA single-point energies at r2SCAN-3c/CPCM level.

One PBS job per conformer per label (4 labels x N conformers, typically up to
4 x 20 = 80 jobs per molecule). Each job runs ORCA on a single geometry so
the SCF guess starts from scratch -- a multi-frame *xyzfile job reuses the
previous frame's MOs as the next guess, which silently corrupts energies for
chemically distinct conformers. Submitted via the user-maintained
suborca.sh wrapper.

See project.md section 10.5.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spiropyran_dr.io_utils import (
    check_orca_normal_termination,
    parse_orca_sp_energy,
)
from spiropyran_dr.pbs_utils import (
    PBSSubmitError,
    submit_via_script,
    write_jobid,
)

LABELS: tuple[str, str, str, str] = ("anti_min", "syn_min", "anti_mecp", "syn_mecp")


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _label_dir(workspace: Path, label: str) -> Path:
    return workspace / "dft_sp" / label


def _conf_dir(workspace: Path, label: str, conf_id: int) -> Path:
    return _label_dir(workspace, label) / f"conf_{conf_id}"


def _job_key(label: str, conf_id: int) -> str:
    return f"{label}/{conf_id}"


def _write_orca_inp(
    path: Path,
    method: str,
    solvent_name: str,
    ncpus: int,
    mem_per_core_mb: int,
    xyz_filename: str,
) -> None:
    """Write an ORCA single-point input file using plain CPCM solvation.

    Solvation model is hardcoded to CPCM (r2SCAN-3c was parametrized with
    CPCM; SMD choice is deferred to a later pipeline version).
    """
    content = (
        f"! {method} CPCM({solvent_name})\n"
        f"\n"
        f"%pal nprocs {ncpus} end\n"
        f"%maxcore {mem_per_core_mb}\n"
        f"\n"
        f"*xyzfile 0 1 {xyz_filename}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def is_ready(manifest: dict[str, Any], workspace: Path) -> bool:
    crest = manifest.get("stages", {}).get("crest", {})
    if crest.get("status") != "done":
        return False
    outputs = crest.get("outputs", {})
    return all(label in outputs and len(outputs[label]) > 0 for label in LABELS)


def submit(
    manifest: dict[str, Any], workspace: Path, config: dict[str, Any]
) -> dict[str, Any]:
    started_at = _now_iso()
    dft_sp_cfg = config["dft_sp"]
    script_path = Path(dft_sp_cfg["script_path"])
    walltime_hours = int(dft_sp_cfg["walltime_hours"])
    method = dft_sp_cfg["method"]
    ncpus = int(dft_sp_cfg["ncpus"])
    mem_per_core_mb = int(dft_sp_cfg["mem_per_core_mb"])
    solvent_name = config["dft"]["solvent"]["name"]

    crest_outputs = manifest["stages"]["crest"]["outputs"]
    pbs_job_ids: dict[str, str] = {}

    for label in LABELS:
        conformers = crest_outputs[label]
        for conf in conformers:
            conf_id = int(conf["conf_id"])
            src_xyz = Path(conf["xyz"])
            if not src_xyz.is_absolute():
                src_xyz = workspace / src_xyz

            conf_dir = _conf_dir(workspace, label, conf_id)
            conf_dir.mkdir(parents=True, exist_ok=True)

            xyz_filename = f"conf_{conf_id}.xyz"
            shutil.copyfile(src_xyz, conf_dir / xyz_filename)

            orca_inp = conf_dir / "orca.inp"
            _write_orca_inp(
                orca_inp, method, solvent_name, ncpus, mem_per_core_mb, xyz_filename
            )

            try:
                jobid, _ = submit_via_script(
                    script_path,
                    ["orca.inp", str(walltime_hours)],
                    cwd=conf_dir,
                )
            except PBSSubmitError as exc:
                return {
                    "status": "failed",
                    "started_at": started_at,
                    "finished_at": _now_iso(),
                    "failure_reason": (
                        f"submission failed for {_job_key(label, conf_id)!r}: {exc}"
                    ),
                }

            write_jobid(conf_dir / "jobid", jobid)
            pbs_job_ids[_job_key(label, conf_id)] = jobid

    return {
        "status": "submitted",
        "started_at": started_at,
        "submitted_at": _now_iso(),
        "pbs_job_ids": pbs_job_ids,
    }


def collect(
    manifest: dict[str, Any], workspace: Path, config: dict[str, Any]
) -> dict[str, Any]:
    crest_outputs = manifest["stages"]["crest"]["outputs"]
    outputs: dict[str, list[dict[str, Any]]] = {}

    for label in LABELS:
        conformers = crest_outputs[label]
        label_entries: list[dict[str, Any]] = []
        for conf in conformers:
            conf_id = int(conf["conf_id"])
            key = _job_key(label, conf_id)
            orca_out = _conf_dir(workspace, label, conf_id) / "orca.out"

            if not orca_out.exists():
                return {
                    "status": "failed",
                    "finished_at": _now_iso(),
                    "failure_reason": f"orca.out missing for {key!r}: {orca_out}",
                }

            if not check_orca_normal_termination(orca_out):
                return {
                    "status": "failed",
                    "finished_at": _now_iso(),
                    "failure_reason": (f"ORCA did not terminate normally for {key!r}"),
                }

            try:
                energy = parse_orca_sp_energy(orca_out)
            except ValueError as exc:
                return {
                    "status": "failed",
                    "finished_at": _now_iso(),
                    "failure_reason": f"{key!r}: {exc}",
                }

            label_entries.append(
                {
                    "conf_id": conf_id,
                    "xyz": conf["xyz"],
                    "energy_hartree": energy,
                    "label": label,
                }
            )

        outputs[label] = label_entries

    return {
        "status": "done",
        "finished_at": _now_iso(),
        "outputs": outputs,
    }
