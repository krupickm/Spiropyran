"""Stage 5: ORCA single-point energies at r2SCAN-3c/CPCM level.

One PBS job per label (anti_min, syn_min, anti_mecp, syn_mecp). Each job
receives a multi-frame XYZ of the filtered CREST conformers and computes SP
energies sequentially via ORCA's *xyzfile directive. Submitted via the
user-maintained suborca.sh wrapper.

See project.md section 10.5.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spiropyran_dr.io_utils import (
    check_orca_normal_termination,
    parse_orca_sp_energies,
    read_xyz,
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


def _concatenate_xyz(src_paths: list[str | Path], dest: Path) -> None:
    """Concatenate individual XYZ files into one multi-frame file.

    Each source is a single-frame XYZ. The frames are appended in order;
    the comment line from each source frame is preserved verbatim.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for p in src_paths:
        symbols, coords, comment = read_xyz(Path(p))
        lines.append(str(len(symbols)))
        lines.append(comment)
        for sym, (x, y, z) in zip(symbols, coords):
            lines.append(f"{sym} {x:.8f} {y:.8f} {z:.8f}")
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        label_dir = _label_dir(workspace, label)
        label_dir.mkdir(parents=True, exist_ok=True)

        xyz_paths = [c["xyz"] for c in conformers]
        # Resolve paths relative to workspace if not absolute.
        abs_xyz_paths = [
            p if Path(p).is_absolute() else workspace / p for p in xyz_paths
        ]
        conformers_xyz = label_dir / "conformers.xyz"
        _concatenate_xyz(abs_xyz_paths, conformers_xyz)

        orca_inp = label_dir / "orca.inp"
        _write_orca_inp(
            orca_inp, method, solvent_name, ncpus, mem_per_core_mb, "conformers.xyz"
        )

        try:
            jobid, _ = submit_via_script(
                script_path,
                ["orca.inp", str(walltime_hours)],
                cwd=label_dir,
            )
        except PBSSubmitError as exc:
            return {
                "status": "failed",
                "started_at": started_at,
                "finished_at": _now_iso(),
                "failure_reason": f"submission failed for label {label!r}: {exc}",
            }

        write_jobid(label_dir / "jobid", jobid)
        pbs_job_ids[label] = jobid

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
        label_dir = _label_dir(workspace, label)
        orca_out = label_dir / "orca.out"

        if not orca_out.exists():
            return {
                "status": "failed",
                "finished_at": _now_iso(),
                "failure_reason": f"orca.out missing for label {label!r}: {orca_out}",
            }

        if not check_orca_normal_termination(orca_out):
            return {
                "status": "failed",
                "finished_at": _now_iso(),
                "failure_reason": f"ORCA did not terminate normally for label {label!r}",
            }

        try:
            energies = parse_orca_sp_energies(orca_out)
        except ValueError as exc:
            return {
                "status": "failed",
                "finished_at": _now_iso(),
                "failure_reason": str(exc),
            }

        conformers = crest_outputs[label]
        if len(energies) != len(conformers):
            return {
                "status": "failed",
                "finished_at": _now_iso(),
                "failure_reason": (
                    f"energy count mismatch for label {label!r}: "
                    f"ORCA reported {len(energies)} energies but "
                    f"{len(conformers)} conformers were submitted"
                ),
            }

        outputs[label] = [
            {
                "conf_id": c["conf_id"],
                "xyz": c["xyz"],
                "energy_hartree": e,
                "label": label,
            }
            for c, e in zip(conformers, energies)
        ]

    return {
        "status": "done",
        "finished_at": _now_iso(),
        "outputs": outputs,
    }
