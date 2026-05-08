"""PBS submission helpers: job submission, qstat polling, PBS script generation."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


class PBSSubmitError(RuntimeError):
    """Raised when a submission script invocation fails or its output is unparseable."""


def parse_jobid_from_qsub_stdout(text: str) -> str:
    """Return the PBS job id from a submission script's stdout.

    `qsub` prints the job id (e.g. ``12345.meta-pbs.metacentrum.cz``) as its
    final non-blank line. Some wrappers (including ``sub_crest.sh``) write
    informational chatter beforehand, so we take the *last* non-blank line
    rather than the first.
    """
    last = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            last = stripped
    if not last:
        raise PBSSubmitError("submission script produced no output to parse jobid from")
    return last


def write_jobid(path: Path, jobid: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(jobid + "\n", encoding="utf-8")


def read_jobid(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def poll_job_state(jobid: str) -> str:
    """Query qstat for a single job. Returns 'running', 'finished', or 'not_found'.

    'finished' covers PBS terminal states C and F.
    'not_found' covers non-zero qstat exit (job purged from queue) and the case
    where qstat is not installed (developer laptop).
    """
    try:
        proc = subprocess.run(
            ["qstat", "-f", jobid],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return "not_found"
    if proc.returncode != 0:
        return "not_found"
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("job_state"):
            state = stripped.split("=", 1)[-1].strip()
            if state in ("C", "F"):
                return "finished"
            return "running"
    return "not_found"


def is_all_jobs_done(pbs_job_ids: dict[str, str]) -> bool:
    """Return True when every job in the dict is finished or not found in qstat."""
    return all(poll_job_state(jid) != "running" for jid in pbs_job_ids.values())


def generate_orchestrator_pbs(
    workspace: Path,
    config: dict[str, Any],
    python_exe: str,
) -> str:
    """Return the PBS script text for the long-running orchestrator job.

    The script cd's into the workspace and runs `predict` in unbuffered mode
    so that orchestrator.log is visible on NFS without extra sync calls.
    """
    pbs = config.get("pbs", {})
    queue = pbs.get("queue_orchestrator", "oven@meta-pbs.metacentrum.cz")
    walltime = pbs.get("walltime_orchestrator", "720:00:00")
    ws = str(workspace.resolve())
    return (
        "#!/bin/bash\n"
        f"#PBS -N spiropyran_dr\n"
        f"#PBS -q {queue}\n"
        f"#PBS -l walltime={walltime}\n"
        "#PBS -l select=1:ncpus=1:mem=4gb\n"
        "\n"
        f"cd {ws}\n"
        f"exec {python_exe} -u -m spiropyran_dr predict --workspace {ws}\n"
    )


def submit_via_script(script: Path, args: list[str], cwd: Path) -> tuple[str, str]:
    """Run a submission script in `cwd`, capture stdout, return (jobid, raw_stdout).

    The script is expected to call ``qsub`` itself and echo the resulting
    job id (this is what ``sub_crest.sh`` does). Any non-zero exit, or
    unparseable stdout, is wrapped as ``PBSSubmitError``.
    """
    cmd = [str(script), *args]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise PBSSubmitError(
            f"submission script {script} failed (exit {exc.returncode}): "
            f"stderr={exc.stderr!r}"
        ) from exc
    except FileNotFoundError as exc:
        raise PBSSubmitError(f"submission script not found: {script}") from exc
    jobid = parse_jobid_from_qsub_stdout(proc.stdout)
    return jobid, proc.stdout
