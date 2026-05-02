"""Minimal PBS submission helpers.

Only what the CREST stage needs today; xTB and ORCA stages will extend this
with template rendering and richer qstat parsing later (see project.md
section 9). Keeping this module intentionally small avoids speculative
abstraction.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


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


def submit_via_script(
    script: Path, args: list[str], cwd: Path
) -> tuple[str, str]:
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
