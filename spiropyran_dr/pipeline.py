"""Blocking orchestrator loop for the spiropyran d.r. pipeline.

Called from inside the PBS orchestrator job via `predict_dr.py predict`.
Reads manifest.json from the workspace, walks STAGE_ORDER, submits PBS jobs,
polls for completion, collects outputs, and exits when all stages are done.

See project.md section 3 for the orchestrator model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from spiropyran_dr.config_utils import compute_config_hash
from spiropyran_dr.io_utils import atomic_write_json
from spiropyran_dr.pbs_utils import is_all_jobs_done
from spiropyran_dr.stages import STAGE_IS_PBS, STAGE_ORDER, get_stage_module


class PipelineError(RuntimeError):
    """Raised when a stage fails or the manifest is in an unrecoverable state."""


class _FlushingFileHandler(logging.FileHandler):
    """FileHandler that flushes after every record.

    NFS caches can hold buffered data for seconds; flushing here ensures
    orchestrator.log is readable from a login node without extra sync calls.
    """

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def molecule_id_from_smiles(smiles_canonical: str) -> str:
    """Return a short stable ID derived from the canonical SMILES."""
    digest = hashlib.sha256(smiles_canonical.encode()).hexdigest()[:8]
    return f"sp_{digest}"


def _setup_logger(workspace: Path) -> logging.Logger:
    logger = logging.getLogger("spiropyran_dr.pipeline")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh = _FlushingFileHandler(workspace / "orchestrator.log", encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def run(workspace: Path, config: dict[str, Any]) -> None:
    """Blocking orchestrator loop.

    Expects manifest.json to exist in `workspace` with at least prep and mm
    marked done (written by `predict_dr.py submit`). Walks STAGE_ORDER from
    the first non-terminal stage, submitting PBS jobs and sleeping between
    polls. Raises PipelineError on any stage failure or config hash mismatch.
    """
    logger = _setup_logger(workspace)
    manifest_path = workspace / "manifest.json"

    if not manifest_path.is_file():
        raise PipelineError(
            f"manifest.json not found in {workspace}. Run 'predict_dr.py submit' first."
        )

    with manifest_path.open(encoding="utf-8") as fh:
        manifest = json.load(fh)

    # Validate config has not changed since submit.
    stored_hash = manifest.get("config_hash")
    if stored_hash:
        current_hash = compute_config_hash(config)
        if current_hash != stored_hash:
            raise PipelineError(
                f"Config hash mismatch: manifest has {stored_hash!r}, "
                f"current config gives {current_hash!r}. "
                "Use the same config as during submit, or reset the stage to pending."
            )

    poll_interval: int = config.get("polling", {}).get("interval_seconds", 60)
    stages = manifest.setdefault("stages", {})
    options: dict[str, Any] = manifest.get("options", {})

    logger.info(
        "Orchestrator starting. molecule_id=%s workspace=%s",
        manifest.get("molecule_id", "<unknown>"),
        workspace,
    )

    while True:
        for stage_name in STAGE_ORDER:
            stage_block = stages.setdefault(stage_name, {"status": "pending"})
            status = stage_block.get("status", "pending")

            if status in ("done", "skipped"):
                continue

            if status == "failed":
                raise PipelineError(
                    f"Stage {stage_name!r} is in failed state: "
                    f"{stage_block.get('failure_reason', '<no reason recorded>')}"
                )

            # dft_freq is skipped when thermal corrections were not requested.
            if stage_name == "dft_freq" and not options.get("thermal", False):
                logger.info("Stage dft_freq: skipped (--no-thermal).")
                stage_block["status"] = "skipped"
                atomic_write_json(manifest_path, manifest)
                continue

            mod = get_stage_module(stage_name)
            if mod is None:
                logger.info("Stage %s: module not implemented, skipping.", stage_name)
                stage_block["status"] = "skipped"
                atomic_write_json(manifest_path, manifest)
                continue

            if status in ("submitted", "running"):
                pbs_ids = stage_block.get("pbs_job_ids", {})
                if is_all_jobs_done(pbs_ids):
                    logger.info(
                        "Stage %s: all PBS jobs finished, collecting.", stage_name
                    )
                    result = mod.collect(manifest, workspace, config)
                    stage_block.update(result)
                    atomic_write_json(manifest_path, manifest)
                    if result.get("status") == "failed":
                        raise PipelineError(
                            f"Stage {stage_name!r} collect failed: "
                            f"{result.get('failure_reason', '<no reason>')}"
                        )
                    logger.info("Stage %s: done.", stage_name)
                    # Collected successfully; continue to submit next stage in
                    # the same pass rather than restarting the for loop.
                    continue
                else:
                    logger.info(
                        "Stage %s: PBS jobs still running (%s). Sleeping %ds.",
                        stage_name,
                        list(pbs_ids.keys()),
                        poll_interval,
                    )
                    time.sleep(poll_interval)
                    break

            else:
                # status == "pending"
                if not mod.is_ready(manifest, workspace):
                    logger.warning(
                        "Stage %s: is_ready() returned False; sleeping.", stage_name
                    )
                    time.sleep(poll_interval)
                    break

                logger.info("Stage %s: submitting.", stage_name)
                result = mod.submit(manifest, workspace, config)
                stage_block.update(result)
                atomic_write_json(manifest_path, manifest)

                if result.get("status") == "failed":
                    raise PipelineError(
                        f"Stage {stage_name!r} submit failed: "
                        f"{result.get('failure_reason', '<no reason>')}"
                    )

                logger.info(
                    "Stage %s: submitted (status=%s).", stage_name, result.get("status")
                )

                if (
                    STAGE_IS_PBS.get(stage_name, False)
                    and result.get("status") == "submitted"
                ):
                    # PBS job is queued; wait before polling.
                    time.sleep(poll_interval)
                    break

                # Local stage completed synchronously; continue to next stage.
                continue

        else:
            # for-loop exhausted without break: all stages done or skipped.
            logger.info("All stages complete.")
            break

    logger.info("Orchestrator finished.")
