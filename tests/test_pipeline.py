from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from spiropyran_dr.pipeline import PipelineError, molecule_id_from_smiles, run


# ---------------------------------------------------------------------------
# molecule_id_from_smiles
# ---------------------------------------------------------------------------


def test_molecule_id_from_smiles_stable() -> None:
    mid = molecule_id_from_smiles("CCO")
    assert mid == molecule_id_from_smiles("CCO")
    assert mid.startswith("sp_")
    assert len(mid) == 11  # "sp_" + 8 hex chars


def test_molecule_id_from_smiles_different_for_different_smiles() -> None:
    assert molecule_id_from_smiles("CCO") != molecule_id_from_smiles("OCC")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_manifest(workspace: Path, manifest: dict[str, Any]) -> None:
    (workspace / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _minimal_config() -> dict[str, Any]:
    return {
        "polling": {"interval_seconds": 0},
        "mecp": {"c_o_distance_angstrom": 3.4, "constraint_force_constant": 1.0},
        "xtb_constr": {},
        "crest": {},
        "dft": {},
        "options": {},
    }


# ---------------------------------------------------------------------------
# run() — manifest missing
# ---------------------------------------------------------------------------


def test_run_raises_on_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(PipelineError, match="manifest.json not found"):
        run(tmp_path, _minimal_config())


# ---------------------------------------------------------------------------
# run() — all stages already done or skipped → exits immediately
# ---------------------------------------------------------------------------


def test_run_exits_immediately_when_all_done(tmp_path: Path) -> None:
    manifest = {
        "molecule_id": "sp_test",
        "options": {"thermal": False},
        "stages": {
            s: {"status": "done"}
            for s in (
                "prep",
                "mm",
                "xtb_constr",
                "crest",
                "dft_sp",
                "dft_freq",
                "aggregate",
            )
        },
    }
    _write_manifest(tmp_path, manifest)
    run(tmp_path, _minimal_config())  # must not raise or block


# ---------------------------------------------------------------------------
# run() — failed stage raises PipelineError
# ---------------------------------------------------------------------------


def test_run_raises_on_failed_stage(tmp_path: Path) -> None:
    manifest = {
        "molecule_id": "sp_test",
        "options": {"thermal": False},
        "stages": {
            "prep": {"status": "done"},
            "mm": {"status": "done"},
            "xtb_constr": {
                "status": "failed",
                "failure_reason": "xTB crashed",
            },
        },
    }
    _write_manifest(tmp_path, manifest)
    with pytest.raises(PipelineError, match="xTB crashed"):
        run(tmp_path, _minimal_config())


# ---------------------------------------------------------------------------
# run() — unimplemented stage is skipped, loop continues
# ---------------------------------------------------------------------------


def test_run_skips_unimplemented_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """get_stage_module returning None marks the stage skipped and moves on."""
    manifest = {
        "molecule_id": "sp_test",
        "options": {"thermal": False},
        "stages": {
            "prep": {"status": "done"},
            "mm": {"status": "done"},
            "xtb_constr": {"status": "done"},
            "crest": {"status": "done"},
            "dft_sp": {"status": "done"},
            "dft_freq": {"status": "pending"},
            "aggregate": {"status": "pending"},
        },
    }
    _write_manifest(tmp_path, manifest)

    import spiropyran_dr.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "get_stage_module", lambda name: None)

    run(tmp_path, _minimal_config())

    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert saved["stages"]["dft_freq"]["status"] == "skipped"
    assert saved["stages"]["aggregate"]["status"] == "skipped"


# ---------------------------------------------------------------------------
# run() — PBS stage: collect called when is_all_jobs_done returns True
# ---------------------------------------------------------------------------


def test_run_collects_when_jobs_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After detecting that PBS jobs are finished, run() calls collect()."""
    manifest = {
        "molecule_id": "sp_test",
        "options": {"thermal": False},
        "stages": {
            "prep": {"status": "done"},
            "mm": {"status": "done"},
            "xtb_constr": {
                "status": "submitted",
                "pbs_job_ids": {"anti": "1.meta", "syn": "2.meta"},
            },
            "crest": {"status": "pending"},
            "dft_sp": {"status": "pending"},
            "dft_freq": {"status": "pending"},
            "aggregate": {"status": "pending"},
        },
    }
    _write_manifest(tmp_path, manifest)

    collect_called: list[str] = []

    def fake_collect(manifest, workspace, config):  # type: ignore[no-untyped-def]
        collect_called.append("xtb_constr")
        return {
            "status": "done",
            "finished_at": "2026-01-01T00:00:00+00:00",
            "outputs": {"anti": [], "syn": []},
        }

    fake_mod = MagicMock()
    fake_mod.is_ready.return_value = True
    fake_mod.collect.side_effect = fake_collect
    fake_mod.submit.return_value = {"status": "submitted", "pbs_job_ids": {}}

    import spiropyran_dr.pipeline as pipeline_mod

    def fake_get_module(name: str):  # type: ignore[no-untyped-def]
        if name in ("xtb_constr", "crest", "dft_sp"):
            return fake_mod
        return None

    monkeypatch.setattr(pipeline_mod, "get_stage_module", fake_get_module)
    monkeypatch.setattr(pipeline_mod, "is_all_jobs_done", lambda ids: True)
    monkeypatch.setattr(time, "sleep", lambda _: None)

    run(tmp_path, _minimal_config())

    assert "xtb_constr" in collect_called
    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert saved["stages"]["xtb_constr"]["status"] == "done"


# ---------------------------------------------------------------------------
# run() — config hash mismatch raises PipelineError
# ---------------------------------------------------------------------------


def test_run_raises_on_config_hash_mismatch(tmp_path: Path) -> None:
    manifest = {
        "molecule_id": "sp_test",
        "config_hash": "sha256:deadbeef",
        "options": {"thermal": False},
        "stages": {"prep": {"status": "done"}},
    }
    _write_manifest(tmp_path, manifest)
    with pytest.raises(PipelineError, match="Config hash mismatch"):
        run(tmp_path, _minimal_config())


# ---------------------------------------------------------------------------
# run() — dft_freq skipped automatically when thermal=False
# ---------------------------------------------------------------------------


def test_run_skips_dft_freq_when_no_thermal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = {
        "molecule_id": "sp_test",
        "options": {"thermal": False},
        "stages": {
            "prep": {"status": "done"},
            "mm": {"status": "done"},
            "xtb_constr": {"status": "done"},
            "crest": {"status": "done"},
            "dft_sp": {"status": "done"},
            "dft_freq": {"status": "pending"},
            "aggregate": {"status": "pending"},
        },
    }
    _write_manifest(tmp_path, manifest)

    import spiropyran_dr.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "get_stage_module", lambda name: None)

    run(tmp_path, _minimal_config())

    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert saved["stages"]["dft_freq"]["status"] == "skipped"
