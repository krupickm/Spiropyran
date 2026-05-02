from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from spiropyran_dr import pbs_utils
from spiropyran_dr.pbs_utils import PBSSubmitError
from spiropyran_dr.stages import crest_stage
from spiropyran_dr.stages.base import Stage

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "crest"


# -- protocol --------------------------------------------------------------


def test_crest_module_satisfies_stage_protocol() -> None:
    stage: Stage = crest_stage
    assert callable(stage.is_ready)
    assert callable(stage.submit)
    assert callable(stage.collect)


# -- is_ready --------------------------------------------------------------


def test_is_ready_false_when_mm_pending(tmp_path: Path) -> None:
    manifest = {"stages": {"mm": {"status": "pending"}}}
    assert crest_stage.is_ready(manifest, tmp_path) is False


def test_is_ready_false_when_mm_has_zero_conformers(tmp_path: Path) -> None:
    manifest = {
        "stages": {
            "mm": {
                "status": "done",
                "outputs": {
                    "n_conformers_anti": 0,
                    "n_conformers_syn": 5,
                },
            }
        }
    }
    assert crest_stage.is_ready(manifest, tmp_path) is False


def test_is_ready_true_when_mm_done_with_both_labels(tmp_path: Path) -> None:
    manifest = {
        "stages": {
            "mm": {
                "status": "done",
                "outputs": {
                    "n_conformers_anti": 3,
                    "n_conformers_syn": 2,
                },
            }
        }
    }
    assert crest_stage.is_ready(manifest, tmp_path) is True


# -- submit ----------------------------------------------------------------


def _seed_mm_outputs(workspace: Path) -> dict[str, Any]:
    """Write fake MM xyz files and return a manifest stub the stage can consume."""
    for label in ("anti", "syn"):
        d = workspace / "mm" / label
        d.mkdir(parents=True, exist_ok=True)
        (d / "conf_0.xyz").write_text(
            "1\nfake\nH 0 0 0\n", encoding="utf-8"
        )
        (d / "conf_1.xyz").write_text(
            "1\nfake\nH 0 0 1\n", encoding="utf-8"
        )
    return {
        "stages": {
            "mm": {
                "status": "done",
                "outputs": {
                    "n_conformers_anti": 2,
                    "n_conformers_syn": 2,
                    "anti": [
                        {"conf_id": 0, "xyz": "mm/anti/conf_0.xyz",
                         "mmff_energy_kcal_mol": 1.0, "label": "anti"},
                        {"conf_id": 1, "xyz": "mm/anti/conf_1.xyz",
                         "mmff_energy_kcal_mol": 2.0, "label": "anti"},
                    ],
                    "syn": [
                        {"conf_id": 0, "xyz": "mm/syn/conf_0.xyz",
                         "mmff_energy_kcal_mol": 1.5, "label": "syn"},
                        {"conf_id": 1, "xyz": "mm/syn/conf_1.xyz",
                         "mmff_energy_kcal_mol": 2.5, "label": "syn"},
                    ],
                },
            }
        }
    }


def _config(script_path: Path = Path("/fake/sub_crest.sh")) -> dict[str, Any]:
    return {
        "crest": {"walltime_hours": 6, "script_path": str(script_path)},
        "ensemble": {"max_conformers_per_diastereomer": 20},
    }


def test_submit_invokes_script_and_records_jobids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _seed_mm_outputs(tmp_path)
    config = _config()

    calls: list[dict[str, object]] = []
    counter = {"n": 0}

    def fake_submit(script, args, cwd):  # type: ignore[no-untyped-def]
        counter["n"] += 1
        jobid = f"{1000 + counter['n']}.meta-pbs"
        calls.append({"script": script, "args": list(args), "cwd": Path(cwd)})
        return jobid, jobid + "\n"

    monkeypatch.setattr(crest_stage, "submit_via_script", fake_submit)

    result = crest_stage.submit(manifest, tmp_path, config)
    assert result["status"] == "submitted", result
    assert set(result["pbs_job_ids"]) == {"anti", "syn"}
    assert result["pbs_job_ids"]["anti"] != result["pbs_job_ids"]["syn"]
    # Both timestamps present per project.md s4 stage interface contract.
    assert "submitted_at" in result
    assert "started_at" in result

    # Two invocations, one per label, each in its own work dir.
    assert len(calls) == 2
    cwds = {c["cwd"] for c in calls}
    assert cwds == {tmp_path / "crest" / "anti", tmp_path / "crest" / "syn"}
    for c in calls:
        assert c["args"] == ["6", "input.xyz"]
        assert Path(str(c["script"])).name == "sub_crest.sh"

    # input.xyz copied into each work dir, jobid written.
    for label in ("anti", "syn"):
        d = tmp_path / "crest" / label
        assert (d / "input.xyz").is_file()
        assert (d / "jobid").is_file()
        assert (d / "jobid").read_text(encoding="utf-8").strip() == result[
            "pbs_job_ids"
        ][label]


def test_submit_marks_failed_when_script_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _seed_mm_outputs(tmp_path)
    config = _config()

    def boom(script, args, cwd):  # type: ignore[no-untyped-def]
        raise PBSSubmitError("qsub: bad queue")

    monkeypatch.setattr(crest_stage, "submit_via_script", boom)
    result = crest_stage.submit(manifest, tmp_path, config)
    assert result["status"] == "failed"
    assert "qsub" in result["failure_reason"]


def test_submit_uses_only_lowest_energy_mm_conformer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _seed_mm_outputs(tmp_path)
    config = _config()

    monkeypatch.setattr(
        crest_stage,
        "submit_via_script",
        lambda script, args, cwd: ("1.meta-pbs", "1.meta-pbs\n"),
    )
    crest_stage.submit(manifest, tmp_path, config)

    # The stage copies mm/<label>/conf_0.xyz; conf_1's payload must not appear.
    for label in ("anti", "syn"):
        copied = (tmp_path / "crest" / label / "input.xyz").read_text(
            encoding="utf-8"
        )
        original = (tmp_path / "mm" / label / "conf_0.xyz").read_text(
            encoding="utf-8"
        )
        assert copied == original


# -- collect ---------------------------------------------------------------


def _seed_crest_outputs(workspace: Path) -> None:
    """Copy fixture CREST outputs into a workspace's crest/{anti,syn}/ dirs."""
    for label in ("anti", "syn"):
        dest = workspace / "crest" / label
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            FIXTURES / label / "crest_conformers.xyz",
            dest / "crest_conformers.xyz",
        )
        shutil.copyfile(
            FIXTURES / label / "crest.energies",
            dest / "crest.energies",
        )


def test_collect_parses_fixtures_and_writes_filtered(tmp_path: Path) -> None:
    _seed_crest_outputs(tmp_path)
    config = _config()
    result = crest_stage.collect({}, tmp_path, config)

    assert result["status"] == "done", result
    out = result["outputs"]
    assert out["n_conformers_anti"] == 3
    assert out["n_conformers_syn"] == 2

    for label in ("anti", "syn"):
        d = tmp_path / "crest" / label / "filtered"
        files = sorted(d.glob("conf_*.xyz"))
        assert len(files) == out[f"n_conformers_{label}"]
        # entries are energy-ascending; conf_0 has the minimum (relative=0).
        entries = out[label]
        assert entries[0]["relative_energy_kcal_mol"] == 0.0
        for i, e in enumerate(entries):
            assert e["conf_id"] == i
            assert e["label"] == label
            assert e["xyz"].startswith(f"crest/{label}/filtered/conf_")


def test_collect_caps_at_max_conformers_per_diastereomer(tmp_path: Path) -> None:
    _seed_crest_outputs(tmp_path)
    config = {
        "crest": {"walltime_hours": 6, "script_path": "/x"},
        "ensemble": {"max_conformers_per_diastereomer": 2},
    }
    result = crest_stage.collect({}, tmp_path, config)
    assert result["outputs"]["n_conformers_anti"] == 2
    assert result["outputs"]["n_conformers_syn"] == 2


def test_collect_fails_when_outputs_missing(tmp_path: Path) -> None:
    # Only seed the anti directory; syn is missing.
    (tmp_path / "crest" / "anti").mkdir(parents=True)
    shutil.copyfile(
        FIXTURES / "anti" / "crest_conformers.xyz",
        tmp_path / "crest" / "anti" / "crest_conformers.xyz",
    )
    shutil.copyfile(
        FIXTURES / "anti" / "crest.energies",
        tmp_path / "crest" / "anti" / "crest.energies",
    )
    result = crest_stage.collect({}, tmp_path, _config())
    assert result["status"] == "failed"
    assert "syn" in result["failure_reason"].lower()


def test_collect_fails_on_count_mismatch(tmp_path: Path) -> None:
    _seed_crest_outputs(tmp_path)
    # Truncate energies for anti to mismatch frames.
    (tmp_path / "crest" / "anti" / "crest.energies").write_text(
        "  1   -76.40000000\n", encoding="utf-8"
    )
    result = crest_stage.collect({}, tmp_path, _config())
    assert result["status"] == "failed"
    assert "mismatch" in result["failure_reason"].lower()


# -- pbs_utils integration sanity -----------------------------------------


def test_pbs_utils_module_is_importable_for_stage() -> None:
    # The stage imports submit_via_script + write_jobid + PBSSubmitError;
    # this guards against accidental rename of the public surface.
    assert hasattr(pbs_utils, "submit_via_script")
    assert hasattr(pbs_utils, "write_jobid")
    assert hasattr(pbs_utils, "PBSSubmitError")
