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


def _ready_manifest() -> dict[str, Any]:
    return {
        "stages": {
            "xtb_constr": {
                "status": "done",
                "outputs": {
                    "anti": [{"conf_id": 0, "xyz": "xtb_constr/anti/xtbopt.xyz"}],
                    "syn": [{"conf_id": 0, "xyz": "xtb_constr/syn/xtbopt.xyz"}],
                },
            },
            "mm": {
                "status": "done",
                "outputs": {
                    "n_conformers_anti": 3,
                    "n_conformers_syn": 2,
                },
            },
        }
    }


def test_is_ready_false_when_mm_pending(tmp_path: Path) -> None:
    manifest = {
        "stages": {
            "xtb_constr": {
                "status": "done",
                "outputs": {
                    "anti": [{"conf_id": 0, "xyz": "x"}],
                    "syn": [{"conf_id": 0, "xyz": "x"}],
                },
            },
            "mm": {"status": "pending"},
        }
    }
    assert crest_stage.is_ready(manifest, tmp_path) is False


def test_is_ready_false_when_xtb_constr_pending(tmp_path: Path) -> None:
    manifest = {
        "stages": {
            "xtb_constr": {"status": "pending"},
            "mm": {
                "status": "done",
                "outputs": {"n_conformers_anti": 2, "n_conformers_syn": 2},
            },
        }
    }
    assert crest_stage.is_ready(manifest, tmp_path) is False


def test_is_ready_false_when_xtb_constr_outputs_empty(tmp_path: Path) -> None:
    manifest = {
        "stages": {
            "xtb_constr": {"status": "done", "outputs": {"anti": [], "syn": []}},
            "mm": {
                "status": "done",
                "outputs": {"n_conformers_anti": 2, "n_conformers_syn": 2},
            },
        }
    }
    assert crest_stage.is_ready(manifest, tmp_path) is False


def test_is_ready_false_when_mm_has_zero_conformers(tmp_path: Path) -> None:
    manifest = {
        "stages": {
            "xtb_constr": {
                "status": "done",
                "outputs": {
                    "anti": [{"conf_id": 0, "xyz": "x"}],
                    "syn": [{"conf_id": 0, "xyz": "x"}],
                },
            },
            "mm": {
                "status": "done",
                "outputs": {"n_conformers_anti": 0, "n_conformers_syn": 5},
            },
        }
    }
    assert crest_stage.is_ready(manifest, tmp_path) is False


def test_is_ready_true_when_mm_done_with_both_labels(tmp_path: Path) -> None:
    assert crest_stage.is_ready(_ready_manifest(), tmp_path) is True


# -- submit ----------------------------------------------------------------


def _seed_mm_outputs(workspace: Path) -> None:
    for label in ("anti", "syn"):
        d = workspace / "mm" / label
        d.mkdir(parents=True, exist_ok=True)
        (d / "conf_0.xyz").write_text("1\nfake\nH 0 0 0\n", encoding="utf-8")
        (d / "conf_1.xyz").write_text("1\nfake\nH 0 0 1\n", encoding="utf-8")


def _seed_xtb_constr_outputs(workspace: Path) -> None:
    for label in ("anti", "syn"):
        d = workspace / "xtb_constr" / label
        d.mkdir(parents=True, exist_ok=True)
        (d / "xtbopt.xyz").write_text("1\nxtb seed\nC 0 0 0\n", encoding="utf-8")


def _full_manifest(workspace: Path) -> dict[str, Any]:
    _seed_mm_outputs(workspace)
    _seed_xtb_constr_outputs(workspace)
    return {
        "stages": {
            "prep": {
                "status": "done",
                "outputs": {"spiro_carbon_idx": 0, "chromene_oxygen_idx": 1},
            },
            "xtb_constr": {
                "status": "done",
                "outputs": {
                    "anti": [
                        {
                            "conf_id": 0,
                            "xyz": "xtb_constr/anti/xtbopt.xyz",
                            "label": "anti",
                        }
                    ],
                    "syn": [
                        {
                            "conf_id": 0,
                            "xyz": "xtb_constr/syn/xtbopt.xyz",
                            "label": "syn",
                        }
                    ],
                },
            },
            "mm": {
                "status": "done",
                "outputs": {
                    "n_conformers_anti": 2,
                    "n_conformers_syn": 2,
                    "anti": [
                        {
                            "conf_id": 0,
                            "xyz": "mm/anti/conf_0.xyz",
                            "mmff_energy_kcal_mol": 1.0,
                            "label": "anti",
                        },
                        {
                            "conf_id": 1,
                            "xyz": "mm/anti/conf_1.xyz",
                            "mmff_energy_kcal_mol": 2.0,
                            "label": "anti",
                        },
                    ],
                    "syn": [
                        {
                            "conf_id": 0,
                            "xyz": "mm/syn/conf_0.xyz",
                            "mmff_energy_kcal_mol": 1.5,
                            "label": "syn",
                        },
                        {
                            "conf_id": 1,
                            "xyz": "mm/syn/conf_1.xyz",
                            "mmff_energy_kcal_mol": 2.5,
                            "label": "syn",
                        },
                    ],
                },
            },
        }
    }


def _config(script_path: Path = Path("/fake/sub_crest.sh")) -> dict[str, Any]:
    return {
        "crest": {"walltime_hours": 6, "script_path": str(script_path)},
        "ensemble": {"max_conformers_per_diastereomer": 20},
        "mecp": {"c_o_distance_angstrom": 3.4, "constraint_force_constant": 1.0},
    }


def test_submit_invokes_script_and_records_jobids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _full_manifest(tmp_path)
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
    assert set(result["pbs_job_ids"]) == {
        "anti_min",
        "syn_min",
        "anti_mecp",
        "syn_mecp",
    }
    assert "submitted_at" in result
    assert "started_at" in result

    assert len(calls) == 4
    cwds = {c["cwd"] for c in calls}
    assert cwds == {
        tmp_path / "crest" / "anti_min",
        tmp_path / "crest" / "syn_min",
        tmp_path / "crest" / "anti_mecp",
        tmp_path / "crest" / "syn_mecp",
    }

    for label in ("anti_min", "syn_min", "anti_mecp", "syn_mecp"):
        d = tmp_path / "crest" / label
        assert (d / "input.xyz").is_file()
        assert (d / "jobid").is_file()


def test_submit_marks_failed_when_script_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _full_manifest(tmp_path)
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
    manifest = _full_manifest(tmp_path)
    config = _config()

    monkeypatch.setattr(
        crest_stage,
        "submit_via_script",
        lambda script, args, cwd: ("1.meta-pbs", "1.meta-pbs\n"),
    )
    crest_stage.submit(manifest, tmp_path, config)

    for base in ("anti", "syn"):
        copied = (tmp_path / "crest" / f"{base}_min" / "input.xyz").read_text(
            encoding="utf-8"
        )
        original = (tmp_path / "mm" / base / "conf_0.xyz").read_text(encoding="utf-8")
        assert copied == original


def test_submit_writes_xcontrol_only_for_mecp_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _full_manifest(tmp_path)
    monkeypatch.setattr(
        crest_stage,
        "submit_via_script",
        lambda script, args, cwd: ("1.meta-pbs", "1.meta-pbs\n"),
    )
    crest_stage.submit(manifest, tmp_path, _config())

    assert (tmp_path / "crest" / "anti_mecp" / ".xcontrol").is_file()
    assert (tmp_path / "crest" / "syn_mecp" / ".xcontrol").is_file()
    assert not (tmp_path / "crest" / "anti_min" / ".xcontrol").exists()
    assert not (tmp_path / "crest" / "syn_min" / ".xcontrol").exists()


def test_submit_passes_cinp_flag_only_for_mecp_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _full_manifest(tmp_path)
    calls: list[dict[str, object]] = []

    def fake_submit(script, args, cwd):  # type: ignore[no-untyped-def]
        calls.append({"args": list(args), "cwd": Path(cwd)})
        return "1.meta-pbs", "1.meta-pbs\n"

    monkeypatch.setattr(crest_stage, "submit_via_script", fake_submit)
    crest_stage.submit(manifest, tmp_path, _config())

    for c in calls:
        label = c["cwd"].name  # type: ignore[union-attr]
        if label.endswith("_mecp"):
            assert "--cinp" in c["args"]
            assert ".xcontrol" in c["args"]
        else:
            assert "--cinp" not in c["args"]


def test_submit_seeds_mecp_from_xtb_constr_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _full_manifest(tmp_path)
    monkeypatch.setattr(
        crest_stage,
        "submit_via_script",
        lambda script, args, cwd: ("1.meta-pbs", "1.meta-pbs\n"),
    )
    crest_stage.submit(manifest, tmp_path, _config())

    for base in ("anti", "syn"):
        copied = (tmp_path / "crest" / f"{base}_mecp" / "input.xyz").read_text(
            encoding="utf-8"
        )
        original = (tmp_path / "xtb_constr" / base / "xtbopt.xyz").read_text(
            encoding="utf-8"
        )
        assert copied == original


def test_submit_seeds_min_from_mm_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _full_manifest(tmp_path)
    monkeypatch.setattr(
        crest_stage,
        "submit_via_script",
        lambda script, args, cwd: ("1.meta-pbs", "1.meta-pbs\n"),
    )
    crest_stage.submit(manifest, tmp_path, _config())

    for base in ("anti", "syn"):
        copied = (tmp_path / "crest" / f"{base}_min" / "input.xyz").read_text(
            encoding="utf-8"
        )
        original = (tmp_path / "mm" / base / "conf_0.xyz").read_text(encoding="utf-8")
        assert copied == original


# -- collect ---------------------------------------------------------------


def _seed_crest_outputs(workspace: Path) -> None:
    """Copy fixture CREST outputs into all 4 label directories."""
    for label in ("anti_min", "syn_min", "anti_mecp", "syn_mecp"):
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

    # anti_min: 3 conformers from fixture; syn_min: 2; _mecp dirs: 2 each.
    assert out["n_conformers_anti_min"] == 3
    assert out["n_conformers_syn_min"] == 2
    assert out["n_conformers_anti_mecp"] == 2
    assert out["n_conformers_syn_mecp"] == 2

    for label in ("anti_min", "syn_min", "anti_mecp", "syn_mecp"):
        d = tmp_path / "crest" / label / "filtered"
        files = sorted(d.glob("conf_*.xyz"))
        assert len(files) == out[f"n_conformers_{label}"]
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
        "ensemble": {"max_conformers_per_diastereomer": 1},
        "mecp": {"c_o_distance_angstrom": 3.4, "constraint_force_constant": 1.0},
    }
    result = crest_stage.collect({}, tmp_path, config)
    assert result["status"] == "done", result
    for label in ("anti_min", "syn_min", "anti_mecp", "syn_mecp"):
        assert result["outputs"][f"n_conformers_{label}"] == 1


def test_collect_fails_when_outputs_missing(tmp_path: Path) -> None:
    # Seed only anti_min; the rest are absent.
    label = "anti_min"
    dest = tmp_path / "crest" / label
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        FIXTURES / label / "crest_conformers.xyz", dest / "crest_conformers.xyz"
    )
    shutil.copyfile(FIXTURES / label / "crest.energies", dest / "crest.energies")

    result = crest_stage.collect({}, tmp_path, _config())
    assert result["status"] == "failed"
    # The second label in LABELS is syn_min — failure reason should mention it.
    assert "syn_min" in result["failure_reason"].lower()


def test_collect_fails_on_count_mismatch(tmp_path: Path) -> None:
    _seed_crest_outputs(tmp_path)
    (tmp_path / "crest" / "anti_min" / "crest.energies").write_text(
        "  1   -76.40000000\n", encoding="utf-8"
    )
    result = crest_stage.collect({}, tmp_path, _config())
    assert result["status"] == "failed"
    assert "mismatch" in result["failure_reason"].lower()


# -- pbs_utils integration sanity -----------------------------------------


def test_pbs_utils_module_is_importable_for_stage() -> None:
    assert hasattr(pbs_utils, "submit_via_script")
    assert hasattr(pbs_utils, "write_jobid")
    assert hasattr(pbs_utils, "PBSSubmitError")
