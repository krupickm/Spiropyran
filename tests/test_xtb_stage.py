from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from spiropyran_dr.pbs_utils import PBSSubmitError
from spiropyran_dr.stages import xtb_stage
from spiropyran_dr.stages.base import Stage

from conftest import fixture_molecule_dir, fixture_molecule_names


# -- protocol --------------------------------------------------------------


def test_xtb_module_satisfies_stage_protocol() -> None:
    stage: Stage = xtb_stage
    assert callable(stage.is_ready)
    assert callable(stage.submit)
    assert callable(stage.collect)


# -- is_ready --------------------------------------------------------------


def _mm_done_manifest(
    n_anti: int = 2,
    n_syn: int = 2,
    with_prep_indices: bool = True,
) -> dict[str, Any]:
    stages: dict[str, Any] = {
        "mm": {
            "status": "done",
            "outputs": {
                "n_conformers_anti": n_anti,
                "n_conformers_syn": n_syn,
                "anti": [{"conf_id": 0, "xyz": "mm/anti/conf_0.xyz", "label": "anti"}],
                "syn": [{"conf_id": 0, "xyz": "mm/syn/conf_0.xyz", "label": "syn"}],
            },
        }
    }
    if with_prep_indices:
        stages["prep"] = {
            "status": "done",
            "outputs": {
                "spiro_carbon_idx": 0,
                "chromene_oxygen_idx": 1,
                "smiles_canonical": "C1CC1",
            },
        }
    else:
        stages["prep"] = {"status": "done", "outputs": {}}
    return {"stages": stages}


def test_is_ready_false_when_mm_pending(tmp_path: Path) -> None:
    manifest = {
        "stages": {
            "mm": {"status": "pending"},
            "prep": {"status": "done", "outputs": {}},
        }
    }
    assert xtb_stage.is_ready(manifest, tmp_path) is False


def test_is_ready_false_when_prep_indices_missing(tmp_path: Path) -> None:
    manifest = _mm_done_manifest(with_prep_indices=False)
    assert xtb_stage.is_ready(manifest, tmp_path) is False


def test_is_ready_true_when_mm_done_and_prep_indices_present(tmp_path: Path) -> None:
    manifest = _mm_done_manifest()
    assert xtb_stage.is_ready(manifest, tmp_path) is True


# -- submit ----------------------------------------------------------------


def _seed_mm_outputs(workspace: Path) -> None:
    for label in ("anti", "syn"):
        d = workspace / "mm" / label
        d.mkdir(parents=True, exist_ok=True)
        (d / "conf_0.xyz").write_text(
            "3\nfake\nC 0 0 0\nO 3.4 0 0\nH 0 1 0\n", encoding="utf-8"
        )


def _full_manifest(workspace: Path) -> dict[str, Any]:
    _seed_mm_outputs(workspace)
    return {
        "stages": {
            "prep": {
                "status": "done",
                "outputs": {
                    "spiro_carbon_idx": 0,
                    "chromene_oxygen_idx": 1,
                    "smiles_canonical": "C1CC1",
                },
            },
            "mm": {
                "status": "done",
                "outputs": {
                    "n_conformers_anti": 1,
                    "n_conformers_syn": 1,
                    "anti": [
                        {"conf_id": 0, "xyz": "mm/anti/conf_0.xyz", "label": "anti"}
                    ],
                    "syn": [{"conf_id": 0, "xyz": "mm/syn/conf_0.xyz", "label": "syn"}],
                },
            },
        }
    }


def _config(script_path: Path = Path("/fake/sub_xtb.sh")) -> dict[str, Any]:
    return {
        "xtb_constr": {
            "walltime_hours": 1,
            "script_path": str(script_path),
            "method": "gfn2",
            "co_distance_tolerance_angstrom": 0.01,
        },
        "mecp": {
            "c_o_distance_angstrom": 3.4,
            "constraint_force_constant": 1.0,
        },
    }


def test_submit_writes_xtb_inp_with_1based_indices_and_config_distance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _full_manifest(tmp_path)
    monkeypatch.setattr(
        xtb_stage,
        "submit_via_script",
        lambda script, args, cwd: ("1.meta-pbs", "1.meta-pbs\n"),
    )
    xtb_stage.submit(manifest, tmp_path, _config())

    for label in ("anti", "syn"):
        inp = (tmp_path / "xtb_constr" / label / "xtb.inp").read_text(encoding="utf-8")
        # spiro_carbon_idx=0 → 1-based = 1; chromene_oxygen_idx=1 → 1-based = 2
        assert "distance: 1,2,3.4" in inp
        assert "force constant=1.0" in inp
        assert "$constrain" in inp
        assert "$end" in inp


def test_submit_invokes_sub_xtb_with_correct_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _full_manifest(tmp_path)
    calls: list[dict[str, object]] = []

    def fake_submit(script, args, cwd):  # type: ignore[no-untyped-def]
        calls.append({"args": list(args), "cwd": Path(cwd)})
        n = len(calls)
        return f"{1000 + n}.meta-pbs", f"{1000 + n}.meta-pbs\n"

    monkeypatch.setattr(xtb_stage, "submit_via_script", fake_submit)
    xtb_stage.submit(manifest, tmp_path, _config())

    assert len(calls) == 2
    for c in calls:
        assert c["args"] == [
            "1",
            "input.xyz",
            "--opt",
            "--gfn",
            "2",
            "--input",
            "xtb.inp",
        ]
    cwds = {c["cwd"] for c in calls}
    assert cwds == {tmp_path / "xtb_constr" / "anti", tmp_path / "xtb_constr" / "syn"}


def test_submit_records_jobids_and_writes_jobid_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _full_manifest(tmp_path)
    counter = {"n": 0}

    def fake_submit(script, args, cwd):  # type: ignore[no-untyped-def]
        counter["n"] += 1
        jobid = f"{2000 + counter['n']}.meta-pbs"
        return jobid, jobid + "\n"

    monkeypatch.setattr(xtb_stage, "submit_via_script", fake_submit)
    result = xtb_stage.submit(manifest, tmp_path, _config())

    assert result["status"] == "submitted", result
    assert set(result["pbs_job_ids"]) == {"anti", "syn"}
    assert result["pbs_job_ids"]["anti"] != result["pbs_job_ids"]["syn"]
    assert "submitted_at" in result
    assert "started_at" in result

    for label in ("anti", "syn"):
        jobid_file = tmp_path / "xtb_constr" / label / "jobid"
        assert jobid_file.is_file()
        assert (
            jobid_file.read_text(encoding="utf-8").strip()
            == result["pbs_job_ids"][label]
        )


def test_submit_marks_failed_when_script_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _full_manifest(tmp_path)

    def boom(script, args, cwd):  # type: ignore[no-untyped-def]
        raise PBSSubmitError("qsub: bad queue")

    monkeypatch.setattr(xtb_stage, "submit_via_script", boom)
    result = xtb_stage.submit(manifest, tmp_path, _config())
    assert result["status"] == "failed"
    assert "qsub" in result["failure_reason"]


# -- collect ---------------------------------------------------------------


def _seed_xtb_outputs(workspace: Path, molecule: str = "water_synthetic") -> None:
    """Copy fixture xtb_constr outputs into a workspace's xtb_constr/{anti,syn}/ dirs."""
    xtb_fixture = fixture_molecule_dir(molecule) / "xtb_constr"
    for label in ("anti", "syn"):
        dest = workspace / "xtb_constr" / label
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            xtb_fixture / label / "input.xtbopt.xyz", dest / "input.xtbopt.xyz"
        )
        shutil.copyfile(xtb_fixture / label / "input.xtb.log", dest / "input.xtb.log")


# Per-fixture (spiro_carbon_idx, chromene_oxygen_idx) for the constrained
# C-O atom pair. water_synthetic is a 3-atom toy with C@0 / O@1; dimethylSP
# is real BIPS xtb output where the spiro C and chromene O sit at idx 10
# and 19 (verified: |r10 - r19| ~ 3.40 Å, matching the MECP target).
_FIXTURE_PREP_INDICES: dict[str, tuple[int, int]] = {
    "water_synthetic": (0, 1),
    "dimethylSP": (10, 19),
}


def _collect_manifest(molecule: str = "water_synthetic") -> dict[str, Any]:
    spiro, oxygen = _FIXTURE_PREP_INDICES[molecule]
    return {
        "stages": {
            "prep": {
                "status": "done",
                "outputs": {
                    "spiro_carbon_idx": spiro,
                    "chromene_oxygen_idx": oxygen,
                },
            }
        }
    }


def _collect_config() -> dict[str, Any]:
    return {
        "mecp": {"c_o_distance_angstrom": 3.4, "constraint_force_constant": 1.0},
        "xtb_constr": {
            "walltime_hours": 1,
            "script_path": "/fake/sub_xtb.sh",
            "method": "gfn2",
            "co_distance_tolerance_angstrom": 0.01,
        },
    }


def test_collect_parses_xtbopt_and_xtb_out(tmp_path: Path) -> None:
    _seed_xtb_outputs(tmp_path)
    result = xtb_stage.collect(_collect_manifest(), tmp_path, _collect_config())

    assert result["status"] == "done", result
    out = result["outputs"]
    for label in ("anti", "syn"):
        assert label in out
        entries = out[label]
        e = entries[0]
        assert abs(e["co_distance_final_ang"] - 3.402) < 1e-6
        assert e["energy_hartree"] < 0
        assert e["xyz"] == f"xtb_constr/{label}/input.xtbopt.xyz"


def test_collect_fails_on_constraint_violation(tmp_path: Path) -> None:
    _seed_xtb_outputs(tmp_path)
    # Overwrite anti xtbopt.xyz with a C-O distance far from 3.4 Å (3.55)
    (tmp_path / "xtb_constr" / "anti" / "input.xtbopt.xyz").write_text(
        "3\nbad geometry\nC 0.0 0.0 0.0\nO 3.55 0.0 0.0\nH 0.0 1.0 0.0\n",
        encoding="utf-8",
    )
    result = xtb_stage.collect(_collect_manifest(), tmp_path, _collect_config())
    assert result["status"] == "failed"
    assert "tolerance" in result["failure_reason"].lower()


def test_collect_fails_when_xtbopt_missing(tmp_path: Path) -> None:
    _seed_xtb_outputs(tmp_path)
    (tmp_path / "xtb_constr" / "anti" / "input.xtbopt.xyz").unlink()
    result = xtb_stage.collect(_collect_manifest(), tmp_path, _collect_config())
    assert result["status"] == "failed"
    assert "input.xtbopt.xyz" in result["failure_reason"]


def test_collect_outputs_single_element_list_per_label(tmp_path: Path) -> None:
    _seed_xtb_outputs(tmp_path)
    result = xtb_stage.collect(_collect_manifest(), tmp_path, _collect_config())
    assert result["status"] == "done"
    for label in ("anti", "syn"):
        entries = result["outputs"][label]
        assert isinstance(entries, list), f"{label} outputs should be a list"
        assert len(entries) == 1, f"{label} should have exactly 1 element"


@pytest.mark.parametrize("mol_name", fixture_molecule_names())
def test_collect_succeeds_for_all_fixture_molecules(
    mol_name: str, tmp_path: Path
) -> None:
    _seed_xtb_outputs(tmp_path, mol_name)
    result = xtb_stage.collect(_collect_manifest(mol_name), tmp_path, _collect_config())
    assert result["status"] == "done", result
