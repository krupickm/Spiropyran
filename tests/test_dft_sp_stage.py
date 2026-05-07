from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from spiropyran_dr import pbs_utils
from spiropyran_dr.stages import dft_sp_stage
from spiropyran_dr.stages.base import Stage


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "molecules" / "water_synthetic" / "dft_sp"

LABELS = ("anti_min", "syn_min", "anti_mecp", "syn_mecp")


# -- helpers ----------------------------------------------------------------


def _minimal_config() -> dict[str, Any]:
    return {
        "dft_sp": {
            "walltime_hours": 1,
            "script_path": "/fake/suborca.sh",
            "ncpus": 2,
            "mem_per_core_mb": 4000,
            "method": "r2SCAN-3c",
        },
        "dft": {
            "solvent": {"name": "acetonitrile"},
        },
    }


def _crest_done_manifest(workspace: Path) -> dict[str, Any]:
    """Manifest with crest done and 3 conformers per label (pointing to tmp xyz files)."""
    outputs: dict[str, list[dict[str, Any]]] = {}
    for label in LABELS:
        outputs[label] = [
            {
                "conf_id": i,
                "xyz": str(workspace / "crest" / label / "filtered" / f"conf_{i}.xyz"),
                "energy_hartree": -76.4 - i * 0.002,
                "label": label,
            }
            for i in range(3)
        ]
    return {
        "stages": {
            "crest": {
                "status": "done",
                "outputs": outputs,
            }
        }
    }


def _write_dummy_xyz(path: Path, atom_index: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"3\nconf {atom_index}\n"
        "O  0.00000000  0.00000000  0.00000000\n"
        "H  0.96000000  0.00000000  0.00000000\n"
        "H -0.24000000  0.93000000  0.00000000\n",
        encoding="utf-8",
    )


def _seed_crest_xyz(workspace: Path) -> None:
    """Write 3 dummy single-frame XYZ files per label under crest/filtered/."""
    for label in LABELS:
        for i in range(3):
            _write_dummy_xyz(
                workspace / "crest" / label / "filtered" / f"conf_{i}.xyz", atom_index=i
            )


# -- protocol ---------------------------------------------------------------


def test_dft_sp_module_satisfies_stage_protocol() -> None:
    stage: Stage = dft_sp_stage
    assert callable(stage.is_ready)
    assert callable(stage.submit)
    assert callable(stage.collect)


# -- is_ready ---------------------------------------------------------------


def test_is_ready_false_when_crest_pending(tmp_path: Path) -> None:
    manifest = {"stages": {"crest": {"status": "pending"}}}
    assert dft_sp_stage.is_ready(manifest, tmp_path) is False


def test_is_ready_false_when_crest_done_but_label_missing(tmp_path: Path) -> None:
    outputs = {label: [{"conf_id": 0}] for label in LABELS if label != "syn_mecp"}
    manifest = {"stages": {"crest": {"status": "done", "outputs": outputs}}}
    assert dft_sp_stage.is_ready(manifest, tmp_path) is False


def test_is_ready_false_when_crest_label_empty(tmp_path: Path) -> None:
    outputs = {label: [{"conf_id": 0}] for label in LABELS}
    outputs["anti_mecp"] = []
    manifest = {"stages": {"crest": {"status": "done", "outputs": outputs}}}
    assert dft_sp_stage.is_ready(manifest, tmp_path) is False


def test_is_ready_true_when_crest_done_all_labels(tmp_path: Path) -> None:
    manifest = _crest_done_manifest(tmp_path)
    assert dft_sp_stage.is_ready(manifest, tmp_path) is True


# -- submit -----------------------------------------------------------------


def test_submit_writes_conformers_xyz(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_crest_xyz(tmp_path)
    manifest = _crest_done_manifest(tmp_path)
    config = _minimal_config()
    monkeypatch.setattr(dft_sp_stage, "submit_via_script", lambda *a, **kw: ("99.meta-pbs", ""))

    dft_sp_stage.submit(manifest, tmp_path, config)

    for label in LABELS:
        conformers_xyz = tmp_path / "dft_sp" / label / "conformers.xyz"
        assert conformers_xyz.exists(), f"conformers.xyz missing for {label}"
        lines = conformers_xyz.read_text(encoding="utf-8").splitlines()
        # 3 frames × (1 count + 1 comment + 3 atoms) = 15 lines
        assert len(lines) == 15, f"{label}: expected 15 lines, got {len(lines)}"


def test_submit_writes_orca_inp_with_cpcm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_crest_xyz(tmp_path)
    manifest = _crest_done_manifest(tmp_path)
    config = _minimal_config()
    monkeypatch.setattr(dft_sp_stage, "submit_via_script", lambda *a, **kw: ("99.meta-pbs", ""))

    dft_sp_stage.submit(manifest, tmp_path, config)

    for label in LABELS:
        orca_inp = (tmp_path / "dft_sp" / label / "orca.inp").read_text(encoding="utf-8")
        assert "r2SCAN-3c" in orca_inp
        assert "CPCM(acetonitrile)" in orca_inp
        assert "nprocs 2" in orca_inp
        assert "4000" in orca_inp
        assert "*xyzfile 0 1 conformers.xyz" in orca_inp
        # Must NOT contain SMD block — plain CPCM only.
        assert "smd" not in orca_inp.lower()


def test_submit_calls_suborca_with_right_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_crest_xyz(tmp_path)
    manifest = _crest_done_manifest(tmp_path)
    config = _minimal_config()

    calls: list[tuple] = []

    def _fake_submit(script: Path, args: list[str], cwd: Path) -> tuple[str, str]:
        calls.append((script, args, cwd))
        return ("99.meta-pbs", "")

    monkeypatch.setattr(dft_sp_stage, "submit_via_script", _fake_submit)
    dft_sp_stage.submit(manifest, tmp_path, config)

    assert len(calls) == 4
    for script, args, cwd in calls:
        assert Path(script).as_posix() == "/fake/suborca.sh"
        assert args == ["orca.inp", "1"]


def test_submit_returns_submitted_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_crest_xyz(tmp_path)
    manifest = _crest_done_manifest(tmp_path)
    config = _minimal_config()
    monkeypatch.setattr(dft_sp_stage, "submit_via_script", lambda *a, **kw: ("42.meta-pbs", ""))

    result = dft_sp_stage.submit(manifest, tmp_path, config)

    assert result["status"] == "submitted"
    assert "started_at" in result
    assert "submitted_at" in result
    assert set(result["pbs_job_ids"].keys()) == set(LABELS)
    assert all(v == "42.meta-pbs" for v in result["pbs_job_ids"].values())


def test_submit_returns_failed_on_pbs_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_crest_xyz(tmp_path)
    manifest = _crest_done_manifest(tmp_path)
    config = _minimal_config()

    def _boom(*a: object, **kw: object) -> None:
        raise pbs_utils.PBSSubmitError("script not found")

    monkeypatch.setattr(dft_sp_stage, "submit_via_script", _boom)

    result = dft_sp_stage.submit(manifest, tmp_path, config)

    assert result["status"] == "failed"
    assert "failure_reason" in result


# -- collect ----------------------------------------------------------------


def _setup_collect_workspace(workspace: Path, label: str, orca_out_src: Path) -> None:
    """Place orca.out and manifest crest conformer XYZ stubs for one label."""
    label_dir = workspace / "dft_sp" / label
    label_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(orca_out_src, label_dir / "orca.out")


def _make_collect_manifest(workspace: Path) -> dict[str, Any]:
    """Manifest suitable for collect: crest done, 3 conformers per label."""
    outputs: dict[str, list[dict[str, Any]]] = {}
    for label in LABELS:
        outputs[label] = [
            {"conf_id": i, "xyz": f"crest/{label}/filtered/conf_{i}.xyz", "label": label}
            for i in range(3)
        ]
    return {"stages": {"crest": {"status": "done", "outputs": outputs}}}


@pytest.mark.parametrize("label", LABELS)
def test_collect_parses_energies_for_water_synthetic(label: str, tmp_path: Path) -> None:
    orca_out_src = FIXTURE_DIR / label / "orca.out"
    _setup_collect_workspace(tmp_path, label, orca_out_src)

    # Also provide the other labels' orca.out so collect() doesn't fail early.
    for other in LABELS:
        if other != label:
            _setup_collect_workspace(tmp_path, other, FIXTURE_DIR / other / "orca.out")

    manifest = _make_collect_manifest(tmp_path)
    result = dft_sp_stage.collect(manifest, tmp_path, _minimal_config())

    assert result["status"] == "done", result.get("failure_reason")
    confs = result["outputs"][label]
    assert len(confs) == 3
    assert all("energy_hartree" in c for c in confs)
    assert all(c["energy_hartree"] < 0 for c in confs)
    assert [c["conf_id"] for c in confs] == [0, 1, 2]


def test_collect_fails_on_missing_orca_out(tmp_path: Path) -> None:
    manifest = _make_collect_manifest(tmp_path)
    # Provide orca.out for all labels except anti_min.
    for label in LABELS[1:]:
        _setup_collect_workspace(tmp_path, label, FIXTURE_DIR / label / "orca.out")

    result = dft_sp_stage.collect(manifest, tmp_path, _minimal_config())

    assert result["status"] == "failed"
    assert "orca.out missing" in result["failure_reason"]


def test_collect_fails_on_abnormal_termination(tmp_path: Path) -> None:
    # anti_min_failed has FINAL SINGLE POINT ENERGY but no ORCA TERMINATED NORMALLY.
    _setup_collect_workspace(tmp_path, "anti_min", FIXTURE_DIR / "anti_min_failed" / "orca.out")
    for label in LABELS[1:]:
        _setup_collect_workspace(tmp_path, label, FIXTURE_DIR / label / "orca.out")

    manifest = _make_collect_manifest(tmp_path)
    result = dft_sp_stage.collect(manifest, tmp_path, _minimal_config())

    assert result["status"] == "failed"
    assert "not terminate normally" in result["failure_reason"]


def test_collect_fails_on_energy_count_mismatch(tmp_path: Path) -> None:
    # Write an orca.out with only 2 energy lines for anti_min (fixture has 3 conformers).
    label_dir = tmp_path / "dft_sp" / "anti_min"
    label_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / "orca.out").write_text(
        "FINAL SINGLE POINT ENERGY       -76.400000000\n"
        "FINAL SINGLE POINT ENERGY       -76.398000000\n"
        "****ORCA TERMINATED NORMALLY****\n",
        encoding="utf-8",
    )
    for label in LABELS[1:]:
        _setup_collect_workspace(tmp_path, label, FIXTURE_DIR / label / "orca.out")

    manifest = _make_collect_manifest(tmp_path)
    result = dft_sp_stage.collect(manifest, tmp_path, _minimal_config())

    assert result["status"] == "failed"
    assert "mismatch" in result["failure_reason"]
