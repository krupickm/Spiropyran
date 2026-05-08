from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from spiropyran_dr import pbs_utils
from spiropyran_dr.stages import dft_sp_stage
from spiropyran_dr.stages.base import Stage


FIXTURE_DIR = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "molecules"
    / "water_synthetic"
    / "dft_sp"
)

LABELS = ("anti_min", "syn_min", "anti_mecp", "syn_mecp")
N_CONF = 3


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
    """Manifest with crest done and N_CONF conformers per label (pointing to tmp xyz files)."""
    outputs: dict[str, list[dict[str, Any]]] = {}
    for label in LABELS:
        outputs[label] = [
            {
                "conf_id": i,
                "xyz": str(workspace / "crest" / label / "filtered" / f"conf_{i}.xyz"),
                "energy_hartree": -76.4 - i * 0.002,
                "label": label,
            }
            for i in range(N_CONF)
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
    """Write N_CONF dummy single-frame XYZ files per label under crest/filtered/."""
    for label in LABELS:
        for i in range(N_CONF):
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


def test_submit_writes_per_conf_input_xyz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_crest_xyz(tmp_path)
    manifest = _crest_done_manifest(tmp_path)
    config = _minimal_config()
    monkeypatch.setattr(
        dft_sp_stage, "submit_via_script", lambda *a, **kw: ("99.meta-pbs", "")
    )

    dft_sp_stage.submit(manifest, tmp_path, config)

    for label in LABELS:
        for i in range(N_CONF):
            conf_xyz = tmp_path / "dft_sp" / label / f"conf_{i}" / f"conf_{i}.xyz"
            assert conf_xyz.exists(), f"{conf_xyz} missing"
            lines = conf_xyz.read_text(encoding="utf-8").splitlines()
            # Single frame: 1 count + 1 comment + 3 atoms = 5 lines
            assert len(lines) == 5, f"{conf_xyz}: expected 5 lines, got {len(lines)}"


def test_submit_writes_orca_inp_with_cpcm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_crest_xyz(tmp_path)
    manifest = _crest_done_manifest(tmp_path)
    config = _minimal_config()
    monkeypatch.setattr(
        dft_sp_stage, "submit_via_script", lambda *a, **kw: ("99.meta-pbs", "")
    )

    dft_sp_stage.submit(manifest, tmp_path, config)

    for label in LABELS:
        for i in range(N_CONF):
            orca_inp = (
                tmp_path / "dft_sp" / label / f"conf_{i}" / "orca.inp"
            ).read_text(encoding="utf-8")
            assert "r2SCAN-3c" in orca_inp
            assert "CPCM(acetonitrile)" in orca_inp
            assert "nprocs 2" in orca_inp
            assert "4000" in orca_inp
            assert f"*xyzfile 0 1 conf_{i}.xyz" in orca_inp
            # Must NOT contain SMD block — plain CPCM only.
            assert "smd" not in orca_inp.lower()


def test_submit_calls_suborca_per_conformer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_crest_xyz(tmp_path)
    manifest = _crest_done_manifest(tmp_path)
    config = _minimal_config()

    calls: list[tuple] = []

    def _fake_submit(script: Path, args: list[str], cwd: Path) -> tuple[str, str]:
        calls.append((script, args, cwd))
        return ("99.meta-pbs", "")

    monkeypatch.setattr(dft_sp_stage, "submit_via_script", _fake_submit)
    dft_sp_stage.submit(manifest, tmp_path, config)

    assert len(calls) == len(LABELS) * N_CONF
    for script, args, cwd in calls:
        assert Path(script).as_posix() == "/fake/suborca.sh"
        assert args == ["orca.inp", "1"]
        # cwd must be a per-conformer directory.
        assert cwd.parent.parent.name == "dft_sp"
        assert cwd.name.startswith("conf_")


def test_submit_returns_submitted_status_with_composite_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_crest_xyz(tmp_path)
    manifest = _crest_done_manifest(tmp_path)
    config = _minimal_config()

    counter = {"n": 0}

    def _fake_submit(*a: object, **kw: object) -> tuple[str, str]:
        counter["n"] += 1
        return (f"{counter['n']}.meta-pbs", "")

    monkeypatch.setattr(dft_sp_stage, "submit_via_script", _fake_submit)

    result = dft_sp_stage.submit(manifest, tmp_path, config)

    assert result["status"] == "submitted"
    assert "started_at" in result
    assert "submitted_at" in result
    expected_keys = {f"{lbl}/{i}" for lbl in LABELS for i in range(N_CONF)}
    assert set(result["pbs_job_ids"].keys()) == expected_keys
    # All values must be unique (one job per conformer).
    assert len(set(result["pbs_job_ids"].values())) == len(expected_keys)


def test_submit_returns_failed_on_pbs_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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


def _seed_collect_outputs(workspace: Path) -> None:
    """Copy per-conf fixture orca.out into workspace/dft_sp/{label}/conf_{i}/."""
    for label in LABELS:
        for i in range(N_CONF):
            dest_dir = workspace / "dft_sp" / label / f"conf_{i}"
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(
                FIXTURE_DIR / label / f"conf_{i}" / "orca.out",
                dest_dir / "orca.out",
            )


def _make_collect_manifest(workspace: Path) -> dict[str, Any]:
    """Manifest suitable for collect: crest done, N_CONF conformers per label."""
    outputs: dict[str, list[dict[str, Any]]] = {}
    for label in LABELS:
        outputs[label] = [
            {
                "conf_id": i,
                "xyz": f"crest/{label}/filtered/conf_{i}.xyz",
                "label": label,
            }
            for i in range(N_CONF)
        ]
    return {"stages": {"crest": {"status": "done", "outputs": outputs}}}


@pytest.mark.parametrize("label", LABELS)
def test_collect_parses_energies_for_water_synthetic(
    label: str, tmp_path: Path
) -> None:
    _seed_collect_outputs(tmp_path)
    manifest = _make_collect_manifest(tmp_path)
    result = dft_sp_stage.collect(manifest, tmp_path, _minimal_config())

    assert result["status"] == "done", result.get("failure_reason")
    confs = result["outputs"][label]
    assert len(confs) == N_CONF
    assert all("energy_hartree" in c for c in confs)
    assert all(c["energy_hartree"] < 0 for c in confs)
    assert [c["conf_id"] for c in confs] == [0, 1, 2]


def test_collect_fails_on_missing_orca_out(tmp_path: Path) -> None:
    _seed_collect_outputs(tmp_path)
    # Remove one conformer's output to trigger the missing-file branch.
    (tmp_path / "dft_sp" / "anti_min" / "conf_1" / "orca.out").unlink()

    manifest = _make_collect_manifest(tmp_path)
    result = dft_sp_stage.collect(manifest, tmp_path, _minimal_config())

    assert result["status"] == "failed"
    assert "orca.out missing" in result["failure_reason"]
    assert "anti_min/1" in result["failure_reason"]


def test_collect_fails_on_abnormal_termination(tmp_path: Path) -> None:
    _seed_collect_outputs(tmp_path)
    # Replace anti_min/conf_0 with the abnormal-termination fixture.
    shutil.copy(
        FIXTURE_DIR / "anti_min_failed" / "conf_0" / "orca.out",
        tmp_path / "dft_sp" / "anti_min" / "conf_0" / "orca.out",
    )

    manifest = _make_collect_manifest(tmp_path)
    result = dft_sp_stage.collect(manifest, tmp_path, _minimal_config())

    assert result["status"] == "failed"
    assert "not terminate normally" in result["failure_reason"]
    assert "anti_min/0" in result["failure_reason"]


def test_collect_fails_on_multi_energy_in_single_orca_out(tmp_path: Path) -> None:
    """Per-conf orca.out with >1 SP energy line is a multi-frame regression."""
    _seed_collect_outputs(tmp_path)
    multi = (
        "FINAL SINGLE POINT ENERGY       -76.400000000\n"
        "FINAL SINGLE POINT ENERGY       -76.398000000\n"
        "****ORCA TERMINATED NORMALLY****\n"
    )
    (tmp_path / "dft_sp" / "anti_min" / "conf_0" / "orca.out").write_text(
        multi, encoding="utf-8"
    )

    manifest = _make_collect_manifest(tmp_path)
    result = dft_sp_stage.collect(manifest, tmp_path, _minimal_config())

    assert result["status"] == "failed"
    assert "expected exactly one" in result["failure_reason"]
