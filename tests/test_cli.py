from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from spiropyran_dr.cli import main

from conftest import fixture_molecule_dir


def _seed_crest_outputs(
    workspace: Path,
    labels=("anti_min", "syn_min", "anti_mecp", "syn_mecp"),
    molecule: str = "water_synthetic",
) -> None:
    """Drop fixture crest_conformers.xyz into workspace/crest/<label>/.

    `crest.energies` is intentionally not seeded: the stage parses absolute
    energies from the xyz comment lines and never reads the sidecar file.
    """
    crest_fixture = fixture_molecule_dir(molecule) / "crest"
    for label in labels:
        dest = workspace / "crest" / label
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            crest_fixture / label / "crest_conformers.xyz",
            dest / "crest_conformers.xyz",
        )


def _seed_xtb_outputs(workspace: Path, molecule: str = "water_synthetic") -> None:
    """Drop fixture input.xtbopt.xyz / input.xtb.log into workspace/xtb_constr/{anti,syn}/."""
    xtb_fixture = fixture_molecule_dir(molecule) / "xtb_constr"
    for label in ("anti", "syn"):
        dest = workspace / "xtb_constr" / label
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            xtb_fixture / label / "input.xtbopt.xyz", dest / "input.xtbopt.xyz"
        )
        shutil.copyfile(xtb_fixture / label / "input.xtb.log", dest / "input.xtb.log")


def _retarget_prep_indices_to_fixture(workspace: Path) -> None:
    """Rewrite manifest prep indices to point at fixture atoms 0 (C) and 1 (O).

    The xtb_constr fixture is a 3-atom toy XYZ. The chain that bootstraps
    the manifest runs prep on a real chiral BIPS SMILES, so spiro_carbon_idx
    and chromene_oxygen_idx index a much larger molecule. xtb_collect must
    measure the C-O distance on the fixture, so we overwrite the indices to
    match the fixture atom layout.
    """
    path = workspace / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["stages"]["prep"]["outputs"]["spiro_carbon_idx"] = 0
    manifest["stages"]["prep"]["outputs"]["chromene_oxygen_idx"] = 1
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _fake_pbs_submitter(prefix: str = "pbs"):
    counter = {"n": 0}

    def fake(script, args, cwd):  # type: ignore[no-untyped-def]
        counter["n"] += 1
        jobid = f"{counter['n']}.{prefix}"
        return jobid, jobid + "\n"

    return fake


def test_cli_prep_bips_succeeds(
    tmp_path: Path,
    bips_smiles: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["prep", bips_smiles, "--workspace", str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert "status: done" in captured.out
    assert "smiles_canonical:" in captured.out
    assert (tmp_path / "prep" / "stereocentres.json").exists()


def test_cli_prep_json_emits_parseable_payload(
    tmp_path: Path,
    bips_smiles: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["prep", bips_smiles, "--workspace", str(tmp_path), "--json"])
    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["status"] == "done"
    assert payload["outputs"]["smiles_canonical"]
    assert payload["outputs"]["spiro_carbon_idx"] >= 0


def test_cli_prep_invalid_smiles_returns_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["prep", "not a smiles", "--workspace", str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "status: failed" in captured.err
    assert "invalid SMILES" in captured.err


def test_cli_prep_non_spiro_returns_failure(
    tmp_path: Path,
    non_spiro_smiles: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["prep", non_spiro_smiles, "--workspace", str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "heavy atoms" in captured.err


def test_cli_no_subcommand_errors(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_cli_mm_chiral_bips_succeeds(
    tmp_path: Path,
    chiral_bips_smiles: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(
        [
            "mm",
            chiral_bips_smiles,
            "--workspace",
            str(tmp_path),
            "--n-embed",
            "20",
            "--seed",
            "42",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert "status: done" in captured.out
    assert "n_conformers_anti:" in captured.out
    assert "n_conformers_syn:" in captured.out
    assert (tmp_path / "mm" / "anti").is_dir()
    assert (tmp_path / "mm" / "syn").is_dir()
    assert (tmp_path / "mm" / "conformers.json").exists()


def test_cli_mm_invalid_smiles_returns_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["mm", "not a smiles", "--workspace", str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "failed" in captured.err.lower()


def test_cli_xtb_constr_smoke_submits_two_pbs_jobs(
    tmp_path: Path,
    chiral_bips_smiles: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Subprocess is monkeypatched -- no real qsub. Verifies the CLI
    # plumbs prep -> mm -> xtb_stage.submit and reports both jobids.
    from spiropyran_dr.stages import xtb_stage

    counter = {"n": 0}

    def fake_submit(script, args, cwd):  # type: ignore[no-untyped-def]
        counter["n"] += 1
        jobid = f"{2000 + counter['n']}.meta-pbs"
        return jobid, jobid + "\n"

    monkeypatch.setattr(xtb_stage, "submit_via_script", fake_submit)

    rc = main(
        [
            "xtb_constr",
            chiral_bips_smiles,
            "--workspace",
            str(tmp_path),
            "--n-embed",
            "20",
            "--seed",
            "42",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert "status: submitted" in captured.out
    assert "anti jobid:" in captured.out
    assert "syn jobid:" in captured.out
    assert (tmp_path / "xtb_constr" / "anti" / "input.xyz").is_file()
    assert (tmp_path / "xtb_constr" / "syn" / "jobid").is_file()


def test_cli_crest_fails_without_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["crest", "--workspace", str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "manifest.json" in captured.err


def test_cli_crest_fails_when_xtb_constr_not_done(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = {
        "stages": {
            "xtb_constr": {"status": "submitted"},
            "mm": {"status": "done", "outputs": {}},
        }
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rc = main(["crest", "--workspace", str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "xtb_constr" in captured.err


def test_cli_prep_respects_custom_smarts_path(
    tmp_path: Path,
    bips_smiles: str,
    smarts_config_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(
        [
            "prep",
            bips_smiles,
            "--workspace",
            str(tmp_path),
            "--smarts",
            str(smarts_config_path),
        ]
    )
    assert rc == 0


def test_cli_xtb_constr_persists_manifest_json(
    tmp_path: Path,
    chiral_bips_smiles: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spiropyran_dr.stages import xtb_stage

    monkeypatch.setattr(xtb_stage, "submit_via_script", _fake_pbs_submitter("meta-pbs"))

    rc = main(
        [
            "xtb_constr",
            chiral_bips_smiles,
            "--workspace",
            str(tmp_path),
            "--n-embed",
            "20",
            "--seed",
            "42",
        ]
    )
    assert rc == 0, capsys.readouterr().err

    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stages = manifest["stages"]
    assert stages["prep"]["status"] == "done"
    assert stages["mm"]["status"] == "done"
    assert stages["xtb_constr"]["status"] == "submitted"
    assert set(stages["xtb_constr"]["pbs_job_ids"]) == {"anti", "syn"}


def test_cli_xtb_collect_happy_path(
    tmp_path: Path,
    chiral_bips_smiles: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spiropyran_dr.stages import xtb_stage

    monkeypatch.setattr(xtb_stage, "submit_via_script", _fake_pbs_submitter("meta-pbs"))

    rc = main(
        [
            "xtb_constr",
            chiral_bips_smiles,
            "--workspace",
            str(tmp_path),
            "--n-embed",
            "20",
            "--seed",
            "42",
        ]
    )
    assert rc == 0
    capsys.readouterr()  # discard xtb_constr output

    submit_pbs_ids = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )["stages"]["xtb_constr"]["pbs_job_ids"]

    _seed_xtb_outputs(tmp_path)
    _retarget_prep_indices_to_fixture(tmp_path)

    rc = main(["xtb_collect", "--workspace", str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert "status: done" in captured.out
    assert "anti" in captured.out
    assert "syn" in captured.out
    # Per-label numerical summary present.
    assert "C-O (Ang)" in captured.out
    assert "3.4020" in captured.out  # fixture distance, formatted to 4 dp

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    xtb_block = manifest["stages"]["xtb_constr"]
    assert xtb_block["status"] == "done"
    assert "submitted_at" in xtb_block
    assert xtb_block["pbs_job_ids"] == submit_pbs_ids
    for label in ("anti", "syn"):
        entries = xtb_block["outputs"][label]
        assert isinstance(entries, list) and len(entries) == 1
        assert abs(entries[0]["co_distance_final_ang"] - 3.402) < 1e-6
        assert entries[0]["energy_hartree"] < 0


def test_cli_xtb_collect_fails_on_constraint_violation(
    tmp_path: Path,
    chiral_bips_smiles: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spiropyran_dr.stages import xtb_stage

    monkeypatch.setattr(xtb_stage, "submit_via_script", _fake_pbs_submitter("meta-pbs"))

    rc = main(
        [
            "xtb_constr",
            chiral_bips_smiles,
            "--workspace",
            str(tmp_path),
            "--n-embed",
            "20",
            "--seed",
            "42",
        ]
    )
    assert rc == 0
    capsys.readouterr()

    _seed_xtb_outputs(tmp_path)
    _retarget_prep_indices_to_fixture(tmp_path)
    # Overwrite anti input.xtbopt.xyz with a clearly off-target geometry (3.55 A).
    (tmp_path / "xtb_constr" / "anti" / "input.xtbopt.xyz").write_text(
        "3\nbad geometry\nC 0.0 0.0 0.0\nO 3.55 0.0 0.0\nH 0.0 1.0 0.0\n",
        encoding="utf-8",
    )

    rc = main(["xtb_collect", "--workspace", str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "tolerance" in captured.err.lower()

    xtb_block = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))[
        "stages"
    ]["xtb_constr"]
    assert xtb_block["status"] == "failed"
    assert "tolerance" in xtb_block.get("failure_reason", "").lower()


def test_cli_xtb_collect_fails_when_manifest_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["xtb_collect", "--workspace", str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "manifest.json" in captured.err


def test_cli_xtb_collect_fails_when_xtb_constr_status_pending(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = {
        "stages": {
            "xtb_constr": {"status": "pending"},
        }
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    rc = main(["xtb_collect", "--workspace", str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "xtb_constr" in captured.err
    assert "pending" in captured.err


def test_cli_chain_xtb_constr_collect_then_crest_resume_succeeds(
    tmp_path: Path,
    chiral_bips_smiles: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spiropyran_dr.stages import crest_stage, xtb_stage

    monkeypatch.setattr(xtb_stage, "submit_via_script", _fake_pbs_submitter("meta-pbs"))

    rc = main(
        [
            "xtb_constr",
            chiral_bips_smiles,
            "--workspace",
            str(tmp_path),
            "--n-embed",
            "20",
            "--seed",
            "42",
        ]
    )
    assert rc == 0
    capsys.readouterr()

    _seed_xtb_outputs(tmp_path)
    _retarget_prep_indices_to_fixture(tmp_path)

    rc = main(["xtb_collect", "--workspace", str(tmp_path)])
    assert rc == 0, capsys.readouterr().err
    capsys.readouterr()

    monkeypatch.setattr(
        crest_stage, "submit_via_script", _fake_pbs_submitter("crest-pbs")
    )

    rc = main(["crest", "--workspace", str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert "status: submitted" in captured.out
    for label in ("anti_min", "syn_min", "anti_mecp", "syn_mecp"):
        assert f"{label} jobid:" in captured.out

    crest_block = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))[
        "stages"
    ]["crest"]
    assert crest_block["status"] == "submitted"
    assert set(crest_block["pbs_job_ids"]) == {
        "anti_min",
        "syn_min",
        "anti_mecp",
        "syn_mecp",
    }


def test_cli_crest_collect_fails_when_manifest_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["crest_collect", "--workspace", str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "manifest.json" in captured.err


def test_cli_crest_collect_fails_when_crest_status_pending(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = {"stages": {"crest": {"status": "pending"}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    rc = main(["crest_collect", "--workspace", str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "crest" in captured.err
    assert "pending" in captured.err


def _chain_through_crest_submit(
    tmp_path: Path,
    chiral_bips_smiles: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Run prep -> mm -> xtb_constr -> xtb_collect -> crest with mocked PBS."""
    from spiropyran_dr.stages import crest_stage, xtb_stage

    monkeypatch.setattr(xtb_stage, "submit_via_script", _fake_pbs_submitter("meta-pbs"))

    rc = main(
        [
            "xtb_constr",
            chiral_bips_smiles,
            "--workspace",
            str(tmp_path),
            "--n-embed",
            "20",
            "--seed",
            "42",
        ]
    )
    assert rc == 0
    capsys.readouterr()

    _seed_xtb_outputs(tmp_path)
    _retarget_prep_indices_to_fixture(tmp_path)

    rc = main(["xtb_collect", "--workspace", str(tmp_path)])
    assert rc == 0, capsys.readouterr().err
    capsys.readouterr()

    monkeypatch.setattr(
        crest_stage, "submit_via_script", _fake_pbs_submitter("crest-pbs")
    )
    rc = main(["crest", "--workspace", str(tmp_path)])
    assert rc == 0, capsys.readouterr().err
    capsys.readouterr()


def test_cli_crest_collect_happy_path(
    tmp_path: Path,
    chiral_bips_smiles: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chain_through_crest_submit(tmp_path, chiral_bips_smiles, monkeypatch, capsys)
    submit_pbs_ids = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )["stages"]["crest"]["pbs_job_ids"]

    _seed_crest_outputs(tmp_path)

    rc = main(["crest_collect", "--workspace", str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert "status: done" in captured.out
    for label in ("anti_min", "syn_min", "anti_mecp", "syn_mecp"):
        assert label in captured.out

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    crest_block = manifest["stages"]["crest"]
    assert crest_block["status"] == "done"
    # pre-existing pbs_job_ids survive the merge
    assert crest_block["pbs_job_ids"] == submit_pbs_ids
    outputs = crest_block["outputs"]
    for label in ("anti_min", "syn_min", "anti_mecp", "syn_mecp"):
        entries = outputs[label]
        assert isinstance(entries, list) and len(entries) >= 1
        assert outputs[f"n_conformers_{label}"] == len(entries)
        assert outputs[f"{label}_xyz_dir"] == f"crest/{label}/filtered"
        for i, e in enumerate(entries):
            assert e["conf_id"] == i
            assert e["label"] == label


def test_cli_crest_collect_fails_on_missing_outputs(
    tmp_path: Path,
    chiral_bips_smiles: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chain_through_crest_submit(tmp_path, chiral_bips_smiles, monkeypatch, capsys)
    # Seed only one of the four label dirs.
    _seed_crest_outputs(tmp_path, labels=("anti_min",))

    rc = main(["crest_collect", "--workspace", str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "failed" in captured.err.lower()

    crest_block = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))[
        "stages"
    ]["crest"]
    assert crest_block["status"] == "failed"
    assert "failure_reason" in crest_block


# ---------------------------------------------------------------------------
# dft_sp / dft_sp_collect CLI
# ---------------------------------------------------------------------------

DFTSP_LABELS = ("anti_min", "syn_min", "anti_mecp", "syn_mecp")


def _seed_dft_sp_outputs(workspace: Path, molecule: str = "water_synthetic") -> None:
    """Copy per-conformer fixture orca.out files into workspace/dft_sp/<label>/conf_<i>/."""
    dft_fixture = fixture_molecule_dir(molecule) / "dft_sp"
    for label in DFTSP_LABELS:
        for i in range(3):
            dest = workspace / "dft_sp" / label / f"conf_{i}"
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                dft_fixture / label / f"conf_{i}" / "orca.out",
                dest / "orca.out",
            )


def _manifest_with_crest_done(workspace: Path) -> dict:
    """Build a manifest dict with crest done and 3 stub conformers per label.

    Writes 3 dummy single-frame XYZ files per label under
    workspace/crest/<label>/filtered/ so dft_sp.submit can concatenate them.
    """
    outputs: dict = {}
    for label in DFTSP_LABELS:
        confs = []
        for i in range(3):
            xyz_rel = f"crest/{label}/filtered/conf_{i}.xyz"
            xyz_abs = workspace / xyz_rel
            xyz_abs.parent.mkdir(parents=True, exist_ok=True)
            xyz_abs.write_text(
                "3\nconf\nO  0.0  0.0  0.0\nH  1.0  0.0  0.0\nH  0.0  1.0  0.0\n",
                encoding="utf-8",
            )
            confs.append({"conf_id": i, "xyz": xyz_rel, "label": label})
        outputs[label] = confs
    return {
        "smiles_input": "CCO",
        "stages": {
            "crest": {"status": "done", "outputs": outputs},
            "dft_sp": {"status": "pending"},
        },
    }


def test_cli_dft_sp_fails_when_manifest_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["dft_sp", "--workspace", str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "manifest.json" in captured.err


def test_cli_dft_sp_fails_when_crest_not_done(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = {"stages": {"crest": {"status": "submitted"}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rc = main(["dft_sp", "--workspace", str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "crest" in captured.err


def test_cli_dft_sp_happy_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spiropyran_dr.stages import dft_sp_stage

    manifest = _manifest_with_crest_done(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        dft_sp_stage, "submit_via_script", _fake_pbs_submitter("orca-pbs")
    )

    rc = main(["dft_sp", "--workspace", str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert "status: submitted" in captured.out
    for label in DFTSP_LABELS:
        assert label in captured.out

    stored = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    block = stored["stages"]["dft_sp"]
    assert block["status"] == "submitted"
    expected_keys = {f"{lbl}/{i}" for lbl in DFTSP_LABELS for i in range(3)}
    assert set(block["pbs_job_ids"].keys()) == expected_keys


def test_cli_dft_sp_collect_fails_when_manifest_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["dft_sp_collect", "--workspace", str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "manifest.json" in captured.err


def test_cli_dft_sp_collect_fails_when_status_pending(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = {"stages": {"dft_sp": {"status": "pending"}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rc = main(["dft_sp_collect", "--workspace", str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "pending" in captured.err


# ---------------------------------------------------------------------------
# submit (dry-run)
# ---------------------------------------------------------------------------


def test_cli_submit_dry_run_prints_pbs_script(
    tmp_path: Path,
    chiral_bips_smiles: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(
        [
            "submit",
            chiral_bips_smiles,
            "--workspace",
            str(tmp_path),
            "--n-embed",
            "10",
            "--seed",
            "42",
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert "#!/bin/bash" in captured.out
    assert "#PBS" in captured.out
    assert str(tmp_path) in captured.out
    assert "molecule_id" in captured.out
    # manifest must have been written
    assert (tmp_path / "manifest.json").is_file()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"]["prep"]["status"] == "done"
    assert manifest["stages"]["mm"]["status"] == "done"
    assert manifest["stages"]["xtb_constr"]["status"] == "pending"


def test_cli_submit_dry_run_writes_pbs_script_file(
    tmp_path: Path,
    chiral_bips_smiles: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(
        [
            "submit",
            chiral_bips_smiles,
            "--workspace",
            str(tmp_path),
            "--n-embed",
            "10",
            "--seed",
            "42",
            "--dry-run",
        ]
    )
    capsys.readouterr()
    pbs_script = tmp_path / "orchestrator.pbs.sh"
    assert pbs_script.is_file()
    content = pbs_script.read_text(encoding="utf-8")
    assert "#!/bin/bash" in content
    assert "predict" in content


def test_cli_submit_dry_run_invalid_smiles_returns_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["submit", "not-a-smiles", "--workspace", str(tmp_path), "--dry-run"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "failed" in captured.err.lower()


def test_cli_submit_dry_run_stores_config_hash(
    tmp_path: Path,
    chiral_bips_smiles: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(
        [
            "submit",
            chiral_bips_smiles,
            "--workspace",
            str(tmp_path),
            "--n-embed",
            "10",
            "--seed",
            "42",
            "--dry-run",
        ]
    )
    capsys.readouterr()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config_hash"].startswith("sha256:")
    assert manifest["molecule_id"].startswith("sp_")
    assert manifest["options"] == {"thermal": False}


def test_cli_submit_dry_run_thermal_flag(
    tmp_path: Path,
    chiral_bips_smiles: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(
        [
            "submit",
            chiral_bips_smiles,
            "--workspace",
            str(tmp_path),
            "--n-embed",
            "10",
            "--seed",
            "42",
            "--thermal",
            "--dry-run",
        ]
    )
    capsys.readouterr()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["options"]["thermal"] is True


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def _manifest_for_status(tmp_path: Path) -> None:
    manifest = {
        "molecule_id": "sp_abc12345",
        "smiles_input": "CCO",
        "stages": {
            "prep": {"status": "done", "finished_at": "2026-05-07T10:00:00+00:00"},
            "mm": {"status": "done", "finished_at": "2026-05-07T10:00:05+00:00"},
            "xtb_constr": {
                "status": "submitted",
                "submitted_at": "2026-05-07T10:01:00+00:00",
                "pbs_job_ids": {"anti": "12345.meta-pbs", "syn": "12346.meta-pbs"},
            },
            "crest": {"status": "pending"},
            "dft_sp": {"status": "pending"},
            "dft_freq": {"status": "skipped"},
            "aggregate": {"status": "pending"},
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_cli_status_prints_all_stages(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _manifest_for_status(tmp_path)
    rc = main(["status", "--workspace", str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    for stage in (
        "prep",
        "mm",
        "xtb_constr",
        "crest",
        "dft_sp",
        "dft_freq",
        "aggregate",
    ):
        assert stage in captured.out
    assert "sp_abc12345" in captured.out
    assert "12345.meta-pbs" in captured.out


def test_cli_status_fails_without_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["status", "--workspace", str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "manifest.json" in captured.err


# ---------------------------------------------------------------------------
# predict / resume — fail without manifest
# ---------------------------------------------------------------------------


def test_cli_predict_fails_without_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["predict", "--workspace", str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "manifest.json" in captured.err


def test_cli_resume_fails_without_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["resume", "--workspace", str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "manifest.json" in captured.err


# ---------------------------------------------------------------------------
# dft_sp_collect happy-path (already defined below, keep existing)
# ---------------------------------------------------------------------------


def test_cli_dft_sp_collect_happy_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest_with_crest_done(tmp_path)
    submit_pbs_ids = {
        f"{label}/{i}": f"{n:02d}.meta-pbs"
        for n, (label, i) in enumerate(
            (lbl, idx) for lbl in DFTSP_LABELS for idx in range(3)
        )
    }
    manifest["stages"]["dft_sp"] = {
        "status": "submitted",
        "pbs_job_ids": submit_pbs_ids,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _seed_dft_sp_outputs(tmp_path)

    rc = main(["dft_sp_collect", "--workspace", str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert "status: done" in captured.out
    for label in DFTSP_LABELS:
        assert label in captured.out

    stored = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    block = stored["stages"]["dft_sp"]
    assert block["status"] == "done"
    # pbs_job_ids from submit must survive the collect merge
    assert block["pbs_job_ids"] == submit_pbs_ids
    for label in DFTSP_LABELS:
        entries = block["outputs"][label]
        assert len(entries) == 3
        assert all("energy_hartree" in e for e in entries)
