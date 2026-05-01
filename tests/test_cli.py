from __future__ import annotations

import json
from pathlib import Path

import pytest

from spiropyran_dr.cli import main


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
