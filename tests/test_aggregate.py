from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from spiropyran_dr.stages import aggregate
from spiropyran_dr.stages.base import Stage

LABELS = ("anti_min", "syn_min", "anti_mecp", "syn_mecp")


def _manifest_with_energies(
    energies: dict[str, list[float]],
    *,
    molecule_id: str = "sp_test",
    smiles_canonical: str = "CCO",
) -> dict[str, Any]:
    """Build a manifest with dft_sp.outputs populated from {label: [E0, E1, ...]}."""
    outputs: dict[str, list[dict[str, Any]]] = {}
    for label, e_list in energies.items():
        outputs[label] = [
            {
                "conf_id": i,
                "xyz": f"dft_sp/{label}/conf_{i}/conf_{i}.xyz",
                "energy_hartree": float(e),
                "label": label,
            }
            for i, e in enumerate(e_list)
        ]
    return {
        "molecule_id": molecule_id,
        "smiles_canonical": smiles_canonical,
        "config_hash": "sha256:fake",
        "stages": {
            "dft_sp": {
                "status": "done",
                "outputs": outputs,
            }
        },
    }


# -- protocol conformance ---------------------------------------------------


def test_module_satisfies_stage_protocol() -> None:
    stage: Stage = aggregate
    assert callable(stage.is_ready)
    assert callable(stage.submit)
    assert callable(stage.collect)


# -- is_ready ----------------------------------------------------------------


def test_is_ready_false_when_dft_sp_missing(tmp_path: Path) -> None:
    assert aggregate.is_ready({"stages": {}}, tmp_path) is False


def test_is_ready_false_when_dft_sp_pending(tmp_path: Path) -> None:
    manifest = {"stages": {"dft_sp": {"status": "submitted", "outputs": {}}}}
    assert aggregate.is_ready(manifest, tmp_path) is False


def test_is_ready_false_when_label_empty(tmp_path: Path) -> None:
    manifest = _manifest_with_energies(
        {"anti_min": [-1.0], "syn_min": [-1.0], "anti_mecp": [-1.0], "syn_mecp": []}
    )
    assert aggregate.is_ready(manifest, tmp_path) is False


def test_is_ready_true_when_all_labels_have_conformers(tmp_path: Path) -> None:
    manifest = _manifest_with_energies({label: [-1.0, -1.001] for label in LABELS})
    assert aggregate.is_ready(manifest, tmp_path) is True


# -- submit: arithmetic -----------------------------------------------------


def test_submit_computes_ddE_lowest_conformer(tmp_path: Path) -> None:
    # Energies engineered so the lowest in each label is unambiguous and the
    # ddE values are easy to verify by hand.
    energies = {
        "anti_min": [-1234.500, -1234.499],  # min = -1234.500
        "syn_min": [-1234.501, -1234.498],  # min = -1234.501
        "anti_mecp": [-1234.450, -1234.448],  # min = -1234.450
        "syn_mecp": [-1234.452, -1234.449],  # min = -1234.452
    }
    manifest = _manifest_with_energies(energies)
    result = aggregate.submit(manifest, tmp_path, config={})

    assert result["status"] == "done"
    ddE = result["outputs"]["ddE"]

    # ground: anti_min - syn_min = -1234.500 - (-1234.501) = +0.001
    assert ddE["ground"]["hartree"] == pytest.approx(0.001, rel=1e-9, abs=1e-12)
    assert ddE["ground"]["kj_mol"] == pytest.approx(
        0.001 * aggregate.HARTREE_TO_KJ_MOL, rel=1e-9
    )
    assert ddE["ground"]["anti_conf_id"] == 0
    assert ddE["ground"]["syn_conf_id"] == 0

    # mecp: anti_mecp - syn_mecp = -1234.450 - (-1234.452) = +0.002
    assert ddE["mecp"]["hartree"] == pytest.approx(0.002, rel=1e-9, abs=1e-12)
    assert ddE["mecp"]["kj_mol"] == pytest.approx(
        0.002 * aggregate.HARTREE_TO_KJ_MOL, rel=1e-9
    )


def test_submit_picks_lowest_not_first(tmp_path: Path) -> None:
    # conf_0 is NOT the lowest for anti_min; aggregate must select conf_2.
    energies = {
        "anti_min": [-1.000, -1.001, -1.002],  # lowest = conf_2
        "syn_min": [-1.000],
        "anti_mecp": [-0.900],
        "syn_mecp": [-0.900],
    }
    manifest = _manifest_with_energies(energies)
    result = aggregate.submit(manifest, tmp_path, config={})
    assert result["outputs"]["ddE"]["ground"]["anti_conf_id"] == 2


# -- ratio formatting -------------------------------------------------------


def test_format_ratio_anti_dominates() -> None:
    # Negative ddE -> anti more stable -> K > 1 -> "K:1"
    s = aggregate._format_ratio(-5.0)
    assert s.endswith(":1")
    K = math.exp(5.0 / (aggregate.R_KJ_PER_MOL_K * aggregate.TEMPERATURE_K))
    assert s == f"{K:.1f}:1"


def test_format_ratio_syn_dominates() -> None:
    # Positive ddE -> syn more stable -> K < 1 -> "1:N"
    s = aggregate._format_ratio(+5.0)
    assert s.startswith("1:")
    K = math.exp(-5.0 / (aggregate.R_KJ_PER_MOL_K * aggregate.TEMPERATURE_K))
    assert s == f"1:{1.0 / K:.1f}"


def test_format_ratio_zero_ddE_is_one_to_one() -> None:
    # K = 1 exactly -> falls into the K >= 1 branch -> "1.0:1"
    assert aggregate._format_ratio(0.0) == "1.0:1"


# -- result.json -----------------------------------------------------------


def test_submit_writes_result_json(tmp_path: Path) -> None:
    manifest = _manifest_with_energies(
        {label: [-1234.5 - 0.001 * i for i in range(2)] for label in LABELS}
    )
    aggregate.submit(manifest, tmp_path, config={})

    result_path = tmp_path / "result.json"
    assert result_path.is_file()
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert payload["molecule_id"] == "sp_test"
    assert payload["smiles_canonical"] == "CCO"
    assert payload["config_hash"] == "sha256:fake"
    assert payload["temperature_k"] == aggregate.TEMPERATURE_K
    for label in LABELS:
        assert len(payload["energies_hartree"][label]) == 2
    assert "ratio_anti_syn" in payload["ddE"]["mecp"]
    assert "ratio_anti_syn" in payload["ddE"]["ground"]


# -- single-conformer ensembles --------------------------------------------


def test_submit_handles_single_conformer_per_label(tmp_path: Path) -> None:
    manifest = _manifest_with_energies({label: [-76.4] for label in LABELS})
    result = aggregate.submit(manifest, tmp_path, config={})
    assert result["status"] == "done"
    # All four labels equal -> both ddE values are zero.
    assert result["outputs"]["ddE"]["mecp"]["hartree"] == pytest.approx(0.0)
    assert result["outputs"]["ddE"]["ground"]["hartree"] == pytest.approx(0.0)


# -- failure modes ---------------------------------------------------------


def test_submit_failed_when_label_empty(tmp_path: Path) -> None:
    manifest = _manifest_with_energies(
        {"anti_min": [-1.0], "syn_min": [-1.0], "anti_mecp": [-1.0], "syn_mecp": []}
    )
    result = aggregate.submit(manifest, tmp_path, config={})
    assert result["status"] == "failed"
    assert "syn_mecp" in result["failure_reason"]
    assert not (tmp_path / "result.json").exists()


# -- collect is no-op ------------------------------------------------------


def test_collect_returns_existing_block(tmp_path: Path) -> None:
    block = {"status": "done", "outputs": {"ddE": {}}}
    manifest = {"stages": {"aggregate": block}}
    out = aggregate.collect(manifest, tmp_path, config={})
    assert out is block
