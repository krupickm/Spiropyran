"""Stage 7: aggregate (local) -- v1 simplistic.

Reads dft_sp per-conformer energies from the manifest, computes the
lowest-conformer energy difference for each ensemble pair (MECP and
ground-state) as

    ddE = E(anti) - E(syn)

at T = 298.15 K, and the corresponding equilibrium constant

    K = exp(-ddE / RT)

reported as an "anti:syn" ratio with anti always first. Writes
``result.json`` to the workspace and prints the same numbers to stdout.

This is the v1 stripped-down implementation. Boltzmann averaging,
ddG (from dft_freq), and the full result.json schema in project.md
section 7 are deferred to a v2 rework. See project.md section 10.7.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LABELS: tuple[str, str, str, str] = ("anti_min", "syn_min", "anti_mecp", "syn_mecp")
PAIRS: tuple[tuple[str, str, str], ...] = (
    ("mecp", "anti_mecp", "syn_mecp"),
    ("ground", "anti_min", "syn_min"),
)

HARTREE_TO_KJ_MOL: float = 2625.4996394798254
R_KJ_PER_MOL_K: float = 8.314462618e-3
TEMPERATURE_K: float = 298.15


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _lowest(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return min(entries, key=lambda e: float(e["energy_hartree"]))


def _format_ratio(ddE_kj_mol: float) -> str:
    """Return "anti:syn" ratio, anti always first.

    K = exp(-ddE / RT) is the anti/syn equilibrium constant given
    ddE = E(anti) - E(syn). When K >= 1 anti dominates and the ratio is
    "K:1"; otherwise syn dominates and we invert to "1:(1/K)".
    """
    K = math.exp(-ddE_kj_mol / (R_KJ_PER_MOL_K * TEMPERATURE_K))
    if K >= 1.0:
        return f"{K:.1f}:1"
    return f"1:{1.0 / K:.1f}"


def is_ready(manifest: dict[str, Any], workspace: Path) -> bool:
    dft_sp = manifest.get("stages", {}).get("dft_sp", {})
    if dft_sp.get("status") != "done":
        return False
    outputs = dft_sp.get("outputs", {})
    return all(label in outputs and len(outputs[label]) > 0 for label in LABELS)


def submit(
    manifest: dict[str, Any], workspace: Path, config: dict[str, Any]
) -> dict[str, Any]:
    started_at = _now_iso()
    dft_sp_outputs = manifest["stages"]["dft_sp"]["outputs"]

    for label in LABELS:
        entries = dft_sp_outputs.get(label) or []
        if not entries:
            return {
                "status": "failed",
                "started_at": started_at,
                "finished_at": _now_iso(),
                "failure_reason": f"no dft_sp conformers for label {label!r}",
            }

    energies_hartree: dict[str, list[dict[str, Any]]] = {
        label: [
            {"conf_id": int(e["conf_id"]), "energy_hartree": float(e["energy_hartree"])}
            for e in dft_sp_outputs[label]
        ]
        for label in LABELS
    }

    ddE: dict[str, dict[str, Any]] = {}
    for pair_name, anti_label, syn_label in PAIRS:
        a = _lowest(dft_sp_outputs[anti_label])
        s = _lowest(dft_sp_outputs[syn_label])
        ddE_h = float(a["energy_hartree"]) - float(s["energy_hartree"])
        ddE_kj = ddE_h * HARTREE_TO_KJ_MOL
        ddE[pair_name] = {
            "hartree": ddE_h,
            "kj_mol": ddE_kj,
            "anti_conf_id": int(a["conf_id"]),
            "syn_conf_id": int(s["conf_id"]),
            "ratio_anti_syn": _format_ratio(ddE_kj),
        }

    result_payload: dict[str, Any] = {
        "molecule_id": manifest.get("molecule_id"),
        "smiles_canonical": manifest.get("smiles_canonical"),
        "config_hash": manifest.get("config_hash"),
        "temperature_k": TEMPERATURE_K,
        "energies_hartree": energies_hartree,
        "ddE": ddE,
    }

    result_path = workspace / "result.json"
    result_path.write_text(
        json.dumps(result_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    _print_summary(result_payload)

    return {
        "status": "done",
        "started_at": started_at,
        "finished_at": _now_iso(),
        "outputs": {
            "result_path": "result.json",
            "ddE": ddE,
            "temperature_k": TEMPERATURE_K,
        },
    }


def collect(
    manifest: dict[str, Any], workspace: Path, config: dict[str, Any]
) -> dict[str, Any]:
    # Local stage: all work happens in submit(); collect is a no-op so the
    # orchestrator can call it without breaking re-entry.
    return manifest.get("stages", {}).get("aggregate", {"status": "done"})


def _print_summary(payload: dict[str, Any]) -> None:
    """Print raw per-conformer energies and the two ddE numbers to stdout."""
    print("=== aggregate (v1) ===")
    print(f"molecule_id:      {payload.get('molecule_id')}")
    print(f"smiles_canonical: {payload.get('smiles_canonical')}")
    print(f"temperature_k:    {payload['temperature_k']}")
    print()
    print(f"  {'label':<10}  {'conf_id':>7}  {'E (Eh)':>16}")
    for label in LABELS:
        for entry in payload["energies_hartree"][label]:
            print(
                f"  {label:<10}  {entry['conf_id']:>7d}  "
                f"{entry['energy_hartree']:>16.8f}"
            )
    print()
    print(f"  {'pair':<8}  {'ddE (Eh)':>14}  {'ddE (kJ/mol)':>14}  {'anti:syn':>12}")
    for pair_name, _, _ in PAIRS:
        d = payload["ddE"][pair_name]
        print(
            f"  {pair_name:<8}  {d['hartree']:>+14.8f}  {d['kj_mol']:>+14.4f}  "
            f"{d['ratio_anti_syn']:>12}"
        )
