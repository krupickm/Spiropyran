from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from rdkit import Chem


def write_xyz(
    path: Path, mol: Chem.Mol, conf_id: int = 0, comment: str = ""
) -> None:
    """Write a single conformer of an RDKit Mol as a standard XYZ file.

    Coordinates are taken from the conformer with id ``conf_id`` (default 0).
    Element symbols come from ``atom.GetSymbol()``; explicit hydrogens are
    written if they are present on the Mol. The caller is responsible for
    AddHs/embedding before calling this.
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("Mol has no conformer; embed (and AddHs) before writing")
    conf = mol.GetConformer(conf_id)

    n = mol.GetNumAtoms()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{n}", comment]
    for idx in range(n):
        pos = conf.GetAtomPosition(idx)
        sym = mol.GetAtomWithIdx(idx).GetSymbol()
        lines.append(f"{sym} {pos.x:.8f} {pos.y:.8f} {pos.z:.8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_xyz(path: Path) -> tuple[list[str], list[tuple[float, float, float]], str]:
    """Parse a single-frame XYZ file into (symbols, coords, comment).

    Minimal reader: assumes the file is well-formed and the header line is
    a non-negative integer. Used by tests for round-trip checks; not yet a
    multi-frame CREST-output reader.
    """
    text = path.read_text(encoding="utf-8").splitlines()
    n = int(text[0].strip())
    comment = text[1] if len(text) > 1 else ""
    symbols: list[str] = []
    coords: list[tuple[float, float, float]] = []
    for raw in text[2 : 2 + n]:
        parts = raw.split()
        symbols.append(parts[0])
        coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return symbols, coords, comment


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write JSON: tempfile in same dir, then os.replace.

    Same-directory tempfile guarantees the rename is on the same filesystem,
    so os.replace is atomic on POSIX and (since Python 3.3) on Windows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
