from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from spiropyran_dr.pbs_utils import (
    PBSSubmitError,
    parse_jobid_from_qsub_stdout,
    read_jobid,
    submit_via_script,
    write_jobid,
)


# -- parse_jobid_from_qsub_stdout -----------------------------------------


def test_parse_jobid_strips_whitespace() -> None:
    assert (
        parse_jobid_from_qsub_stdout("  12345.meta-pbs.metacentrum.cz\n")
        == "12345.meta-pbs.metacentrum.cz"
    )


def test_parse_jobid_takes_last_nonblank_line() -> None:
    # sub_crest.sh prints other lines before qsub's output; the jobid is the
    # last non-blank line.
    text = (
        "Submitting CREST job for input.xyz\n"
        "\n"
        "67890.meta-pbs.metacentrum.cz\n"
    )
    assert parse_jobid_from_qsub_stdout(text) == "67890.meta-pbs.metacentrum.cz"


def test_parse_jobid_rejects_empty() -> None:
    with pytest.raises(PBSSubmitError):
        parse_jobid_from_qsub_stdout("   \n\n")


# -- jobid file I/O -------------------------------------------------------


def test_write_then_read_jobid(tmp_path: Path) -> None:
    target = tmp_path / "jobid"
    write_jobid(target, "42.meta-pbs")
    assert read_jobid(target) == "42.meta-pbs"


# -- submit_via_script ----------------------------------------------------


def test_submit_via_script_invokes_subprocess_and_parses_jobid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_run(cmd, cwd, capture_output, text, check):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="999.meta-pbs\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    script = Path("/fake/sub_crest.sh")
    jobid, raw = submit_via_script(script, ["24", "input.xyz"], cwd=tmp_path)
    assert jobid == "999.meta-pbs"
    assert raw == "999.meta-pbs\n"
    assert captured["cmd"] == [str(script), "24", "input.xyz"]
    assert captured["cwd"] == str(tmp_path)


def test_submit_via_script_wraps_subprocess_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(cmd, cwd, capture_output, text, check):  # type: ignore[no-untyped-def]
        raise subprocess.CalledProcessError(
            returncode=1, cmd=cmd, output="", stderr="qsub: bad queue"
        )

    monkeypatch.setattr(subprocess, "run", boom)

    with pytest.raises(PBSSubmitError, match="qsub"):
        submit_via_script(
            Path("/fake/sub_crest.sh"), ["24", "input.xyz"], cwd=tmp_path
        )
