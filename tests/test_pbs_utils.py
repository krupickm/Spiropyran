from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from spiropyran_dr.pbs_utils import (
    PBSSubmitError,
    generate_orchestrator_pbs,
    is_all_jobs_done,
    parse_jobid_from_qsub_stdout,
    poll_job_state,
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
    text = "Submitting CREST job for input.xyz\n\n67890.meta-pbs.metacentrum.cz\n"
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


# -- poll_job_state -------------------------------------------------------


def test_poll_job_state_running(monkeypatch: pytest.MonkeyPatch) -> None:
    output = "Job Id: 123.meta-pbs\n    job_state = R\n    queue = default\n"

    def fake_run(cmd, capture_output, text):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=output, stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert poll_job_state("123.meta-pbs") == "running"


def test_poll_job_state_finished_C(monkeypatch: pytest.MonkeyPatch) -> None:
    output = "Job Id: 123.meta-pbs\n    job_state = C\n"

    def fake_run(cmd, capture_output, text):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=output, stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert poll_job_state("123.meta-pbs") == "finished"


def test_poll_job_state_finished_F(monkeypatch: pytest.MonkeyPatch) -> None:
    output = "Job Id: 123.meta-pbs\n    job_state = F\n"

    def fake_run(cmd, capture_output, text):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=output, stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert poll_job_state("123.meta-pbs") == "finished"


def test_poll_job_state_not_found_on_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(cmd, capture_output, text):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(
            args=cmd, returncode=1, stdout="", stderr="Unknown Job Id"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert poll_job_state("999.meta-pbs") == "not_found"


def test_poll_job_state_not_found_when_qstat_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(cmd, capture_output, text):  # type: ignore[no-untyped-def]
        raise FileNotFoundError("qstat not found")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert poll_job_state("123.meta-pbs") == "not_found"


def test_is_all_jobs_done_true_when_all_finished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "spiropyran_dr.pbs_utils.poll_job_state", lambda jid: "finished"
    )
    assert is_all_jobs_done({"anti": "1.meta", "syn": "2.meta"}) is True


def test_is_all_jobs_done_false_when_one_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = {"1.meta": "finished", "2.meta": "running"}
    monkeypatch.setattr(
        "spiropyran_dr.pbs_utils.poll_job_state", lambda jid: states[jid]
    )
    assert is_all_jobs_done({"anti": "1.meta", "syn": "2.meta"}) is False


def test_is_all_jobs_done_true_for_empty_dict() -> None:
    assert is_all_jobs_done({}) is True


# -- generate_orchestrator_pbs --------------------------------------------


def test_generate_orchestrator_pbs_contains_required_fields(tmp_path: Path) -> None:
    config = {
        "pbs": {
            "queue_orchestrator": "oven@meta-pbs.metacentrum.cz",
            "walltime_orchestrator": "720:00:00",
        }
    }
    script = generate_orchestrator_pbs(
        workspace=tmp_path,
        config=config,
        python_exe="/path/to/python",
    )
    assert "#!/bin/bash" in script
    assert "#PBS -q oven@meta-pbs.metacentrum.cz" in script
    assert "#PBS -l walltime=720:00:00" in script
    assert str(tmp_path.resolve()) in script
    assert "/path/to/python" in script
    assert "predict" in script


def test_generate_orchestrator_pbs_uses_config_defaults(tmp_path: Path) -> None:
    script = generate_orchestrator_pbs(
        workspace=tmp_path,
        config={},
        python_exe="/usr/bin/python3",
    )
    assert "oven@meta-pbs.metacentrum.cz" in script
    assert "720:00:00" in script


# -- submit_via_script ----------------------------------------------------


def test_submit_via_script_wraps_subprocess_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(cmd, cwd, capture_output, text, check):  # type: ignore[no-untyped-def]
        raise subprocess.CalledProcessError(
            returncode=1, cmd=cmd, output="", stderr="qsub: bad queue"
        )

    monkeypatch.setattr(subprocess, "run", boom)

    with pytest.raises(PBSSubmitError, match="qsub"):
        submit_via_script(Path("/fake/sub_crest.sh"), ["24", "input.xyz"], cwd=tmp_path)
