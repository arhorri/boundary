"""Tests for src/session.py's run/fold distinction. RUN THESE IN notebooks/06.

The checkpoint directory is named for the RUN (`dev-no-uhcs1`), while the
identity stored inside the checkpoint is the FOLD (`dev`). They are equal
whenever nothing is excluded, which is exactly why conflating them went
unnoticed until the first reduced-mixture run reported a correctly written
checkpoint as missing.

Nothing here needs a GPU, torch or the datasets, so nothing here can skip.
"""

from __future__ import annotations

import pytest

from src import session as session_mod


def _session(tmp_path, platform="local"):
    return session_mod.for_host({"platform": platform,
                                 "persistent_dir": tmp_path})


def _write_checkpoint(tmp_path, run_name, name="last.pt"):
    directory = tmp_path / "checkpoints" / run_name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"not a real checkpoint, but a real file")
    return path


def test_checkpoint_dir_is_named_for_the_run_not_the_fold(tmp_path):
    session = _session(tmp_path)
    assert session.checkpoint_dir("dev").name == "dev"
    assert session.checkpoint_dir("dev-no-uhcs1").name == "dev-no-uhcs1", (
        "a reduced-mixture run keeps its checkpoints under its own name")


def test_checks_find_a_checkpoint_written_under_a_reduced_mixture_run(tmp_path):
    """The regression: this check failed on a run that was perfectly fine.

    Trainer writes to checkpoints/<run_name>/, so asking the session about the
    FOLD looks in a directory that was never written and reports a missing
    checkpoint for a run that saved one correctly.
    """
    _write_checkpoint(tmp_path, "dev-no-uhcs1")
    session = _session(tmp_path)

    by_run = dict((name, ok) for name, ok, _ in session.checks("dev-no-uhcs1"))
    assert all(by_run.values()), f"the run's own checkpoint was not found: {by_run}"

    # ...and addressing it by the fold name is what used to happen, and fails.
    by_fold = dict((name, ok) for name, ok, _ in session.checks("dev"))
    assert not any(by_fold.values()), (
        "checks('dev') found something under a run named 'dev-no-uhcs1'; the "
        "test can no longer tell the regression from correct behaviour")


def test_checks_still_work_when_run_and_fold_coincide(tmp_path):
    """The ordinary case: no exclusion, so the run IS the fold."""
    _write_checkpoint(tmp_path, "dev")
    session = _session(tmp_path)
    assert all(ok for _, ok, _ in session.checks("dev"))


def test_checks_fail_when_there_is_genuinely_no_checkpoint(tmp_path):
    session = _session(tmp_path)
    (tmp_path / "checkpoints" / "dev-no-uhcs1").mkdir(parents=True)
    assert not any(ok for _, ok, _ in session.checks("dev-no-uhcs1")), (
        "an empty directory must not read as a saved checkpoint")


def test_stage_resume_reports_the_run_and_the_fold_separately(tmp_path):
    _write_checkpoint(tmp_path, "dev-no-uhcs1")
    session = _session(tmp_path)
    report = session.stage_resume("dev-no-uhcs1", fold="dev", verbose=False)
    assert report["run"] == "dev-no-uhcs1"
    assert report["fold"] == "dev", (
        "the fold identity must survive alongside the run name -- it is what "
        "the checkpoint's own header is validated against")
    assert report["resumable"], "the run's checkpoint is right there"
    assert report["checkpoint_dir"].endswith("dev-no-uhcs1")


def test_fold_defaults_to_the_run_when_they_are_the_same(tmp_path):
    session = _session(tmp_path)
    report = session.stage_resume("dev", verbose=False)
    assert report["fold"] == "dev"


def test_survival_report_is_addressed_by_run(tmp_path, capsys):
    """Addressed by run, checked on the empty path so no torch load is needed.

    Whether it can READ a checkpoint is torch's business and is covered where a
    real one exists; what fix this test pins is that it looks under the run
    directory at all.
    """
    (tmp_path / "checkpoints" / "dev-no-uhcs1").mkdir(parents=True)
    session = _session(tmp_path)
    report = session.survival_report("dev-no-uhcs1")
    assert report["run"] == "dev-no-uhcs1"
    assert report["checkpoint_dir"].endswith("dev-no-uhcs1")
    assert report["checkpoints"]["last.pt"] is None
    out = capsys.readouterr().out
    assert "dev-no-uhcs1" in out


def test_the_generic_host_needs_no_action_and_kaggle_does(tmp_path):
    """The dispatch itself, since every method above depends on it."""
    assert type(_session(tmp_path, "local")) is session_mod.SessionSupport
    assert _session(tmp_path, "local").persists
    assert not _session(tmp_path, "local").time_limited
    assert session_mod.HOSTS["kaggle"] is session_mod.KaggleSession
