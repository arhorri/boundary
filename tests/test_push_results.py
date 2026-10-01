"""push_results and the GH_TOKEN lookup, against real git repositories in tmp dirs.

The failure these cover (Colab, hours into a session): the secret store, which is asked
through the browser, stopped answering, so the token lookup at the END of push_results
raised after ``git commit`` had already run. The retry then saw a clean tree, reported
"nothing changed" and never pushed the stranded commit.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

import scripts.bootstrap_session as bootstrap
import scripts.push_results as pr


def _git(cwd, *args):
    out = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def _repo_with_remote(tmp_path):
    """A bare 'origin', and a clone of it with one commit pushed, on its default branch."""
    remote = tmp_path / "remote.git"
    _git(tmp_path, "init", "--bare", str(remote))
    seed = tmp_path / "seed"
    _git(tmp_path, "clone", str(remote), str(seed))
    _git(seed, "config", "user.email", "t@example.com")
    _git(seed, "config", "user.name", "t")
    (seed / "reports").mkdir()
    (seed / "reports" / "base.txt").write_text("base")
    _git(seed, "add", "-A")
    _git(seed, "commit", "-m", "base")
    branch = _git(seed, "rev-parse", "--abbrev-ref", "HEAD")
    _git(seed, "push", "origin", f"HEAD:{branch}")
    work = tmp_path / "work"
    _git(tmp_path, "clone", str(remote), str(work))
    _git(work, "config", "user.email", "t@example.com")
    _git(work, "config", "user.name", "t")
    return remote, work, branch


class _FakeBootstrap:
    """Stands in for scripts.bootstrap_session: a token that works, or one that cannot be read."""

    def __init__(self, fail=False):
        self.fail = fail

    def get_github_token(self, platform):
        if self.fail:
            raise RuntimeError("GH_TOKEN not available on colab (TimeoutException)")
        return "tok"

    def detect_platform(self):
        return "colab"

    def _auth_url(self, url, token):
        return url        # a local path: no credentials to embed


def _commits(repo):
    return int(_git(repo, "rev-list", "--count", "HEAD"))


def test_an_unreadable_token_fails_with_nothing_committed(tmp_path, monkeypatch):
    remote, work, branch = _repo_with_remote(tmp_path)
    (work / "reports" / "new.txt").write_text("results")
    monkeypatch.setattr(pr, "_bootstrap_module", lambda root: _FakeBootstrap(fail=True))
    before = _commits(work)

    with pytest.raises(RuntimeError, match="GH_TOKEN"):
        pr.push_results("step: x", repo_root=work, branch=branch)

    assert _commits(work) == before, "the token must be fetched BEFORE git commit"


def test_a_retry_after_that_failure_commits_and_pushes_the_results(tmp_path, monkeypatch):
    remote, work, branch = _repo_with_remote(tmp_path)
    (work / "reports" / "new.txt").write_text("results")
    monkeypatch.setattr(pr, "_bootstrap_module", lambda root: _FakeBootstrap(fail=True))
    with pytest.raises(RuntimeError):
        pr.push_results("step: x", repo_root=work, branch=branch)

    monkeypatch.setattr(pr, "_bootstrap_module", lambda root: _FakeBootstrap())
    assert pr.push_results("step: x", repo_root=work, branch=branch) is True
    assert _git(remote, "log", "-1", "--format=%s", branch) == "step: x"
    assert "reports/new.txt" in _git(remote, "ls-tree", "-r", "--name-only", branch)


def test_a_commit_stranded_by_an_earlier_failed_push_is_pushed_not_called_a_no_op(tmp_path, monkeypatch):
    """The exact state the Colab failure left behind: committed locally, never pushed."""
    remote, work, branch = _repo_with_remote(tmp_path)
    (work / "reports" / "new.txt").write_text("results")
    _git(work, "add", "-A")
    _git(work, "commit", "-m", "stranded")
    assert _git(remote, "rev-list", "--count", branch) == "1"
    monkeypatch.setattr(pr, "_bootstrap_module", lambda root: _FakeBootstrap())

    assert pr.push_results("step: x", repo_root=work, branch=branch) is True

    assert _git(remote, "rev-list", "--count", branch) == "2"
    assert _git(remote, "log", "-1", "--format=%s", branch) == "stranded"


def test_a_clean_tree_that_is_in_sync_is_still_a_no_op_and_needs_no_token(tmp_path, monkeypatch):
    remote, work, branch = _repo_with_remote(tmp_path)

    def must_not_be_asked(root):
        raise AssertionError("nothing to push, so no token should be fetched")

    monkeypatch.setattr(pr, "_bootstrap_module", must_not_be_asked)
    assert pr.push_results("step: x", repo_root=work, branch=branch) is False


def test_unpushed_commits_counts_only_local_commits(tmp_path):
    remote, work, branch = _repo_with_remote(tmp_path)
    assert pr._unpushed_commits(work, branch) == 0
    (work / "reports" / "a.txt").write_text("a")
    _git(work, "add", "-A")
    _git(work, "commit", "-m", "a")
    assert pr._unpushed_commits(work, branch) == 1
    assert pr._unpushed_commits(work, "no-such-branch") == 0


# --------------------------------------------------------------------------
# the token lookup: read once, reuse -- the secret store can go away mid-session
# --------------------------------------------------------------------------
def _clean_env(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "placeholder")
    monkeypatch.delenv("GH_TOKEN")        # registered, so the test's own write is undone


def test_the_token_is_read_from_the_store_once_then_reused_when_the_store_stops_answering(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setitem(bootstrap.SECRET_READERS, "colab", lambda: "  tok  ")
    assert bootstrap.get_github_token("colab") == "tok"
    assert os.environ["GH_TOKEN"] == "tok"

    def timed_out():
        raise TimeoutError("the browser never answered")

    monkeypatch.setitem(bootstrap.SECRET_READERS, "colab", timed_out)
    assert bootstrap.get_github_token("colab") == "tok", (
        "a later call must not depend on the secret store still answering")


def test_with_no_token_anywhere_the_lookup_still_fails_loudly(monkeypatch):
    _clean_env(monkeypatch)

    def timed_out():
        raise TimeoutError("the browser never answered")

    monkeypatch.setitem(bootstrap.SECRET_READERS, "colab", timed_out)
    monkeypatch.setitem(bootstrap.SECRET_READERS, "kaggle", timed_out)
    with pytest.raises(bootstrap.BootstrapError, match="GH_TOKEN not available"):
        bootstrap.get_github_token("colab")
