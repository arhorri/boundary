#!/usr/bin/env python3
"""Push host-generated results back into the repo.

Notebooks call this in their final cell. Results produced on Colab or Kaggle
(audit reports, updated configs, the executed notebook itself) only become
visible to later steps once they are committed here.

    from scripts.push_results import push_results
    push_results("step 1: dataset audit", paths=PATHS)

Only ``reports/``, ``configs/`` and ``notebooks/`` are ever staged. Data,
checkpoints and outputs are never pushed -- they are large, they are
gitignored, and they live in PERSISTENT_DIR instead.

Authentication uses the GH_TOKEN host secret, exactly as the bootstrap does.
The token is written into the remote URL only for the duration of the push
and scrubbed out immediately afterwards; it is never printed and never
committed.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

TRACKED_DIRS = ("reports", "configs", "notebooks")

DEFAULT_AUTHOR_NAME = "phase11-session"
DEFAULT_AUTHOR_EMAIL = "phase11-session@users.noreply.github.com"


class PushError(RuntimeError):
    """Raised when results cannot be pushed. Never fails silently."""


def _bootstrap_module(repo_root: Path):
    """Reuse the bootstrap's platform + secret logic; never duplicate it."""
    try:
        import scripts.bootstrap_session as mod  # namespace package

        return mod
    except Exception:
        pass
    path = repo_root / "scripts" / "bootstrap_session.py"
    if not path.is_file():
        raise PushError(f"cannot locate {path}; is this a full checkout?")
    spec = importlib.util.spec_from_file_location("bootstrap_session", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(cmd: Sequence, cwd: Path, check: bool = True, quiet: bool = False):
    cmd = [str(c) for c in cmd]
    if not quiet:
        print("  $ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise PushError(
            "command failed: " + " ".join(cmd) + "\n"
            + (proc.stderr or proc.stdout or "").strip()
        )
    return proc


def _ensure_identity(repo_root: Path) -> None:
    """Give the ephemeral host a committer identity if it has none."""
    for key, value in (("user.name", DEFAULT_AUTHOR_NAME),
                       ("user.email", DEFAULT_AUTHOR_EMAIL)):
        got = _run(["git", "config", "--get", key], repo_root, check=False, quiet=True)
        if not got.stdout.strip():
            _run(["git", "config", key, value], repo_root, quiet=True)


def _current_branch(repo_root: Path) -> str:
    proc = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_root, quiet=True)
    branch = proc.stdout.strip()
    if not branch or branch == "HEAD":
        raise PushError(
            "the checkout is in detached HEAD state; pass branch= explicitly."
        )
    return branch


def push_results(
    message: str,
    repo_root: Optional[Path] = None,
    branch: Optional[str] = None,
    paths: Optional[dict] = None,
    dirs: Iterable[str] = TRACKED_DIRS,
) -> bool:
    """Stage, commit and push generated results. Returns True if it pushed.

    ``message`` is the commit message. ``paths`` is the dict returned by
    ``bootstrap_session.bootstrap()``; it supplies repo_root and branch when
    they are not given explicitly. A clean tree is a no-op, not an error.
    """
    if not message or not message.strip():
        raise PushError("a commit message is required.")

    if repo_root is None:
        repo_root = Path(paths["repo_root"]) if paths else Path(__file__).resolve().parent.parent
    repo_root = Path(repo_root)
    if not (repo_root / ".git").is_dir():
        raise PushError(f"not a git checkout: {repo_root}")

    branch = branch or (paths.get("branch") if paths else None) or _current_branch(repo_root)

    existing = [d for d in dirs if (repo_root / d).is_dir()]
    if not existing:
        raise PushError(
            f"none of {', '.join(dirs)} exist under {repo_root}; nothing can be staged."
        )

    _ensure_identity(repo_root)
    _run(["git", "add", "--", *existing], repo_root, quiet=True)

    staged = _run(["git", "diff", "--cached", "--name-only"], repo_root, quiet=True)
    changed = [ln for ln in staged.stdout.splitlines() if ln.strip()]
    if not changed:
        print("push_results: nothing changed in "
              + ", ".join(existing) + " - nothing to commit or push.")
        return False

    print("push_results: committing " + str(len(changed)) + " file(s):")
    for name in changed:
        print(f"    {name}")

    _run(["git", "commit", "-m", message], repo_root, quiet=True)

    mod = _bootstrap_module(repo_root)
    token = mod.get_github_token(mod.detect_platform())
    url = _run(["git", "remote", "get-url", "origin"], repo_root, quiet=True).stdout.strip()
    if not url:
        raise PushError("origin remote is not configured.")
    auth = mod._auth_url(url, token)

    _run(["git", "remote", "set-url", "origin", auth], repo_root, quiet=True)
    try:
        _run(["git", "push", "origin", f"HEAD:{branch}"], repo_root, quiet=True)
    finally:
        _run(["git", "remote", "set-url", "origin", url], repo_root, quiet=True)

    head = _run(["git", "rev-parse", "--short", "HEAD"], repo_root, quiet=True).stdout.strip()
    print(f"push_results: pushed {head} to origin/{branch}")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Commit and push generated results")
    ap.add_argument("-m", "--message", required=True)
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--branch", default=None)
    args = ap.parse_args()
    ok = push_results(args.message, repo_root=args.repo_root, branch=args.branch)
    sys.exit(0)
