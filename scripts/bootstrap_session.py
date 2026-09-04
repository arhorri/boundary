#!/usr/bin/env python3
"""Idempotent remote-session bootstrap for Colab / Kaggle.

Safe to run any number of times in one session and after any disconnect:
every step inspects the current state before acting.

Standard use, from cell 1 of any notebook (the repo is not cloned yet, so
this file is fetched from GitHub first)::

    import bootstrap_session
    PATHS = bootstrap_session.bootstrap(
        repo_url="https://github.com/<owner>/<repo>.git", branch="main")

From a shell inside an existing checkout::

    python scripts/bootstrap_session.py

What it does, in order:
    1. detect the platform (colab | kaggle | local)
    2. read the GitHub PAT from the host secret store -- never a literal
    3. clone the repo, or fetch + hard reset an existing clone
    4. pip install requirements-notebook.txt, skipping what already imports
    5. mount Google Drive (colab only)
    6. resolve and VERIFY DATA_ROOT against session.expected_datasets
    7. resolve PERSISTENT_DIR and symlink outputs/ + checkpoints/ into it
    8. print the session summary and return every resolved path

Host-specific absolute paths are never written here. They live in
configs/default.yaml under ``session:`` and are read through src/paths.py.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

REQUIREMENTS_FILE = "requirements-notebook.txt"

#: pip distribution name -> module name, where they differ.
IMPORT_NAMES = {
    "segmentation-models-pytorch": "segmentation_models_pytorch",
    "opencv-python": "cv2",
    "opencv-python-headless": "cv2",
    "pyyaml": "yaml",
    "scikit-image": "skimage",
    "scikit-learn": "sklearn",
    "pillow": "PIL",
}

SECRET_SETUP = {
    "colab": (
        "GH_TOKEN secret is missing.\n"
        "  1. Click the key icon (Secrets) in the Colab left sidebar.\n"
        "  2. Add a secret named exactly  GH_TOKEN  whose value is a GitHub\n"
        "     fine-grained PAT with Contents: read and write on this repo.\n"
        "  3. Turn 'Notebook access' ON for this notebook.\n"
        "  4. Re-run this cell."
    ),
    "kaggle": (
        "GH_TOKEN secret is missing.\n"
        "  1. In the Kaggle notebook editor open  Add-ons -> Secrets.\n"
        "  2. Add a secret named exactly  GH_TOKEN  whose value is a GitHub\n"
        "     fine-grained PAT with Contents: read and write on this repo.\n"
        "  3. Tick the checkbox that attaches it to this notebook.\n"
        "  4. Re-run this cell."
    ),
    "local": (
        "GH_TOKEN is not set. Export it before running:\n"
        "    export GH_TOKEN=<github personal access token>"
    ),
}


class BootstrapError(RuntimeError):
    """Anything that makes this session unusable. Always raised loudly."""


# --------------------------------------------------------------------------
# platform
# --------------------------------------------------------------------------
def detect_platform() -> str:
    """Return ``"colab"``, ``"kaggle"`` or ``"local"``.

    Kaggle is tested FIRST and by evidence that only Kaggle has. Its image can
    carry an importable ``google.colab`` shim, so "does google.colab import?"
    answers yes on both hosts and cannot be the deciding test. Getting this
    backwards sends the session to the wrong secret store, the wrong data root
    and the wrong branch.
    """
    for var in ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE",
                "KAGGLE_DATA_PROXY_TOKEN", "KAGGLE_CONTAINER_NAME"):
        if os.environ.get(var):
            return "kaggle"
    for marker in ("input", "working"):
        if Path(os.sep, "kaggle", marker).is_dir():
            return "kaggle"

    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:
        return "colab"
    if Path(os.sep, "content").is_dir():
        try:
            import google.colab  # noqa: F401

            return "colab"
        except Exception:
            pass
    return "local"


def platform_evidence() -> str:
    """Why detect_platform() answered as it did -- printed, never guessed at."""
    hits = [f"{v}={os.environ[v]!r}" for v in (
        "KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE", "COLAB_RELEASE_TAG",
        "COLAB_GPU") if os.environ.get(v)]
    for path in (Path(os.sep, "kaggle", "input"), Path(os.sep, "kaggle", "working"),
                 Path(os.sep, "content")):
        if path.is_dir():
            hits.append(f"{path} exists")
    return ", ".join(hits) or "no host markers found"


def _run(cmd, cwd: Optional[Path] = None, check: bool = True, quiet: bool = False):
    cmd = [str(c) for c in cmd]
    if not quiet:
        print("  $ " + " ".join(cmd))
    proc = subprocess.run(
        cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True
    )
    if check and proc.returncode != 0:
        raise BootstrapError(
            "command failed: " + " ".join(cmd) + "\n"
            + (proc.stderr or proc.stdout or "").strip()
        )
    return proc


# --------------------------------------------------------------------------
# secrets
# --------------------------------------------------------------------------
def _colab_secret() -> Optional[str]:
    from google.colab import userdata

    return userdata.get("GH_TOKEN")


def _kaggle_secret() -> Optional[str]:
    from kaggle_secrets import UserSecretsClient

    return UserSecretsClient().get_secret("GH_TOKEN")


SECRET_READERS = {"colab": _colab_secret, "kaggle": _kaggle_secret}


def get_github_token(platform: str) -> str:
    """Read GH_TOKEN from the host secret store. Never accepts a literal.

    The detected platform's own store is tried first, then the others, then the
    environment. A secret that is present in a store this host was not thought
    to have is a detection bug, not a missing secret -- so it is used, and the
    disagreement is printed rather than turned into a dead end.
    """
    tried = []
    order = [platform] + [name for name in SECRET_READERS if name != platform]
    for name in order:
        reader = SECRET_READERS.get(name)
        if reader is None:
            continue
        try:
            token = reader()
        except Exception as exc:
            tried.append(f"{name} store: {type(exc).__name__}")
            continue
        if token:
            if name != platform:
                print(f"  ! GH_TOKEN came from the {name} secret store, but the "
                      f"platform was detected as {platform}.")
                print(f"  ! evidence: {platform_evidence()}")
            return token.strip()
        tried.append(f"{name} store: no secret named GH_TOKEN")

    token = os.environ.get("GH_TOKEN")
    if token:
        return token.strip()
    tried.append("environment: GH_TOKEN not set")

    print(SECRET_SETUP.get(platform, SECRET_SETUP["local"]))
    print("  tried: " + "; ".join(tried))
    print(f"  detected platform: {platform} ({platform_evidence()})")
    raise BootstrapError(f"GH_TOKEN not available on {platform}.")


# --------------------------------------------------------------------------
# repo
# --------------------------------------------------------------------------
def _auth_url(repo_url: str, token: str) -> str:
    if not repo_url.startswith("https://"):
        raise BootstrapError(
            f"repo_url must be an https clone URL, got: {repo_url}. "
            "ssh remotes cannot authenticate from a notebook."
        )
    return "https://x-access-token:" + token + "@" + repo_url[len("https://"):]


def rescue_generated_results(dest: Path) -> list:
    """Copy uncommitted reports/ and configs/ out of harm's way before a reset.

    ``git reset --hard`` reverts tracked files. A notebook that has just
    regenerated reports/ and configs/ but not yet pushed them loses that work
    the moment cell 1 is re-run -- silently, because the reset succeeds and the
    later push then finds "nothing changed". Anything dirty is copied beside
    the clone first and the paths are printed, so the results still exist and
    the situation is visible instead of mysterious.
    """
    import shutil

    proc = _run(["git", "status", "--porcelain", "--", "reports", "configs"],
                cwd=dest, check=False, quiet=True)
    dirty = sorted({line[3:].strip().strip('"') for line in proc.stdout.splitlines()
                    if line.strip()})
    if not dirty:
        return []

    backup = dest.parent / ("rescued-results-" + time.strftime("%Y%m%d-%H%M%S"))
    saved = []
    for rel in dirty:
        source = dest / rel
        if not source.is_file():
            continue
        target = backup / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        saved.append(rel)
    if saved:
        print(f"  ! {len(saved)} uncommitted result file(s) would be destroyed by "
              f"the sync; copied to {backup}")
        for rel in saved[:10]:
            print(f"      {rel}")
        if len(saved) > 10:
            print(f"      ... and {len(saved) - 10} more")
        print("  ! if these were the results of a run you had not pushed yet, "
              "re-run the notebook's steps rather than only its push cell.")
    return saved


def _fetch_branch(dest: Path, branch: str) -> None:
    """Fetch one branch into refs/remotes/origin/<branch>, whatever the clone did.

    ``git clone --branch main --depth 1`` writes a remote refspec that maps ONLY
    main, so a later ``git fetch origin kaggle`` updates FETCH_HEAD and creates
    no ``origin/kaggle`` at all -- and every command that names it then fails
    with "not a commit". Naming the destination ref explicitly makes the fetch
    independent of how the clone was set up.
    """
    proc = _run(["git", "fetch", "--depth", "1", "origin",
                 f"+refs/heads/{branch}:refs/remotes/origin/{branch}"],
                cwd=dest, check=False)
    if proc.returncode != 0:
        raise BootstrapError(
            f"cannot fetch branch '{branch}' from origin:\n"
            + (proc.stderr or proc.stdout or "").strip()
            + f"\nDoes origin have a branch named '{branch}'? It is named in "
              "configs/default.yaml under session.<platform>.branch.")


def current_branch(dest: Path) -> Optional[str]:
    """The branch this checkout is actually on, or None if there is no checkout."""
    if not (dest / ".git").is_dir():
        return None
    proc = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=dest,
                check=False, quiet=True)
    name = proc.stdout.strip()
    return name if name and name != "HEAD" else None


def clone_or_update(repo_url: str, branch: str, dest: Path, token: str) -> Path:
    """Clone the repo, or fetch + hard-reset an existing clone. Idempotent.

    The token-bearing remote URL is written only for the duration of the
    network call and scrubbed back out of .git/config afterwards. Uncommitted
    results are rescued before the reset, never silently discarded.
    """
    auth = _auth_url(repo_url, token)
    if (dest / ".git").is_dir():
        # Sync the branch this checkout is ON, never a hardcoded one. A session
        # that silently resets a feature branch to main is worse than no branch
        # at all: the work is gone and nothing says so.
        checked_out = current_branch(dest)
        if checked_out and checked_out != branch:
            print(f"  checkout is on '{checked_out}', not '{branch}': syncing "
                  f"'{checked_out}'")
            branch = checked_out
        rescue_generated_results(dest)
        _run(["git", "remote", "set-url", "origin", auth], cwd=dest, quiet=True)
        try:
            _fetch_branch(dest, branch)
            _run(["git", "reset", "--hard", f"origin/{branch}"], cwd=dest)
            _run(["git", "checkout", "-B", branch, f"origin/{branch}"], cwd=dest)
        finally:
            _run(["git", "remote", "set-url", "origin", repo_url], cwd=dest, quiet=True)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  $ git clone --branch {branch} --depth 1 <repo> {dest}")
        _run(["git", "clone", "--branch", branch, "--depth", "1", auth, str(dest)],
             quiet=True)
        # A single-branch clone can only ever see the branch it cloned; this
        # host may need another one (its config overlay lives there).
        _run(["git", "config", "remote.origin.fetch",
              "+refs/heads/*:refs/remotes/origin/*"], cwd=dest, quiet=True)
        _run(["git", "remote", "set-url", "origin", repo_url], cwd=dest, quiet=True)
    return dest


def switch_branch(repo_url: str, branch: str, dest: Path, token: str) -> str:
    """Move an existing checkout onto ``branch``. Idempotent."""
    if current_branch(dest) == branch:
        return branch
    auth = _auth_url(repo_url, token)
    _run(["git", "remote", "set-url", "origin", auth], cwd=dest, quiet=True)
    try:
        _fetch_branch(dest, branch)
        _run(["git", "checkout", "-B", branch, f"origin/{branch}"], cwd=dest)
        _run(["git", "reset", "--hard", f"origin/{branch}"], cwd=dest)
    finally:
        _run(["git", "remote", "set-url", "origin", repo_url], cwd=dest, quiet=True)
    return branch


def repo_commit(dest: Path) -> str:
    proc = _run(["git", "rev-parse", "--short", "HEAD"], cwd=dest, check=False, quiet=True)
    return proc.stdout.strip() or "unknown"


def _repo_name(repo_url: str) -> str:
    name = repo_url.rstrip("/").split("/")[-1]
    return name[:-4] if name.endswith(".git") else name


# --------------------------------------------------------------------------
# dependencies
# --------------------------------------------------------------------------
def parse_requirements(path: Path) -> list:
    """Return the distribution names listed in a requirements file."""
    if not path.is_file():
        raise BootstrapError(
            f"{path} not found. The repo checkout is incomplete; delete the "
            "clone directory and re-run this cell."
        )
    names = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[<>=!~;\[ ]", line, 1)[0].strip()
        if name:
            names.append(name)
    return names


def _importable(dist: str) -> bool:
    module = IMPORT_NAMES.get(dist.lower(), dist.replace("-", "_"))
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


def install_requirements(repo_root: Path) -> dict:
    """pip install only what is not already importable. Idempotent."""
    req = repo_root / REQUIREMENTS_FILE
    wanted = parse_requirements(req)
    missing = [d for d in wanted if not _importable(d)]
    skipped = [d for d in wanted if d not in missing]

    if missing:
        _run([sys.executable, "-m", "pip", "install", "-q", *missing])
        importlib.invalidate_caches()
        still_missing = [d for d in missing if not _importable(d)]
        if still_missing:
            raise BootstrapError(
                "pip reported success but these are still not importable: "
                + ", ".join(still_missing)
            )
    print(f"  installed: {', '.join(missing) if missing else '(nothing)'}")
    print(f"  skipped (already importable): {', '.join(skipped) if skipped else '(nothing)'}")
    return {"installed": missing, "skipped": skipped}


# --------------------------------------------------------------------------
# drive
# --------------------------------------------------------------------------
def mount_drive(mount_point: str) -> None:
    """Mount Google Drive. A no-op if it is already mounted."""
    from google.colab import drive

    mount = Path(mount_point)
    if mount.is_dir() and any(mount.iterdir()):
        print(f"  Drive already mounted at {mount_point}")
        return
    drive.mount(mount_point, force_remount=False)


# --------------------------------------------------------------------------
# data root
# --------------------------------------------------------------------------
def _listing(path: Path, limit: int = 30) -> str:
    if not path.exists():
        return "(does not exist)"
    if not path.is_dir():
        return "(not a directory)"
    entries = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())
    if not entries:
        return "(empty)"
    shown = entries[:limit]
    more = "" if len(entries) <= limit else f" ... (+{len(entries) - limit} more)"
    return ", ".join(shown) + more


def verify_data_root(data_root: Path, expected: list, paths_mod) -> Path:
    """Return the directory that actually holds the dataset folders.

    Accepts either ``data_root`` itself or a single subdirectory of it (host
    dataset archives often add one wrapping folder). Raises with the full
    listing of what was found when the expected folders are not there.
    """
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise BootstrapError(
            f"DATA_ROOT does not exist: {data_root}\n"
            f"  parent {data_root.parent} contains: {_listing(data_root.parent)}\n"
            "Fix session.*.data_root / dataset_slug in configs/default.yaml."
        )

    candidates = [data_root] + [p for p in sorted(data_root.iterdir()) if p.is_dir()]
    best, best_found = None, {}
    for cand in candidates:
        found_as, missing = paths_mod.match_datasets(cand, expected)
        if not missing:
            return cand
        if len(found_as) > len(best_found):
            best, best_found = cand, found_as

    _, missing = paths_mod.match_datasets(best or data_root, expected)
    raise BootstrapError(
        f"DATA_ROOT is missing expected dataset folders: {', '.join(missing)}\n"
        f"  looked in : {data_root}\n"
        f"  found     : {_listing(data_root)}\n"
        f"  expected  : {', '.join(expected)}\n"
        "Either the upload is incomplete or session.expected_datasets in "
        "configs/default.yaml does not match how the data is packaged."
    )


def verify_gt_root(gt_root: Path, expected: list, paths_mod, platform: str) -> Path:
    """Locate and check the boundary maps step 2 produced.

    On Kaggle this is a separate READ-ONLY input dataset: if it is missing or
    incomplete nothing in the session can fix it, so the run stops here with
    the listing rather than failing later inside a data loader.

    On Colab it lives in Drive and step 2 is what creates it, so notebooks 00
    to 02 legitimately run before it exists. There the directory is created and
    a warning printed -- refusing to boot would make step 2 impossible to run.
    """
    gt_root = Path(gt_root)
    if not gt_root.is_dir():
        if platform == "kaggle":
            raise BootstrapError(
                f"GT_BOUNDARIES_ROOT does not exist: {gt_root}\n"
                f"  parent {gt_root.parent} contains: {_listing(gt_root.parent)}\n"
                "On Kaggle the boundary maps are a separate read-only input "
                "dataset. Attach it, and check session.kaggle.gt_dataset_slug "
                "in configs/kaggle.yaml."
            )
        gt_root.mkdir(parents=True, exist_ok=True)
        print(f"  ! {gt_root} is empty: step 2 has not run on this host yet")
        return gt_root

    found_as, missing = paths_mod.match_datasets(gt_root, expected)
    root = gt_root
    if missing:
        # Some upload paths add one wrapping folder; accept exactly one level.
        for cand in sorted(p for p in gt_root.iterdir() if p.is_dir()):
            _, cand_missing = paths_mod.match_datasets(cand, expected)
            if not cand_missing:
                return cand
    if missing:
        message = (
            f"GT_BOUNDARIES_ROOT is missing boundary maps for: "
            f"{', '.join(missing)}\n"
            f"  looked in : {gt_root}\n"
            f"  found     : {_listing(gt_root)}\n"
            f"  expected  : {', '.join(expected)}"
        )
        if platform == "kaggle":
            raise BootstrapError(
                message + "\nThe uploaded dataset is incomplete; nothing in a "
                "Kaggle session can produce these.")
        print(f"  ! {message}")
        print("  ! run notebooks/02_boundary_gt.ipynb to produce the missing ones")
    return root


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------
def link_into_persistent(link: Path, target: Path) -> None:
    """Idempotently point ``link`` at ``target`` so nothing lives only in RAM."""
    link, target = Path(link), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if Path(os.readlink(link)).resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        if link.is_dir() and not any(link.iterdir()):
            link.rmdir()
        else:
            raise BootstrapError(
                f"{link} exists, is not a symlink, and is not empty. Refusing "
                "to touch it. Move it aside and re-run."
            )
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)
    print(f"  {link} -> {target}")


# --------------------------------------------------------------------------
# gpu
# --------------------------------------------------------------------------
def gpu_summary() -> tuple:
    """Return (gpu name, vram string, torch version). Never raises."""
    try:
        import torch
    except Exception:
        return "unknown", "-", "not importable"
    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            return name, f"{vram:.1f} GB", torch.__version__
        return "none (CPU only)", "-", torch.__version__
    except Exception:
        return "unknown", "-", torch.__version__


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def _find_repo_root(repo_url: str, clone_name: Optional[str]) -> Path:
    """Existing checkout containing this file, else a clone dir under CWD."""
    here = Path(__file__).resolve()
    for parent in (here.parent.parent, here.parent):
        if (parent / ".git").is_dir():
            return parent
    return Path.cwd() / (clone_name or _repo_name(repo_url))


def bootstrap(
    repo_url: Optional[str] = None,
    branch: Optional[str] = None,
    config_path: Optional[str] = None,
) -> dict:
    """Prepare the session and return every resolved path. Idempotent."""
    platform = detect_platform()
    print(f"[bootstrap] platform: {platform}  ({platform_evidence()})")

    token = get_github_token(platform)

    if not repo_url or not branch:
        raise BootstrapError(
            "repo_url and branch are required: the repo must be located before "
            "its config can be read. Pass them from the notebook bootstrap cell."
        )

    repo_root = _find_repo_root(repo_url, None)
    if platform == "local":
        print(f"  local platform: using checkout at {repo_root} as-is (no reset)")
        if not (repo_root / ".git").is_dir():
            raise BootstrapError(f"no git checkout at {repo_root}")
    else:
        clone_or_update(repo_url, branch, repo_root, token)
        branch = current_branch(repo_root) or branch

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    install_requirements(repo_root)

    # src/paths.py owns every host path; it is only importable after the clone.
    importlib.invalidate_caches()
    paths_mod = importlib.import_module("src.paths")
    cfg = paths_mod.load_config(config_path, platform=platform)

    # A host may need a branch of its own -- not different logic, just the
    # configuration overlay that names its roots. The notebooks are identical
    # on every branch, so the switch is made here, once, from config.
    wanted = (cfg.get("session", {}).get(platform) or {}).get("branch")
    if platform != "local" and wanted and wanted != branch:
        print(f"  {platform} prefers branch '{wanted}' (session.{platform}.branch); "
              f"switching from '{branch}'")
        branch = switch_branch(repo_url, wanted, repo_root, token)
        importlib.invalidate_caches()
        paths_mod = importlib.reload(paths_mod)
        cfg = paths_mod.load_config(config_path, platform=platform)
    if cfg.get("_overlay"):
        print(f"  config overlay applied: {cfg['_overlay']}")

    if platform == "colab":
        mount_drive(paths_mod._need(cfg, "colab", "drive_mount"))

    resolved = paths_mod.resolve_paths(config=cfg, config_path=config_path,
                                       platform=platform)
    data_root = verify_data_root(
        resolved["data_root"], resolved["expected_datasets"], paths_mod
    )
    gt_root = verify_gt_root(
        resolved["gt_boundaries_root"], resolved["expected_datasets"],
        paths_mod, platform
    )
    persistent_dir = Path(resolved["persistent_dir"])
    persistent_dir.mkdir(parents=True, exist_ok=True)

    outputs_dir = persistent_dir / "outputs"
    checkpoints_dir = persistent_dir / "checkpoints"
    logs_dir = persistent_dir / "logs"
    for d in (outputs_dir, checkpoints_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Repo-relative names point at the persistent locations, so a killed
    # session loses nothing and code can always say repo_root/"outputs".
    if persistent_dir.resolve() != repo_root.resolve():
        link_into_persistent(repo_root / "outputs", outputs_dir)
        link_into_persistent(repo_root / "checkpoints", checkpoints_dir)
    if data_root.resolve() != (repo_root / "data").resolve():
        link_into_persistent(repo_root / "data", data_root)

    gpu_name, vram, torch_version = gpu_summary()
    commit = repo_commit(repo_root)

    print("\n==================== session summary ====================")
    print(f"  platform        : {platform}")
    print(f"  GPU             : {gpu_name}")
    print(f"  VRAM            : {vram}")
    print(f"  torch           : {torch_version}")
    print(f"  repo commit     : {commit} ({branch})")
    print(f"  repo root       : {repo_root}")
    print(f"  DATA_ROOT       : {data_root}")
    print(f"  GT_BOUNDARIES   : {gt_root}")
    print(f"  PERSISTENT_DIR  : {persistent_dir}")
    print("=========================================================\n")

    if platform == "kaggle":
        print("  " + "!" * 70)
        print("  ! /kaggle/working DOES NOT SURVIVE THIS SESSION.")
        print("  ! Everything written there -- checkpoints, logs, outputs -- is")
        print("  ! discarded when the kernel stops, UNLESS you use Save Version")
        print("  ! (Save & Run All, or Quick Save) to persist the output.")
        print("  ! Push reports and configs to GitHub as you go; treat a")
        print("  ! checkpoint left only in /kaggle/working as already lost.")
        print("  " + "!" * 70 + "\n")

    return {
        "platform": platform,
        "repo_root": repo_root,
        "repo_url": repo_url,
        "branch": branch,
        "commit": commit,
        "config_path": resolved["config_path"],
        "data_root": data_root,
        "gt_boundaries_root": gt_root,
        "persistent_dir": persistent_dir,
        "outputs_dir": outputs_dir,
        "checkpoints_dir": checkpoints_dir,
        "logs_dir": logs_dir,
        "reports_dir": repo_root / "reports",
        "expected_datasets": resolved["expected_datasets"],
        "gpu": gpu_name,
        "vram": vram,
        "torch": torch_version,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Idempotent Colab/Kaggle bootstrap")
    ap.add_argument("--repo-url", required=True, help="https clone URL")
    ap.add_argument("--branch", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    bootstrap(repo_url=args.repo_url, branch=args.branch, config_path=args.config)
