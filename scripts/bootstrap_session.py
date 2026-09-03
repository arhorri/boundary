#!/usr/bin/env python3
"""Idempotent remote-session bootstrap for Colab / Kaggle / local.

Safe to run any number of times in one session and after any disconnect:
every step checks current state before acting.

From a notebook (repo not yet cloned)::

    import urllib.request, pathlib
    url = "https://raw.githubusercontent.com/<owner>/<repo>/<branch>/scripts/bootstrap_session.py"
    urllib.request.urlretrieve(url, "bootstrap_session.py")
    import bootstrap_session
    paths = bootstrap_session.bootstrap(
        repo_url="https://github.com/<owner>/<repo>.git", branch="<branch>")

From a shell inside the repo::

    python scripts/bootstrap_session.py --repo-url https://github.com/<owner>/<repo>.git --branch main

The GitHub PAT is ALWAYS read from the host secret store, never a literal:
    colab   google.colab.userdata.get('GH_TOKEN')
    kaggle  kaggle_secrets.UserSecretsClient().get_secret('GH_TOKEN')
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Keep in sync with environment.yml / requirements-colab.txt.
TORCH_MIN = (2, 1, 2)
TORCHVISION_MIN = (0, 16, 2)

REQUIRED_IMPORTS = [
    "numpy",
    "scipy",
    "skimage",
    "yaml",
    "tqdm",
    "matplotlib",
    "pandas",
    "tensorboard",
    "cv2",
    "albumentations",
    "segmentation_models_pytorch",
]

SECRET_SETUP = {
    "colab": (
        "GH_TOKEN secret is missing.\n"
        "  1. Click the key icon (Secrets) in the Colab left sidebar.\n"
        "  2. Add a secret named exactly  GH_TOKEN  whose value is a GitHub PAT\n"
        "     with repo contents read access.\n"
        "  3. Turn 'Notebook access' ON for this notebook.\n"
        "  4. Re-run this cell."
    ),
    "kaggle": (
        "GH_TOKEN secret is missing.\n"
        "  1. In the Kaggle notebook editor open  Add-ons -> Secrets.\n"
        "  2. Add a secret named exactly  GH_TOKEN  whose value is a GitHub PAT\n"
        "     with repo contents read access.\n"
        "  3. Tick the checkbox to attach it to this notebook.\n"
        "  4. Re-run this cell."
    ),
}


# --------------------------------------------------------------------------
# platform
# --------------------------------------------------------------------------
def detect_platform() -> str:
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:
        return "colab"
    try:
        import google.colab  # noqa: F401

        return "colab"
    except Exception:
        pass
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.isdir("/kaggle"):
        return "kaggle"
    return "local"


def _run(cmd, cwd: Optional[Path] = None, check: bool = True, quiet: bool = False):
    if not quiet:
        printable = " ".join(str(c) for c in cmd)
        print(f"  $ {printable}")
    return subprocess.run(
        [str(c) for c in cmd], cwd=str(cwd) if cwd else None, check=check
    )


# --------------------------------------------------------------------------
# torch version gate
# --------------------------------------------------------------------------
def _version_tuple(v: str):
    out = []
    for part in v.split("+")[0].split("."):
        try:
            out.append(int(part))
        except ValueError:
            break
    return tuple(out)


def assert_host_torch() -> None:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - host is expected to ship it
        raise SystemExit(
            "torch is not importable on this host. Colab and Kaggle ship it "
            "preinstalled; do not run this off those hosts without torch."
        ) from exc
    tv = _version_tuple(torch.__version__)
    if tv < TORCH_MIN:
        raise SystemExit(
            f"host torch {torch.__version__} is older than the required "
            f"{'.'.join(map(str, TORCH_MIN))}. Update the host runtime."
        )
    try:
        import torchvision

        vv = _version_tuple(torchvision.__version__)
        if vv < TORCHVISION_MIN:
            raise SystemExit(
                f"host torchvision {torchvision.__version__} is older than the "
                f"required {'.'.join(map(str, TORCHVISION_MIN))}."
            )
    except ImportError:
        raise SystemExit("torchvision is not importable on this host.")


# --------------------------------------------------------------------------
# secrets
# --------------------------------------------------------------------------
def get_github_token(platform: str) -> str:
    token = None
    if platform == "colab":
        try:
            from google.colab import userdata

            token = userdata.get("GH_TOKEN")
        except Exception:
            token = None
    elif platform == "kaggle":
        try:
            from kaggle_secrets import UserSecretsClient

            token = UserSecretsClient().get_secret("GH_TOKEN")
        except Exception:
            token = None
    else:
        token = os.environ.get("GH_TOKEN")

    if not token:
        print(SECRET_SETUP.get(platform, "Set the GH_TOKEN environment variable."))
        sys.exit(1)
    return token


# --------------------------------------------------------------------------
# drive
# --------------------------------------------------------------------------
def mount_drive(mount_point: str = "/content/drive") -> None:
    from google.colab import drive

    # force_remount=False makes this a no-op if already mounted.
    drive.mount(mount_point, force_remount=False)


# --------------------------------------------------------------------------
# repo
# --------------------------------------------------------------------------
def _auth_url(repo_url: str, token: str) -> str:
    if repo_url.startswith("https://"):
        return "https://x-access-token:" + token + "@" + repo_url[len("https://"):]
    return repo_url


def clone_or_update(repo_url: str, branch: str, dest: Path, token: str) -> Path:
    auth = _auth_url(repo_url, token)
    if (dest / ".git").is_dir():
        _run(["git", "remote", "set-url", "origin", auth], cwd=dest)
        _run(["git", "fetch", "--depth", "1", "origin", branch], cwd=dest)
        _run(["git", "reset", "--hard", f"origin/{branch}"], cwd=dest)
        _run(["git", "checkout", "-B", branch, f"origin/{branch}"], cwd=dest, check=False)
        # scrub the token back out of .git/config
        _run(["git", "remote", "set-url", "origin", repo_url], cwd=dest, quiet=True)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--branch", branch, "--depth", "1", auth, str(dest)])
        _run(["git", "remote", "set-url", "origin", repo_url], cwd=dest, quiet=True)
    return dest


def repo_commit(dest: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(dest), capture_output=True, text=True
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------
# dependencies
# --------------------------------------------------------------------------
def imports_satisfied() -> bool:
    for mod in REQUIRED_IMPORTS:
        try:
            importlib.import_module(mod)
        except Exception:
            return False
    return True


def pip_install_requirements(repo_root: Path) -> None:
    if imports_satisfied():
        print("  dependencies already satisfied - skipping pip install")
        return
    req = repo_root / "requirements-colab.txt"
    _run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)])
    importlib.invalidate_caches()


# --------------------------------------------------------------------------
# symlinks
# --------------------------------------------------------------------------
def _link(target: Path, link: Path) -> None:
    """Idempotently point ``link`` at ``target``."""
    target = Path(target)
    link = Path(link)
    if link.is_symlink():
        if Path(os.readlink(link)) == target:
            return
        link.unlink()
    elif link.exists():
        if link.is_dir() and not any(link.iterdir()):
            link.rmdir()
        else:
            print(f"  ! {link} exists and is not empty - leaving it untouched")
            return
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)
    print(f"  {link} -> {target}")


# --------------------------------------------------------------------------
# config-driven path resolution
# --------------------------------------------------------------------------
def _load_config(repo_root: Path, config_path: Optional[str] = None) -> dict:
    import yaml

    path = Path(config_path) if config_path else repo_root / "configs" / "default.yaml"
    if not path.is_file():
        return {}
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def _need(value, what: str, platform: str):
    if value in (None, ""):
        raise SystemExit(
            f"configs/default.yaml: session.{what} is null but is required on "
            f"{platform}. Fill it in and re-run."
        )
    return value


def resolve_data_root(platform: str, cfg: dict, repo_root: Path) -> Path:
    session = cfg.get("session", {}) or {}
    if platform == "colab":
        return Path(_need((session.get("colab") or {}).get("data_root"), "colab.data_root", platform))
    if platform == "kaggle":
        slug = _need((session.get("kaggle") or {}).get("dataset_slug"), "kaggle.dataset_slug", platform)
        return Path("/kaggle/input") / slug
    return repo_root / "data"


def resolve_persistent_dir(platform: str, cfg: dict, repo_root: Path) -> Path:
    session = cfg.get("session", {}) or {}
    if platform == "colab":
        return Path(_need((session.get("colab") or {}).get("persistent_dir"), "colab.persistent_dir", platform))
    if platform == "kaggle":
        return Path("/kaggle/working")
    return repo_root


# --------------------------------------------------------------------------
# gpu summary
# --------------------------------------------------------------------------
def gpu_summary() -> tuple:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return name, f"{vram_gb:.1f} GB", torch.__version__
        return "none (CPU only)", "-", torch.__version__
    except Exception:
        return "unknown", "-", "unknown"


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------
def bootstrap(
    repo_url: Optional[str] = None,
    branch: Optional[str] = None,
    config_path: Optional[str] = None,
) -> dict:
    platform = detect_platform()
    print(f"[bootstrap] platform: {platform}")

    assert_host_torch()

    token = get_github_token(platform)

    if platform == "colab":
        mount_drive()

    # Where does the repo live?  If this file is already inside a checkout,
    # reuse it; otherwise clone under the host working area.
    here = Path(__file__).resolve()
    in_repo = (here.parent.parent / ".git").is_dir()
    if in_repo:
        repo_root = here.parent.parent
    else:
        base = Path("/content") if platform == "colab" else (
            Path("/kaggle/working") if platform == "kaggle" else Path.cwd()
        )
        repo_root = base / _default_clone_name(repo_url)

    # config may only be readable after the clone; try repo first, else CWD file
    pre_cfg = _load_config(repo_root, config_path) if (repo_root / "configs").is_dir() else {}
    repo_url = repo_url or (pre_cfg.get("session", {}) or {}).get("repo_url")
    branch = branch or (pre_cfg.get("session", {}) or {}).get("branch")
    if not repo_url or not branch:
        raise SystemExit(
            "repo_url and branch must be given as arguments or set in "
            "configs/default.yaml (session.repo_url / session.branch)."
        )

    clone_or_update(repo_url, branch, repo_root, token)

    cfg = _load_config(repo_root, config_path)
    pip_install_requirements(repo_root)

    data_root = resolve_data_root(platform, cfg, repo_root)
    persistent_dir = resolve_persistent_dir(platform, cfg, repo_root)
    persistent_dir.mkdir(parents=True, exist_ok=True)

    outputs_dir = persistent_dir / "outputs"
    checkpoints_dir = persistent_dir / "checkpoints"
    logs_dir = persistent_dir / "logs"
    for d in (outputs_dir, checkpoints_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Wire the repo-relative names to their real, persistent locations.
    if data_root.resolve() != (repo_root / "data").resolve():
        _link(data_root, repo_root / "data")
    if persistent_dir.resolve() != repo_root.resolve():
        _link(outputs_dir, repo_root / "outputs")
        _link(checkpoints_dir, repo_root / "checkpoints")

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    gpu_name, vram, torch_version = gpu_summary()
    commit = repo_commit(repo_root)

    print("\n==================== session summary ====================")
    print(f"  platform        : {platform}")
    print(f"  GPU             : {gpu_name}")
    print(f"  VRAM            : {vram}")
    print(f"  torch           : {torch_version}")
    print(f"  repo commit     : {commit}")
    print(f"  repo root       : {repo_root}")
    print(f"  data root       : {data_root}")
    print(f"  persistent dir  : {persistent_dir}")
    print("========================================================\n")

    return {
        "platform": platform,
        "repo_root": repo_root,
        "data_root": data_root,
        "persistent_dir": persistent_dir,
        "outputs_dir": outputs_dir,
        "checkpoints_dir": checkpoints_dir,
        "logs_dir": logs_dir,
        "commit": commit,
    }


def _default_clone_name(repo_url: Optional[str]) -> str:
    if not repo_url:
        return "repo"
    return repo_url.rstrip("/").split("/")[-1].removesuffix(".git") or "repo"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Idempotent Colab/Kaggle bootstrap")
    ap.add_argument("--repo-url", default=None)
    ap.add_argument("--branch", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    bootstrap(repo_url=args.repo_url, branch=args.branch, config_path=args.config)
