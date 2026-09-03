"""Path resolution for local / Colab / Kaggle sessions.

Library code must never hardcode ``/content`` or ``/kaggle``. Call
:func:`resolve_paths` instead. It only *computes* locations from the current
platform and ``configs/default.yaml``; it does not mount Drive, clone the
repo, or install anything (that is ``scripts/bootstrap_session.py``'s job).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def detect_platform() -> str:
    """Return ``"colab"``, ``"kaggle"`` or ``"local"``."""
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


def load_config(config_path: Optional[os.PathLike] = None) -> dict:
    """Load a YAML config; missing file yields an empty dict."""
    import yaml

    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.is_file():
        return {}
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def _get(d: dict, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def resolve_paths(
    config: Optional[dict] = None,
    config_path: Optional[os.PathLike] = None,
    platform: Optional[str] = None,
) -> dict:
    """Resolve every path library code might need.

    Returns a dict with keys: ``platform``, ``repo_root``, ``data_root``,
    ``persistent_dir``, ``outputs_dir``, ``checkpoints_dir``, ``logs_dir``.
    All values are :class:`pathlib.Path`. Nothing is created here.
    """
    if config is None:
        config = load_config(config_path)
    platform = platform or detect_platform()
    session = config.get("session", {}) if isinstance(config, dict) else {}

    if platform == "colab":
        data_root = _get(session, "colab", "data_root")
        persistent_dir = _get(session, "colab", "persistent_dir")
        data_root = Path(data_root) if data_root else None
        persistent_dir = Path(persistent_dir) if persistent_dir else None
    elif platform == "kaggle":
        slug = _get(session, "kaggle", "dataset_slug")
        data_root = Path("/kaggle/input") / slug if slug else Path("/kaggle/input")
        persistent_dir = Path("/kaggle/working")
    else:  # local
        data_root = REPO_ROOT / "data"
        persistent_dir = REPO_ROOT

    resolved = {
        "platform": platform,
        "repo_root": REPO_ROOT,
        "data_root": data_root,
        "persistent_dir": persistent_dir,
    }
    if persistent_dir is not None:
        resolved["outputs_dir"] = persistent_dir / "outputs"
        resolved["checkpoints_dir"] = persistent_dir / "checkpoints"
        resolved["logs_dir"] = persistent_dir / "logs"
    else:
        resolved["outputs_dir"] = None
        resolved["checkpoints_dir"] = None
        resolved["logs_dir"] = None
    return resolved
