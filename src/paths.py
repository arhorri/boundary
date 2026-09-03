"""Path resolution for remote (Colab / Kaggle) and local sessions.

Single source of truth for every location the pipeline touches. Library code
must never contain a host-specific absolute path: those live only in
``configs/default.yaml`` under ``session:``, and are read from here.

This module only *computes* locations. It does not mount Drive, clone the
repo, create directories or install anything -- that is the job of
``scripts/bootstrap_session.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

#: Folder names that mean the same dataset across differently-packaged copies.
DATASET_ALIASES = {
    "uhcs1": ("uhcs1", "uhcs"),
    "uhcs": ("uhcs", "uhcs1"),
}


class PathConfigError(RuntimeError):
    """Raised when configs/default.yaml is missing a value this host needs."""


def detect_platform() -> str:
    """Return ``"colab"``, ``"kaggle"`` or ``"local"``."""
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:
        return "colab"
    try:
        import google.colab  # noqa: F401

        return "colab"
    except Exception:
        pass
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or "KAGGLE_URL_BASE" in os.environ:
        return "kaggle"
    if Path(os.sep, "kaggle").is_dir():
        return "kaggle"
    return "local"


def load_config(config_path: Optional[os.PathLike] = None) -> dict:
    """Load a YAML config. A missing file is an error, not an empty dict."""
    import yaml

    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.is_file():
        raise PathConfigError(
            f"config not found: {path}. Every path in this project is resolved "
            "from configs/default.yaml; it must exist."
        )
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise PathConfigError(f"config {path} did not parse to a mapping.")
    return cfg


def _get(d, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _need(cfg: dict, *keys) -> str:
    value = _get(cfg, "session", *keys)
    if value in (None, ""):
        dotted = ".".join(keys)
        raise PathConfigError(
            f"configs/default.yaml: session.{dotted} is not set, but this host "
            "needs it. Fill it in, commit, and re-run the bootstrap cell."
        )
    return value


def expected_datasets(config: Optional[dict] = None) -> list:
    """Dataset folder names that must exist under the data root."""
    cfg = config if config is not None else load_config()
    names = _get(cfg, "session", "expected_datasets")
    if not names:
        raise PathConfigError(
            "configs/default.yaml: session.expected_datasets is empty. The "
            "bootstrap cannot verify the data root without it."
        )
    return list(names)


def dataset_candidates(name: str) -> tuple:
    """All accepted folder names for one dataset (packaging differs by host)."""
    return DATASET_ALIASES.get(name.lower(), (name,))


def match_datasets(root: Path, names: Iterable[str]) -> tuple:
    """Split ``names`` into (found_as, missing) against the subdirs of ``root``.

    ``found_as`` maps the expected name to the folder name actually present.
    Matching is case-insensitive and alias-aware; nothing is created or read.
    """
    root = Path(root)
    present = {}
    if root.is_dir():
        for child in root.iterdir():
            if child.is_dir():
                present[child.name.lower()] = child.name
    found_as, missing = {}, []
    for name in names:
        hit = None
        for candidate in dataset_candidates(name):
            if candidate.lower() in present:
                hit = present[candidate.lower()]
                break
        if hit is None:
            missing.append(name)
        else:
            found_as[name] = hit
    return found_as, missing


def resolve_paths(
    config: Optional[dict] = None,
    config_path: Optional[os.PathLike] = None,
    platform: Optional[str] = None,
) -> dict:
    """Resolve every path the pipeline needs, for the current platform.

    Returns a dict with keys ``platform``, ``repo_root``, ``config_path``,
    ``data_root``, ``persistent_dir``, ``outputs_dir``, ``checkpoints_dir``,
    ``logs_dir``, ``reports_dir``, ``expected_datasets``. All locations are
    :class:`pathlib.Path`. Nothing is created and nothing is checked for
    existence here -- ``scripts/bootstrap_session.py`` does the asserting.

    Raises :class:`PathConfigError` if a value this host needs is unset.
    """
    if config is None:
        config = load_config(config_path)
    platform = platform or detect_platform()

    if platform == "colab":
        data_root = Path(_need(config, "colab", "data_root"))
        persistent_dir = Path(_need(config, "colab", "persistent_dir"))
    elif platform == "kaggle":
        input_root = Path(_need(config, "kaggle", "input_root"))
        data_root = input_root / _need(config, "kaggle", "dataset_slug")
        persistent_dir = Path(_need(config, "kaggle", "working_dir"))
    else:
        data_root = REPO_ROOT / "data"
        persistent_dir = REPO_ROOT

    return {
        "platform": platform,
        "repo_root": REPO_ROOT,
        "config_path": Path(config_path) if config_path else DEFAULT_CONFIG,
        "data_root": data_root,
        "persistent_dir": persistent_dir,
        "outputs_dir": persistent_dir / "outputs",
        "checkpoints_dir": persistent_dir / "checkpoints",
        "logs_dir": persistent_dir / "logs",
        "reports_dir": REPO_ROOT / "reports",
        "expected_datasets": expected_datasets(config),
    }
