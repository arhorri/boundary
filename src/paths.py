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
    """Return ``"colab"``, ``"kaggle"`` or ``"local"``.

    Both host images carry the other's markers, so most "obvious" tests lie:

    * Colab ships an EMPTY ``/kaggle/input`` directory.
    * Kaggle sets ``COLAB_RELEASE_TAG``, and has ``/content``.
    * Kaggle's image has an importable ``google.colab`` shim.

    What actually discriminates: the ``KAGGLE_*`` environment variables and
    ``/kaggle/working``, which exist only on Kaggle. Everything else is
    checked afterwards, and only as a fallback. Getting this wrong sends the
    session to the wrong secret store, the wrong data root and the wrong
    branch -- all of which happened before this ordering was pinned down.

    Kept deliberately identical to the copy in
    scripts/bootstrap_session.py, which must run before this module
    exists on the host.
    """
    for var in ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE",
                "KAGGLE_DATA_PROXY_TOKEN", "KAGGLE_CONTAINER_NAME"):
        if os.environ.get(var):
            return "kaggle"
    if Path(os.sep, "kaggle", "working").is_dir():
        return "kaggle"

    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:
        return "colab"
    try:
        import google.colab  # noqa: F401

        return "colab"
    except Exception:
        pass
    return "local"


def platform_evidence() -> str:
    """Why detect_platform() answered as it did -- printed, never guessed at.

    Markers both hosts have are labelled as such, so a future reader does not
    repeat the mistake of treating one of them as decisive.
    """
    parts = [f"{var}={os.environ[var]!r}" for var in (
        "KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE", "KAGGLE_DATA_PROXY_TOKEN",
        "COLAB_RELEASE_TAG", "COLAB_GPU") if os.environ.get(var)]
    for path, shared in ((Path(os.sep, "kaggle", "working"), False),
                         (Path(os.sep, "kaggle", "input"), True),
                         (Path(os.sep, "content"), True)):
        if path.is_dir():
            parts.append(f"{path} exists" + (" [both hosts]" if shared else ""))
    return ", ".join(parts) or "no host markers found"


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge ``overlay`` over ``base`` without mutating either."""
    out = dict(base)
    for key, value in (overlay or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        elif value is not None:
            out[key] = value
    return out


def load_config(
    config_path: Optional[os.PathLike] = None,
    platform: Optional[str] = None,
) -> dict:
    """Load configs/default.yaml, then overlay configs/<platform>.yaml if present.

    The overlay is how a host that needs different roots gets them WITHOUT a
    second copy of the pipeline: ``configs/kaggle.yaml`` carries only the keys
    Kaggle sees differently, and everything else stays shared. A missing
    overlay is normal, not an error; a missing default.yaml is an error.
    """
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

    platform = platform or detect_platform()
    overlay_path = path.parent / f"{platform}.yaml"
    if overlay_path.is_file():
        with open(overlay_path) as fh:
            overlay = yaml.safe_load(fh)
        if overlay is not None and not isinstance(overlay, dict):
            raise PathConfigError(
                f"config overlay {overlay_path} did not parse to a mapping.")
        cfg = _deep_merge(cfg, overlay or {})
        cfg["_overlay"] = str(overlay_path)
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
    ``data_root``, ``gt_boundaries_root``, ``persistent_dir``, ``outputs_dir``,
    ``checkpoints_dir``, ``logs_dir``, ``reports_dir``, ``expected_datasets``.

    ``gt_boundaries_root`` is resolved INDEPENDENTLY of ``persistent_dir``: on
    Colab step 2 writes it into Drive beside the checkpoints, but on Kaggle it
    is a separate read-only input dataset. Deriving it from PERSISTENT_DIR
    would point at /kaggle/working, which is empty. All locations are
    :class:`pathlib.Path`. Nothing is created and nothing is checked for
    existence here -- ``scripts/bootstrap_session.py`` does the asserting.

    Raises :class:`PathConfigError` if a value this host needs is unset.
    """
    if config is None:
        config = load_config(config_path)
    platform = platform or detect_platform()

    gt_subdir = _get(config, "session", "gt_subdir") or "gt_boundaries"
    if platform == "colab":
        data_root = Path(_need(config, "colab", "data_root"))
        persistent_dir = Path(_need(config, "colab", "persistent_dir"))
        # Written by step 2 into the same Drive folder that survives a restart.
        gt_boundaries_root = persistent_dir / gt_subdir
    elif platform == "kaggle":
        input_root = Path(_need(config, "kaggle", "input_root"))
        data_root = input_root / _need(config, "kaggle", "dataset_slug")
        persistent_dir = Path(_need(config, "kaggle", "working_dir"))
        # A SEPARATE read-only input dataset, not something under
        # /kaggle/working: the boundary maps were produced elsewhere and
        # uploaded, so this root is resolved independently of PERSISTENT_DIR.
        gt_boundaries_root = input_root / _need(config, "kaggle", "gt_dataset_slug")
    else:
        data_root = REPO_ROOT / "data"
        persistent_dir = REPO_ROOT
        gt_boundaries_root = REPO_ROOT / gt_subdir

    return {
        "platform": platform,
        "repo_root": REPO_ROOT,
        "config_path": Path(config_path) if config_path else DEFAULT_CONFIG,
        "data_root": data_root,
        "gt_boundaries_root": gt_boundaries_root,
        "persistent_dir": persistent_dir,
        "outputs_dir": persistent_dir / "outputs",
        "checkpoints_dir": persistent_dir / "checkpoints",
        "logs_dir": persistent_dir / "logs",
        "reports_dir": REPO_ROOT / "reports",
        "expected_datasets": expected_datasets(config),
    }
