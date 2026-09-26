"""Post-processing of a predicted boundary probability map into a binary map.

The network outputs a probability per pixel; everything downstream that
wants a boundary LINE has to binarize it. How that is done is not neutral.
A plain threshold keeps whatever width the model happened to paint, and a
model trained on 4 px ground truth that paints 6 px bands has bands wide
enough to swallow a thin region between two boundaries outright -- which is
exactly what an under-segmenting region metric would look like.

``skeleton_redilate`` normalises that width the same way the ground truth's
own width was normalised in step 2: collapse to a 1 px centreline, then grow
back to ``target_width_px`` with ``boundary_gt._dilate_to_width`` -- the very
function that produced the ground truth's uniform line. Prediction and truth
therefore end up the same width by the same operation, not by two
implementations that could quietly disagree.

Deliberately torch-free: this is a pure numpy function, importable (and
testable) wherever numpy and scikit-image are, and usable at inference time
by a downstream stage without pulling in the training stack.

``mode="none"`` is the default everywhere and is exactly ``prob >= threshold``
-- byte-identical to how every evaluation in this repository binarized a
prediction before this module existed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

#: ``none`` -- binarize only (today's behaviour). ``skeleton_redilate`` --
#: binarize, skeletonize, dilate back to ``target_width_px``.
MODES = ("none", "skeleton_redilate")

# --------------------------------------------------------------------------
# defaults -- overridable from configs/default.yaml under ``postprocess:``.
# No training or tiling stage reads this; the default stays "none" so every
# existing evaluation keeps its exact numbers. ``target_width_px: None``
# means "the width the ground truth was generated at", which is recorded in
# reports/gt_extraction.json and must be passed in by the caller -- this
# module does no I/O of its own.
# --------------------------------------------------------------------------
DEFAULTS = {
    "mode": "none",
    "target_width_px": None,
}


class PostprocessError(RuntimeError):
    """Raised when a boundary cannot be post-processed. Never silent."""


def load_config(config_path: Optional[Path] = None) -> dict:
    """Merge ``postprocess:`` from configs/default.yaml over DEFAULTS."""
    from src import paths as paths_mod

    cfg = paths_mod.load_config(config_path)
    settings = dict(DEFAULTS)
    section = cfg.get("postprocess") or {}
    if not isinstance(section, dict):
        raise PostprocessError(
            f"configs/default.yaml: postprocess must be a mapping, got "
            f"{type(section).__name__}")
    unknown = set(section) - set(DEFAULTS)
    if unknown:
        raise PostprocessError(
            f"configs/default.yaml: unknown postprocess keys {sorted(unknown)}; "
            f"known keys are {sorted(DEFAULTS)}")
    for key, value in section.items():
        if value is not None:
            settings[key] = value
    if settings["mode"] not in MODES:
        raise PostprocessError(
            f"postprocess.mode must be one of {MODES}, got {settings['mode']!r}")
    return settings


def postprocess_boundary(prob, threshold, mode: str = "none",
                         target_width_px: Optional[int] = None) -> np.ndarray:
    """Binarize a boundary probability map, optionally normalising its width.

    ``mode="none"``: returns ``np.asarray(prob) >= float(threshold)`` and
    nothing else -- the exact expression every evaluation used before this
    function existed, so passing through here with the default changes no
    number anywhere.

    ``mode="skeleton_redilate"``: the same binarization, then
    ``skimage.morphology.skeletonize`` to a 1 px centreline, then
    ``boundary_gt._dilate_to_width(skeleton, target_width_px)``.
    ``target_width_px`` is REQUIRED for this mode -- pass the width the ground
    truth was generated at (``reports/gt_extraction.json``'s recorded
    ``settings.line_width_px``), not a guess. An empty binarization stays
    empty: there is no centreline to grow, and nothing is fabricated.

    Returns a boolean array the same shape as ``prob``. Pure: no I/O, no
    randomness, no state.
    """
    if mode not in MODES:
        raise PostprocessError(f"postprocess mode must be one of {MODES}, got {mode!r}")
    binary = np.asarray(prob) >= float(threshold)
    if mode == "none":
        return binary

    if target_width_px is None:
        raise PostprocessError(
            "skeleton_redilate needs target_width_px -- pass the width the "
            "ground truth was generated at (reports/gt_extraction.json "
            "settings.line_width_px).")
    width = int(round(float(target_width_px)))
    if width < 1:
        raise PostprocessError(f"target_width_px must be >= 1, got {target_width_px}")
    if binary.ndim != 2:
        raise PostprocessError(
            f"skeleton_redilate works on one 2-D map at a time, got shape {binary.shape}")
    if not binary.any():
        return binary

    from skimage.morphology import skeletonize

    from src import boundary_gt

    return boundary_gt._dilate_to_width(skeletonize(binary), width)
