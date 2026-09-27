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

``morph_close`` and ``skeleton_bridge`` target a DIFFERENT failure than width:
a genuine short GAP in the predicted line, of the kind that lets a
marker-controlled watershed (or the region metric read off the thresholded
boundary directly) flood two true regions into one. ``morph_close`` seals a
gap by dilating the boundary by ``closing_radius`` and eroding it back (a
literal binary closing) -- simple, but it also fattens every surviving
boundary, which is why it is swept independently of ``skeleton_redilate``
rather than combined with it. ``skeleton_bridge`` instead finds the loose
ENDS of the skeleton and, where two ends from DIFFERENT fragments are within
``bridge_px`` of each other, draws a straight line between them and redilates
to ``target_width_px`` -- closing exactly the gap and nothing else, at the
cost of doing nothing for a gap wider than ``bridge_px``.

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
#: binarize, skeletonize, dilate back to ``target_width_px``. ``morph_close``
#: -- binarize, then a literal binary closing at ``closing_radius``.
#: ``skeleton_bridge`` -- binarize, skeletonize, bridge nearby loose ends
#: within ``bridge_px``, dilate back to ``target_width_px``.
MODES = ("none", "skeleton_redilate", "morph_close", "skeleton_bridge")

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


def _skeleton_endpoints(skeleton: np.ndarray) -> np.ndarray:
    """8-connected degree-1 pixels of a binary skeleton.

    A skeleton pixel with exactly one skeleton neighbour in its 3x3
    neighbourhood is a loose end -- the far tip of a fragment, whether that
    fragment is a whole boundary or a piece cut off by a gap.
    """
    import cv2

    sk = skeleton.astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbour_count = cv2.filter2D(sk, ddepth=cv2.CV_8U, kernel=kernel,
                                   borderType=cv2.BORDER_CONSTANT)
    return skeleton & (neighbour_count == 1)


def _bridge_skeleton_endpoints(skeleton: np.ndarray, bridge_px: float) -> np.ndarray:
    """Connect loose skeleton ends from DIFFERENT fragments within ``bridge_px``.

    Deterministic and greedy: every candidate pair (one endpoint from each of
    two still-separate fragments, within ``bridge_px``) is considered in order
    of increasing distance, ties broken by pixel coordinate, and a pair is
    bridged with a straight line only if its two fragments have not already
    been joined by a shorter bridge. A fragment can end up bridged to more
    than one neighbour if it has more than one loose end; it is never bridged
    to itself.

    Endpoints are found on the ORIGINAL skeleton and fragment membership is
    tracked with union-find as bridges are added, rather than re-skeletonizing
    after every bridge -- so a long chain of short gaps bridges in one pass
    without the drawn lines themselves changing what counts as an endpoint.
    """
    from skimage.draw import line as draw_line
    from skimage.measure import label as cc_label

    out = skeleton.copy()
    if bridge_px <= 0 or not skeleton.any():
        return out

    fragments = cc_label(skeleton, connectivity=2)
    points = np.argwhere(_skeleton_endpoints(skeleton))
    if len(points) < 2:
        return out
    fragment_of = fragments[points[:, 0], points[:, 1]]

    parent = {int(f): int(f) for f in np.unique(fragment_of)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n = len(points)
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            if fragment_of[i] == fragment_of[j]:
                continue
            dist = float(np.hypot(*(points[i] - points[j])))
            if dist <= float(bridge_px):
                candidates.append((dist, tuple(points[i]), tuple(points[j]), i, j))
    candidates.sort(key=lambda c: (c[0], c[1], c[2]))

    for _, (r0, c0), (r1, c1), i, j in candidates:
        root_i, root_j = find(int(fragment_of[i])), find(int(fragment_of[j]))
        if root_i == root_j:
            continue
        rr, cc = draw_line(int(r0), int(c0), int(r1), int(c1))
        out[rr, cc] = True
        parent[root_j] = root_i
    return out


def postprocess_boundary(prob, threshold, mode: str = "none",
                         target_width_px: Optional[int] = None,
                         closing_radius: Optional[int] = None,
                         bridge_px: Optional[float] = None) -> np.ndarray:
    """Binarize a boundary probability map, optionally closing gaps or
    normalising its width.

    ``mode="none"``: returns ``np.asarray(prob) >= float(threshold)`` and
    nothing else -- the exact expression every evaluation used before this
    function existed, so passing through here with the default changes no
    number anywhere.

    ``mode="skeleton_redilate"``: the same binarization, then
    ``skimage.morphology.skeletonize`` to a 1 px centreline, then
    ``boundary_gt._dilate_to_width(skeleton, target_width_px)``.
    ``target_width_px`` is REQUIRED -- pass the width the ground truth was
    generated at (``reports/gt_extraction.json``'s recorded
    ``settings.line_width_px``), not a guess.

    ``mode="morph_close"``: the same binarization, then a literal binary
    closing (dilate by ``closing_radius``, then erode by the same amount) --
    ``skimage.morphology.binary_closing`` with a disk footprint.
    ``closing_radius`` is REQUIRED. This also fattens every surviving
    boundary by roughly ``closing_radius`` on each side; it is not combined
    with ``skeleton_redilate`` because the two would then be indistinguishable
    from each other in a sweep.

    ``mode="skeleton_bridge"``: the same binarization, skeletonize, bridge
    loose ends across different fragments within ``bridge_px`` with a
    straight line (see :func:`_bridge_skeleton_endpoints`), then redilate to
    ``target_width_px``. Both ``bridge_px`` and ``target_width_px`` are
    REQUIRED. A gap wider than ``bridge_px`` is left open.

    Every mode: an empty binarization stays empty -- there is no centreline
    or gap to act on, and nothing is fabricated. Returns a boolean array the
    same shape as ``prob``. Pure: no I/O, no randomness, no state.
    """
    if mode not in MODES:
        raise PostprocessError(f"postprocess mode must be one of {MODES}, got {mode!r}")
    binary = np.asarray(prob) >= float(threshold)
    if mode == "none":
        return binary
    if binary.ndim != 2:
        raise PostprocessError(
            f"{mode} works on one 2-D map at a time, got shape {binary.shape}")
    if not binary.any():
        return binary

    def _width(name):
        if target_width_px is None:
            raise PostprocessError(
                f"{name} needs target_width_px -- pass the width the ground "
                "truth was generated at (reports/gt_extraction.json "
                "settings.line_width_px).")
        width = int(round(float(target_width_px)))
        if width < 1:
            raise PostprocessError(f"target_width_px must be >= 1, got {target_width_px}")
        return width

    if mode == "skeleton_redilate":
        from skimage.morphology import skeletonize

        from src import boundary_gt

        return boundary_gt._dilate_to_width(skeletonize(binary), _width(mode))

    if mode == "morph_close":
        if closing_radius is None:
            raise PostprocessError(
                "morph_close needs closing_radius (px) -- the gap-closing "
                "distance to sweep.")
        radius = int(round(float(closing_radius)))
        if radius < 1:
            raise PostprocessError(f"closing_radius must be >= 1, got {closing_radius}")

        from skimage.morphology import binary_closing, disk

        return binary_closing(binary, footprint=disk(radius))

    if bridge_px is None:
        raise PostprocessError(
            "skeleton_bridge needs bridge_px -- the maximum gap it may close.")
    bridge = float(bridge_px)
    if bridge <= 0:
        raise PostprocessError(f"bridge_px must be > 0, got {bridge_px}")

    from skimage.morphology import skeletonize

    from src import boundary_gt

    bridged = _bridge_skeleton_endpoints(skeletonize(binary), bridge)
    return boundary_gt._dilate_to_width(bridged, _width(mode))
