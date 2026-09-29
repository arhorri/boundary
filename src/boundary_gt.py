"""Boundary ground truth extraction: MODE A (painted colour) and MODE B (labels).

The mode of every dataset folder is READ from ``reports/audit.json`` -- never
assumed, never inferred here. Step 1 measured it; this step obeys it, with an
explicit per-folder override available for a verdict a human disagrees with.

    MODE A  the mask paints boundaries in one colour. The HSV window for that
            colour is DERIVED from the pixels themselves (2nd/98th percentile
            of H, S, V over sampled masks) and written to
            configs/hsv_ranges.yaml so it can be hand-tuned afterwards.
            Yields phase interfaces AND intra-phase grain boundaries.

    MODE B  the mask is a phase-label map. Every pixel is snapped to the
            nearest of the K palette colours derived from the audit, and
            ``find_boundaries`` marks where labels meet. Yields phase
            interfaces ONLY -- grain boundaries are absent from the source
            data and are not invented here.

Both modes then share one cleanup: OPEN to drop speckle, CLOSE to seal 1-2 px
gaps, skeletonize to a single-pixel centreline, dilate back to one uniform
width. The output is a uint8 PNG whose pixels are exactly 0 or 255.

No learned model appears anywhere in this path, and nothing is ever resized:
every operation is per-pixel or morphological, at the stored resolution.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from src import audit as audit_mod
from src.paths import REPO_ROOT

# --------------------------------------------------------------------------
# defaults -- every one of these is overridable from configs/default.yaml
# under the ``boundary_gt:`` key, and the values actually used are echoed
# into reports/gt_extraction.md.
# --------------------------------------------------------------------------
DEFAULTS = {
    "line_width_px": 2,          # uniform width of the final boundary line
    "open_kernel": 3,            # speckle removal, see open_mode
    "open_mode": "speckle",      # "speckle" | "morph"; see _despeckle
    "close_kernel": 3,           # seals 1-2 px gaps
    "hsv_percentiles": [2, 98],  # MODE A window, derived not hardcoded
    "hsv_sample_files": 12,      # masks sampled when deriving the window
    "palette_merge_distance": 48,  # RGB distance below which two mask colours
                                   # are the same class (swallows JPEG /
                                   # anti-alias ramps; real classes in this
                                   # collection are >190 apart)
    "min_palette_share": 0.002,  # a peak must hold this share of pixels
    "max_unsnapped_fraction": 0.02,  # reject if this much of the mask sits far
                                     # from every palette peak
    "output_subdir": "gt_boundaries",
    # Mask and image dimensions that differ by at most this many pixels in
    # either axis are centre-cropped to their common size instead of being
    # rejected: an off-by-one row in a source dataset is a packaging slip, not
    # a broken annotation. Set to 0 to reject every mismatch.
    "size_tolerance_px": 2,
    "exclusions": {},            # folder -> [mask filename, ...] never processed
    # folder -> [[r, g, b], ...] colours suspected of being scale bars, tool
    # marks or other non-phase annotation. ALWAYS measured and reported;
    # folded into the background class only when fold_artifact_colours is on.
    "artifact_colours": {},
    "fold_artifact_colours": False,
    # MODE B only. folder -> minimum connected-component area in pixels; a
    # phase component smaller than this is merged into its surrounding phase
    # (see merge_small_regions) BEFORE find_boundaries runs, so it draws no
    # boundary loop of its own. 0 (absent from the mapping) is OFF -- a
    # dataset that does not set this here is byte-identical to before this
    # setting existed. Not a global default: Steel1 and every other dataset
    # must stay untouched while Steel2's speckle is cleaned up.
    "min_region_area_px": {},
}

#: A boundary map covering less/more than this is not a boundary map.
#: Calibrated at DEFAULTS["line_width_px"] (2 px): a uniform-width skeleton
#: dilation covers area roughly proportional to width for a fixed line
#: network, so the upper bound must scale with the configured width via
#: ``sane_fraction_band()`` rather than being read off directly.
SANE_FRACTION_BAND = (0.005, 0.25)


def sane_fraction_band(settings: dict) -> tuple:
    """Width-scaled (lo, hi) boundary-fraction band for the given settings.

    A thicker configured line covers proportionally more pixels for the same
    boundary network, so the fixed SANE_FRACTION_BAND (calibrated at the
    reference width) is scaled by ``line_width_px / reference width``.
    """
    lo, hi = SANE_FRACTION_BAND
    scale = float(settings["line_width_px"]) / float(DEFAULTS["line_width_px"])
    return (lo, min(0.95, hi * scale))


class ExtractionError(RuntimeError):
    """Raised when extraction cannot proceed. Never fails silently."""


# --------------------------------------------------------------------------
# config + audit input
# --------------------------------------------------------------------------
def load_config(config_path: Optional[Path] = None) -> dict:
    """Merge ``boundary_gt:`` from configs/default.yaml over DEFAULTS."""
    from src import paths as paths_mod

    cfg = paths_mod.load_config(config_path)
    settings = dict(DEFAULTS)
    section = cfg.get("boundary_gt") or {}
    if not isinstance(section, dict):
        raise ExtractionError(
            "configs/default.yaml: boundary_gt must be a mapping, got "
            f"{type(section).__name__}"
        )
    for key, value in section.items():
        if value is not None:
            settings[key] = value
    unknown = set(section) - set(DEFAULTS)
    if unknown:
        raise ExtractionError(
            "configs/default.yaml: unknown boundary_gt keys "
            f"{sorted(unknown)}; known keys are {sorted(DEFAULTS)}"
        )
    return settings


def load_audit(reports_dir: Optional[Path] = None) -> dict:
    """Read reports/audit.json. The mode of a folder comes from here only."""
    path = Path(reports_dir or (REPO_ROOT / "reports")) / "audit.json"
    if not path.is_file():
        raise ExtractionError(
            f"{path} does not exist. Run notebooks/01_audit.ipynb first: this "
            "step reads each folder's mask mode from the audit and must never "
            "guess it."
        )
    report = json.loads(path.read_text())
    if not report.get("datasets"):
        raise ExtractionError(f"{path} contains no audited datasets.")
    return report


def folder_modes(audit: dict, overrides: Optional[dict] = None) -> dict:
    """Mode per folder, audit verdict first, explicit override second."""
    modes = {name: d["mode"]["inferred"] for name, d in audit["datasets"].items()}
    for name, mode in (overrides or {}).items():
        if name not in modes:
            raise ExtractionError(
                f"override names folder '{name}', which the audit does not "
                f"contain. Audited folders: {', '.join(sorted(modes))}"
            )
        if mode not in ("A", "B"):
            raise ExtractionError(f"override for {name} must be 'A' or 'B', got {mode!r}")
        modes[name] = mode
    return modes


# --------------------------------------------------------------------------
# palette: the K real classes, not the raw unique-colour count
# --------------------------------------------------------------------------
def _as_rgb(colour: Sequence) -> tuple:
    """Normalise a 1-channel grey to a 3-channel triple.

    Half of Steel2's masks are stored 1-channel and half 3-channel, so the
    same physical grey appears twice in the audit under two spellings. Every
    comparison in this module happens after this normalisation.
    """
    vals = [int(v) for v in colour]
    if len(vals) == 1:
        return (vals[0], vals[0], vals[0])
    return tuple(vals[:3])


def pooled_colours(folder_report: dict) -> list:
    """Pixel share per colour, pooled over the audited sample of one folder."""
    pooled, total = {}, 0
    for rec in folder_report["files"]:
        area = rec["mask"]["width"] * rec["mask"]["height"]
        total += area
        for entry in rec["palette"]["top_colours"]:
            key = _as_rgb(entry["colour"])
            pooled[key] = pooled.get(key, 0) + int(entry["pixels"])
    if not total:
        raise ExtractionError("the audit recorded no sampled masks for this folder")
    return sorted(((c, n / total) for c, n in pooled.items()),
                  key=lambda t: -t[1])


def derive_palette(folder_report: dict, settings: dict) -> list:
    """The K palette colours of a folder: peaks, not raw unique colours.

    A raw unique-colour count is not K. Steel2's masks report 45-52 colours
    where there are two real classes: the rest is an anti-aliasing ramp
    decaying smoothly from each class colour. So colours are walked from the
    most common down, and one is kept as a new class only if it is further
    than ``palette_merge_distance`` from every class already kept and holds at
    least ``min_palette_share`` of the pixels. Everything else is a ramp value
    and gets snapped to whichever class it is nearest.
    """
    merge_d = float(settings["palette_merge_distance"])
    min_share = float(settings["min_palette_share"])
    peaks = []
    for colour, share in pooled_colours(folder_report):
        if share < min_share:
            continue
        arr = np.array(colour, dtype=float)
        if all(np.linalg.norm(arr - np.array(p, dtype=float)) > merge_d for p in peaks):
            peaks.append(colour)
    if len(peaks) < 2:
        raise ExtractionError(
            f"only {len(peaks)} palette class(es) survive merging at distance "
            f"{merge_d} with a {min_share} share floor: "
            f"{peaks}. A mask with one class has no boundaries; lower "
            "min_palette_share or check the folder."
        )
    return peaks


def _centre_crop(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    """Centre-crop an array to (height, width). Never resamples."""
    top = (arr.shape[0] - height) // 2
    left = (arr.shape[1] - width) // 2
    return arr[top:top + height, left:left + width]


def reconcile_size(mask: np.ndarray, image_wh: tuple, tolerance: int) -> tuple:
    """Bring a mask and its image to a common size, or refuse to.

    Returns ``(mask, info, error)``. ``info`` is None when the sizes already
    agree; ``error`` is a message when the mismatch is larger than
    ``tolerance`` and the pair must be rejected. The crop is centred and
    applied to the mask here; ``info["image_crop"]`` records the identical
    crop the image needs, so a later step can apply it without guessing.

    Cropping, not resizing: a resampled mask would invent label values, and a
    resampled image would break the physical micron scale.
    """
    img_w, img_h = int(image_wh[0]), int(image_wh[1])
    mask_h, mask_w = int(mask.shape[0]), int(mask.shape[1])
    if (img_w, img_h) == (mask_w, mask_h):
        return mask, None, None

    d_w, d_h = abs(img_w - mask_w), abs(img_h - mask_h)
    if d_w > tolerance or d_h > tolerance:
        return mask, None, (
            f"mask {mask_w}x{mask_h} does not match image {img_w}x{img_h} "
            f"(differs by {d_w}x{d_h} px, tolerance {tolerance})"
        )

    common_w, common_h = min(img_w, mask_w), min(img_h, mask_h)
    info = {
        "image_size": [img_w, img_h],
        "mask_size": [mask_w, mask_h],
        "final_size": [common_w, common_h],
        "delta_px": [d_w, d_h],
        # top-left offset of the centre crop, for whoever loads the image
        "image_crop": {"left": (img_w - common_w) // 2,
                       "top": (img_h - common_h) // 2,
                       "width": common_w, "height": common_h},
        "mask_crop": {"left": (mask_w - common_w) // 2,
                      "top": (mask_h - common_h) // 2,
                      "width": common_w, "height": common_h},
    }
    return _centre_crop(mask, common_h, common_w), info, None


def watched_colours(folder: str, settings: dict) -> list:
    """Colours to measure (and possibly fold) for one folder."""
    listed = (settings.get("artifact_colours") or {}).get(folder) or []
    return [_as_rgb(c) for c in listed]


def fold_targets(palette: Sequence, folder: str, settings: dict) -> list:
    """Palette indices to fold into the background class.

    Empty unless ``fold_artifact_colours`` is on. Index 0 is the background:
    ``derive_palette`` returns peaks in descending pixel share, so the first
    entry is the majority colour of the folder.
    """
    if not settings.get("fold_artifact_colours"):
        return []
    watch = {tuple(c) for c in watched_colours(folder, settings)}
    return [i for i, c in enumerate(palette) if tuple(_as_rgb(c)) in watch and i != 0]


def colour_hit(rgb: np.ndarray, colour: Sequence) -> np.ndarray:
    """Exact-match mask for one colour, grayscale or RGB."""
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    target = np.array(_as_rgb(colour), dtype=np.int32)
    return np.all(rgb[..., :3].astype(np.int32) == target, axis=-1)


def artifact_boundary_overlap(
    mask: np.ndarray,
    colour: Sequence,
    boundary: np.ndarray,
    return_masks: bool = False,
):
    """Is this colour present, and is a boundary being drawn around it?

    ``find_boundaries`` cannot tell a phase from a scale bar: any region whose
    colour is its own class gets a closed boundary loop drawn around its
    outline. This measures that directly -- how much of the extracted boundary
    lies on the colour's own pixels or in the 1-px ring around them.
    """
    from scipy.ndimage import binary_dilation

    hit = colour_hit(mask, colour)
    bnd = boundary > 0
    n_bnd = int(bnd.sum())
    stats = {
        "colour": [int(v) for v in _as_rgb(colour)],
        "present": bool(hit.any()),
        "pixels": int(hit.sum()),
        "fraction": float(hit.mean()),
        "boundary_pixels_on_outline": 0,
        "share_of_boundary": 0.0,
    }
    ring = np.zeros_like(hit)
    if hit.any() and n_bnd:
        ring = binary_dilation(hit, structure=np.ones((3, 3), bool)) & ~hit
        touching = bnd & (ring | hit)
        stats["boundary_pixels_on_outline"] = int(touching.sum())
        stats["share_of_boundary"] = float(touching.sum()) / float(n_bnd)
    if return_masks:
        return stats, hit, ring
    return stats


def snap_to_palette(rgb: np.ndarray, palette: Sequence) -> tuple:
    """Snap every pixel to its nearest palette colour.

    Returns ``(labels, unsnapped_fraction, n_levels)``. ``unsnapped_fraction``
    is the share of pixels further from every palette colour than half the
    smallest gap between two palette colours -- i.e. pixels the palette does
    not really explain, which is what makes a mask untrustworthy rather than
    merely noisy.
    """
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    centres = np.array([list(_as_rgb(c)) for c in palette], dtype=np.int32)
    flat = rgb.reshape(-1, rgb.shape[2])[:, :3].astype(np.int32)

    labels = np.empty(flat.shape[0], dtype=np.int32)
    best = np.empty(flat.shape[0], dtype=np.float64)
    chunk = 65536
    for start in range(0, flat.shape[0], chunk):
        block = flat[start:start + chunk]
        dist = ((block[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        labels[start:start + chunk] = np.argmin(dist, axis=1)
        best[start:start + chunk] = np.sqrt(dist.min(axis=1))

    gaps = [np.linalg.norm(centres[i] - centres[j])
            for i in range(len(centres)) for j in range(i + 1, len(centres))]
    tolerance = float(min(gaps)) / 2.0 if gaps else 0.0
    unsnapped = float((best > tolerance).mean())
    labels = labels.reshape(rgb.shape[:2])
    return labels, unsnapped, int(len(np.unique(labels)))


def merge_small_regions(labels: np.ndarray, min_area_px: int) -> tuple:
    """Merge every connected component smaller than ``min_area_px`` into its
    surrounding phase, so it produces no boundary of its own.

    Steel2's label map is dominated by speckle-sized regions -- 53% of the
    regions ``true_regions_from_boundary`` finds in its extracted boundaries
    are under 100 px, against 18.5% for Steel1 (``reports/gt_noise_steel.md``)
    -- and every one of them draws a closed loop that
    :func:`boundaries_from_labels` has no way to distinguish from a real
    grain or phase interface. This runs BEFORE that call, on the palette-
    snapped label map itself, not on the boundary it would produce: erasing a
    boundary loop after the fact would still leave the interior mislabelled
    for anything downstream that reads the label map, and would not correctly
    handle a speckle that touches more than one neighbouring phase.

    For every connected component (of ANY class, computed per class since a
    plain ``skimage.measure.label`` call has no notion of "background" among
    K phase labels) smaller than ``min_area_px``, every pixel of it is
    reassigned to the MAJORITY label along its immediate border -- the phase
    it is embedded in, not an arbitrary neighbour. Processed in ascending
    component-id order (deterministic, not dependent on dict/set iteration):
    if two small components are mutual neighbours, the first one processed
    absorbs into whatever surrounds IT at that point, and the second then
    sees the first's new (now larger) label as part of its own border tally --
    a single deterministic pass, not a search for a globally stable
    partition, which is not needed here.

    ``min_area_px <= 0`` is off: returns ``labels`` UNCHANGED (the same array,
    not a copy) and a stats dict with ``enabled: False``, so a dataset that
    does not set ``boundary_gt.min_region_area_px`` is guaranteed byte-
    identical to before this function existed -- callers must not skip this
    function to get that guarantee; it is built in.

    Returns ``(merged_labels, stats)``. ``stats`` records how many components
    existed, how many were merged, and how many pixels were reassigned --
    the boundary-pixel counts before/after are added by the caller, which
    already computes ``boundaries_from_labels`` on both.
    """
    if min_area_px <= 0:
        return labels, {"enabled": False, "min_area_px": int(min_area_px),
                        "n_components_total": None, "n_components_merged": 0,
                        "n_pixels_reassigned": 0}

    from scipy import ndimage as ndi
    from skimage.measure import label as cc_label

    merged = labels.copy()
    instance_map = np.zeros(labels.shape, dtype=np.int64)
    offset = 0
    for value in np.unique(labels):
        cls_mask = labels == value
        cc = cc_label(cls_mask, connectivity=2)
        if cc.max() == 0:
            continue
        instance_map[cls_mask] = cc[cls_mask] + offset
        offset += int(cc.max())

    areas = np.bincount(instance_map.ravel())
    small_instances = sorted(
        i for i in range(1, len(areas)) if 0 < areas[i] < min_area_px)

    n_pixels_reassigned = 0
    n_components_merged = 0
    for inst_id in small_instances:
        comp_mask = instance_map == inst_id
        border = ndi.binary_dilation(comp_mask) & ~comp_mask
        border_values, border_counts = np.unique(merged[border], return_counts=True)
        if border_values.size == 0:
            # The component covers the entire tile -- nothing surrounds it.
            # Leaving it alone is correct: there is no "surrounding phase" to
            # merge into, and forcing one would fabricate a label. NOT counted
            # as merged: nothing about the labels changed for it.
            continue
        majority = border_values[int(np.argmax(border_counts))]
        merged[comp_mask] = majority
        n_pixels_reassigned += int(comp_mask.sum())
        n_components_merged += 1

    return merged, {
        "enabled": True,
        "min_area_px": int(min_area_px),
        "n_components_total": int(len(areas) - 1),
        "n_components_merged": n_components_merged,
        "n_pixels_reassigned": n_pixels_reassigned,
    }


def boundaries_from_labels(labels: np.ndarray) -> np.ndarray:
    """Where labels meet. Phase interfaces only -- by construction."""
    from skimage.segmentation import find_boundaries

    return find_boundaries(labels, mode="thick")


# --------------------------------------------------------------------------
# MODE A: HSV window derived from the data
# --------------------------------------------------------------------------
def _to_hsv(rgb: np.ndarray) -> np.ndarray:
    """OpenCV HSV (H 0-179, S/V 0-255) from an RGB array, via BGR."""
    import cv2

    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    bgr = np.ascontiguousarray(rgb[..., :3][..., ::-1]).astype(np.uint8)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def boundary_colour(folder_report: dict) -> list:
    """The painted-boundary colour the audit identified for this folder."""
    ev = folder_report["mode"]["evidence"]
    if ev.get("passing_colours"):
        return list(ev["passing_colours"][0])
    if ev.get("candidate_colour"):
        return list(ev["candidate_colour"])
    raise ExtractionError(
        "MODE A was requested but the audit found no boundary colour for this "
        "folder at all. Either the override is wrong, or re-run the audit."
    )


def derive_hsv_window(
    folder_report: dict,
    pairs: Sequence,
    settings: dict,
    colour: Optional[Sequence] = None,
) -> dict:
    """Derive the HSV window for a folder's boundary colour from its pixels.

    Nothing is hardcoded: the pixels carrying the boundary colour are gathered
    from up to ``hsv_sample_files`` masks and the window is the 2nd/98th
    percentile of H, S and V over them. Hue is circular, so a colour straddling
    the 0/179 wrap (reds) is detected and emitted as two windows to be OR-ed.
    """
    colour = list(colour or boundary_colour(folder_report))
    lo_p, hi_p = settings["hsv_percentiles"]
    target = np.array(_as_rgb(colour), dtype=np.int32)

    samples = []
    for _, mask_path in list(pairs)[: int(settings["hsv_sample_files"])]:
        rgb = audit_mod.read_array(mask_path)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[:, :, None], 3, axis=2)
        hit = np.all(rgb[..., :3].astype(np.int32) == target, axis=-1)
        if not hit.any():
            continue
        samples.append(_to_hsv(rgb)[hit])
    if not samples:
        raise ExtractionError(
            f"colour {colour} does not occur in any of the "
            f"{settings['hsv_sample_files']} sampled masks, so no HSV window "
            "can be derived from the data."
        )

    px = np.concatenate(samples, axis=0).astype(np.float64)
    h, s, v = px[:, 0], px[:, 1], px[:, 2]

    # circular hue: if the spread is implausibly wide, the colour wraps
    h_lo, h_hi = np.percentile(h, [lo_p, hi_p])
    wraps = bool((h_hi - h_lo) > 90.0)
    if wraps:
        shifted = (h + 90.0) % 180.0
        s_lo, s_hi = np.percentile(shifted, [lo_p, hi_p])
        h_lo, h_hi = (s_lo - 90.0) % 180.0, (s_hi - 90.0) % 180.0

    window = {
        "colour_rgb": [int(c) for c in _as_rgb(colour)],
        "n_pixels_sampled": int(px.shape[0]),
        "percentiles": [lo_p, hi_p],
        "wraps_hue": wraps,
        "lower": [float(h_lo), float(np.percentile(s, lo_p)), float(np.percentile(v, lo_p))],
        "upper": [float(h_hi), float(np.percentile(s, hi_p)), float(np.percentile(v, hi_p))],
        "derived_from": "2nd/98th percentile of the boundary colour's own pixels",
        "manual": False,
    }
    return window


def scale_hsv_window(window: dict, factor: float) -> dict:
    """Widen (>1) or tighten (<1) a window around its own centre."""
    out = dict(window)
    lo, hi = np.array(window["lower"], float), np.array(window["upper"], float)
    centre = (lo + hi) / 2.0
    half = (hi - lo) / 2.0 * float(factor)
    new_lo, new_hi = centre - half, centre + half
    out["lower"] = [float(max(0.0, new_lo[0])), float(np.clip(new_lo[1], 0, 255)),
                    float(np.clip(new_lo[2], 0, 255))]
    out["upper"] = [float(min(179.0, new_hi[0])), float(np.clip(new_hi[1], 0, 255)),
                    float(np.clip(new_hi[2], 0, 255))]
    out["scaled_by"] = float(factor)
    return out


def apply_hsv_window(rgb: np.ndarray, window: dict) -> np.ndarray:
    """cv2.inRange on the HSV image; handles the hue wrap as two windows."""
    import cv2

    hsv = _to_hsv(rgb)
    lo = np.array(window["lower"], dtype=np.float64)
    hi = np.array(window["upper"], dtype=np.float64)
    if window.get("wraps_hue") and lo[0] > hi[0]:
        a = cv2.inRange(hsv, np.array([lo[0], lo[1], lo[2]], np.uint8),
                        np.array([179, hi[1], hi[2]], np.uint8))
        b = cv2.inRange(hsv, np.array([0, lo[1], lo[2]], np.uint8),
                        np.array([hi[0], hi[1], hi[2]], np.uint8))
        hit = cv2.bitwise_or(a, b)
    else:
        hit = cv2.inRange(hsv, lo.astype(np.uint8), hi.astype(np.uint8))
    return hit.astype(bool)


# --------------------------------------------------------------------------
# shared cleanup
# --------------------------------------------------------------------------
def _dilate_to_width(skeleton: np.ndarray, width: int) -> np.ndarray:
    """Grow a 1-px skeleton to exactly ``width`` px.

    A 3x3 dilation adds one pixel on each side, so it can only make odd
    widths; an even width needs one asymmetric 2x2 step first. The checks cell
    of the notebook re-measures the width of the written files.
    """
    import cv2

    if width < 1:
        raise ExtractionError(f"line_width_px must be >= 1, got {width}")
    out = skeleton.astype(np.uint8)
    if width == 1:
        return out.astype(bool)
    if width % 2 == 0:
        out = cv2.dilate(out, np.ones((2, 2), np.uint8), iterations=1)
        extra = (width - 2) // 2
    else:
        extra = (width - 1) // 2
    if extra:
        out = cv2.dilate(out, np.ones((3, 3), np.uint8), iterations=extra)
    return out.astype(bool)


def _despeckle(binary: np.ndarray, settings: dict) -> np.ndarray:
    """Remove speckle without destroying the line it is speckling.

    A literal morphological OPEN with a 3x3 kernel CANNOT be used here. Its
    erosion step keeps a pixel only when all nine of its neighbours are set,
    and the thing being cleaned is a boundary map one to three pixels wide --
    ``find_boundaries(mode="thick")`` output is 2 px, a painted annotation
    line is 1-3 px. A 3x3 OPEN erases all of it and leaves an empty image.

    So the default (``open_mode: "speckle"``) achieves what OPEN was asked for
    -- drop isolated noise blobs -- by removing connected components smaller
    than ``open_kernel ** 2`` pixels, which is exactly the speckle a 3x3 OPEN
    was meant to catch, while leaving thin lines untouched. Set
    ``open_mode: "morph"`` in configs/default.yaml to get the literal 3x3
    OPEN instead; expect it to empty the map on this data.
    """
    import cv2
    from skimage.morphology import remove_small_objects

    k = int(settings["open_kernel"])
    if k <= 1:
        return binary
    mode = str(settings["open_mode"]).lower()
    if mode == "morph":
        img = cv2.morphologyEx((binary.astype(np.uint8)) * 255, cv2.MORPH_OPEN,
                               np.ones((k, k), np.uint8))
        return img > 0
    if mode != "speckle":
        raise ExtractionError(
            f"boundary_gt.open_mode must be 'speckle' or 'morph', got {mode!r}"
        )
    return remove_small_objects(binary, min_size=k * k, connectivity=2)


def clean_boundary(raw: np.ndarray, settings: dict) -> tuple:
    """Despeckle, CLOSE, skeletonize, dilate to one uniform width.

    Returns ``(uint8 image of 0/255, fraction before, fraction after, stages)``.
    ``stages`` carries the foreground fraction after each step, so a map that
    was destroyed by cleanup says which step destroyed it instead of arriving
    as a bare "nothing left".
    """
    import cv2
    from skimage.morphology import skeletonize

    frac_before = float(raw.mean())
    binary = raw.astype(bool)
    stages = {"raw": frac_before}

    binary = _despeckle(binary, settings)
    stages["after_open"] = float(binary.mean())

    k_close = int(settings["close_kernel"])
    if k_close > 1:
        closed = cv2.morphologyEx((binary.astype(np.uint8)) * 255, cv2.MORPH_CLOSE,
                                  np.ones((k_close, k_close), np.uint8))
        binary = closed > 0
    stages["after_close"] = float(binary.mean())

    skel = skeletonize(binary)
    stages["after_skeletonize"] = float(skel.mean())

    grown = _dilate_to_width(skel, int(settings["line_width_px"]))
    out = (grown.astype(np.uint8)) * 255
    stages["after_dilate"] = float((out > 0).mean())
    return out, frac_before, stages["after_dilate"], stages


def measured_line_width(binary: np.ndarray, width_hint: Optional[int] = None
                         ) -> Optional[float]:
    """Median line thickness via the distance transform at skeleton pixels.

    area / skeleton_length was tried first and rejected: a boundary network
    has triple points where three lines of width W meet, and each junction
    contributes a roughly W x W patch of area to the numerator while adding
    almost nothing to skeleton length (a branch point, not a branch). That
    bias grows with W, so it passed at line_width_px=2 by luck and failed at
    line_width_px=4 even though the dilation itself was correct. The distance
    transform instead measures thickness locally, at each skeleton pixel, and
    is insensitive to junction area.

    ``skeletonize`` always returns a strictly 1-px skeleton, even for a W-px
    strip. At the single surviving skeleton pixel of a straight strip, the
    distance transform (distance to the nearest background pixel, in pixel
    units) works out to ``ceil(W / 2)`` for EITHER parity of W: a width-3 and
    a width-4 strip both give a skeleton-point distance of 2. So the raw
    distance alone cannot tell a width-3 line from a width-4 one -- the
    correction back to true width (``2*dist - 1`` for odd W, ``2*dist`` for
    even W) depends on which parity W actually is. ``width_hint`` supplies
    that parity (the configured ``line_width_px`` this measurement is meant
    to verify); with no hint, odd is assumed to preserve prior behaviour.
    """
    import cv2
    from skimage.morphology import skeletonize

    hit = (binary > 0)
    if not hit.any():
        return None
    skel = skeletonize(hit)
    if not skel.any():
        return None
    dist = cv2.distanceTransform(hit.astype(np.uint8), cv2.DIST_L2, 5)
    offset = 0.0 if (width_hint is not None and int(width_hint) % 2 == 0) else 1.0
    widths = 2.0 * dist[skel] - offset
    return float(np.median(widths)) if widths.size else None


# --------------------------------------------------------------------------
# mode-check diagnostic: is a MODE B palette class actually a painted line?
#
# A MODE B label is assumed to be a PHASE: an area with an inside. Nothing
# checks that assumption. If a class is instead a thin network traced over
# the raw image (an etched grain-boundary line, say), find_boundaries still
# runs on it without complaint -- it just draws a boundary loop on BOTH
# edges of every line segment, trapping the line's own pixels as a spurious
# strip "region" between them. The functions below measure whether a class
# looks like that (this is read-only: nothing here changes what
# extract_folder writes) and, if so, build the alternative boundary MODE A
# would have produced had this class been the painted colour all along.
# --------------------------------------------------------------------------
def label_value_histogram(pairs: Sequence, sample: Optional[int] = None) -> list:
    """Exact-value pixel share, pooled over a sample of a folder's masks.

    Unlike :func:`derive_palette`'s peaks (colours merged within
    ``palette_merge_distance`` of each other) or the audit's per-file
    ``top_colours`` (only the most common few per file), this tallies every
    EXACT value across full masks read fresh from disk -- the raw label
    alphabet a folder's masks actually use, before any snapping or merging
    decision narrows it to K classes.
    """
    from src import audit as audit_mod

    chosen = list(pairs)[:sample] if sample else list(pairs)
    if not chosen:
        raise ExtractionError("label_value_histogram: no pairs given")
    pooled: dict = {}
    total = 0
    for _, mask_path in chosen:
        arr = audit_mod.read_array(mask_path)
        arr3 = np.repeat(arr[:, :, None], 3, axis=2) if arr.ndim == 2 else arr[:, :, :3]
        flat = arr3.reshape(-1, 3)
        values, counts = np.unique(flat, axis=0, return_counts=True)
        total += int(flat.shape[0])
        for v, c in zip(values.tolist(), counts.tolist()):
            key = tuple(v)
            pooled[key] = pooled.get(key, 0) + int(c)
    return sorted(([list(c), n / total] for c, n in pooled.items()), key=lambda t: -t[1])


def _skeleton_widths(mask: np.ndarray) -> np.ndarray:
    """2 x distance-transform at every skeleton pixel of a binary mask.

    Empty when the mask has no foreground or skeletonizes to nothing. No
    width_hint parity correction (contrast :func:`measured_line_width`):
    this is a DISTRIBUTION to look at, not one calibrated width to trust.
    """
    import cv2
    from skimage.morphology import skeletonize

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros(0, dtype=float)
    skel = skeletonize(mask)
    if not skel.any():
        return np.zeros(0, dtype=float)
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return 2.0 * dist[skel]


def class_shape_profile(class_mask: np.ndarray, gray: Optional[np.ndarray] = None,
                        thin_width_px: float = 8.0) -> dict:
    """Shape (and, given ``gray``, intensity) profile of ONE binary class mask.

    ``share_thin`` is the fraction of SKELETON pixels narrower than
    ``thin_width_px`` -- weighted by network length, which is what "is this
    class a thin line network" turns on, not by raw area (a single wide
    blob would otherwise dilute a genuinely thin network's share for no
    good reason). ``largest_cc_share`` is the single largest 8-connected
    component's share of the class's OWN area: near 1.0 means one connected
    network; well below 1.0 means many separate blobs. ``mean_inside``/
    ``mean_outside`` need ``gray`` (same shape as ``class_mask``) and are
    ``None`` without it.
    """
    from skimage.measure import label as cc_label

    mask = np.asarray(class_mask, dtype=bool)
    out = {
        "n_pixels": int(mask.sum()), "width_p50": None, "width_p95": None,
        "share_thin": None, "n_skeleton_px": 0, "largest_cc_share": None,
        "n_components": 0, "mean_inside": None, "mean_outside": None,
    }
    if not mask.any():
        return out

    widths = _skeleton_widths(mask)
    out["n_skeleton_px"] = int(widths.size)
    if widths.size:
        out["width_p50"] = float(np.percentile(widths, 50))
        out["width_p95"] = float(np.percentile(widths, 95))
        out["share_thin"] = float((widths < thin_width_px).mean())

    cc = cc_label(mask, connectivity=2)
    if cc.max() > 0:
        areas = np.bincount(cc.ravel())[1:]
        out["n_components"] = int(areas.size)
        out["largest_cc_share"] = float(areas.max() / areas.sum())

    if gray is not None:
        gray = np.asarray(gray, dtype=float)
        out["mean_inside"] = float(gray[mask].mean())
        if (~mask).any():
            out["mean_outside"] = float(gray[~mask].mean())
    return out


def pooled_class_shape_profile(pairs: Sequence, palette: Sequence, class_index: int,
                               sample: Optional[int] = None,
                               thin_width_px: float = 8.0) -> dict:
    """:func:`class_shape_profile`, pooled over a sample of (image, mask) pairs.

    Every skeleton pixel's width and every class pixel's raw intensity is
    POOLED across files before a percentile, share or mean is taken -- a
    file with more class pixels contributes proportionally more, the same
    way one big profile would, rather than an average of per-file averages
    that would weight a nearly-empty file the same as a dense one. The
    largest-component share is pooled the same way: summed component/class
    area across files, not a mean of per-file ratios.
    """
    from src import audit as audit_mod

    chosen = list(pairs)[:sample] if sample else list(pairs)
    if not chosen:
        raise ExtractionError("pooled_class_shape_profile: no pairs given")

    widths, inside, outside = [], [], []
    total_class_px, total_px = 0, 0
    largest_cc_px_sum, n_components_sum = 0, 0
    n_files_with_class = 0
    for img_path, mask_path in chosen:
        mask = audit_mod.read_array(mask_path)
        labels, _, _ = snap_to_palette(mask, palette)
        class_mask = labels == class_index
        total_class_px += int(class_mask.sum())
        total_px += int(class_mask.size)
        if not class_mask.any():
            continue
        n_files_with_class += 1

        w = _skeleton_widths(class_mask)
        if w.size:
            widths.append(w)

        from skimage.measure import label as cc_label
        cc = cc_label(class_mask, connectivity=2)
        if cc.max() > 0:
            areas = np.bincount(cc.ravel())[1:]
            largest_cc_px_sum += int(areas.max())
            n_components_sum += int(areas.size)

        gray = audit_mod.read_array(img_path)
        gray = gray[..., :3].mean(axis=-1) if gray.ndim == 3 else gray.astype(float)
        inside.append(gray[class_mask])
        if (~class_mask).any():
            outside.append(gray[~class_mask])

    all_widths = np.concatenate(widths) if widths else np.zeros(0, dtype=float)
    all_inside = np.concatenate(inside) if inside else np.zeros(0, dtype=float)
    all_outside = np.concatenate(outside) if outside else np.zeros(0, dtype=float)
    return {
        "n_files_sampled": len(chosen),
        "n_files_with_class": n_files_with_class,
        "pixel_share": (total_class_px / total_px) if total_px else None,
        "width_p50": float(np.percentile(all_widths, 50)) if all_widths.size else None,
        "width_p95": float(np.percentile(all_widths, 95)) if all_widths.size else None,
        "share_thin": (float((all_widths < thin_width_px).mean())
                      if all_widths.size else None),
        "n_skeleton_px": int(all_widths.size),
        "largest_cc_share": ((largest_cc_px_sum / total_class_px)
                             if total_class_px else None),
        "n_components_total": int(n_components_sum),
        "mean_inside": float(all_inside.mean()) if all_inside.size else None,
        "mean_outside": float(all_outside.mean()) if all_outside.size else None,
    }


def classify_line_class_components(labels: np.ndarray, class_index: int,
                                   blob_thickness_px: float) -> tuple:
    """Split one palette class's connected components into LINE (thin) and
    BLOB (thick) by each component's OWN median local width
    (:func:`measured_line_width`, which a lone pixel or short spur still
    answers sensibly: distance-to-background 1, width ~1, correctly thin).

    Returns ``(line_mask, blob_mask, widths_by_component)`` -- the last a
    ``{component_id: width_px}`` dict for a diagnostic to inspect the split
    it produced, not just trust it.
    """
    from skimage.measure import label as cc_label

    class_mask = np.asarray(labels) == class_index
    line_mask = np.zeros_like(class_mask)
    blob_mask = np.zeros_like(class_mask)
    widths_by_component = {}
    if not class_mask.any():
        return line_mask, blob_mask, widths_by_component

    cc = cc_label(class_mask, connectivity=2)
    for cid in range(1, int(cc.max()) + 1):
        comp = cc == cid
        width = measured_line_width(comp)
        widths_by_component[cid] = width
        if width is not None and width <= float(blob_thickness_px):
            line_mask |= comp
        else:
            blob_mask |= comp
    return line_mask, blob_mask, widths_by_component


def mode_a_raw_from_line_class(labels: np.ndarray, class_index: int,
                               blob_thickness_px: float) -> np.ndarray:
    """The MODE-A-style raw boundary for a palette class that IS ITSELF the
    painted boundary, not a phase -- e.g. Steel2's dark class (see
    ``reports/steel2_gt_mode_check.md``), where MODE B's ``find_boundaries``
    draws a loop on BOTH edges of every line segment and traps the line's
    own pixels as a spurious thin "region" between them.

    Every connected component of the class is classified LINE or BLOB by
    :func:`classify_line_class_components` first:

    - LINE (width <= ``blob_thickness_px``): its own pixels ARE the
      boundary already -- passed straight through, unskeletonized, exactly
      the way ``apply_hsv_window``'s colour-hit output feeds real MODE A's
      shared :func:`clean_boundary` (which does its own despeckle, CLOSE,
      skeletonize and redilate). Skeletonizing here first would just be
      redone by clean_boundary a moment later.
    - BLOB (thicker): a genuinely filled phase region -- skeletonizing it
      would fabricate a boundary through its interior, colinear with
      nothing painted in the raw image. Left as a phase instead: it gets
      the SAME outline :func:`boundaries_from_labels` already gives every
      MODE B phase, via a two-value map of ``{this blob, everything else}``.

    Returns the raw (pre-cleanup) binary array; pass it through
    :func:`clean_boundary` for a result comparable to MODE B's own ``raw``.
    """
    line_mask, blob_mask, _ = classify_line_class_components(
        labels, class_index, blob_thickness_px)
    raw = line_mask.copy()
    if blob_mask.any():
        raw |= boundaries_from_labels(blob_mask.astype(np.int32))
    return raw


# --------------------------------------------------------------------------
# per-folder extraction
# --------------------------------------------------------------------------
def _save_png(path: Path, image: np.ndarray) -> None:
    from PIL import Image

    values = np.unique(image)
    if not set(values.tolist()) <= {0, 255}:
        raise ExtractionError(
            f"refusing to write {path.name}: values {values.tolist()[:8]} are "
            "not strictly 0/255."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8), mode="L").save(path)


def extract_folder(
    folder_report: dict,
    mode: str,
    out_dir: Path,
    settings: dict,
    limit: Optional[int] = None,
    progress: Optional[Callable] = None,
    hsv_window: Optional[dict] = None,
) -> dict:
    """Extract every pair of one folder. Returns the folder's statistics."""
    name = folder_report["folder"]
    pairing = audit_mod.discover_pairs(Path(folder_report["path"]))
    pairs = pairing["pairs"]
    if limit:
        pairs = pairs[:limit]

    excluded_names = {str(n) for n in (settings["exclusions"] or {}).get(name, [])}
    excluded, kept = [], []
    for img_path, mask_path in pairs:
        if mask_path.name in excluded_names or img_path.name in excluded_names:
            excluded.append({"pair": [img_path.name, mask_path.name],
                             "why": "listed in boundary_gt.exclusions"})
        else:
            kept.append((img_path, mask_path))

    palette, window = None, hsv_window
    watch = watched_colours(name, settings)
    folded_idx = []
    min_region_area = int((settings["min_region_area_px"] or {}).get(name, 0))
    if mode == "B":
        palette = derive_palette(folder_report, settings)
        folded_idx = fold_targets(palette, name, settings)
    else:
        window = window or derive_hsv_window(folder_report, kept, settings)
        if min_region_area:
            raise ExtractionError(
                f"boundary_gt.min_region_area_px is set for {name!r} but it "
                "is a MODE A dataset; region-area merging is defined on the "
                "phase-label map MODE B produces, not on painted colour.")

    processed, rejected, reconciled = [], [], []
    out_dir = Path(out_dir) / name
    iterator = progress(kept, desc=name) if progress else kept
    for img_path, mask_path in iterator:
        try:
            mask = audit_mod.read_array(mask_path)
            image_meta = audit_mod.read_meta(img_path)
            mask, size_info, size_error = reconcile_size(
                mask,
                (image_meta["width"], image_meta["height"]),
                int(settings["size_tolerance_px"]),
            )
            if size_error:
                rejected.append({"pair": [img_path.name, mask_path.name],
                                 "why": size_error})
                continue
            if size_info:
                reconciled.append({"pair": [img_path.name, mask_path.name],
                                   **size_info})

            if mode == "B":
                labels, unsnapped, n_levels = snap_to_palette(mask, palette)
                if n_levels > len(palette):
                    rejected.append({"pair": [img_path.name, mask_path.name],
                                     "why": f"snapped to {n_levels} levels, more "
                                            f"than the K={len(palette)} palette "
                                            "colours"})
                    continue
                if unsnapped > float(settings["max_unsnapped_fraction"]):
                    rejected.append({"pair": [img_path.name, mask_path.name],
                                     "why": f"{unsnapped:.3f} of pixels sit far "
                                            "from every palette colour "
                                            f"(limit {settings['max_unsnapped_fraction']})"})
                    continue
                if folded_idx:
                    # Fold suspected annotation colours into the background
                    # class so no boundary loop is drawn around them.
                    labels[np.isin(labels, folded_idx)] = 0
                    n_levels = int(len(np.unique(labels)))
                if n_levels < 2:
                    rejected.append({"pair": [img_path.name, mask_path.name],
                                     "why": "mask carries a single label: no "
                                            "boundary exists in it"})
                    continue

                region_merge = None
                if min_region_area:
                    raw_before_merge = boundaries_from_labels(labels)
                    labels, region_merge = merge_small_regions(
                        labels, min_region_area)
                    n_levels = int(len(np.unique(labels)))
                    if n_levels < 2:
                        rejected.append({
                            "pair": [img_path.name, mask_path.name],
                            "why": "min_region_area_px merged every phase "
                                   "into one label: no boundary would remain"})
                        continue
                raw = boundaries_from_labels(labels)
                if region_merge is not None:
                    region_merge["boundary_px_before"] = int(raw_before_merge.sum())
                    region_merge["boundary_px_after"] = int(raw.sum())
                    removed = region_merge["boundary_px_before"] - region_merge["boundary_px_after"]
                    region_merge["boundary_px_removed"] = int(removed)
                    region_merge["boundary_px_removed_fraction"] = (
                        float(removed) / region_merge["boundary_px_before"]
                        if region_merge["boundary_px_before"] else 0.0)
            else:
                unsnapped = None
                region_merge = None
                raw = apply_hsv_window(mask, window)

            out, frac_before, frac_after, stages = clean_boundary(raw, settings)
            if frac_after <= 0.0:
                dead = next((k for k, v in stages.items() if v == 0.0), "unknown")
                rejected.append({"pair": [img_path.name, mask_path.name],
                                 "why": "cleanup left no boundary pixels; "
                                        f"foreground died at '{dead}' "
                                        f"(stage fractions {stages})"})
                continue

            artifacts = [artifact_boundary_overlap(mask, c, out) for c in watch]

            _save_png(out_dir / (img_path.stem + ".png"), out)
            processed.append({
                "pair": [img_path.name, mask_path.name],
                "output": img_path.stem + ".png",
                "artifacts": artifacts,
                "size_reconciled": size_info,
                "fraction_before": frac_before,
                "fraction_after": frac_after,
                "stages": stages,
                "unsnapped": unsnapped,
                "region_merge": region_merge,
            })
        except ExtractionError:
            raise
        except Exception as exc:
            rejected.append({"pair": [img_path.name, mask_path.name],
                             "why": f"{type(exc).__name__}: {exc}"})

    return {
        "folder": name,
        "mode": mode,
        "out_dir": str(out_dir),
        "n_pairs": len(pairs),
        "n_processed": len(processed),
        "n_rejected": len(rejected),
        "n_excluded": len(excluded),
        "palette": [list(c) for c in palette] if palette else None,
        "k": len(palette) if palette else None,
        "hsv_window": window,
        "fraction_before": audit_mod._minmedmax([p["fraction_before"] for p in processed]),
        "fraction_after": audit_mod._minmedmax([p["fraction_after"] for p in processed]),
        "fraction_after_open": audit_mod._minmedmax(
            [p["stages"]["after_open"] for p in processed]),
        "fraction_after_close": audit_mod._minmedmax(
            [p["stages"]["after_close"] for p in processed]),
        "artifacts": _summarise_artifacts(watch, processed, folded_idx, palette),
        "folded_colours": [list(palette[i]) for i in folded_idx] if palette else [],
        "region_merge": _summarise_region_merge(processed, min_region_area),
        "excluded": excluded,
        "rejected": rejected,
        "reconciled": reconciled,
        "n_reconciled": len(reconciled),
        "files": processed,
    }


def _summarise_artifacts(watch, processed, folded_idx, palette) -> list:
    """Per watched colour: how often it appears and how much boundary it causes."""
    folded = {tuple(_as_rgb(palette[i])) for i in folded_idx} if palette else set()
    out = []
    for colour in watch:
        rows = [a for rec in processed for a in rec["artifacts"]
                if tuple(a["colour"]) == tuple(colour)]
        present = [r for r in rows if r["present"]]
        out.append({
            "colour": [int(v) for v in colour],
            "folded_into_background": tuple(colour) in folded,
            "files_present": len(present),
            "files_checked": len(rows),
            "pixel_fraction": audit_mod._minmedmax([r["fraction"] for r in present]),
            "share_of_boundary": audit_mod._minmedmax(
                [r["share_of_boundary"] for r in present]),
        })
    return out


def _summarise_region_merge(processed: list, min_region_area: int) -> dict:
    """Folder-level roll-up of ``merge_small_regions``: how much of the
    speckle this run actually removed, printable straight into
    ``reports/gt_extraction.md`` without a caller re-deriving it per file.

    ``min_region_area == 0`` (the default -- off) returns
    ``{"enabled": False}`` regardless of what ``processed`` holds, so a
    dataset that never set ``min_region_area_px`` reports itself untouched
    even before any per-file record is inspected.
    """
    if not min_region_area:
        return {"enabled": False, "min_area_px": 0}
    merges = [rec["region_merge"] for rec in processed if rec.get("region_merge")]
    if not merges:
        return {"enabled": True, "min_area_px": int(min_region_area),
                "n_files": 0, "n_components_merged_total": 0,
                "n_pixels_reassigned_total": 0,
                "boundary_px_removed_fraction": None}
    removed_fracs = [m["boundary_px_removed_fraction"] for m in merges]
    return {
        "enabled": True,
        "min_area_px": int(min_region_area),
        "n_files": len(merges),
        "n_components_total": sum(m["n_components_total"] for m in merges),
        "n_components_merged_total": sum(m["n_components_merged"] for m in merges),
        "n_pixels_reassigned_total": sum(m["n_pixels_reassigned"] for m in merges),
        "boundary_px_removed_fraction": audit_mod._minmedmax(removed_fracs),
    }


def extract_all(
    audit: dict,
    out_root: Path,
    settings: dict,
    mode_overrides: Optional[dict] = None,
    datasets: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    progress: Optional[Callable] = None,
) -> dict:
    """Extract every audited folder in the mode the audit assigned it."""
    modes = folder_modes(audit, mode_overrides)
    names = [n for n in audit["datasets"] if not datasets or n in set(datasets)]
    if not names:
        raise ExtractionError(
            f"no folder matches {datasets}; audited: {', '.join(audit['datasets'])}"
        )

    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "audit_generated_utc": audit.get("generated_utc"),
        "out_root": str(out_root),
        "settings": {k: settings[k] for k in sorted(settings)},
        "mode_overrides": dict(mode_overrides or {}),
        "sane_fraction_band": list(sane_fraction_band(settings)),
        "datasets": {},
        "failures": {},
    }
    for name in names:
        try:
            report["datasets"][name] = extract_folder(
                audit["datasets"][name], modes[name], Path(out_root), settings,
                limit=limit, progress=progress)
        except Exception as exc:
            report["failures"][name] = f"{type(exc).__name__}: {exc}"
    if not report["datasets"]:
        raise ExtractionError("every folder failed: "
                              + json.dumps(report["failures"], indent=2))
    return report


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------
def write_hsv_ranges(report: dict, configs_dir: Optional[Path] = None) -> Path:
    """Write configs/hsv_ranges.yaml, preserving any hand-tuned window.

    An entry carrying ``manual: true`` is never overwritten -- that is the
    point of the file: derive once, then tune by hand and keep the tuning.
    """
    import yaml

    configs_dir = Path(configs_dir or (REPO_ROOT / "configs"))
    configs_dir.mkdir(parents=True, exist_ok=True)
    path = configs_dir / "hsv_ranges.yaml"

    existing = {}
    if path.is_file():
        loaded = yaml.safe_load(path.read_text()) or {}
        existing = loaded.get("folders") or {}

    folders = dict(existing)
    kept_manual = []
    for name, d in report["datasets"].items():
        if d["mode"] != "A" or not d.get("hsv_window"):
            continue
        if existing.get(name, {}).get("manual"):
            kept_manual.append(name)
            continue
        folders[name] = d["hsv_window"]

    header = (
        "# HSV windows for MODE A folders, in OpenCV units: H 0-179, S/V 0-255.\n"
        "#\n"
        "# DERIVED, not hardcoded: each window is the 2nd/98th percentile of H,\n"
        "# S and V over the pixels that actually carry the folder's painted\n"
        "# boundary colour, measured by src/boundary_gt.py.\n"
        "#\n"
        "# To hand-tune: edit lower/upper and set  manual: true  on that folder.\n"
        "# A folder marked manual is never overwritten by a later extraction.\n"
        "# Widen the window when boundaries come out broken; tighten it when\n"
        "# neighbouring phases bleed into the boundary map.\n"
    )
    if not folders:
        header += (
            "#\n"
            "# No folder is currently MODE A -- every dataset in reports/audit.json\n"
            "# is a phase-label map -- so there is no window to derive yet. This\n"
            "# file exists so that the moment a folder is MODE A (a new dataset, or\n"
            "# an override in notebooks/02_boundary_gt.ipynb) its window lands here.\n"
        )
    path.write_text(header + yaml.safe_dump({"folders": folders}, sort_keys=True))
    if kept_manual:
        print(f"  kept hand-tuned HSV windows for: {', '.join(kept_manual)}")
    return path


def write_extraction_report(report: dict, reports_dir: Optional[Path] = None) -> tuple:
    """Write reports/gt_extraction.md and its machine-readable twin."""
    reports_dir = Path(reports_dir or (REPO_ROOT / "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    md_path = reports_dir / "gt_extraction.md"
    json_path = reports_dir / "gt_extraction.json"

    slim = json.loads(json.dumps(report))
    for d in slim["datasets"].values():
        d.pop("files", None)          # per-file rows stay out of the tracked report
    json_path.write_text(json.dumps(slim, indent=2))
    md_path.write_text(render_markdown(report))
    return md_path, json_path


def _fmt(value, spec: str = ".4f") -> str:
    return "-" if value is None else format(value, spec)


def render_markdown(report: dict) -> str:
    s = report["settings"]
    lines = [
        "# Boundary ground truth extraction",
        "",
        f"- generated: {report['generated_utc']}",
        f"- from audit: {report['audit_generated_utc']}",
        f"- output root: `{report['out_root']}`",
        f"- line width: {s['line_width_px']} px, speckle removal "
        f"'{s['open_mode']}' at {s['open_kernel']}x{s['open_kernel']}, "
        f"CLOSE {s['close_kernel']}x{s['close_kernel']}",
        "- speckle removal drops connected components smaller than "
        f"{int(s['open_kernel']) ** 2} px. A literal 3x3 morphological OPEN "
        "would erase the whole map: its erosion needs a full 3x3 block of "
        "foreground, and a boundary line is 1-3 px wide. Set "
        "`boundary_gt.open_mode: morph` to force the literal version.",
        f"- size tolerance: {s['size_tolerance_px']} px — a mask within this "
        "many pixels of its image is centre-cropped to the common size, not "
        "rejected. Cropped, never resized.",
        f"- mode overrides applied: {report['mode_overrides'] or 'none'}",
        f"- artifact colour folding: "
        f"{'ON' if s.get('fold_artifact_colours') else 'off'} "
        f"(watched: {s.get('artifact_colours') or 'none'})",
        "",
        "MODE A extracts a painted boundary colour by HSV thresholding and "
        "yields phase interfaces AND grain boundaries. MODE B derives "
        "boundaries from a phase-label map with `find_boundaries` and yields "
        "phase interfaces ONLY -- grain boundaries are absent from the source "
        "masks and are not invented here.",
        "",
        "| folder | mode | K | processed | reconciled | rejected | excluded | frac before | frac after |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, d in report["datasets"].items():
        lines.append(
            f"| {name} | **{d['mode']}** | {d['k'] or '-'} | {d['n_processed']} | "
            f"{d.get('n_reconciled', 0)} | "
            f"{d['n_rejected']} | {d['n_excluded']} | "
            f"{_fmt(d['fraction_before']['median'])} | "
            f"{_fmt(d['fraction_after']['median'])} |"
        )
    if report["failures"]:
        lines += ["", "### Folders that could not be extracted", ""]
        lines += [f"- **{k}**: {v}" for k, v in report["failures"].items()]

    for name, d in report["datasets"].items():
        lines += [
            "", f"## {name}", "",
            f"- mode: **{d['mode']}**"
            + ("  (overridden)" if name in report["mode_overrides"] else ""),
            f"- output: `{d['out_dir']}`",
            f"- pairs: {d['n_pairs']} -> processed {d['n_processed']} "
            f"({d.get('n_reconciled', 0)} size-reconciled), "
            f"rejected {d['n_rejected']}, excluded {d['n_excluded']}",
            f"- boundary pixel fraction before cleanup (min/median/max): "
            f"{_fmt(d['fraction_before']['min'])}/"
            f"{_fmt(d['fraction_before']['median'])}/"
            f"{_fmt(d['fraction_before']['max'])}",
            f"- boundary pixel fraction after cleanup (min/median/max): "
            f"{_fmt(d['fraction_after']['min'])}/"
            f"{_fmt(d['fraction_after']['median'])}/"
            f"{_fmt(d['fraction_after']['max'])}",
            f"- intermediate medians: after speckle removal "
            f"{_fmt(d['fraction_after_open']['median'])}, after CLOSE "
            f"{_fmt(d['fraction_after_close']['median'])}",
        ]
        if d["mode"] == "B":
            lines += [
                f"- K = {d['k']} palette classes: "
                + ", ".join(f"`{c}`" for c in (d["palette"] or [])),
                "- K is the number of colour *peaks*, not the raw unique-colour "
                "count: colours within "
                f"{report['settings']['palette_merge_distance']} RGB of a heavier "
                "colour are anti-aliasing or JPEG ramp values and snap to it.",
            ]
        elif d.get("hsv_window"):
            w = d["hsv_window"]
            lines += [
                f"- boundary colour `{w['colour_rgb']}`, window derived from "
                f"{w['n_pixels_sampled']} pixels at percentiles {w['percentiles']}",
                f"- HSV lower {[round(v, 1) for v in w['lower']]}, "
                f"upper {[round(v, 1) for v in w['upper']]}"
                + (" (hue wraps)" if w.get("wraps_hue") else ""),
            ]
        rm = d.get("region_merge") or {}
        if rm.get("enabled"):
            frac = rm.get("boundary_px_removed_fraction") or {}
            lines += [
                "", "### Small-region cleanup (`boundary_gt.min_region_area_px`)", "",
                f"- threshold: {rm['min_area_px']} px -- every phase component "
                "smaller than this was merged into its surrounding phase (the "
                "majority label along its border) before boundary extraction, "
                "so it draws no boundary loop of its own",
                f"- {rm.get('n_components_merged_total', 0)} of "
                f"{rm.get('n_components_total', 0)} components merged across "
                f"{rm.get('n_files', 0)} files "
                f"({rm.get('n_pixels_reassigned_total', 0)} px reassigned)",
                f"- boundary pixels removed by the merge (min/median/max of the "
                f"per-file fraction): {_fmt(frac.get('min'))}/"
                f"{_fmt(frac.get('median'))}/{_fmt(frac.get('max'))}",
                "- the region-count/area-percentile before/after and the "
                "4-tile visual live in notebooks/02_boundary_gt.ipynb, not "
                "here: they need the extracted PNGs on disk, which this "
                "report only describes.",
            ]
        if d.get("artifacts"):
            lines += [
                "", "### Watched artifact colours", "",
                "A colour kept as its own class gets a closed boundary loop drawn "
                "around every region of it. `share_of_boundary` is how much of "
                "this folder's extracted boundary lies on that colour's outline "
                "-- the cost of treating it as a phase.",
                "",
                "| colour | folded | present in | pixel frac (med) | share of boundary (med) |",
                "| --- | --- | --- | --- | --- |",
            ]
            for a in d["artifacts"]:
                lines.append(
                    f"| `{a['colour']}` | {'yes' if a['folded_into_background'] else 'no'} | "
                    f"{a['files_present']}/{a['files_checked']} | "
                    f"{_fmt(a['pixel_fraction']['median'])} | "
                    f"{_fmt(a['share_of_boundary']['median'])} |"
                )
        if d.get("reconciled"):
            lines += [
                "", f"### Size-reconciled pairs ({d['n_reconciled']})", "",
                "Mask and image dimensions disagreed by no more than the "
                f"{s['size_tolerance_px']} px tolerance, so both were "
                "centre-cropped to their common size and the pair was kept. The "
                "boundary PNG is written at the final size; `image crop` is the "
                "identical crop the raw image needs when it is loaded.",
                "",
                "| pair | image | mask | final | image crop (l,t,w,h) |",
                "| --- | --- | --- | --- | --- |",
            ]
            for rec in d["reconciled"]:
                c = rec["image_crop"]
                lines.append(
                    f"| `{rec['pair'][1]}` | {rec['image_size'][0]}x{rec['image_size'][1]} "
                    f"| {rec['mask_size'][0]}x{rec['mask_size'][1]} "
                    f"| {rec['final_size'][0]}x{rec['final_size'][1]} "
                    f"| {c['left']},{c['top']},{c['width']},{c['height']} |"
                )
        if d["excluded"]:
            lines += ["", "### Excluded before processing", ""]
            lines += [f"- `{e['pair'][1]}` — {e['why']}" for e in d["excluded"]]
        if d["rejected"]:
            lines += ["", f"### Rejected during processing ({d['n_rejected']})", ""]
            lines += [f"- `{r['pair'][1]}` — {r['why']}" for r in d["rejected"][:50]]
            if d["n_rejected"] > 50:
                lines.append(f"- ... and {d['n_rejected'] - 50} more")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# mode-check report (reports/steel2_gt_mode_check.{md,json})
#
# Diagnostic-only, read-only: the payload is assembled by the notebook cell
# from the functions above plus train.true_regions_from_boundary (a step-6
# function; kept out of this step-2 module so boundary_gt never depends on
# train), and this just persists it. Nothing here writes a boundary PNG,
# touches a manifest, or changes what extract_folder produces.
# --------------------------------------------------------------------------
def render_mode_check_markdown(payload: dict) -> str:
    lines = [
        "# GT mode check: is a MODE B palette class actually a painted line?",
        "",
        f"- generated: {payload['generated_utc']}",
        f"- settings: {payload['settings']}",
        "",
        "MODE B assumes every palette class is a PHASE (an area with an inside) and "
        "draws `find_boundaries` around it. Nothing checks that assumption. This "
        "report measures, per dataset, whether the class actually looks like a thin "
        "painted line instead -- which `find_boundaries` would still happily outline "
        "on both edges, trapping the line's own pixels as a spurious strip \"region\".",
    ]

    for name, d in payload["datasets"].items():
        lines += [
            "", f"## {name}", "",
            f"- K = {d['k']} palette classes: "
            + ", ".join(f"`{c}`" for c in d["palette"]),
            f"- sampled {d['n_files_sampled']} mask files fresh from disk",
            "",
            "Raw label alphabet (top values, exact-match pixel share, before any "
            "palette-merge snapping):",
            "",
            "| value | pixel share |", "| --- | --- |",
        ]
        for value, share in d["label_value_histogram_top"]:
            lines.append(f"| `{value}` | {share:.4f} |")

        lines += [
            "", "Per-palette-class shape/intensity profile:", "",
            "| class idx | colour | pixel share | width p50 | width p95 | "
            "share thin | components | largest CC share | mean inside | mean outside |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for idx, prof in d["class_profiles"].items():
            colour = d["palette"][int(idx)]
            lines.append(
                f"| {idx} | `{colour}` | {_fmt(prof['pixel_share'])} | "
                f"{_fmt(prof['width_p50'], '.2f')} | {_fmt(prof['width_p95'], '.2f')} | "
                f"{_fmt(prof['share_thin'], '.2%')} | {prof['n_components_total']} | "
                f"{_fmt(prof['largest_cc_share'], '.2%')} | "
                f"{_fmt(prof['mean_inside'], '.1f')} | {_fmt(prof['mean_outside'], '.1f')} |"
            )

    v = payload["verdict"]
    lines += [
        "", "## Verdict", "",
        f"- **{v['dataset']}** class {v['class_index']} (colour `{v['colour']}`): "
        + ("LINE-LIKE" if v["line_like"] else "not line-like"),
        f"- {v['reason']}",
    ]

    mc = payload.get("mode_comparison")
    if mc:
        lines += [
            "", "## MODE B vs proposed MODE A -- side by side", "",
            "Proposed MODE A: this class's own pixels ARE the boundary (thin "
            "components passed straight through to the shared cleanup, unskeletonized "
            "here since clean_boundary skeletonizes anyway; thick components keep "
            "MODE B's own outline treatment -- see "
            "`boundary_gt.mode_a_raw_from_line_class`). Nothing regenerated on disk: "
            "this is computed in memory for comparison only.",
            "", "### The 4 gallery tiles (most GT regions under current MODE B)", "",
            "| tile | regions (B) | regions (A) | area p50 (B) | area p50 (A) |",
            "| --- | --- | --- | --- | --- |",
        ]
        for t in mc["gallery"]:
            lines.append(
                f"| `{t['source_image']}` | {t['n_regions_mode_b']} | "
                f"{t['n_regions_mode_a']} | {_fmt(t['area_p50_mode_b'], '.0f')} | "
                f"{_fmt(t['area_p50_mode_a'], '.0f')} |"
            )

        full = mc["full_steel2"]
        lines += [
            "", f"### Full Steel2 profile ({full['n_files']} files)", "",
            "| | regions/tile | area p25 | area p50 | area p75 | share < 50 px | share < 100 px |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for label, key in (("MODE B (current)", "mode_b"), ("MODE A (proposed)", "mode_a")):
            m = full[key]
            pct = m["area_percentiles"]
            lines.append(
                f"| {label} | {m['regions_per_tile']:.1f} | {_fmt(pct.get('p25'), '.0f')} | "
                f"{_fmt(pct.get('p50'), '.0f')} | {_fmt(pct.get('p75'), '.0f')} | "
                f"{m['share_lt_50']:.1%} | {m['share_lt_100']:.1%} |"
            )
    else:
        lines += ["", "No MODE B/A comparison was run (the class did not verify as "
                      "line-like enough to warrant one)."]

    lines += [
        "", "## What this report does NOT do", "",
        "- does not regenerate any boundary PNG under `GT_BOUNDARIES_ROOT`",
        "- does not rebuild `fold_steel_combined`'s manifests or `configs/fold_stats.yaml`",
        "- does not retrain or touch any checkpoint",
        "- proposes a per-dataset MODE option; does not turn it on anywhere",
    ]
    return "\n".join(lines) + "\n"


def write_mode_check_report(payload: dict, reports_dir: Optional[Path] = None) -> tuple:
    """Write reports/steel2_gt_mode_check.{md,json}. Read-only diagnostic:
    see the module-level comment above render_mode_check_markdown."""
    reports_dir = Path(reports_dir or (REPO_ROOT / "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    md_path = reports_dir / "steel2_gt_mode_check.md"
    json_path = reports_dir / "steel2_gt_mode_check.json"
    json_path.write_text(json.dumps(payload, indent=2))
    md_path.write_text(render_mode_check_markdown(payload))
    return md_path, json_path
