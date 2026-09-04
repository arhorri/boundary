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
}

#: A boundary map covering less/more than this is not a boundary map.
SANE_FRACTION_BAND = (0.005, 0.25)


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


def measured_line_width(binary: np.ndarray) -> Optional[float]:
    """Mean thickness of a boundary map: area / skeleton length."""
    from skimage.morphology import skeletonize

    hit = binary > 0
    area = int(hit.sum())
    if area == 0:
        return None
    skel = int(skeletonize(hit).sum())
    return (float(area) / float(skel)) if skel else None


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
    if mode == "B":
        palette = derive_palette(folder_report, settings)
        folded_idx = fold_targets(palette, name, settings)
    else:
        window = window or derive_hsv_window(folder_report, kept, settings)

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
                raw = boundaries_from_labels(labels)
            else:
                unsnapped = None
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
        "sane_fraction_band": list(SANE_FRACTION_BAND),
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
