"""Dataset audit: discover image/mask pairs and classify each folder MODE A/B.

The whole downstream pipeline branches on one question per dataset folder:

    MODE A  the mask paints boundaries as their own distinct colour, so
            phase interfaces AND intra-phase grain boundaries are present
            and are extracted by HSV colour thresholding.
    MODE B  the mask is a phase-label map only. Boundaries must be derived
            with ``find_boundaries`` on the label map, which yields phase
            interfaces ONLY -- grain boundaries are simply not in the ground
            truth and cannot be recovered.

This module answers that question from evidence and records the evidence
alongside the verdict, so a human can disagree with it by reading
``reports/audit.md``. Nothing here trains or applies a model, and nothing
here writes inside ``data/`` -- masks and images are opened read-only.

Every folder's layout and naming convention is INFERRED, never assumed: the
five datasets are packaged differently from one another.
"""

from __future__ import annotations

import json
import os
import platform
import random
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------
# tunables -- every threshold that a verdict depends on is named here, and
# every one of them is echoed into audit.json so a verdict can be re-judged.
# --------------------------------------------------------------------------
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pgm")

#: directory names that mark the two halves of a pair, lowercased
MASK_DIR_HINTS = ("mask", "masks", "label", "labels", "annotation", "annotations",
                  "gt", "ground_truth", "groundtruth", "seg", "segmentation",
                  "target", "targets", "class", "classes")
IMAGE_DIR_HINTS = ("image", "images", "img", "imgs", "raw", "micrograph",
                   "micrographs", "input", "inputs", "original", "originals",
                   "data", "photo", "photos")

#: affixes tried when image and mask stems differ, longest first
AFFIX_CANDIDATES = ("_ground_truth", "_groundtruth", "_segmentation", "_annotation",
                    "_labels", "_label", "_masks", "_mask", "_seg", "_gt", "_anno",
                    "_target", "_class", "-ground_truth", "-mask", "-label", "-gt",
                    "-seg", ".mask", ".label")

SAMPLE_SIZE = 24          # pairs opened per folder; pairing stats use ALL files
SAMPLE_SEED = 0

PALETTE_TOP_N = 20        # colours reported per folder
MAX_LABELS = 32           # colours kept when building a MODE B label map
QUANTIZED_COVERAGE = 0.995   # top-MAX_LABELS coverage above this = quantized
QUANTIZED_MAX_COLOURS = 512  # more distinct colours than this = continuous

MODE_A_MIN_FRAC = 0.0005  # a painted line covers at least this much of the image
MODE_A_MAX_FRAC = 0.30    # ... and at most this much, or it is a phase, not a line
MODE_A_MAX_THICKNESS = 4.0   # px; area / skeleton length
MODE_A_MIN_SPAN = 0.50    # candidate bbox must cover this fraction of the image
MODE_A_FILE_QUORUM = 0.60 # fraction of sampled files that must agree

MAX_CANDIDATE_COLOURS = 12   # thinness is measured for at most this many colours


class AuditError(RuntimeError):
    """Raised when a folder cannot be audited. Never fails silently."""


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _holds_images(path: Path) -> bool:
    """True as soon as one image file is seen anywhere under ``path``."""
    try:
        return any(_is_image(p) for p in path.rglob("*"))
    except OSError:
        return False


def _dir_role(name: str) -> Optional[str]:
    """Classify a directory name as ``"mask"``, ``"image"`` or ``None``."""
    low = name.lower().strip("_- ")
    if any(h == low or h in low.split("_") or h in low.split("-") for h in MASK_DIR_HINTS):
        return "mask"
    if any(h == low or h in low.split("_") or h in low.split("-") for h in IMAGE_DIR_HINTS):
        return "image"
    if any(h in low for h in MASK_DIR_HINTS):
        return "mask"
    if any(h in low for h in IMAGE_DIR_HINTS):
        return "image"
    return None


def _median(values: Sequence) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return float(np.median(vals)) if vals else None


def _minmedmax(values: Sequence) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"min": None, "median": None, "max": None, "n": 0}
    arr = np.asarray(vals, dtype=float)
    return {
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


# --------------------------------------------------------------------------
# 1. layout + pairing discovery
# --------------------------------------------------------------------------
def _split_dirs(folder: Path) -> tuple:
    """Find (image_dir, mask_dir) anywhere under ``folder`` by directory name."""
    image_dirs, mask_dirs = [], []
    for root, dirnames, _ in os.walk(folder):
        for d in dirnames:
            role = _dir_role(d)
            path = Path(root) / d
            if role is None or not _holds_images(path):
                continue
            if role == "mask":
                mask_dirs.append(path)
            elif role == "image":
                image_dirs.append(path)
    if len(image_dirs) == 1 and len(mask_dirs) == 1:
        return image_dirs[0], mask_dirs[0]
    if image_dirs and mask_dirs:
        # more than one of each: prefer siblings under the shallowest parent
        image_dirs.sort(key=lambda p: len(p.parts))
        mask_dirs.sort(key=lambda p: len(p.parts))
        return image_dirs[0], mask_dirs[0]
    return None, None


def _strip_affix(stem: str, affix: str) -> str:
    if affix and stem.endswith(affix):
        return stem[: -len(affix)]
    if affix and stem.startswith(affix):
        return stem[len(affix):]
    return stem


def infer_affix(image_stems: Iterable[str], mask_stems: Iterable[str]) -> str:
    """Infer the affix that turns a mask filename into its image filename.

    Returns ``""`` when the stems already match. Chosen by counting matches,
    not by assuming any one dataset's convention.
    """
    images = set(image_stems)
    masks = list(mask_stems)
    if not images or not masks:
        return ""

    def score(affix: str) -> int:
        return sum(1 for m in masks if _strip_affix(m, affix) in images)

    best, best_score = "", score("")
    for affix in AFFIX_CANDIDATES:
        s = score(affix)
        if s > best_score:
            best, best_score = affix, s
    if best_score >= len(masks) * 0.5:
        return best

    # Nothing from the known list fits: derive the common suffix of the mask
    # stems and try every trailing slice of it.
    common = os.path.commonprefix([m[::-1] for m in masks])[::-1][-24:]
    for k in range(len(common), 0, -1):
        cand = common[-k:]
        s = score(cand)
        if s > best_score:
            best, best_score = cand, s
    return best


def discover_pairs(folder: Path) -> dict:
    """Discover the image/mask convention of one dataset folder.

    Handles two layouts: sibling image/ and mask/ directories (at any depth),
    and a single flat directory where masks are distinguished by an affix.
    Returns pairing counts over ALL files -- nothing is sampled here.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise AuditError(f"not a directory: {folder}")

    image_dir, mask_dir = _split_dirs(folder)
    if image_dir is not None and mask_dir is not None:
        layout = "split-dirs"
        images = sorted(p for p in image_dir.rglob("*") if _is_image(p))
        masks = sorted(p for p in mask_dir.rglob("*") if _is_image(p))
    else:
        layout = "flat-affix"
        every = sorted(p for p in folder.rglob("*") if _is_image(p))
        affix_hit = {p for p in every
                     if any(a in p.stem.lower() for a in AFFIX_CANDIDATES)}
        masks = sorted(affix_hit)
        images = [p for p in every if p not in affix_hit]

    if not images or not masks:
        raise AuditError(
            f"{folder.name}: could not identify both images and masks.\n"
            f"  layout guess : {layout}\n"
            f"  images found : {len(images)}\n"
            f"  masks found  : {len(masks)}\n"
            f"  subdirs      : {sorted(p.name for p in folder.iterdir() if p.is_dir())}\n"
            "Add the folder's directory name to MASK_DIR_HINTS / IMAGE_DIR_HINTS "
            "in src/audit.py, or its suffix to AFFIX_CANDIDATES."
        )

    affix = infer_affix((p.stem for p in images), (p.stem for p in masks))
    by_stem = {}
    for p in images:
        by_stem.setdefault(p.stem, p)
        by_stem.setdefault(p.stem.lower(), p)

    pairs, unmatched_masks = [], []
    matched_images = set()
    for m in masks:
        key = _strip_affix(m.stem, affix)
        img = by_stem.get(key) or by_stem.get(key.lower())
        if img is None:
            unmatched_masks.append(m.name)
        else:
            pairs.append((img, m))
            matched_images.add(img)
    unmatched_images = [p.name for p in images if p not in matched_images]

    return {
        "layout": layout,
        "image_dir": str(image_dir) if image_dir else str(folder),
        "mask_dir": str(mask_dir) if mask_dir else str(folder),
        "mask_affix": affix,
        "convention": (
            f"{layout}: mask stem == image stem"
            + (f" + '{affix}'" if affix else "")
        ),
        "n_images": len(images),
        "n_masks": len(masks),
        "n_pairs": len(pairs),
        "unmatched_images": sorted(unmatched_images),
        "unmatched_masks": sorted(unmatched_masks),
        "pairs": sorted(pairs, key=lambda t: t[1].name),
    }


# --------------------------------------------------------------------------
# 2. reading
# --------------------------------------------------------------------------
def read_meta(path: Path) -> dict:
    """File-level metadata, read without decoding pixel data."""
    from PIL import Image

    with Image.open(path) as im:
        width, height = im.size
        return {
            "name": path.name,
            "format": im.format,
            "pil_mode": im.mode,
            "width": int(width),
            "height": int(height),
        }


def read_array(path: Path) -> np.ndarray:
    """Read an image as an array, exactly as stored. Never resized.

    Palette images are expanded to RGB because the palette entries are the
    semantic colours. Alpha is dropped. The file is never written to.
    """
    from PIL import Image

    with Image.open(path) as im:
        if im.mode == "P":
            im = im.convert("RGB")
        elif im.mode in ("RGBA", "LA"):
            im = im.convert("RGB" if im.mode == "RGBA" else "L")
        arr = np.asarray(im)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    return arr


def _bit_depth(arr: np.ndarray) -> int:
    return int(arr.dtype.itemsize * 8)


def _colour_mode(arr: np.ndarray) -> str:
    if arr.ndim == 2:
        return "grayscale"
    if arr.shape[2] == 3:
        flat = arr.reshape(-1, 3)
        step = max(1, flat.shape[0] // 100000)
        sub = flat[::step]
        if np.array_equal(sub[:, 0], sub[:, 1]) and np.array_equal(sub[:, 1], sub[:, 2]):
            return "rgb-but-gray"
        return "rgb"
    return f"{arr.shape[2]}-channel"


# --------------------------------------------------------------------------
# 3. palette
# --------------------------------------------------------------------------
def colour_counts(arr: np.ndarray) -> tuple:
    """Return (colours, counts) with colours as tuples, most frequent first."""
    if arr.ndim == 2:
        vals, counts = np.unique(arr.ravel(), return_counts=True)
        colours = [(int(v),) for v in vals]
    elif arr.dtype == np.uint8 and arr.shape[2] == 3:
        flat = arr.reshape(-1, 3).astype(np.int64)
        packed = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
        vals, counts = np.unique(packed, return_counts=True)
        colours = [(int(v >> 16), int((v >> 8) & 255), int(v & 255)) for v in vals]
    else:
        flat = arr.reshape(-1, arr.shape[2])
        vals, counts = np.unique(flat, axis=0, return_counts=True)
        colours = [tuple(int(c) for c in row) for row in vals]
    order = np.argsort(-counts)
    return [colours[i] for i in order], counts[order]


def palette_report(arr: np.ndarray, top_n: int = PALETTE_TOP_N) -> dict:
    """Exact unique-colour count, the top colours, and a quantization verdict."""
    colours, counts = colour_counts(arr)
    total = int(arr.shape[0] * arr.shape[1])
    top = [
        {"colour": list(c), "pixels": int(n), "fraction": float(n) / total}
        for c, n in zip(colours[:top_n], counts[:top_n])
    ]
    head_coverage = float(counts[:MAX_LABELS].sum()) / total
    n_unique = len(colours)
    quantized = n_unique <= MAX_LABELS or (
        head_coverage >= QUANTIZED_COVERAGE and n_unique <= QUANTIZED_MAX_COLOURS
    )
    return {
        "n_unique_colours": int(n_unique),
        "top_colours": top,
        "top_labels_coverage": head_coverage,
        "quantized": bool(quantized),
        "continuous_evidence": (
            None if quantized else
            f"{n_unique} distinct colours, top {MAX_LABELS} cover only "
            f"{head_coverage:.4f} of pixels -- consistent with JPEG contamination"
        ),
    }


def label_map(arr: np.ndarray, max_labels: int = MAX_LABELS) -> tuple:
    """Quantize a mask to a small integer label map.

    Every pixel is assigned to the nearest of the ``max_labels`` most frequent
    colours (nearest in RGB, or in value for grayscale). For an already
    quantized mask this is exact; for a JPEG-contaminated one it is the
    documented repair. No model is involved.
    """
    colours, counts = colour_counts(arr)
    keep = colours[:max_labels]
    flat = (arr.reshape(-1, 1) if arr.ndim == 2
            else arr.reshape(-1, arr.shape[2])).astype(np.int32)
    centres = np.array([list(c) for c in keep], dtype=np.int32)

    if len(colours) <= max_labels:
        # Already quantized: assign exactly, no distance search needed.
        index = {c: i for i, c in enumerate(keep)}
        labels = np.empty(flat.shape[0], dtype=np.int32)
        for i, c in enumerate(keep):
            hit = np.all(flat == centres[i][None, :], axis=1)
            labels[hit] = index[c]
    else:
        # Continuous (JPEG-contaminated): nearest of the dominant colours,
        # in chunks so a large mask cannot blow up memory.
        labels = np.empty(flat.shape[0], dtype=np.int32)
        chunk = 65536
        for start in range(0, flat.shape[0], chunk):
            block = flat[start:start + chunk]
            dist = ((block[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
            labels[start:start + chunk] = np.argmin(dist, axis=1)
    labels = labels.reshape(arr.shape[:2])
    return labels, [tuple(int(v) for v in c) for c in keep]


# --------------------------------------------------------------------------
# 4. MODE A evidence
# --------------------------------------------------------------------------
def _rgb_to_hsv(colour: Sequence) -> tuple:
    import colorsys

    if len(colour) == 1:
        return (0.0, 0.0, colour[0] / 255.0)
    r, g, b = (c / 255.0 for c in colour[:3])
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return (h * 360.0, s, v)


def colour_geometry(mask_of_colour: np.ndarray) -> dict:
    """Geometric evidence for 'is this colour a painted line?'.

    ``mean_thickness`` is area / skeleton length: ~1-3 px for a drawn
    boundary, large for a filled phase region. ``span`` is how much of the
    image the colour's bounding box covers: a boundary network spans the
    whole frame, a corner artefact does not.
    """
    from skimage.measure import label as cc_label
    from skimage.morphology import skeletonize

    area = int(mask_of_colour.sum())
    h, w = mask_of_colour.shape
    out = {
        "pixels": area,
        "fraction": float(area) / float(h * w),
        "skeleton_pixels": 0,
        "skeleton_to_area": None,
        "mean_thickness_px": None,
        "n_components": 0,
        "span": 0.0,
    }
    if area == 0:
        return out
    skel = skeletonize(mask_of_colour)
    skel_px = int(skel.sum())
    out["skeleton_pixels"] = skel_px
    if skel_px:
        out["skeleton_to_area"] = float(skel_px) / float(area)
        out["mean_thickness_px"] = float(area) / float(skel_px)
    out["n_components"] = int(cc_label(mask_of_colour, connectivity=2).max())
    ys, xs = np.nonzero(mask_of_colour)
    bbox = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    out["span"] = float(bbox) / float(h * w)
    return out


def mode_a_candidates(arr: np.ndarray) -> list:
    """Score every plausible painted-boundary colour in one mask."""
    colours, counts = colour_counts(arr)
    total = float(arr.shape[0] * arr.shape[1])
    shortlist = [
        (c, n) for c, n in zip(colours, counts)
        if MODE_A_MIN_FRAC <= (n / total) <= MODE_A_MAX_FRAC
    ][:MAX_CANDIDATE_COLOURS]

    results = []
    for colour, _ in shortlist:
        if arr.ndim == 2:
            hit = arr == colour[0]
        else:
            hit = np.all(arr[..., :len(colour)] == np.array(colour), axis=-1)
        geo = colour_geometry(hit)
        hsv = _rgb_to_hsv(colour)
        thin = (geo["mean_thickness_px"] is not None
                and geo["mean_thickness_px"] <= MODE_A_MAX_THICKNESS)
        qualifies = bool(thin and geo["span"] >= MODE_A_MIN_SPAN
                         and MODE_A_MIN_FRAC <= geo["fraction"] <= MODE_A_MAX_FRAC)
        results.append({
            "colour": list(colour),
            "hsv": [round(hsv[0], 1), round(hsv[1], 3), round(hsv[2], 3)],
            "saturation": round(hsv[1], 3),
            **geo,
            "qualifies": qualifies,
        })
    results.sort(key=lambda r: (not r["qualifies"],
                                r["mean_thickness_px"] if r["mean_thickness_px"] else 1e9))
    return results


# --------------------------------------------------------------------------
# 5. MODE B evidence
# --------------------------------------------------------------------------
def mode_b_stats(labels: np.ndarray, colours: Sequence) -> dict:
    """Per-label connected components and areas, plus islands vs tessellation."""
    from skimage.measure import label as cc_label

    total = float(labels.size)
    per_label = []
    majority = None
    for idx in range(len(colours)):
        hit = labels == idx
        n_px = int(hit.sum())
        if n_px == 0:
            continue
        cc = cc_label(hit, connectivity=1)
        n_cc = int(cc.max())
        areas = np.bincount(cc.ravel())[1:] if n_cc else np.array([], dtype=int)
        entry = {
            "label": idx,
            "colour": list(colours[idx]),
            "pixels": n_px,
            "fraction": n_px / total,
            "n_components": n_cc,
            "component_area": _minmedmax(areas.tolist()),
        }
        per_label.append(entry)
        if majority is None or n_px > majority["pixels"]:
            majority = entry

    structure = None
    if majority is not None:
        structure = "islands" if majority["n_components"] <= 1 else "tessellation"
    return {
        "n_labels": len(per_label),
        "per_label": per_label,
        "majority_label": None if majority is None else majority["label"],
        "majority_n_components": None if majority is None else majority["n_components"],
        "majority_structure": structure,
    }


def boundary_fraction_mode_b(labels: np.ndarray) -> float:
    """Fraction of pixels find_boundaries would mark on the label map."""
    from skimage.segmentation import find_boundaries

    return float(find_boundaries(labels, mode="thick").mean())


# --------------------------------------------------------------------------
# 6. per-file and per-folder audit
# --------------------------------------------------------------------------
def audit_pair(image_path: Path, mask_path: Path) -> dict:
    """Audit one image/mask pair. Read-only."""
    image_meta = read_meta(image_path)
    mask_meta = read_meta(mask_path)
    mask = read_array(mask_path)
    image = read_array(image_path)

    palette = palette_report(mask)
    candidates = mode_a_candidates(mask)
    labels, label_colours = label_map(mask)
    b_stats = mode_b_stats(labels, label_colours)

    best = candidates[0] if candidates and candidates[0]["qualifies"] else None
    frac_a = best["fraction"] if best else 0.0

    return {
        "image": image_meta | {
            "bit_depth": _bit_depth(image),
            "colour_mode": _colour_mode(image),
        },
        "mask": mask_meta | {
            "bit_depth": _bit_depth(mask),
            "colour_mode": _colour_mode(mask),
        },
        "size_match": (image_meta["width"] == mask_meta["width"]
                       and image_meta["height"] == mask_meta["height"]),
        "palette": palette,
        "mode_a_candidates": candidates[:5],
        "mode_a_colour": None if best is None else best["colour"],
        "mode_b": b_stats,
        "boundary_fraction": {
            "mode_a": frac_a,
            "mode_b": boundary_fraction_mode_b(labels),
        },
    }


def audit_folder(
    folder: Path,
    sample_size: int = SAMPLE_SIZE,
    seed: int = SAMPLE_SEED,
    progress: Optional[Callable] = None,
) -> dict:
    """Audit one dataset folder: pairing over all files, pixels over a sample.

    ``progress`` is called as ``progress(sequence, desc=folder_name)`` and must
    return an iterable -- e.g. ``tqdm``. It is optional: the audit is identical
    without it.
    """
    folder = Path(folder)
    pairing = discover_pairs(folder)
    pairs = pairing.pop("pairs")
    if not pairs:
        raise AuditError(
            f"{folder.name}: {pairing['n_images']} images and "
            f"{pairing['n_masks']} masks were found but none paired under the "
            f"inferred convention '{pairing['convention']}'."
        )

    rng = random.Random(seed)
    chosen = pairs if len(pairs) <= sample_size else rng.sample(pairs, sample_size)
    chosen = sorted(chosen, key=lambda t: t[1].name)

    per_file, errors = [], []
    iterator = progress(chosen, desc=folder.name) if progress else chosen
    for img, msk in iterator:
        try:
            rec = audit_pair(img, msk)
        except Exception as exc:  # a corrupt file must not hide the rest
            errors.append({"mask": msk.name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        rec["pair"] = [img.name, msk.name]
        per_file.append(rec)

    if not per_file:
        raise AuditError(
            f"{folder.name}: every sampled pair failed to read. First error: "
            f"{errors[0]['error'] if errors else 'unknown'}"
        )

    # ---- aggregate ------------------------------------------------------
    widths = [r["image"]["width"] for r in per_file]
    heights = [r["image"]["height"] for r in per_file]

    # MODE A verdict: which colour qualifies, and in how many sampled files
    votes = Counter(tuple(r["mode_a_colour"]) for r in per_file if r["mode_a_colour"])
    quorum = MODE_A_FILE_QUORUM * len(per_file)
    winner, n_votes = (votes.most_common(1)[0] if votes else (None, 0))
    is_a = winner is not None and n_votes >= quorum

    ev = [c for r in per_file for c in r["mode_a_candidates"]
          if winner and tuple(c["colour"]) == winner]
    evidence = {
        "candidate_colour": list(winner) if winner else None,
        "candidate_hsv": ev[0]["hsv"] if ev else None,
        "files_qualifying": int(n_votes),
        "files_sampled": len(per_file),
        "quorum_required": round(quorum, 2),
        "pixel_fraction": _minmedmax([c["fraction"] for c in ev]),
        "skeleton_to_area": _minmedmax([c["skeleton_to_area"] for c in ev]),
        "mean_thickness_px": _minmedmax([c["mean_thickness_px"] for c in ev]),
        "n_components": _minmedmax([c["n_components"] for c in ev]),
        "span": _minmedmax([c["span"] for c in ev]),
        "thresholds": {
            "min_fraction": MODE_A_MIN_FRAC,
            "max_fraction": MODE_A_MAX_FRAC,
            "max_mean_thickness_px": MODE_A_MAX_THICKNESS,
            "min_span": MODE_A_MIN_SPAN,
            "file_quorum": MODE_A_FILE_QUORUM,
        },
    }
    if is_a:
        reason = (
            f"colour {list(winner)} is thin (median "
            f"{evidence['mean_thickness_px']['median']:.2f} px thick), sparse "
            f"({evidence['pixel_fraction']['median']:.4f} of pixels) and spans "
            f"the frame, in {n_votes}/{len(per_file)} sampled masks"
        )
    elif winner is not None:
        reason = (
            f"colour {list(winner)} looks line-like in only {n_votes}/"
            f"{len(per_file)} sampled masks, below the "
            f"{MODE_A_FILE_QUORUM:.0%} quorum"
        )
    else:
        reason = ("no colour is simultaneously sparse, thin and frame-spanning: "
                  "the mask is a phase-label map, so grain boundaries are absent")

    majority_structures = Counter(r["mode_b"]["majority_structure"] for r in per_file)
    mode_b_summary = {
        "n_labels": _minmedmax([r["mode_b"]["n_labels"] for r in per_file]),
        "majority_structure": majority_structures.most_common(1)[0][0],
        "majority_structure_votes": dict(majority_structures),
        "majority_n_components": _minmedmax(
            [r["mode_b"]["majority_n_components"] for r in per_file]),
        "per_label_component_count": _minmedmax(
            [pl["n_components"] for r in per_file for pl in r["mode_b"]["per_label"]]),
        "per_label_component_area": _minmedmax(
            [pl["component_area"]["median"] for r in per_file
             for pl in r["mode_b"]["per_label"]
             if pl["component_area"]["median"] is not None]),
        "per_label_area_min": _minmedmax(
            [pl["component_area"]["min"] for r in per_file
             for pl in r["mode_b"]["per_label"]
             if pl["component_area"]["min"] is not None]),
        "per_label_area_max": _minmedmax(
            [pl["component_area"]["max"] for r in per_file
             for pl in r["mode_b"]["per_label"]
             if pl["component_area"]["max"] is not None]),
    }

    return {
        "folder": folder.name,
        "path": str(folder),
        "pairing": pairing,
        "sampling": {
            "sample_size": len(per_file),
            "seed": seed,
            "read_errors": errors,
        },
        "image": {
            "formats": dict(Counter(r["image"]["format"] for r in per_file)),
            "bit_depths": dict(Counter(r["image"]["bit_depth"] for r in per_file)),
            "colour_modes": dict(Counter(r["image"]["colour_mode"] for r in per_file)),
            "width": _minmedmax(widths),
            "height": _minmedmax(heights),
        },
        "mask": {
            "formats": dict(Counter(r["mask"]["format"] for r in per_file)),
            "bit_depths": dict(Counter(r["mask"]["bit_depth"] for r in per_file)),
            "colour_modes": dict(Counter(r["mask"]["colour_mode"] for r in per_file)),
            "width": _minmedmax([r["mask"]["width"] for r in per_file]),
            "height": _minmedmax([r["mask"]["height"] for r in per_file]),
            "size_matches_image": all(r["size_match"] for r in per_file),
        },
        "palette": {
            "n_unique_colours": _minmedmax(
                [r["palette"]["n_unique_colours"] for r in per_file]),
            "quantized": all(r["palette"]["quantized"] for r in per_file),
            "quantized_votes": dict(Counter(r["palette"]["quantized"] for r in per_file)),
            "top_colours": _merge_palettes(per_file),
            "continuous_evidence": next(
                (r["palette"]["continuous_evidence"] for r in per_file
                 if r["palette"]["continuous_evidence"]), None),
        },
        "mode": {
            "inferred": "A" if is_a else "B",
            "reason": reason,
            "evidence": evidence,
        },
        "mode_b": mode_b_summary,
        "boundary_fraction": {
            "mode_a": _minmedmax([r["boundary_fraction"]["mode_a"] for r in per_file]),
            "mode_b": _minmedmax([r["boundary_fraction"]["mode_b"] for r in per_file]),
        },
        "files": per_file,
    }


def _merge_palettes(per_file: Sequence, top_n: int = PALETTE_TOP_N) -> list:
    """Pixel frequency per colour, pooled over the sampled masks."""
    pooled, total = Counter(), 0
    for rec in per_file:
        for entry in rec["palette"]["top_colours"]:
            pooled[tuple(entry["colour"])] += entry["pixels"]
            total += entry["pixels"]
    return [
        {"colour": list(c), "pixels": int(n),
         "fraction": (float(n) / total) if total else 0.0}
        for c, n in pooled.most_common(top_n)
    ]


# --------------------------------------------------------------------------
# 7. driver + reports
# --------------------------------------------------------------------------
def audit_datasets(
    data_root: Path,
    datasets: Optional[Sequence[str]] = None,
    sample_size: int = SAMPLE_SIZE,
    seed: int = SAMPLE_SEED,
    progress: Optional[Callable] = None,
) -> dict:
    """Audit every dataset folder under ``data_root``. Never writes to data."""
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise AuditError(f"DATA_ROOT does not exist: {data_root}")

    folders = sorted(p for p in data_root.iterdir() if p.is_dir()
                     and not p.name.startswith("."))
    if datasets:
        wanted = {d.lower() for d in datasets}
        folders = [f for f in folders if f.name.lower() in wanted]
    if not folders:
        raise AuditError(
            f"no dataset folders under {data_root}; found: "
            + (", ".join(p.name for p in data_root.iterdir()) or "(empty)")
        )

    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_root": str(data_root),
        "host": platform.platform(),
        "sample_size": sample_size,
        "seed": seed,
        "datasets": {},
        "failures": {},
    }
    for folder in folders:
        try:
            report["datasets"][folder.name] = audit_folder(
                folder, sample_size=sample_size, seed=seed, progress=progress)
        except Exception as exc:
            report["failures"][folder.name] = f"{type(exc).__name__}: {exc}"
    if not report["datasets"]:
        raise AuditError("every dataset folder failed: "
                         + json.dumps(report["failures"], indent=2))
    return report


def write_reports(report: dict, reports_dir: Path) -> tuple:
    """Write reports/audit.json and reports/audit.md. Returns both paths."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "audit.json"
    md_path = reports_dir / "audit.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=False))
    md_path.write_text(render_markdown(report))
    return json_path, md_path


def _fmt(value, spec: str = ".4f") -> str:
    return "-" if value is None else format(value, spec)


def render_markdown(report: dict) -> str:
    """Human-readable audit: verdict first, then the evidence behind it."""
    lines = [
        "# Dataset audit",
        "",
        f"- generated: {report['generated_utc']}",
        f"- data root: `{report['data_root']}`",
        f"- pairing counted over ALL files; pixel statistics over "
        f"{report['sample_size']} sampled pairs per folder (seed {report['seed']})",
        "",
        "MODE A masks paint boundaries as their own colour, so phase interfaces "
        "AND intra-phase grain boundaries are present. MODE B masks are phase-label "
        "maps: `find_boundaries` recovers phase interfaces only, and grain "
        "boundaries are absent from the ground truth.",
        "",
        "## Verdicts",
        "",
        "| folder | mode | pairs | unmatched | mask colours (median) | boundary frac A | boundary frac B |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, d in report["datasets"].items():
        p = d["pairing"]
        lines.append(
            f"| {name} | **{d['mode']['inferred']}** | {p['n_pairs']} | "
            f"{len(p['unmatched_images'])} img / {len(p['unmatched_masks'])} mask | "
            f"{_fmt(d['palette']['n_unique_colours']['median'], '.0f')} | "
            f"{_fmt(d['boundary_fraction']['mode_a']['median'])} | "
            f"{_fmt(d['boundary_fraction']['mode_b']['median'])} |"
        )
    if report["failures"]:
        lines += ["", "### Folders that could not be audited", ""]
        lines += [f"- **{k}**: {v}" for k, v in report["failures"].items()]

    for name, d in report["datasets"].items():
        p, m, img, msk = d["pairing"], d["mode"], d["image"], d["mask"]
        lines += [
            "", f"## {name}", "",
            f"**MODE {m['inferred']}** — {m['reason']}", "",
            "### Pairing", "",
            f"- convention inferred: `{p['convention']}`",
            f"- images dir: `{p['image_dir']}`",
            f"- masks dir: `{p['mask_dir']}`",
            f"- {p['n_images']} images, {p['n_masks']} masks, "
            f"**{p['n_pairs']} matched pairs**",
            f"- unmatched images ({len(p['unmatched_images'])}): "
            + (", ".join(p["unmatched_images"][:20]) or "none")
            + (" ..." if len(p["unmatched_images"]) > 20 else ""),
            f"- unmatched masks ({len(p['unmatched_masks'])}): "
            + (", ".join(p["unmatched_masks"][:20]) or "none")
            + (" ..." if len(p["unmatched_masks"]) > 20 else ""),
            "",
            "### Files", "",
            f"- image: format {img['formats']}, bit depth {img['bit_depths']}, "
            f"colour {img['colour_modes']}",
            f"- image size w {_fmt(img['width']['min'], '.0f')}/"
            f"{_fmt(img['width']['median'], '.0f')}/{_fmt(img['width']['max'], '.0f')}, "
            f"h {_fmt(img['height']['min'], '.0f')}/"
            f"{_fmt(img['height']['median'], '.0f')}/{_fmt(img['height']['max'], '.0f')} "
            "(min/median/max)",
            f"- mask: format {msk['formats']}, bit depth {msk['bit_depths']}, "
            f"colour {msk['colour_modes']}",
            f"- mask size matches image: {msk['size_matches_image']}",
            "",
            "### Mask palette", "",
            f"- unique colours per mask (min/median/max): "
            f"{_fmt(d['palette']['n_unique_colours']['min'], '.0f')}/"
            f"{_fmt(d['palette']['n_unique_colours']['median'], '.0f')}/"
            f"{_fmt(d['palette']['n_unique_colours']['max'], '.0f')}",
            f"- quantized: {d['palette']['quantized']}"
            + (f" — {d['palette']['continuous_evidence']}"
               if d["palette"]["continuous_evidence"] else ""),
            "",
            "| colour | pixels | fraction |",
            "| --- | --- | --- |",
        ]
        for entry in d["palette"]["top_colours"]:
            lines.append(f"| `{entry['colour']}` | {entry['pixels']} | "
                         f"{entry['fraction']:.6f} |")

        ev = m["evidence"]
        lines += [
            "", "### MODE A evidence", "",
            f"- candidate colour: `{ev['candidate_colour']}` "
            f"(HSV {ev['candidate_hsv']})",
            f"- qualifying in {ev['files_qualifying']}/{ev['files_sampled']} "
            f"sampled masks (quorum {ev['quorum_required']})",
            f"- pixel fraction (min/median/max): "
            f"{_fmt(ev['pixel_fraction']['min'])}/"
            f"{_fmt(ev['pixel_fraction']['median'])}/"
            f"{_fmt(ev['pixel_fraction']['max'])}",
            f"- skeleton-to-area: {_fmt(ev['skeleton_to_area']['median'])} "
            f"(mean thickness {_fmt(ev['mean_thickness_px']['median'], '.2f')} px)",
            f"- components: {_fmt(ev['n_components']['median'], '.0f')}, "
            f"frame span {_fmt(ev['span']['median'], '.3f')}",
            "",
            "### MODE B structure", "",
            f"- labels per mask (min/median/max): "
            f"{_fmt(d['mode_b']['n_labels']['min'], '.0f')}/"
            f"{_fmt(d['mode_b']['n_labels']['median'], '.0f')}/"
            f"{_fmt(d['mode_b']['n_labels']['max'], '.0f')}",
            f"- majority label is **{d['mode_b']['majority_structure']}** "
            f"({d['mode_b']['majority_structure_votes']}), median "
            f"{_fmt(d['mode_b']['majority_n_components']['median'], '.0f')} components",
            f"- components per label (min/median/max): "
            f"{_fmt(d['mode_b']['per_label_component_count']['min'], '.0f')}/"
            f"{_fmt(d['mode_b']['per_label_component_count']['median'], '.0f')}/"
            f"{_fmt(d['mode_b']['per_label_component_count']['max'], '.0f')}",
            f"- component area, median per label (min/median/max): "
            f"{_fmt(d['mode_b']['per_label_component_area']['min'], '.0f')}/"
            f"{_fmt(d['mode_b']['per_label_component_area']['median'], '.0f')}/"
            f"{_fmt(d['mode_b']['per_label_component_area']['max'], '.0f')} px",
            f"- smallest component area seen: "
            f"{_fmt(d['mode_b']['per_label_area_min']['min'], '.0f')} px; largest: "
            f"{_fmt(d['mode_b']['per_label_area_max']['max'], '.0f')} px",
            "",
            "### Boundary pixel fraction each mode would produce", "",
            f"- MODE A (painted colour): "
            f"{_fmt(d['boundary_fraction']['mode_a']['median'])}",
            f"- MODE B (find_boundaries on the label map): "
            f"{_fmt(d['boundary_fraction']['mode_b']['median'])}",
        ]
    return "\n".join(lines) + "\n"


def mode_of(dataset: str, reports_dir: Optional[Path] = None) -> str:
    """Read a folder's MODE from reports/audit.json. Never guess it."""
    from src.paths import REPO_ROOT

    path = Path(reports_dir or (REPO_ROOT / "reports")) / "audit.json"
    if not path.is_file():
        raise AuditError(
            f"{path} does not exist. Run notebooks/01_audit.ipynb first: the "
            "mask mode of a folder must be read from the audit, never assumed."
        )
    report = json.loads(path.read_text())
    if dataset not in report["datasets"]:
        raise AuditError(
            f"{dataset} is not in {path}. Audited folders: "
            + ", ".join(report["datasets"])
        )
    return report["datasets"][dataset]["mode"]["inferred"]
