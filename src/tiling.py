"""Tiling, parent grouping, and the fold protocol.

Three facts about this data set the design, and every one of them is a
correctness constraint rather than a preference:

1. Images are never resized. A micrograph carries a physical micron scale;
   resampling it destroys the very thing the network must learn. Size
   variation is handled by cutting 256 px tiles with 50% overlap, and the
   LAST tile of a row or column is clamped so its far edge lands exactly on
   the image edge. Overlap at that last position is uneven; nothing is
   fabricated. Padding happens only when an image is smaller than the patch
   in an axis -- no dataset here is -- and every padded pixel is counted and
   reported. Steel1 and Steel2 are already exactly 256x256 and must produce
   exactly one tile each with zero padding -- asserted, not assumed.

2. Tiles are not independent samples. Steel1's 907 tiles come from 19 source
   micrographs, 48 tiles each. Two tiles 128 px apart show the same grains
   under the same etch and illumination. Splitting them across train and val
   leaks, and the resulting validation score measures memorisation. Every
   tile therefore carries a ``parent_id`` and splits are made BY PARENT.

3. Datasets are not interchangeable. Steel2 is optical/colour among four
   grayscale SEM sets and its masks are auto-thresholded; it is held out as
   the domain-shift test fold and never trained on.

Nothing here writes tile images. A tile is a row in a manifest -- dataset,
parent, source image, x, y, boundary fraction -- and the loader crops it on
the fly. That keeps one copy of the pixels on disk and makes a fold a text
file that can be read, diffed and argued with.
"""

from __future__ import annotations

import csv
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from src import audit as audit_mod
from src.paths import REPO_ROOT

# --------------------------------------------------------------------------
# defaults -- overridable from configs/default.yaml under ``tiling:``
# --------------------------------------------------------------------------
DEFAULTS = {
    "patch_size": 256,
    "stride": 128,               # 50% overlap
    "pad_mode": "reflect",       # right/bottom only, and only where needed
    "min_boundary_frac": 0.005,  # a tile with less boundary than this teaches
                                 # the network nothing but background

    # stem -> parent_id. A dataset absent from this map is its own parent.
    "parent_rules": {
        "Steel1": r"_\d+_\d+$",        # ..._4000x14_1_3 -> ..._4000x14
        "Steel2": r"_tile_\d+_\d+$",   # 87431041_tile_0_256 -> 87431041
    },
    "expected_parents": {"Steel1": 19},   # asserted after exclusions
    "assert_single_tile": ["Steel1", "Steel2"],  # 256x256, one tile, no padding

    # Split protocol
    "test_only": ["Steel2"],             # never in any train or val split
    "parent_split": {"Steel1": {"val_parents": 4}},   # split by parent, seeded
    "lodo_datasets": ["MetalDam", "uhcs1", "uhcs2"],  # leave-one-dataset-out
    "seed": 0,

    # Sampling weights: "parents" makes each independent scene contribute
    # equally, "dataset" makes each dataset contribute equally, "none" leaves
    # raw tile counts alone.
    "weight_mode": "parents",

    "exclusions": {},            # dataset -> [image filename, ...]
    "gt_subdir": "gt_boundaries",
    "manifest_subdir": "manifests",

    # fold_steel_combined: a SEPARATE experiment-specific fold that pools
    # Steel1 and Steel2 into one fresh, parent-split 70/15/15 train/val/test
    # split, replacing the LODO protocol for this experiment only. It is
    # built alongside build_folds(), never inside it -- see
    # build_steel_combined_fold(). test_frac's slice is Steel2/Steel1 tiles
    # this fold pools and re-splits itself; it is written to its own
    # fold_steel_combined_test.csv, never to the shared test.csv that
    # test_only/lodo_datasets above produce.
    "steel_combined": {
        "name": "fold_steel_combined",
        "datasets": ["Steel1", "Steel2"],
        "train_frac": 0.70,
        "val_frac": 0.15,
        "test_frac": 0.15,
        # Steel2 has only 4 parents total; this guarantees that many of them
        # land in EVERY split rather than leaving it to chance where a small
        # dataset could end up entirely on one side.
        "min_parents_per_split": 1,
        "held_out_label": "mixed(Steel1+Steel2)",
    },
}

#: pos_weight outside this band means the class balance is not what we think.
#: Calibrated at the reference line width (2 px, ``_POS_WEIGHT_REFERENCE_WIDTH``
#: below): a uniform-width skeleton dilation covers area roughly proportional
#: to width for a fixed boundary network (same reasoning as
#: ``boundary_gt.SANE_FRACTION_BAND``), so boundary_fraction scales
#: proportionally with the configured width and pos_weight -- which is
#: (1-fraction)/fraction -- scales roughly INVERSELY with it. The raw constant
#: must not be compared against directly once line_width_px has changed; use
#: ``sane_pos_weight_band()``.
SANE_POS_WEIGHT_BAND = (5.0, 200.0)
_POS_WEIGHT_REFERENCE_WIDTH = 2.0


def sane_pos_weight_band(line_width_px) -> tuple:
    """Width-scaled (lo, hi) pos_weight band for the given ``line_width_px``.

    A thicker configured line covers proportionally more boundary pixels for
    the same network, which proportionally LOWERS pos_weight (more positives,
    same total) -- the inverse of how ``boundary_gt.sane_fraction_band``
    scales its band. Confirmed against a real run: doubling line_width_px
    2 -> 4 took pos_weight from 15.974/7.45/7.111 (MetalDam/uhcs1/uhcs2) to
    6.928/3.016/2.84, a factor of ~2.2-2.5 -- close enough to this band's own
    2x that the scaled band still has margin on every fold, not just the ones
    that happened to clear the old fixed one.

    Pass the width that ACTUALLY PRODUCED the tiles being checked --
    ``reports/gt_extraction.json``'s recorded ``settings.line_width_px``, not
    whatever ``configs/default.yaml`` currently says, which can have moved on
    since. The lower bound is floored at 1.0: a pos_weight below that would
    mean more positive than negative pixels, which no boundary-map width makes
    sane.
    """
    lo, hi = SANE_POS_WEIGHT_BAND
    width = float(line_width_px)
    if width <= 0:
        raise TilingError(f"line_width_px must be positive, got {line_width_px}")
    scale = _POS_WEIGHT_REFERENCE_WIDTH / width
    return (max(1.0, lo * scale), hi * scale)

MANIFEST_COLUMNS = [
    "tile_id", "dataset", "parent_id", "source_image", "x", "y",
    "boundary_fraction", "split",
    # everything below is what a loader needs to actually read the tile
    "patch", "image_path", "gt_path",
    "crop_left", "crop_top", "crop_width", "crop_height",
    "pad_right", "pad_bottom",
]


class TilingError(RuntimeError):
    """Raised when tiling cannot proceed. Never fails silently."""


# --------------------------------------------------------------------------
# config + inputs
# --------------------------------------------------------------------------
def load_config(config_path: Optional[Path] = None) -> dict:
    """Merge ``tiling:`` from configs/default.yaml over DEFAULTS."""
    from src import paths as paths_mod

    cfg = paths_mod.load_config(config_path)
    settings = dict(DEFAULTS)
    section = cfg.get("tiling") or {}
    if not isinstance(section, dict):
        raise TilingError(
            f"configs/default.yaml: tiling must be a mapping, got "
            f"{type(section).__name__}"
        )
    unknown = set(section) - set(DEFAULTS)
    if unknown:
        raise TilingError(
            f"configs/default.yaml: unknown tiling keys {sorted(unknown)}; "
            f"known keys are {sorted(DEFAULTS)}"
        )
    for key, value in section.items():
        if value is not None:
            settings[key] = value
    if int(settings["stride"]) > int(settings["patch_size"]):
        raise TilingError(
            f"stride {settings['stride']} exceeds patch size "
            f"{settings['patch_size']}: tiles would leave gaps in the image."
        )
    return settings


def load_extraction(reports_dir: Optional[Path] = None) -> dict:
    """Read reports/gt_extraction.json. Boundary maps and crops come from here."""
    path = Path(reports_dir or (REPO_ROOT / "reports")) / "gt_extraction.json"
    if not path.is_file():
        raise TilingError(
            f"{path} does not exist. Run notebooks/02_boundary_gt.ipynb first: "
            "tiling indexes the boundary maps that step produced."
        )
    report = json.loads(path.read_text())
    if not report.get("datasets"):
        raise TilingError(f"{path} contains no extracted datasets.")
    return report


def crop_lookup(extraction: dict) -> dict:
    """(dataset, image filename) -> the image_crop recorded by step 2.

    Step 2 reconciled a handful of pairs whose mask was a row shorter than its
    image. The crop it applied is recorded, not recomputed here: re-deriving it
    would risk cropping a different row and silently misaligning the pair.
    """
    out = {}
    for name, d in extraction["datasets"].items():
        for rec in d.get("reconciled") or []:
            out[(name, rec["pair"][0])] = rec["image_crop"]
    return out


# --------------------------------------------------------------------------
# parents
# --------------------------------------------------------------------------
def parent_id(dataset: str, stem: str, settings: dict) -> str:
    """Group key for a source image: the micrograph it was cut from.

    Datasets that ship pre-tiled encode the tile index in the filename; strip
    it and what remains identifies the specimen. A dataset with no rule is its
    own parent, one image per scene.
    """
    rule = (settings["parent_rules"] or {}).get(dataset)
    if not rule:
        return stem
    return re.sub(rule, "", stem)


def parent_map(tiles: Sequence) -> dict:
    """dataset -> parent_id -> {"images": [...], "tiles": n}."""
    out = defaultdict(lambda: defaultdict(lambda: {"images": set(), "tiles": 0}))
    for t in tiles:
        entry = out[t["dataset"]][t["parent_id"]]
        entry["images"].add(t["source_image"])
        entry["tiles"] += 1
    return {
        ds: {p: {"images": sorted(v["images"]), "n_images": len(v["images"]),
                 "tiles": v["tiles"]}
             for p, v in sorted(parents.items())}
        for ds, parents in sorted(out.items())
    }


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def tile_positions(size: int, patch: int, stride: int) -> tuple:
    """Start offsets covering ``size``, and the padding needed (normally none).

    Positions step by ``stride`` while the tile still fits. When the tail of
    the axis is left uncovered, the final tile is CLAMPED to ``size - patch``
    so its far edge sits exactly on the image edge, rather than overrunning
    and being reflection-padded.

    Padding a 645 px axis to 768 would have made 48% of the last tile column
    mirrored texture -- microstructure that exists in no micrograph, complete
    with an artificial mirror-symmetric boundary down the seam. Clamping costs
    only an uneven overlap at one position, and the network never sees an
    invented pixel.

    When the clamped position would sit within half a stride of the previous
    one, it REPLACES it instead of being appended: two tiles 5 px apart are
    the same tile twice.

    Padding survives for exactly one case -- an image smaller than the patch
    in an axis, where there is no position that covers it. The caller reports
    those loudly; no dataset in this collection has one.
    """
    if patch <= 0 or stride <= 0:
        raise TilingError(f"patch {patch} and stride {stride} must be positive")
    if size <= patch:
        return [0], patch - size

    positions = list(range(0, size - patch + 1, stride))
    last = size - patch
    if positions[-1] < last:
        if last - positions[-1] < stride / 2:
            positions[-1] = last
        else:
            positions.append(last)
    return positions, 0


def plan_tiles(width: int, height: int, settings: dict) -> dict:
    """Tile grid for one image: x/y offsets and the right/bottom padding."""
    patch = int(settings["patch_size"])
    stride = int(settings["stride"])
    xs, pad_right = tile_positions(int(width), patch, stride)
    ys, pad_bottom = tile_positions(int(height), patch, stride)
    return {
        "xs": xs, "ys": ys,
        "pad_right": int(pad_right), "pad_bottom": int(pad_bottom),
        "n_tiles": len(xs) * len(ys),
        "patch": patch, "stride": stride,
        # pixels that exist only because an image is smaller than the patch
        "padded_pixels": int((int(width) + pad_right) * (int(height) + pad_bottom)
                             - int(width) * int(height)),
    }


def pad_to_grid(arr: np.ndarray, plan: dict, mode: str = "reflect") -> np.ndarray:
    """Reflection-pad right and bottom so every planned tile is in bounds."""
    pad_r, pad_b = plan["pad_right"], plan["pad_bottom"]
    if not pad_r and not pad_b:
        return arr
    if mode == "reflect" and (pad_b >= arr.shape[0] or pad_r >= arr.shape[1]):
        # np.pad's reflect cannot mirror more than the array holds.
        mode = "symmetric"
    pad = [(0, pad_b), (0, pad_r)] + [(0, 0)] * (arr.ndim - 2)
    return np.pad(arr, pad, mode=mode)


# --------------------------------------------------------------------------
# indexing
# --------------------------------------------------------------------------
def index_dataset(
    dataset: str,
    folder: Path,
    gt_dir: Path,
    settings: dict,
    crops: Optional[dict] = None,
    progress: Optional[Callable] = None,
) -> dict:
    """Enumerate every tile of one dataset. Reads GT maps, never writes."""
    from PIL import Image

    crops = crops or {}
    pairing = audit_mod.discover_pairs(Path(folder))
    excluded_names = {str(n) for n in (settings["exclusions"] or {}).get(dataset, [])}

    tiles, dropped, excluded, skipped, single_tile_drops = [], [], [], [], []
    padded_images = []   # only possible when an image is smaller than a patch
    # Images with no mask never reach the pairing loop below, so they are
    # enumerated here explicitly: an orphan must be reported, not simply absent.
    orphans = [
        {"image": name,
         "why": "image has no matching mask (never had ground truth)",
         "listed_in_exclusions": name in excluded_names}
        for name in pairing["unmatched_images"]
    ]
    listed_but_absent = sorted(
        excluded_names
        - {o["image"] for o in orphans}
        - {p.name for pair in pairing["pairs"] for p in pair}
    )

    pairs = pairing["pairs"]
    iterator = progress(pairs, desc=dataset) if progress else pairs
    for img_path, mask_path in iterator:
        if img_path.name in excluded_names or mask_path.name in excluded_names:
            excluded.append({"image": img_path.name,
                             "why": "listed in tiling.exclusions"})
            continue

        gt_path = Path(gt_dir) / (img_path.stem + ".png")
        if not gt_path.is_file():
            skipped.append({"image": img_path.name,
                            "why": f"no boundary map at {gt_path.name}; step 2 "
                                   "rejected or excluded this pair"})
            continue

        meta = audit_mod.read_meta(img_path)
        crop = crops.get((dataset, img_path.name))
        if crop:
            width, height = int(crop["width"]), int(crop["height"])
        else:
            width, height = int(meta["width"]), int(meta["height"])

        gt = np.asarray(Image.open(gt_path))
        if (gt.shape[1], gt.shape[0]) != (width, height):
            raise TilingError(
                f"{dataset}/{img_path.name}: image is {width}x{height} after "
                f"the recorded crop but its boundary map is "
                f"{gt.shape[1]}x{gt.shape[0]}. Step 2 and step 3 disagree about "
                "this pair; do not tile it until they do not."
            )

        plan = plan_tiles(width, height, settings)
        if dataset in (settings["assert_single_tile"] or []):
            patch = int(settings["patch_size"])
            if (width, height) != (patch, patch) or plan["n_tiles"] != 1 \
                    or plan["pad_right"] or plan["pad_bottom"]:
                raise TilingError(
                    f"{dataset}/{img_path.name} is {width}x{height} and plans "
                    f"{plan['n_tiles']} tile(s) with padding "
                    f"({plan['pad_right']},{plan['pad_bottom']}). "
                    f"{dataset} is declared pre-tiled at {patch}x{patch} in "
                    "tiling.assert_single_tile: one tile, zero padding."
                )

        if plan["padded_pixels"]:
            padded_images.append({
                "image": img_path.name,
                "size": [width, height],
                "pad_right": plan["pad_right"],
                "pad_bottom": plan["pad_bottom"],
                "padded_pixels": plan["padded_pixels"],
                "why": f"image is smaller than the {plan['patch']} px patch in "
                       "an axis, so no tile position can cover it without "
                       f"{settings['pad_mode']} padding",
            })

        padded = pad_to_grid(gt > 0, plan, settings["pad_mode"])
        patch = plan["patch"]
        parent = parent_id(dataset, img_path.stem, settings)
        kept_here = 0
        planned_here = []
        for y in plan["ys"]:
            for x in plan["xs"]:
                frac = float(padded[y:y + patch, x:x + patch].mean())
                rec = {
                    "tile_id": f"{dataset}/{img_path.stem}/x{x}_y{y}",
                    "dataset": dataset,
                    "parent_id": parent,
                    "source_image": img_path.name,
                    "x": int(x), "y": int(y),
                    "boundary_fraction": round(frac, 6),
                    "split": "",
                    "patch": patch,
                    "image_path": str(img_path),
                    "gt_path": str(gt_path),
                    "crop_left": int(crop["left"]) if crop else 0,
                    "crop_top": int(crop["top"]) if crop else 0,
                    "crop_width": width, "crop_height": height,
                    "pad_right": plan["pad_right"],
                    "pad_bottom": plan["pad_bottom"],
                }
                planned_here.append(rec)
                if frac < float(settings["min_boundary_frac"]):
                    dropped.append({"tile_id": rec["tile_id"],
                                    "boundary_fraction": rec["boundary_fraction"]})
                else:
                    tiles.append(rec)
                    kept_here += 1

        if kept_here == 0 and len(planned_here) == 1:
            # A pre-tiled dataset losing its only tile loses the whole image.
            single_tile_drops.append({
                "dataset": dataset,
                "image": img_path.name,
                "boundary_fraction": planned_here[0]["boundary_fraction"],
                "why": f"the image's only tile is below min_boundary_frac "
                       f"({settings['min_boundary_frac']}) -- the image is gone "
                       "from every split",
            })

    parents = sorted({t["parent_id"] for t in tiles})
    expected = (settings["expected_parents"] or {}).get(dataset)
    if expected is not None and len(parents) != int(expected):
        raise TilingError(
            f"{dataset} resolved to {len(parents)} parents, expected "
            f"{expected}. Parents found: {parents}. Either tiling.exclusions is "
            "wrong or tiling.parent_rules no longer matches the filenames."
        )

    return {
        "dataset": dataset,
        "folder": str(folder),
        "gt_dir": str(gt_dir),
        "n_images": len({t["source_image"] for t in tiles}),
        "n_parents": len(parents),
        "parents": parents,
        "n_tiles": len(tiles),
        "n_dropped": len(dropped),
        "n_excluded": len(excluded),
        "n_skipped": len(skipped),
        "n_padded_images": len(padded_images),
        "padded_pixels": int(sum(x["padded_pixels"] for x in padded_images)),
        "padded_images": padded_images,
        "n_orphans": len(orphans),
        "n_orphans_listed": sum(1 for o in orphans if o["listed_in_exclusions"]),
        "listed_but_absent": listed_but_absent,
        "dropped": dropped,
        "excluded": excluded,
        "skipped": skipped,
        "orphans": orphans,
        "single_tile_drops": single_tile_drops,
        "tiles": tiles,
    }


def build_index(
    extraction: dict,
    audit: dict,
    gt_root: Path,
    settings: dict,
    datasets: Optional[Sequence[str]] = None,
    progress: Optional[Callable] = None,
) -> dict:
    """Index every dataset that step 2 produced boundary maps for."""
    crops = crop_lookup(extraction)
    names = [n for n in extraction["datasets"]
             if not datasets or n in set(datasets)]
    if not names:
        raise TilingError("no datasets to tile; check --datasets")

    out = {"datasets": {}, "settings": settings, "failures": {}}
    for name in names:
        folder = Path(audit["datasets"][name]["path"])
        gt_dir = Path(gt_root) / name
        if not gt_dir.is_dir():
            raise TilingError(
                f"{gt_dir} does not exist. Step 2 wrote boundary maps into "
                "PERSISTENT_DIR; is this the same host / the same Drive?"
            )
        out["datasets"][name] = index_dataset(
            name, folder, gt_dir, settings, crops=crops, progress=progress)
    return out


def all_tiles(index: dict) -> list:
    return [t for d in index["datasets"].values() for t in d["tiles"]]


# --------------------------------------------------------------------------
# folds
# --------------------------------------------------------------------------
def split_parents(parents: Sequence, n_val: int, seed: int) -> tuple:
    """Deterministic parent-level split. Same seed, same parents, same answer."""
    ordered = sorted(parents)
    if n_val >= len(ordered):
        raise TilingError(
            f"asked for {n_val} validation parents out of {len(ordered)}; "
            "that would leave nothing to train on."
        )
    rng = random.Random(seed)
    val = sorted(rng.sample(ordered, n_val))
    train = [p for p in ordered if p not in set(val)]
    return train, val


def build_folds(index: dict, settings: dict) -> dict:
    """Build every fold manifest plus the test manifest.

    Each fold is:  val   = one held-out dataset, entire
                         + the fixed validation parents of the parent-split
                           datasets
                   train = the remaining datasets
                         + the training parents of the parent-split datasets
                   test  = the test-only datasets, in no fold at all
    """
    tiles = all_tiles(index)
    by_dataset = defaultdict(list)
    for t in tiles:
        by_dataset[t["dataset"]].append(t)

    test_only = list(settings["test_only"] or [])
    lodo = [d for d in (settings["lodo_datasets"] or []) if d in by_dataset]
    if not lodo:
        raise TilingError(
            "none of tiling.lodo_datasets are present in the index: "
            f"{settings['lodo_datasets']}"
        )

    # Fixed, seeded parent split for the datasets that are split by parent.
    parent_splits = {}
    for name, spec in (settings["parent_split"] or {}).items():
        if name not in by_dataset:
            continue
        parents = sorted({t["parent_id"] for t in by_dataset[name]})
        train_p, val_p = split_parents(
            parents, int(spec["val_parents"]), int(settings["seed"]))
        parent_splits[name] = {"train": train_p, "val": val_p}

    folds = {}
    for held in lodo:
        rows = []
        for t in tiles:
            ds = t["dataset"]
            if ds in test_only:
                continue
            if ds in parent_splits:
                side = "val" if t["parent_id"] in set(parent_splits[ds]["val"]) \
                    else "train"
            elif ds == held:
                side = "val"
            else:
                side = "train"
            rows.append(dict(t, split=side))
        folds[f"fold_{held}"] = {
            "name": f"fold_{held}",
            "held_out": held,
            "rows": rows,
            "parent_splits": parent_splits,
        }

    # "dev" is the fold with the smallest validation set: same protocol, least
    # waiting. It is an alias, not a different split.
    smallest = min(folds.values(),
                   key=lambda f: sum(1 for r in f["rows"] if r["split"] == "val"))
    folds["dev"] = {
        "name": "dev",
        "held_out": smallest["held_out"],
        "alias_of": smallest["name"],
        "rows": [dict(r) for r in smallest["rows"]],
        "parent_splits": parent_splits,
    }

    test_rows = [dict(t, split="test") for t in tiles if t["dataset"] in test_only]
    return {"folds": folds, "test": {"name": "test", "datasets": test_only,
                                     "rows": test_rows},
            "parent_splits": parent_splits}


# --------------------------------------------------------------------------
# fold_steel_combined -- a separate, pooled train/val/test split
# --------------------------------------------------------------------------
def allocate_parents_by_tile_ratio(
    parent_tiles: dict, fracs: dict, seed: int, min_parents_per_split: int = 1,
) -> dict:
    """Assign every (dataset, parent) to train/val/test by TILE COUNT ratio.

    ``parent_tiles`` is ``{dataset: {parent_id: n_tiles}}``. The split is made
    BY PARENT -- a parent's tiles move together -- but the TARGET ratio
    (``fracs``, a ``{"train": f, "val": f, "test": f}`` summing to 1.0) is
    measured in tiles, because parents are not equal-sized: pooling Steel1
    (19 parents, ~47 tiles each) with Steel2 (4 parents, ~126 tiles each)
    means a parent-count split and a tile-count split disagree, and the tile
    count is what a training epoch actually sees.

    Deterministic in two stages, both driven by one ``random.Random(seed)``
    and both iterating in a fixed (sorted) order so the result never depends
    on dict or set iteration order:

    1. **Minimum guarantee.** For each dataset, ``min_parents_per_split``
       parents are handed to EACH split before anything else is decided, by
       shuffling that dataset's own parent list and popping from it. This is
       what keeps a small dataset (Steel2's 4 parents) from landing entirely
       on one side of the split by chance -- without it, a dataset with only
       a few parents could easily miss a split altogether.
    2. **The rest.** Every remaining (dataset, parent) pair, pooled across
       datasets, is shuffled once and then assigned ONE PARENT AT A TIME to
       whichever split is currently furthest BELOW its target tile-count
       share. This greedy balance converges on the requested ratio without
       ever moving a parent once it is placed, which is what makes stage 2
       reproducible: the same seed always shuffles the remainder into the
       same order and hands out the same sequence of "furthest behind"
       decisions.

    Raises if a dataset does not have enough parents to give every split its
    minimum -- silently shrinking the guarantee would defeat the point of it.

    Returns ``{"train": {dataset: [parent_id, ...]}, "val": {...}, "test": {...}}``,
    every list sorted.
    """
    splits = ("train", "val", "test")
    missing_frac = sorted(set(splits) - set(fracs))
    if missing_frac:
        raise TilingError(f"allocate_parents_by_tile_ratio: fracs is missing {missing_frac}")
    if abs(sum(fracs[s] for s in splits) - 1.0) > 1e-6:
        raise TilingError(
            f"allocate_parents_by_tile_ratio: fracs must sum to 1.0, got {fracs}")
    if min_parents_per_split < 0:
        raise TilingError(
            f"min_parents_per_split must be >= 0, got {min_parents_per_split}")

    rng = random.Random(seed)
    assigned = {s: defaultdict(list) for s in splits}
    used = {ds: set() for ds in parent_tiles}

    for ds in sorted(parent_tiles):
        parents = sorted(parent_tiles[ds])
        needed = min_parents_per_split * len(splits)
        if len(parents) < needed:
            raise TilingError(
                f"{ds} has {len(parents)} parents, too few to guarantee "
                f"{min_parents_per_split} per split across {list(splits)} "
                f"({needed} needed).")
        pool = list(parents)
        rng.shuffle(pool)
        for s in splits:
            for _ in range(min_parents_per_split):
                p = pool.pop()
                assigned[s][ds].append(p)
                used[ds].add(p)

    remainder = [(ds, p) for ds in sorted(parent_tiles)
                 for p in sorted(parent_tiles[ds]) if p not in used[ds]]
    rng.shuffle(remainder)

    totals = {s: sum(parent_tiles[ds][p] for ds, plist in assigned[s].items()
                     for p in plist) for s in splits}
    grand_total = sum(sum(v.values()) for v in parent_tiles.values())

    for ds, p in remainder:
        n = parent_tiles[ds][p]
        deficit = {s: fracs[s] * grand_total - totals[s] for s in splits}
        target = max(splits, key=lambda s: deficit[s])
        assigned[target][ds].append(p)
        totals[target] += n

    return {s: {ds: sorted(plist) for ds, plist in assigned[s].items()}
            for s in splits}


def build_steel_combined_fold(index: dict, settings: dict) -> dict:
    """``fold_steel_combined``: pooled Steel1+Steel2, its own 70/15/15 split.

    This is a deliberate, SELF-CONTAINED exception to the global
    ``tiling.test_only`` / ``tiling.lodo_datasets`` protocol every other fold
    obeys, not a change to it:

    - Steel2 is globally ``test_only`` -- it never appears in any LODO fold's
      train or val split, only in the shared ``test`` manifest, whole. This
      fold pools Steel2 WITH Steel1 and gives the pool a fresh parent-level
      split instead, so Steel2 tiles DO appear in this fold's train and val,
      despite the global policy. That is only sound because this fold's own
      held-out slice is a genuine, disjoint parent-level test split of its
      own -- it is not reusing Steel2's "held out" status as license to peek.
    - MetalDam, uhcs1 and uhcs2 play no part here; their tiles are simply
      absent from every one of this fold's three splits, and their existing
      folds (which this function never touches) are unaffected.
    - Its held-out test slice is written to ITS OWN manifest
      (``fold_steel_combined_test.csv``, by the caller), never merged into
      the shared ``test.csv`` -- the two test sets answer different
      questions (domain shift onto an unseen microscope vs. generalisation
      to unseen parents of the SAME two microscopes) and must not be
      confused for one another downstream.

    Returns a dict shaped like a ``build_folds()`` fold entry --
    ``name``/``held_out``/``alias_of``/``rows`` (train+val, split-tagged) --
    plus this fold's own extras: ``test_rows`` (its held-out test slice,
    never in ``rows``) and ``parent_allocation`` (the full
    ``{split: {dataset: [parent_id, ...]}}`` this run produced, so it can be
    reported and checked for reproducibility). Callers insert the returned
    entry into ``build_folds()``'s ``folds["folds"]`` dict themselves --
    this function never mutates a LODO fold or the shared ``test`` manifest.
    """
    cfg = settings["steel_combined"]
    datasets = list(cfg["datasets"])

    by_dataset = defaultdict(list)
    for t in all_tiles(index):
        if t["dataset"] in datasets:
            by_dataset[t["dataset"]].append(t)
    missing = [d for d in datasets if not by_dataset.get(d)]
    if missing:
        raise TilingError(
            f"{cfg['name']} needs tiles from {datasets}; none indexed for "
            f"{missing}. Check --datasets or that both datasets extracted "
            "boundary maps in reports/gt_extraction.json.")

    parent_tiles = {}
    for ds in datasets:
        counts = defaultdict(int)
        for t in by_dataset[ds]:
            counts[t["parent_id"]] += 1
        parent_tiles[ds] = dict(counts)

    fracs = {"train": float(cfg["train_frac"]), "val": float(cfg["val_frac"]),
             "test": float(cfg["test_frac"])}
    if abs(sum(fracs.values()) - 1.0) > 1e-6:
        raise TilingError(
            f"tiling.steel_combined train/val/test_frac must sum to 1.0, "
            f"got {fracs} (sum {sum(fracs.values())})")

    assignment = allocate_parents_by_tile_ratio(
        parent_tiles, fracs, seed=int(settings["seed"]),
        min_parents_per_split=int(cfg["min_parents_per_split"]))

    # Loud, not assumed: no parent may appear on more than one side. The
    # allocator cannot produce this by construction (every parent is popped
    # from a single pool once), but the fold this replaces exists precisely
    # because that invariant was worth asserting rather than trusting.
    seen = {}
    for split in ("train", "val", "test"):
        for ds, plist in assignment[split].items():
            for p in plist:
                key = (ds, p)
                if key in seen:
                    raise TilingError(
                        f"{cfg['name']}: parent {key} was assigned to both "
                        f"{seen[key]} and {split}.")
                seen[key] = split

    rows_by_split = {"train": [], "val": [], "test": []}
    for split, per_dataset in assignment.items():
        wanted = {ds: set(plist) for ds, plist in per_dataset.items()}
        for ds in datasets:
            keep = wanted.get(ds, set())
            for t in by_dataset[ds]:
                if t["parent_id"] in keep:
                    rows_by_split[split].append(dict(t, split=split))

    return {
        "name": cfg["name"],
        # A descriptive label, not a real dataset name: this fold has no
        # single held-out DATASET in the LODO sense, so nothing downstream
        # that keys per-dataset metrics by held_out (best_key(), the FiLM
        # inference-mode report, the final-epoch table) finds a match for
        # it -- which is exactly what makes each of them fall back to their
        # existing "held-out dataset absent" pooled behaviour instead of
        # crashing or mislabelling a real dataset as this fold's held-out one.
        "held_out": cfg["held_out_label"],
        "alias_of": None,
        "rows": rows_by_split["train"] + rows_by_split["val"],
        "test_rows": rows_by_split["test"],
        "parent_allocation": assignment,
        "datasets": datasets,
    }


# --------------------------------------------------------------------------
# statistics + sampling weights
# --------------------------------------------------------------------------
def pos_weight(rows: Sequence) -> Optional[float]:
    """n_negative / n_positive over a split, from the tile boundary fractions."""
    if not rows:
        return None
    mean_frac = float(np.mean([r["boundary_fraction"] for r in rows]))
    if mean_frac <= 0:
        return None
    return float((1.0 - mean_frac) / mean_frac)


def sampling_weights(rows: Sequence, settings: dict) -> dict:
    """Per-tile weight per dataset, so an epoch is not 91% one specimen.

    ``parents``: each dataset's share of an epoch is its share of the
    independent scenes -- 19 Steel1 micrographs count as 19, not as 907.
    ``dataset``: every dataset contributes equally. ``none``: raw counts.
    """
    mode = settings["weight_mode"]
    counts = Counter(r["dataset"] for r in rows)
    parents = {ds: len({r["parent_id"] for r in rows if r["dataset"] == ds})
               for ds in counts}
    if not counts:
        return {"mode": mode, "weights": {}, "raw_counts": {},
                "target_share": {}, "effective_counts": {}}

    if mode == "none":
        target = {ds: n / sum(counts.values()) for ds, n in counts.items()}
    elif mode == "dataset":
        target = {ds: 1.0 / len(counts) for ds in counts}
    elif mode == "parents":
        total_parents = sum(parents.values())
        target = {ds: parents[ds] / total_parents for ds in counts}
    else:
        raise TilingError(
            f"tiling.weight_mode must be parents|dataset|none, got {mode!r}")

    total = sum(counts.values())
    weights = {ds: target[ds] / counts[ds] for ds in counts}
    # normalise so the mean per-tile weight is 1: readable in a config, and
    # numerically harmless to a WeightedRandomSampler.
    scale = total / sum(weights[r["dataset"]] for r in rows)
    weights = {ds: w * scale for ds, w in weights.items()}
    return {
        "mode": mode,
        "weights": {ds: round(w, 6) for ds, w in sorted(weights.items())},
        "raw_counts": dict(sorted(counts.items())),
        "n_parents": dict(sorted(parents.items())),
        "target_share": {ds: round(v, 4) for ds, v in sorted(target.items())},
        "effective_counts": {ds: int(round(target[ds] * total))
                             for ds in sorted(counts)},
    }


def fold_entry_statistics(fold: dict, settings: dict) -> dict:
    """The per-fold summary, for ONE fold entry.

    Split out of ``fold_statistics`` so that a fold built on its own -- the
    pooled ``fold_steel_combined``, which a partial Steel1+Steel2 index can
    produce without ever calling ``build_folds()`` -- is summarised by the
    SAME code that summarises a LODO fold, rather than by a second copy that
    would be free to drift. ``pos_weight`` and the sampler weights are
    therefore measured from ``fold["rows"]`` the one way, for every fold.
    """
    train = [r for r in fold["rows"] if r["split"] == "train"]
    val = [r for r in fold["rows"] if r["split"] == "val"]
    pw = pos_weight(train)
    return {
        "held_out": fold["held_out"],
        "alias_of": fold.get("alias_of"),
        "n_train_tiles": len(train),
        "n_val_tiles": len(val),
        "train_datasets": sorted({r["dataset"] for r in train}),
        "val_datasets": sorted({r["dataset"] for r in val}),
        "n_train_parents": len({(r["dataset"], r["parent_id"]) for r in train}),
        "n_val_parents": len({(r["dataset"], r["parent_id"]) for r in val}),
        "train_boundary_fraction": _stats([r["boundary_fraction"] for r in train]),
        "val_boundary_fraction": _stats([r["boundary_fraction"] for r in val]),
        "pos_weight": None if pw is None else round(pw, 3),
        "sampling": sampling_weights(train, settings),
    }


def fold_statistics(folds: dict, settings: dict) -> dict:
    """Everything a training run needs to know about a fold before it starts."""
    stats = {name: fold_entry_statistics(fold, settings)
             for name, fold in folds["folds"].items()}
    stats["test"] = test_slice_statistics(
        folds["test"]["rows"], folds["test"]["datasets"],
        "domain-shift fold: evaluated, never trained or validated on")
    return stats


def steel_combined_statistics(fold: dict, settings: dict) -> dict:
    """The complete ``fold_stats.yaml`` entry for ``fold_steel_combined``.

    The base summary every fold gets, plus the two things only this fold has:
    its OWN held-out test slice (``test``) and the parent allocation that
    produced all three splits (``parent_allocation``).

    It lives here rather than inline in a caller because there are now two
    callers -- ``scripts/build_tiles.py``, which builds this fold alongside
    the LODO folds from a full index, and the pooled-split notebook, which
    builds it alone from a Steel1+Steel2-only index -- and an entry assembled
    twice is an entry that can disagree with itself. ``pos_weight`` here comes
    from ``fold["rows"]``, this fold's own train split, exactly as it does for
    every other fold; it is never copied from Steel1's or Steel2's value in
    any other fold's entry.
    """
    entry = fold_entry_statistics(fold, settings)
    entry["test"] = test_slice_statistics(
        fold["test_rows"], fold["datasets"],
        f"{fold['name']}'s own held-out test slice: parent-level, never "
        "touched by training or checkpoint selection, and never merged into "
        "the shared test manifest.")
    entry["parent_allocation"] = fold["parent_allocation"]
    return entry


def test_slice_statistics(rows: Sequence, datasets: Sequence, note: str) -> dict:
    """The same summary shape used for every held-out test slice.

    Shared by the global ``test`` manifest (the LODO folds' test_only
    datasets, whole) and by ``fold_steel_combined``'s own held-out test
    parents, so the two are reported identically even though they are
    produced by different code paths and never share rows.
    """
    return {
        "datasets": list(datasets),
        "n_tiles": len(rows),
        "n_parents": len({(r["dataset"], r["parent_id"]) for r in rows}),
        "boundary_fraction": _stats([r["boundary_fraction"] for r in rows]),
        "note": note,
    }


def _stats(values: Sequence) -> dict:
    if not values:
        return {"min": None, "mean": None, "median": None, "max": None, "n": 0}
    arr = np.asarray(values, dtype=float)
    return {"min": round(float(arr.min()), 6),
            "mean": round(float(arr.mean()), 6),
            "median": round(float(np.median(arr)), 6),
            "max": round(float(arr.max()), 6),
            "n": int(arr.size)}


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------
def write_manifest(rows: Sequence, path: Path) -> Path:
    """One CSV per fold. A fold is a text file, not a hidden code path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS,
                                extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["dataset"], r["tile_id"])):
            writer.writerow(row)
    return path


def write_manifests(folds: dict, manifest_dir: Path) -> dict:
    manifest_dir = Path(manifest_dir)
    written = {}
    for name, fold in folds["folds"].items():
        written[name] = str(write_manifest(fold["rows"], manifest_dir / f"{name}.csv"))
    written["test"] = str(write_manifest(
        folds["test"]["rows"], manifest_dir / "test.csv"))
    return written


def write_fold_stats(
    stats: dict,
    folds: dict,
    settings: dict,
    configs_dir: Optional[Path] = None,
) -> Path:
    """configs/fold_stats.yaml -- read by the training sampler, not by a human."""
    import yaml

    configs_dir = Path(configs_dir or (REPO_ROOT / "configs"))
    configs_dir.mkdir(parents=True, exist_ok=True)
    path = configs_dir / "fold_stats.yaml"
    doc = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "patch_size": int(settings["patch_size"]),
        "stride": int(settings["stride"]),
        "min_boundary_frac": float(settings["min_boundary_frac"]),
        "seed": int(settings["seed"]),
        "weight_mode": settings["weight_mode"],
        "parent_splits": folds["parent_splits"],
        "folds": stats,
    }
    header = (
        "# Fold statistics and sampling weights, generated by src/tiling.py.\n"
        "# Do not hand-edit: notebooks/03_tiling.ipynb overwrites this file.\n"
        "#\n"
        "# sampling.weights is a PER-TILE weight keyed by dataset, for a\n"
        "# WeightedRandomSampler. Mean weight is 1, so an epoch keeps its size\n"
        "# while each dataset's share of it becomes sampling.target_share.\n"
        "# pos_weight is n_negative/n_positive over the train split, for\n"
        "# BCEWithLogitsLoss.\n"
    )
    path.write_text(header + yaml.safe_dump(doc, sort_keys=False))
    return path


#: Top-level keys in configs/fold_stats.yaml that describe the TILE GEOMETRY
#: every fold in the file was cut under. A merge that changed one of these
#: would leave folds in one file that disagree about what a tile even is.
_GEOMETRY_KEYS = ("patch_size", "stride", "min_boundary_frac", "seed")


def merge_fold_stats(
    entries: dict,
    settings: dict,
    configs_dir: Optional[Path] = None,
) -> tuple:
    """Add or replace named folds in configs/fold_stats.yaml, keeping the rest.

    ``write_fold_stats`` rewrites the whole file from one run's ``stats``, which
    is right when that run indexed every dataset. It is WRONG for a partial
    run: a Steel1+Steel2-only index produces stats for those folds alone, and
    writing them would silently delete ``fold_MetalDam``, ``fold_uhcs1``,
    ``fold_uhcs2``, ``dev`` and ``test`` from the file that training reads.
    The deletion would not raise anywhere -- it would surface later as
    ``fold 'dev' is not in configs/fold_stats.yaml``, a long way from its
    cause. This function is the read-modify-write that makes a partial run
    safe: it touches ONLY the fold keys named in ``entries``.

    ``entries`` is ``{fold_name: stats_entry}``. Refuses, rather than
    guessing, when:

    - the file does not exist yet (there is nothing to merge into; a first
      full run must produce it with ``write_fold_stats``)
    - the file's recorded tile geometry disagrees with ``settings``. Folds in
      one ``fold_stats.yaml`` must all describe tiles cut the same way; a
      merge under a different ``patch_size``/``stride``/``min_boundary_frac``/
      ``seed`` would put two incompatible fold definitions in one file and
      nothing downstream would notice.
    - an entry is not a mapping, or names no fold.

    Returns ``(path, summary)`` where summary records which fold keys were
    added, which were replaced, and which were left untouched -- so a caller
    can PRINT the additive-ness rather than assert it on faith.
    """
    import yaml

    configs_dir = Path(configs_dir or (REPO_ROOT / "configs"))
    path = configs_dir / "fold_stats.yaml"
    if not path.is_file():
        raise TilingError(
            f"{path} does not exist, so there is nothing to merge into. "
            "Run the full tiling step (notebooks/03_tiling.ipynb, every "
            "dataset) once to create it before merging a single fold in.")
    if not entries:
        raise TilingError("merge_fold_stats was given no fold entries.")
    for name, entry in entries.items():
        if not isinstance(entry, dict):
            raise TilingError(
                f"merge_fold_stats: entry for {name!r} must be a mapping, got "
                f"{type(entry).__name__}")

    doc = yaml.safe_load(path.read_text())
    if not isinstance(doc, dict) or not isinstance(doc.get("folds"), dict):
        raise TilingError(
            f"{path} is not a fold-stats document (no top-level 'folds' "
            "mapping). Refusing to merge into a file this did not write.")

    mismatched = []
    for key in _GEOMETRY_KEYS:
        if key not in doc or key not in settings:
            continue
        have, want = doc[key], settings[key]
        if key == "min_boundary_frac":
            same = abs(float(have) - float(want)) < 1e-12
        else:
            same = int(have) == int(want)
        if not same:
            mismatched.append(f"{key}: file has {have!r}, this run has {want!r}")
    if mismatched:
        raise TilingError(
            f"{path} was written under a different tile geometry:\n  "
            + "\n  ".join(mismatched)
            + "\nMerging a fold cut one way into a file describing folds cut "
              "another way would put two incompatible definitions of a tile in "
              "one file. Re-run the full tiling step instead.")

    before = set(doc["folds"])
    added = sorted(set(entries) - before)
    replaced = sorted(set(entries) & before)
    for name, entry in entries.items():
        doc["folds"][name] = entry
    untouched = sorted(before - set(entries))

    doc["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    doc["merged_folds"] = sorted(entries)

    header = (
        "# Fold statistics and sampling weights, generated by src/tiling.py.\n"
        "# Do not hand-edit: notebooks/03_tiling.ipynb overwrites this file,\n"
        "# and merge_fold_stats() rewrites individual fold entries in place.\n"
        "#\n"
        "# sampling.weights is a PER-TILE weight keyed by dataset, for a\n"
        "# WeightedRandomSampler. Mean weight is 1, so an epoch keeps its size\n"
        "# while each dataset's share of it becomes sampling.target_share.\n"
        "# pos_weight is n_negative/n_positive over the train split, for\n"
        "# BCEWithLogitsLoss.\n"
    )
    path.write_text(header + yaml.safe_dump(doc, sort_keys=False))
    return path, {"added": added, "replaced": replaced, "untouched": untouched,
                  "n_folds_after": len(doc["folds"])}


def write_fold_report(
    fold: dict,
    entry: dict,
    index: dict,
    manifests: dict,
    reports_dir: Optional[Path] = None,
) -> tuple:
    """reports/tiling_<fold>.{md,json} -- one fold's own report.

    A partial run must not write ``reports/tiling.md``: that file describes
    EVERY dataset and every fold, and regenerating it from a two-dataset
    index would quietly shrink it to those two. This writes a report scoped
    to the one fold instead, beside the repo-wide one rather than over it.

    Takes the ``fold`` itself, not just its stats entry, because the per
    dataset per split TILE counts -- the thing the task asks to be made
    visible rather than silently absorbed -- are only derivable by counting
    rows; the stats entry records totals and parent lists, not the crossing
    of the two.
    """
    name = fold["name"]
    reports_dir = Path(reports_dir or (REPO_ROOT / "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    datasets = list(fold["datasets"])

    rows_by_split = {
        "train": [r for r in fold["rows"] if r["split"] == "train"],
        "val": [r for r in fold["rows"] if r["split"] == "val"],
        "test": list(fold["test_rows"]),
    }
    alloc = fold["parent_allocation"]
    tiles = {s: {d: sum(1 for r in rows if r["dataset"] == d) for d in datasets}
             for s, rows in rows_by_split.items()}
    parents = {s: {d: len((alloc.get(s) or {}).get(d, [])) for d in datasets}
               for s in rows_by_split}

    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fold": name,
        "settings": {k: index["settings"][k] for k in sorted(index["settings"])},
        "datasets_indexed": {
            ds: {k: v for k, v in d.items() if k != "tiles"}
            for ds, d in index["datasets"].items()
        },
        "tiles_per_split_per_dataset": tiles,
        "parents_per_split_per_dataset": parents,
        "parent_allocation": alloc,
        "stats": entry,
        "manifests": manifests,
    }
    json_path = _write(reports_dir / f"tiling_{name}.json",
                       json.dumps(payload, indent=2))

    totals = {s: sum(tiles[s].values()) for s in rows_by_split}
    grand = sum(totals.values())
    cfg = index["settings"]["steel_combined"]

    lines = [
        f"# {name}",
        "",
        f"- generated: {payload['generated_utc']}",
        f"- pooled from {datasets}, split BY PARENT with seed "
        f"{index['settings']['seed']}",
        f"- held out: `{entry['held_out']}` — this fold has no single held-out "
        "DATASET in the LODO sense. The label is descriptive and matches no "
        "dataset in any per-dataset metrics breakdown by construction, which "
        "is what makes the held-out-specific paths in `src/train.py` fall "
        "back to their pooled behaviour instead of mislabelling a real "
        "dataset as this fold's held-out one.",
        f"- pos_weight **{entry['pos_weight']}**, measured on THIS fold's own "
        "train split, never inherited from another fold",
        "",
        "## Parents and tiles per split",
        "",
        "Printed per dataset so an imbalance is visible rather than absorbed "
        "into a total. Steel2 has only 4 parents, so "
        f"`min_parents_per_split: {cfg['min_parents_per_split']}` is what "
        "guarantees it appears in every split at all.",
        "",
        "| split | " + " | ".join(f"{d} p/t" for d in datasets)
        + " | tiles | share |",
        "| --- | " + " | ".join("---" for _ in datasets) + " | --- | --- |",
    ]
    for s in ("train", "val", "test"):
        cells = [f"{parents[s][d]}p / {tiles[s][d]}t" for d in datasets]
        share = totals[s] / grand if grand else 0.0
        lines.append(f"| {s} | " + " | ".join(cells)
                     + f" | {totals[s]} | {share:.3f} |")
    lines += ["",
              f"Total {grand} tiles. Target "
              f"{cfg['train_frac']}/{cfg['val_frac']}/{cfg['test_frac']}, "
              "achieved "
              + "/".join(f"{totals[s] / grand:.3f}" for s in
                         ("train", "val", "test"))
              + " by TILE count (the ratio is measured in tiles, not parents: "
              "Steel1 and Steel2 have very different tiles-per-parent).",
              ""]

    lines += ["## Boundary fraction", "",
              "| split | min | mean | median | max | n |",
              "| --- | --- | --- | --- | --- | --- |"]
    for label, s in (("train", entry["train_boundary_fraction"]),
                     ("val", entry["val_boundary_fraction"]),
                     ("test", entry["test"]["boundary_fraction"])):
        lines.append(f"| {label} | {s['min']} | {s['mean']} | {s['median']} | "
                     f"{s['max']} | {s['n']} |")

    lines += ["", "## Parent allocation", "",
              "Every parent appears in exactly one split; a parent's tiles "
              "never straddle two splits. Listed in full so the split can be "
              "checked by eye and reproduced.", ""]
    for s in ("train", "val", "test"):
        lines.append(f"- **{s}**")
        for d in datasets:
            plist = (alloc.get(s) or {}).get(d, [])
            lines.append(f"  - {d} ({len(plist)}): "
                         + (", ".join(f"`{p}`" for p in plist) or "—"))

    lines += ["", "## Manifests", ""]
    lines += [f"- `{k}`: `{v}`" for k, v in manifests.items()]
    md_path = _write(reports_dir / f"tiling_{name}.md", "\n".join(lines) + "\n")
    return md_path, json_path


def write_parents_md(index: dict, reports_dir: Optional[Path] = None) -> Path:
    """reports/parents.md -- the grouping, in a form a human can check by eye."""
    reports_dir = Path(reports_dir or (REPO_ROOT / "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    pmap = parent_map(all_tiles(index))

    lines = [
        "# Parent grouping",
        "",
        "A *parent* is the source micrograph a tile was cut from. Splits are made "
        "by parent, never by tile: two tiles 128 px apart show the same grains "
        "under the same etch, so a validation tile from a training parent is "
        "leakage and its score means nothing.",
        "",
        "| dataset | parents | source images | tiles | tiles per parent (min/median/max) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for ds, parents in pmap.items():
        counts = [p["tiles"] for p in parents.values()]
        lines.append(
            f"| {ds} | {len(parents)} | "
            f"{sum(p['n_images'] for p in parents.values())} | {sum(counts)} | "
            f"{min(counts)}/{int(np.median(counts))}/{max(counts)} |"
        )

    for ds, parents in pmap.items():
        d = index["datasets"][ds]
        lines += ["", f"## {ds}", "",
                  f"- rule: `{(index['settings']['parent_rules'] or {}).get(ds) or 'filename is the parent'}`",
                  f"- {len(parents)} parents, {d['n_tiles']} tiles kept, "
                  f"{d['n_dropped']} dropped below min_boundary_frac, "
                  f"{d['n_excluded']} excluded, {d['n_orphans']} images with no mask",
                  ""]
        if len(parents) <= 60:
            lines += ["| parent | source images | tiles |", "| --- | --- | --- |"]
            for name, p in parents.items():
                lines.append(f"| `{name}` | {p['n_images']} | {p['tiles']} |")
        else:
            lines.append(f"({len(parents)} parents, one per source image — "
                         "not listed individually)")
        if d["excluded"]:
            lines += ["", f"### Excluded ({d['n_excluded']})", ""]
            lines += [f"- `{e['image']}` — {e['why']}" for e in d["excluded"][:60]]
            if d["n_excluded"] > 60:
                lines.append(f"- ... and {d['n_excluded'] - 60} more")
        if d["orphans"]:
            lines += ["", f"### Images with no mask ({d['n_orphans']})", "",
                      "These never had ground truth, so they cannot be tiled. "
                      f"{d['n_orphans_listed']} of them are named explicitly in "
                      "`tiling.exclusions`; the rest are listed here so that "
                      "none of them vanishes silently."]
            lines += [f"- `{o['image']}`" for o in d["orphans"][:60]]
            if d["n_orphans"] > 60:
                lines.append(f"- ... and {d['n_orphans'] - 60} more")
    return _write(reports_dir / "parents.md", "\n".join(lines) + "\n")


def write_tiling_report(
    index: dict,
    folds: dict,
    stats: dict,
    manifests: dict,
    reports_dir: Optional[Path] = None,
) -> tuple:
    """reports/tiling.md + tiling.json: what was cut, what was dropped, what folds."""
    reports_dir = Path(reports_dir or (REPO_ROOT / "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    s = index["settings"]

    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "settings": {k: s[k] for k in sorted(s)},
        "datasets": {
            name: {k: v for k, v in d.items() if k != "tiles"}
            for name, d in index["datasets"].items()
        },
        "folds": stats,
        "manifests": manifests,
    }
    json_path = _write(reports_dir / "tiling.json", json.dumps(payload, indent=2))

    lines = [
        "# Tiling and folds",
        "",
        f"- generated: {payload['generated_utc']}",
        f"- patch {s['patch_size']} px, stride {s['stride']} px "
        f"({100 - int(100 * s['stride'] / s['patch_size'])}% overlap); the last "
        "tile of each row and column is clamped to the image edge, so no pixel "
        "is fabricated. Padding applies only to an image smaller than the patch "
        f"in an axis ({s['pad_mode']} mode), and every such pixel is counted below.",
        f"- tiles below {s['min_boundary_frac']} boundary fraction are dropped",
        f"- nothing is resized and no tile images are written: a tile is a row "
        "in a manifest and the loader crops it on the fly",
        "",
        "## Tiles per dataset",
        "",
        "| dataset | parents | images | tiles | padded px | dropped (low boundary) | excluded | no mask | no boundary map |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, d in payload["datasets"].items():
        lines.append(
            f"| {name} | {d['n_parents']} | {d['n_images']} | {d['n_tiles']} | "
            f"{d['padded_pixels']} | "
            f"{d['n_dropped']} | {d['n_excluded']} | {d['n_orphans']} | "
            f"{d['n_skipped']} |"
        )
    fabricated = sum(d["padded_pixels"] for d in payload["datasets"].values())
    lines += ["", f"**Fabricated pixels: {fabricated}.** "
              + ("Every tile is real image data." if not fabricated else
                 "Some image is smaller than the patch in an axis — see below.")]
    padded_any = [x | {"dataset": name}
                  for name, d in payload["datasets"].items()
                  for x in d["padded_images"]]
    if padded_any:
        lines += ["", "### Images that required padding", "",
                  "| dataset | image | size | pad right | pad bottom | padded px |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for x in padded_any:
            lines.append(
                f"| {x['dataset']} | `{x['image']}` | {x['size'][0]}x{x['size'][1]} "
                f"| {x['pad_right']} | {x['pad_bottom']} | {x['padded_pixels']} |")

    loud = [x for d in payload["datasets"].values() for x in d["single_tile_drops"]]
    if loud:
        lines += ["", "## Images lost entirely to the boundary filter", "",
                  "Each of these is a pre-tiled image whose ONLY tile fell below "
                  "the threshold, so the image is absent from every split.", ""]
        for x in loud:
            lines.append(f"- **{x['dataset']}/{x['image']}** — boundary fraction "
                         f"{x['boundary_fraction']:.5f}")

    lines += ["", "## Folds", "",
              "Steel2 is test-only: the domain-shift fold, evaluated but never "
              "learned from. Steel1 is split by parent with a fixed seed, so its "
              "tiles never straddle train and val. The remaining three datasets "
              "rotate as the held-out validation set. `fold_steel_combined` is a "
              "separate experiment: it pools Steel1+Steel2 into one fresh, "
              "parent-split 70/15/15 train/val/test split (its `held_out` is a "
              "descriptive label, not a real dataset -- it has no single "
              "held-out dataset in the LODO sense), replacing the LODO protocol "
              "for that experiment only. MetalDam/uhcs1/uhcs2 and the LODO "
              "folds above are untouched by it.",
              "",
              "| fold | held out | train tiles | val tiles | train parents | val parents | pos_weight | train frac (mean) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, f in stats.items():
        if name == "test":
            continue
        alias = f" (= {f['alias_of']})" if f.get("alias_of") else ""
        lines.append(
            f"| {name}{alias} | {f['held_out']} | {f['n_train_tiles']} | "
            f"{f['n_val_tiles']} | {f['n_train_parents']} | {f['n_val_parents']} | "
            f"{f['pos_weight']} | {f['train_boundary_fraction']['mean']} |"
        )
    t = stats["test"]
    lines += ["", f"- test manifest: {t['datasets']}, {t['n_tiles']} tiles from "
              f"{t['n_parents']} parents, mean boundary fraction "
              f"{t['boundary_fraction']['mean']}"]

    # Any fold with its own held-out test slice (currently just
    # fold_steel_combined) gets its own line here, distinct from the global
    # test manifest above: same shape, different rows, never confused.
    for name, f in stats.items():
        if name == "test" or "test" not in f:
            continue
        ft = f["test"]
        lines += ["", f"- {name} test manifest: {ft['datasets']}, "
                  f"{ft['n_tiles']} tiles from {ft['n_parents']} parents, "
                  f"mean boundary fraction {ft['boundary_fraction']['mean']} "
                  "(this fold's own held-out parents, never the shared test "
                  "manifest above)"]

    lines += ["", "## Sampling weights", "",
              "Raw tile counts do not measure independent information: Steel1's "
              "tiles come from a handful of micrographs. The sampler weights each "
              "dataset so its share of an epoch matches its share of the "
              "independent scenes.", ""]
    for name, f in stats.items():
        if name == "test":
            continue
        sm = f["sampling"]
        lines += [f"### {name}", "",
                  f"- mode: `{sm['mode']}`", "",
                  "| dataset | parents | raw tiles | weight | effective tiles/epoch |",
                  "| --- | --- | --- | --- | --- |"]
        for ds in sm["raw_counts"]:
            lines.append(
                f"| {ds} | {sm['n_parents'][ds]} | {sm['raw_counts'][ds]} | "
                f"{sm['weights'][ds]} | {sm['effective_counts'][ds]} |"
            )
        lines.append("")

    lines += ["## Manifests", ""]
    lines += [f"- `{name}`: `{path}`" for name, path in manifests.items()]
    md_path = _write(reports_dir / "tiling.md", "\n".join(lines) + "\n")
    return md_path, json_path


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path
