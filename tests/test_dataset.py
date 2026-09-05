"""Tests for src/dataset.py. RUN THESE IN notebooks/04_dataset.ipynb.

There is no local Python environment for this project, so nothing here is
executed on the machine that wrote it. The notebook runs pytest on the host,
where torch, albumentations and the mounted datasets exist, and prints the
result as part of its checks.

Tests that need the real data skip themselves when it is absent, so the file
still runs anywhere; the notebook's checks cell fails if they were skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src import dataset as ds
from src.paths import REPO_ROOT

MANIFEST = REPO_ROOT / "reports" / "manifests" / "dev.csv"
EXTRACTION = REPO_ROOT / "reports" / "gt_extraction.json"
PATCH = 256


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def settings():
    return ds.load_config()


@pytest.fixture(scope="module")
def rows():
    if not MANIFEST.is_file():
        pytest.skip(f"{MANIFEST} not present; run step 3 first")
    return ds.load_manifest(MANIFEST)


@pytest.fixture(scope="module")
def roots():
    from src import paths as paths_mod

    resolved = paths_mod.resolve_paths()
    return {"data_root": resolved["data_root"],
            "gt_root": Path(resolved["gt_boundaries_root"])}


def _synthetic_pair(size: int = PATCH, seed: int = 0):
    """A grayscale tile and a thin binary boundary network over it."""
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, size=(size, size), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    for k in range(20, size, 37):
        mask[k:k + 2, :] = 1
        mask[:, k:k + 2] = 1
    return image, mask


def _first_available(rows, roots, predicate=lambda r: True, limit=40):
    for row in rows[:limit]:
        if not predicate(row):
            continue
        try:
            ds.read_pair(row, crops=ds.load_crops(), roots=roots)
        except Exception:
            continue
        return row
    return None


# --------------------------------------------------------------------------
# masks stay binary, over many seeds
# --------------------------------------------------------------------------
def test_mask_stays_binary_over_many_seeds(settings):
    """One lucky seed proves nothing: interpolation errors are random."""
    import albumentations  # noqa: F401  (skip cleanly if absent)

    image, mask = _synthetic_pair()
    transform = ds.spatial_transform(settings)
    seen_changes = 0
    for seed in range(60):
        np.random.seed(seed)
        try:
            import random

            random.seed(seed)
        except Exception:
            pass
        out = transform(image=image, mask=mask)
        got = np.asarray(out["mask"])
        assert set(np.unique(got).tolist()) <= {0, 1}, (
            f"seed {seed} produced mask values {np.unique(got)[:5]}")
        ds.assert_binary(got.astype(np.float32), f"seed {seed}")
        if not np.array_equal(got, mask):
            seen_changes += 1
    assert seen_changes > 0, "no seed changed the mask; the transform is inert"


def test_full_pipeline_mask_stays_binary(settings, rows, roots):
    """The same guarantee end to end, through __getitem__ rather than a Compose."""
    train = [r for r in rows if r["split"] == "train"]
    if not train:
        pytest.skip("no train rows in the manifest")
    row = _first_available(train, roots)
    if row is None:
        pytest.skip("source images are not mounted on this host")
    dataset = ds.TileDataset([row] * 25, settings=settings, augment=True,
                             crops=ds.load_crops(), roots=roots)
    for i in range(len(dataset)):
        sample = dataset[i]
        values = np.unique(sample["mask"])
        assert set(values.tolist()) <= {0.0, 1.0}, values[:5]


# --------------------------------------------------------------------------
# image and mask receive the same spatial transform
# --------------------------------------------------------------------------
def test_spatial_transform_is_synchronized(settings):
    """Pass the mask through the IMAGE path and require an identical result.

    ``additional_targets={"mask_image": "image"}`` routes a copy of the mask
    through the image pipeline. With both sides forced to nearest-neighbour,
    a synchronized transform must produce byte-identical arrays; any drift
    means image and mask were sampled with different geometry, which is the
    failure that silently destroys a segmentation model.
    """
    import cv2

    image, mask = _synthetic_pair(seed=3)
    transform = ds.spatial_transform(settings, image_interpolation=cv2.INTER_NEAREST)
    for seed in range(25):
        np.random.seed(seed)
        import random

        random.seed(seed)
        out = transform(image=image, mask=mask, mask_image=mask)
        through_mask = np.asarray(out["mask"])
        through_image = np.asarray(out["mask_image"])
        assert through_mask.shape == through_image.shape
        assert np.array_equal(through_mask, through_image), (
            f"seed {seed}: mask and image paths disagree on "
            f"{(through_mask != through_image).sum()} pixels")


def test_photometric_never_receives_the_mask(settings):
    """The mask is not an argument to the photometric pipeline at all."""
    photometric = ds.photometric_transform(settings)
    out = photometric(image=_synthetic_pair()[0])
    assert set(out) == {"image"}, f"photometric returned {sorted(out)}"


def test_border_mode_is_reflect_not_constant(settings):
    """A constant fill would label a fabricated straight edge as background."""
    import cv2

    report = ds.verify_interpolation(ds.spatial_transform(settings))
    assert report["border_verified"], (
        "no geometric op exposed a border mode; cannot verify structurally")
    assert not report["border_unverified"], report["border_unverified"]
    assert cv2.BORDER_REFLECT_101 == 4


def test_rotation_leaves_no_constant_fill(settings):
    """Measured on the produced tile, over many seeds, not read off the config."""
    import random

    image, mask = _synthetic_pair(seed=11)
    transform = ds.spatial_transform(settings)
    worst = 0.0
    for seed in range(40):
        np.random.seed(seed)
        random.seed(seed)
        out = transform(image=image, mask=mask)
        fill = ds.constant_border_region(np.asarray(out["image"]))
        worst = max(worst, fill["fraction"])
    assert worst < 0.01, (
        f"largest uniform border-touching region {worst:.4f} of the tile; "
        "a geometric transform is filling with a constant")


def test_constant_fill_detector_finds_a_real_wedge():
    """The detector must fire on fill it is shown, or it proves nothing above."""
    tile = np.random.default_rng(0).integers(20, 240, size=(256, 256)).astype(np.float32)
    assert ds.constant_border_region(tile)["fraction"] < 0.01
    tile[:60, :60] = 0.0            # a corner wedge of constant fill
    found = ds.constant_border_region(tile)
    assert found["fraction"] > 0.05, found
    assert found["value"] == 0.0


def test_interpolation_is_configured_as_claimed(settings):
    import cv2

    report = ds.verify_interpolation(ds.spatial_transform(settings))
    assert report["verified"], (
        "no spatial op exposed mask_interpolation; cannot verify structurally")
    for name in report["verified"]:
        assert name  # every verified op passed the INTER_NEAREST check
    assert cv2.INTER_NEAREST == 0


# --------------------------------------------------------------------------
# shapes, dtypes, keys
# --------------------------------------------------------------------------
def test_sample_shapes_dtypes_and_keys(settings, rows, roots):
    row = _first_available(rows, roots)
    if row is None:
        pytest.skip("source images are not mounted on this host")
    dataset = ds.TileDataset([row], settings=settings, augment=False,
                             crops=ds.load_crops(), roots=roots)
    sample = dataset[0]

    assert set(sample) == {"image", "mask", "dataset", "parent_id", "tile_id"}
    assert sample["image"].shape == (1, PATCH, PATCH)
    assert sample["mask"].shape == (1, PATCH, PATCH)
    assert sample["image"].dtype == np.float32
    assert sample["mask"].dtype == np.float32
    for key in ("dataset", "parent_id", "tile_id"):
        assert isinstance(sample[key], str) and sample[key], key
    assert abs(float(sample["image"].mean())) < 1e-3
    assert abs(float(sample["image"].std()) - 1.0) < 1e-2


def test_val_split_returns_the_raw_tile(settings, rows, roots):
    val = [r for r in rows if r["split"] == "val"]
    if not val:
        pytest.skip("no val rows in the manifest")
    row = _first_available(val, roots)
    if row is None:
        pytest.skip("source images are not mounted on this host")
    dataset = ds.TileDataset([row], settings=settings, crops=ds.load_crops(),
                             roots=roots)
    assert dataset.augment is False
    a = dataset[0]["image"]
    b = dataset[0]["image"]
    assert np.array_equal(a, b), "val tiles must be deterministic"


# --------------------------------------------------------------------------
# the two reconciled MetalDam pairs
# --------------------------------------------------------------------------
def test_reconciled_pairs_load_at_the_cropped_size(settings, rows, roots):
    if not EXTRACTION.is_file():
        pytest.skip(f"{EXTRACTION} not present; run step 2 first")
    crops = ds.load_crops()
    if not crops:
        pytest.skip("no reconciled pairs recorded")

    checked = 0
    for (name, image_name), crop in crops.items():
        matching = [r for r in rows
                    if r["dataset"] == name and r["source_image"] == image_name]
        if not matching:
            continue
        try:
            image, gt = ds.read_pair(matching[0], crops=crops, roots=roots)
        except ds.DatasetError as exc:
            if "cannot find" in str(exc):
                pytest.skip("source images are not mounted on this host")
            raise
        assert image.shape == gt.shape, (image.shape, gt.shape)
        assert image.shape == (crop["height"], crop["width"]), (
            f"{name}/{image_name} loaded at {image.shape}, expected "
            f"{(crop['height'], crop['width'])}")
        sample = ds.TileDataset([matching[0]], settings=settings, augment=False,
                                crops=crops, roots=roots)[0]
        assert sample["image"].shape == (1, PATCH, PATCH)
        checked += 1
    if not checked:
        pytest.skip("no reconciled pair appears in this manifest")


# --------------------------------------------------------------------------
# a stale manifest must be refused
# --------------------------------------------------------------------------
def test_padded_manifest_row_raises(tmp_path, rows):
    """Padding was eliminated in step 3; its reappearance is a stale manifest."""
    import csv

    stale = tmp_path / "stale.csv"
    row = dict(rows[0])
    row["pad_right"] = 123
    with open(stale, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(ds.DatasetError) as err:
        ds.load_manifest(stale)
    assert "stale" in str(err.value).lower()
    assert "123" in str(err.value)


def test_manifest_missing_columns_raises(tmp_path, rows):
    import csv

    bad = tmp_path / "bad.csv"
    row = {k: v for k, v in rows[0].items() if k != "parent_id"}
    with open(bad, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ds.DatasetError) as err:
        ds.load_manifest(bad)
    assert "parent_id" in str(err.value)


def test_tile_outside_the_image_raises():
    array = np.zeros((300, 300), dtype=np.uint8)
    with pytest.raises(ds.DatasetError):
        ds.crop_tile(array, x=200, y=0, patch=PATCH)


# --------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------
def test_normalize_is_per_image():
    a = ds.normalize(np.full((32, 32), 40, dtype=np.uint8) + np.arange(32, dtype=np.uint8))
    b = ds.normalize(np.full((32, 32), 200, dtype=np.uint8) + np.arange(32, dtype=np.uint8))
    assert abs(float(a.mean())) < 1e-4 and abs(float(b.mean())) < 1e-4
    assert abs(float(a.std()) - 1.0) < 1e-3 and abs(float(b.std()) - 1.0) < 1e-3
    # Two tiles that differ only by a constant offset normalise identically:
    # that is what "no global statistics" buys across SEM and optical sources.
    assert np.allclose(a, b, atol=1e-5)


def test_normalize_flat_tile_does_not_divide_by_zero():
    out = ds.normalize(np.full((16, 16), 7, dtype=np.uint8))
    assert np.all(np.isfinite(out)) and np.allclose(out, 0.0)


# --------------------------------------------------------------------------
# image cache
# --------------------------------------------------------------------------
def test_cache_estimate_matches_the_manifest(rows):
    """The footprint is arithmetic on recorded dimensions, not a measurement."""
    est = ds.estimate_cache_bytes(rows)
    unique = {(r["dataset"], r["source_image"]) for r in rows}
    assert est["images"] == len(unique)
    expected = sum(int(r["crop_width"]) * int(r["crop_height"]) * 2
                   for r in {(r["dataset"], r["source_image"]): r
                             for r in rows}.values())
    assert est["bytes"] == expected
    assert sum(v["images"] for v in est["per_dataset"].values()) == est["images"]


def test_preload_fills_the_cache_and_reports_its_size(settings, rows, roots):
    row = _first_available(rows, roots)
    if row is None:
        pytest.skip("source images are not mounted on this host")
    dataset = ds.TileDataset([row] * 5, settings=settings, augment=False,
                             crops=ds.load_crops(), roots=roots)
    assert dataset.cache_bytes() == 0
    stats = dataset.preload()
    assert stats["images_decoded"] == 1
    assert dataset.cache_bytes() > 0
    assert stats["measured_mb"] > 0
    # A second preload is a no-op: the images are already held.
    assert dataset.preload()["images_decoded"] == 0


def test_preload_refuses_to_exceed_the_cap(settings, rows, roots):
    """Better to stop than to exhaust a session's memory silently."""
    tiny = dict(settings)
    tiny["cache_max_mb"] = 0.000001
    dataset = ds.TileDataset(rows[:5], settings=tiny, augment=False,
                             crops={}, roots=roots)
    with pytest.raises(ds.DatasetError) as err:
        dataset.preload()
    assert "cache_max_mb" in str(err.value)


def test_cache_can_be_switched_off(settings, rows, roots):
    row = _first_available(rows, roots)
    if row is None:
        pytest.skip("source images are not mounted on this host")
    dataset = ds.TileDataset([row], settings=settings, augment=False,
                             crops=ds.load_crops(), roots=roots,
                             cache_images=False)
    dataset[0]
    assert dataset.cache_bytes() == 0
    with pytest.raises(ds.DatasetError):
        dataset.preload()


# --------------------------------------------------------------------------
# sampler
# --------------------------------------------------------------------------
def test_sampler_weights_come_from_fold_stats(settings, rows, roots):
    stats_path = REPO_ROOT / "configs" / "fold_stats.yaml"
    if not stats_path.is_file():
        pytest.skip("configs/fold_stats.yaml not present; run step 3")
    stats = ds.load_fold_stats()
    train = [r for r in rows if r["split"] == "train"]
    if not train:
        pytest.skip("no train rows")

    dataset = ds.TileDataset(train, settings=settings, augment=False,
                             crops={}, roots=roots)
    sampler = ds.build_sampler(dataset, "dev", fold_stats=stats)
    weights = stats["folds"]["dev"]["sampling"]["weights"]
    assert len(sampler.weights) == len(train)
    for row, w in zip(train, sampler.weights.tolist()):
        assert abs(w - float(weights[row["dataset"]])) < 1e-6
    assert ds.pos_weight("dev", fold_stats=stats) > 1.0
