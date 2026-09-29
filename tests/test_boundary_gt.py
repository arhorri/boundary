"""Tests for src/boundary_gt.py's small-region cleanup (min_region_area_px).

Scoped to merge_small_regions() and boundaries_from_labels() together -- the
two functions the cleanup actually touches -- and to load_config()'s wiring
of the new key. extract_folder()'s full orchestration needs a real
audit-report-shaped folder_report and image/mask files on disk to exercise
end to end; that fixture is not built here, so the byte-identical guarantee
is verified at the function it is implemented in (merge_small_regions itself
returns the input unchanged when off), not by re-running the whole pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import boundary_gt


def test_min_region_area_off_leaves_labels_byte_identical():
    """min_area_px <= 0 is OFF: the exact array, unchanged, every time."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, size=(30, 30))

    merged, stats = boundary_gt.merge_small_regions(labels, 0)
    assert np.array_equal(merged, labels)
    assert stats == {"enabled": False, "min_area_px": 0,
                     "n_components_total": None, "n_components_merged": 0,
                     "n_pixels_reassigned": 0}

    # And boundaries_from_labels on the untouched labels is unaffected too --
    # the guarantee that matters is on what actually gets written to disk.
    before = boundary_gt.boundaries_from_labels(labels)
    after = boundary_gt.boundaries_from_labels(merged)
    assert np.array_equal(before, after)


def test_min_region_area_negative_is_also_off():
    labels = np.zeros((10, 10), dtype=np.int32)
    merged, stats = boundary_gt.merge_small_regions(labels, -5)
    assert np.array_equal(merged, labels)
    assert stats["enabled"] is False


def _labels_with_one_speckle(size=40, speckle_area=20):
    """A large phase (label 0) with one ~speckle_area px island (label 1),
    entirely enclosed and not touching the image border, so it has exactly
    one bordering phase to be merged into.
    """
    labels = np.zeros((size, size), dtype=np.int32)
    # A 4x5 block = 20 px, placed well inside the tile.
    h, w = 4, speckle_area // 4
    assert h * w == speckle_area
    r0, c0 = size // 2 - h // 2, size // 2 - w // 2
    labels[r0:r0 + h, c0:c0 + w] = 1
    return labels, (r0, c0, h, w)


def test_a_20px_speckle_loses_its_boundary_at_min_50_and_keeps_it_at_min_0():
    """The exact case named in the task: a 20 px speckle region inside a
    large phase must produce no boundary loop once min_region_area_px is set
    above its area, and must still produce one when the cleanup is off.
    """
    labels, (r0, c0, h, w) = _labels_with_one_speckle(size=40, speckle_area=20)

    # Off (min 0): the speckle survives and draws a closed boundary loop.
    kept, kept_stats = boundary_gt.merge_small_regions(labels, 0)
    assert kept_stats["enabled"] is False
    raw_kept = boundary_gt.boundaries_from_labels(kept)
    assert raw_kept.sum() > 0, "the speckle's boundary loop must still be drawn"
    assert raw_kept[r0 - 1:r0 + h + 1, c0 - 1:c0 + w + 1].any(), (
        "the boundary loop must actually surround the speckle, not appear "
        "somewhere unrelated in the tile")

    # On at min 50 (> the speckle's 20 px area): merged away, no loop at all.
    merged, merge_stats = boundary_gt.merge_small_regions(labels, 50)
    assert merge_stats["enabled"] is True
    assert merge_stats["min_area_px"] == 50
    assert merge_stats["n_components_merged"] == 1
    assert merge_stats["n_pixels_reassigned"] == 20
    assert (merged == 0).all(), "the speckle must be fully absorbed into label 0"
    raw_merged = boundary_gt.boundaries_from_labels(merged)
    assert raw_merged.sum() == 0, (
        "no label transition remains anywhere in the tile, so there must be "
        "no boundary pixel left")


def test_min_region_area_below_the_speckle_area_leaves_it_alone():
    """40 < 20 is false: a threshold BELOW the speckle's area must not touch
    it -- the boundary between "kept" and "merged" is the area itself, not
    merely "cleanup is enabled at all".
    """
    labels, _ = _labels_with_one_speckle(size=40, speckle_area=20)
    merged, stats = boundary_gt.merge_small_regions(labels, 10)
    assert stats["enabled"] is True
    assert stats["n_components_merged"] == 0
    assert np.array_equal(merged, labels)
    assert boundary_gt.boundaries_from_labels(merged).sum() > 0


def test_merge_reassigns_to_the_majority_border_label_not_an_arbitrary_one():
    """A speckle bordering TWO phases must absorb into whichever one actually
    surrounds most of it, not the first or the smallest one found.

    Geometry chosen so the majority is unambiguous under a 4-connected
    (cross) border, which is what ``scipy.ndimage.binary_dilation`` uses by
    default: a 4x4 speckle's border ring is exactly its four edges, 4 px
    each, no corners. Label 2 occupies only the two columns immediately past
    the speckle's RIGHT edge, so only that one edge (4 px) borders label 2;
    the other three edges (12 px) border label 0 -- a clean 12:4 majority.
    """
    size = 20
    labels = np.zeros((size, size), dtype=np.int32)
    labels[:, 18:] = 2                # label 2 starts right past the speckle
    labels[8:12, 14:18] = 1           # 4x4 = 16 px speckle, cols 14-17 (< 18)

    merged, stats = boundary_gt.merge_small_regions(labels, 20)
    assert stats["n_components_merged"] == 1
    assert not (merged == 1).any(), "the speckle label must be gone"
    assert (merged[8:12, 14:18] == 0).all(), (
        "the speckle must merge into label 0, which borders 12 of its 16 "
        "ring pixels, not label 2, which borders only 4")
    assert (merged[:, 18:] == 2).all(), "label 2's own territory must be untouched"


def test_a_component_covering_the_whole_tile_is_left_alone():
    """No neighbour exists to merge into -- nothing should be fabricated."""
    labels = np.ones((10, 10), dtype=np.int32)
    merged, stats = boundary_gt.merge_small_regions(labels, 1000)
    assert stats["n_components_merged"] == 0
    assert np.array_equal(merged, labels)


def test_load_config_accepts_a_per_dataset_min_region_area_px(tmp_path):
    """The new key follows the exact same folder-keyed-dict pattern
    exclusions/artifact_colours already use, and load_config must not reject
    it as unknown.
    """
    config_path = tmp_path / "default.yaml"
    config_path.write_text(
        "boundary_gt:\n"
        "  min_region_area_px:\n"
        "    Steel2: 50\n"
    )
    settings = boundary_gt.load_config(config_path)
    assert settings["min_region_area_px"] == {"Steel2": 50}
    # Every dataset NOT listed reads as off (0), matching exclusions'/
    # artifact_colours' own "absent means untouched" convention.
    assert settings["min_region_area_px"].get("Steel1", 0) == 0


def test_load_config_min_region_area_px_defaults_to_empty():
    assert boundary_gt.DEFAULTS["min_region_area_px"] == {}


# --------------------------------------------------------------------------
# mode-check diagnostic: is a MODE B class actually a painted line?
# --------------------------------------------------------------------------
def _thin_line_mask(size=40, col=5):
    """A full-height, 1 px wide vertical line -- one component, width ~1.

    Every pixel has a background neighbour immediately left/right, so the
    uncorrected ``2 x distance-transform`` (no width_hint parity fix, unlike
    measured_line_width) works out to ~2.0, not ~1.0 -- that offset is
    exactly what this module's width measurements deliberately skip (see
    class_shape_profile's docstring), so the raw value, not the calibrated
    one, is what a test of it must expect.
    """
    m = np.zeros((size, size), dtype=bool)
    m[:, col] = True
    return m


def _blob_mask(size=40, cy=20, cx=20, radius=10):
    """A solid disk. Its skeleton collapses to a tight cluster near the
    centre (not a ridge with tapering ends, the way a rectangle's would),
    where distance-to-background is ~radius everywhere -- a robust,
    unambiguous "thick" shape to test against, far from `col=5` above.
    """
    rr, cc = np.ogrid[:size, :size]
    return (rr - cy) ** 2 + (cc - cx) ** 2 <= radius ** 2


def test_class_shape_profile_tells_a_thin_line_from_a_blob():
    line = boundary_gt.class_shape_profile(_thin_line_mask())
    assert line["n_pixels"] == 40
    assert line["width_p50"] == pytest.approx(2.0, abs=0.5)
    assert line["share_thin"] == 1.0
    assert line["n_components"] == 1
    assert line["largest_cc_share"] == 1.0

    blob = boundary_gt.class_shape_profile(_blob_mask())
    assert blob["width_p50"] > 8.0, "a solid disk of radius 10 is not a thin line"
    assert blob["share_thin"] == 0.0
    assert blob["n_components"] == 1
    assert blob["largest_cc_share"] == 1.0


def test_class_shape_profile_empty_mask_is_all_none():
    prof = boundary_gt.class_shape_profile(np.zeros((10, 10), dtype=bool))
    assert prof["n_pixels"] == 0
    assert prof["width_p50"] is None and prof["width_p95"] is None
    assert prof["share_thin"] is None
    assert prof["largest_cc_share"] is None
    assert prof["mean_inside"] is None and prof["mean_outside"] is None


def test_class_shape_profile_reads_intensity_inside_vs_outside():
    mask = _blob_mask(size=20, cy=10, cx=10, radius=5)
    gray = np.full((20, 20), 200.0)
    gray[mask] = 10.0
    prof = boundary_gt.class_shape_profile(mask, gray=gray)
    assert prof["mean_inside"] == pytest.approx(10.0)
    assert prof["mean_outside"] == pytest.approx(200.0)


def test_classify_line_class_components_splits_thin_from_thick():
    size = 40
    labels = np.zeros((size, size), dtype=np.int32)
    labels[:, 5] = 1                              # thin line, width ~1
    blob = _blob_mask(size=size, cy=25, cx=25, radius=10)   # far from col 5
    labels[blob] = 1

    line_mask, blob_mask, widths = boundary_gt.classify_line_class_components(
        labels, class_index=1, blob_thickness_px=8.0)

    assert line_mask[:, 5].all()
    assert not line_mask[blob].any(), "the blob must not land in line_mask"
    assert blob_mask[blob].all()
    assert not blob_mask[:, 5].any(), "the line must not land in blob_mask"
    assert len(widths) == 2
    assert sorted(widths.values())[0] < 8.0 < sorted(widths.values())[1]


def test_classify_line_class_components_empty_class_returns_empty_masks():
    labels = np.zeros((10, 10), dtype=np.int32)
    line_mask, blob_mask, widths = boundary_gt.classify_line_class_components(
        labels, class_index=1, blob_thickness_px=8.0)
    assert not line_mask.any() and not blob_mask.any()
    assert widths == {}


def test_mode_a_raw_from_line_class_keeps_line_pixels_and_outlines_the_blob():
    """The thin component's OWN pixels must survive unchanged (a real MODE A
    extraction would feed them straight to clean_boundary, unskeletonized
    here). The thick component must NOT survive as a filled blob -- only its
    OUTLINE, exactly what boundaries_from_labels already draws for any MODE B
    phase -- because skeletonizing a genuine filled region would fabricate a
    boundary through its interior.
    """
    size = 40
    labels = np.zeros((size, size), dtype=np.int32)
    labels[:, 5] = 1                              # thin line
    blob = _blob_mask(size=size, cy=25, cx=25, radius=10)   # thick, far from col 5
    labels[blob] = 1

    raw = boundary_gt.mode_a_raw_from_line_class(labels, class_index=1,
                                                 blob_thickness_px=8.0)
    assert raw[:, 5].all(), "the line's own pixels must pass straight through"
    centre = _blob_mask(size=size, cy=25, cx=25, radius=6)   # well inside the disk
    assert not raw[centre].any(), (
        "the blob's INTERIOR must not be marked -- only its outline may be")
    assert raw[blob].any(), "the blob must still draw SOME boundary around its outline"


def test_mode_a_raw_from_line_class_empty_class_is_empty():
    labels = np.zeros((10, 10), dtype=np.int32)
    raw = boundary_gt.mode_a_raw_from_line_class(labels, class_index=1,
                                                 blob_thickness_px=8.0)
    assert not raw.any()


def _write_png(path, array):
    from PIL import Image

    Image.fromarray(array.astype(np.uint8)).save(path)


def test_label_value_histogram_pools_exact_values_across_files(tmp_path):
    a = np.zeros((4, 4), dtype=np.uint8)
    a[:2, :] = 10                                  # half the pixels: value 10
    b = np.full((4, 4), 10, dtype=np.uint8)        # all pixels: value 10
    b[0, 0] = 20
    path_a, path_b = tmp_path / "a.png", tmp_path / "b.png"
    _write_png(path_a, a)
    _write_png(path_b, b)

    hist = boundary_gt.label_value_histogram([(None, path_a), (None, path_b)])
    shares = {tuple(v): s for v, s in hist}
    # 8 (a) + 16 (b) = 24 pixels total; value 10 covers 8 + 15 = 23 of them.
    assert shares[(10, 10, 10)] == pytest.approx(23 / 32)
    assert shares[(0, 0, 0)] == pytest.approx(8 / 32)
    assert shares[(20, 20, 20)] == pytest.approx(1 / 32)


def test_pooled_class_shape_profile_pools_across_files(tmp_path):
    """Two masks, same palette; one all-background, one with a 4x4 class-1
    block. Pooling must weight by actual pixel count, not average the two
    files' shares as if they carried equal weight -- the empty file
    contributes 0 class pixels, not "half of nothing happened".
    """
    palette = [(0, 0, 0), (255, 255, 255)]
    size = 10
    empty_mask = np.zeros((size, size, 3), dtype=np.uint8)          # all class 0
    filled_mask = np.zeros((size, size, 3), dtype=np.uint8)
    filled_mask[2:6, 2:6] = 255                                    # 16 px of class 1
    empty_img = np.full((size, size), 50, dtype=np.uint8)
    filled_img = np.full((size, size), 50, dtype=np.uint8)
    filled_img[2:6, 2:6] = 5                                       # dark under class 1

    p_mask_empty, p_img_empty = tmp_path / "m0.png", tmp_path / "i0.png"
    p_mask_full, p_img_full = tmp_path / "m1.png", tmp_path / "i1.png"
    _write_png(p_mask_empty, empty_mask)
    _write_png(p_img_empty, empty_img)
    _write_png(p_mask_full, filled_mask)
    _write_png(p_img_full, filled_img)

    prof = boundary_gt.pooled_class_shape_profile(
        [(p_img_empty, p_mask_empty), (p_img_full, p_mask_full)], palette,
        class_index=1)
    assert prof["n_files_sampled"] == 2
    assert prof["n_files_with_class"] == 1
    assert prof["pixel_share"] == pytest.approx(16 / (2 * size * size))
    assert prof["mean_inside"] == pytest.approx(5.0)
    assert prof["mean_outside"] == pytest.approx(50.0)
