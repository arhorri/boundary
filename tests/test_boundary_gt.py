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
