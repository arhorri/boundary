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


# --------------------------------------------------------------------------
# forced mode-check: threshold sensitivity + mode_b_vs_a_raw
# --------------------------------------------------------------------------
def test_threshold_labels_basic():
    gray = np.array([[10, 200], [250, 5]], dtype=float)
    labels = boundary_gt.threshold_labels(gray, 128)
    assert labels.tolist() == [[1, 0], [0, 1]]


def test_mode_b_vs_a_raw_matches_the_two_underlying_calls():
    size = 20
    labels = np.zeros((size, size), dtype=np.int32)
    labels[:, 5] = 1
    current, proposed = boundary_gt.mode_b_vs_a_raw(labels, class_index=1,
                                                     blob_thickness_px=8.0)
    assert np.array_equal(current, boundary_gt.boundaries_from_labels(labels))
    assert np.array_equal(
        proposed, boundary_gt.mode_a_raw_from_line_class(labels, 1, 8.0))


def test_threshold_sensitivity_profile_shows_fragmentation_dropping_toward_white(tmp_path):
    """One line, one 5 px "faint" gap at intensity 200 -- neither fully dark
    nor fully white. At threshold=128 the gap is NOT dark, splitting the
    line into 2 components; at threshold=220 the gap IS dark, reconnecting
    it into 1 -- exactly the mechanism the task suspects is inflating
    Steel2's component count, made concrete and checkable without
    executing anything on real data.
    """
    size = 50
    gray = np.full((size, size), 250, dtype=np.uint8)
    gray[25, 0:20] = 10
    gray[25, 20:25] = 200          # the "faint anti-aliased" gap
    gray[25, 25:50] = 10
    path = tmp_path / "m.png"
    _write_png(path, gray)

    result = boundary_gt.threshold_sensitivity_profile(
        [(None, path)], [128, 220], thin_width_px=8.0)

    assert result["128"]["n_components_total"] == 2
    assert result["128"]["largest_cc_share"] == pytest.approx(25 / 45)
    assert result["128"]["dark_share"] == pytest.approx(45 / (size * size))

    assert result["220"]["n_components_total"] == 1
    assert result["220"]["largest_cc_share"] == pytest.approx(1.0)
    assert result["220"]["dark_share"] == pytest.approx(50 / (size * size))

    assert result["128"]["per_file"][0]["n_components"] == 2
    assert result["220"]["per_file"][0]["n_components"] == 1

    # the mechanism this task is checking for, stated as an assertion: the
    # fragmentation measured at the low cut must be worse than at the high one.
    assert result["220"]["largest_cc_share"] > result["128"]["largest_cc_share"]


def test_threshold_sensitivity_profile_no_pairs_raises():
    with pytest.raises(boundary_gt.ExtractionError):
        boundary_gt.threshold_sensitivity_profile([], [128])


def test_threshold_sensitivity_profile_reads_each_file_once_per_call(tmp_path, monkeypatch):
    """Every threshold in the sweep must reuse the SAME read, not re-read
    the file from disk once per threshold -- the whole point of sweeping in
    one function instead of calling a per-threshold profile in a loop.
    """
    from src import audit as audit_mod

    size = 20
    gray = np.full((size, size), 250, dtype=np.uint8)
    gray[10, :] = 10
    path = tmp_path / "m.png"
    _write_png(path, gray)

    calls = []
    real_read = audit_mod.read_array

    def counting_read(p):
        calls.append(p)
        return real_read(p)

    monkeypatch.setattr(audit_mod, "read_array", counting_read)
    boundary_gt.threshold_sensitivity_profile([(None, path)], [64, 128, 192, 224, 240])
    assert calls == [path], f"expected exactly one read, got {calls}"


# --------------------------------------------------------------------------
# mode_a_line_class wired into extract_folder (per-dataset, default off)
# --------------------------------------------------------------------------
def _crossing_lines_mask(size=48):
    """White background, a 6 px black vertical and horizontal bar crossing,
    plus a filled black disk far from both -- a line network and a blob in the
    same class, as Steel2's dark class is suspected to be. The bars are 6 px
    wide on purpose: a 1 px line's find_boundaries band is 3 px of solid
    boundary that skeletonizes straight back to its centreline, so the
    both-edges artefact this option removes only exists once a line is wide
    enough for its two edge outlines to leave a gap between them."""
    m = np.full((size, size, 3), 255, dtype=np.uint8)
    m[:, 10:16] = 0
    m[32:38, :] = 0
    rr, cc = np.ogrid[:size, :size]
    m[(rr - 14) ** 2 + (cc - 34) ** 2 <= 36] = 0
    return m


def _make_folder(tmp_path, name, masks):
    """An audit-report-shaped ``folder_report`` over real files on disk, which
    is all extract_folder needs to run end to end."""
    root = tmp_path / name
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    files = []
    for i, m in enumerate(masks):
        _write_png(root / "images" / f"t{i}.png", np.full(m.shape[:2], 128, np.uint8))
        _write_png(root / "masks" / f"t{i}.png", m)
        colours, counts = np.unique(m.reshape(-1, 3), axis=0, return_counts=True)
        files.append({"mask": {"width": m.shape[1], "height": m.shape[0]},
                      "palette": {"top_colours": [
                          {"colour": c.tolist(), "pixels": int(n)}
                          for c, n in zip(colours, counts)]}})
    return {"folder": name, "path": str(root), "files": files}


def _settings_with(**overrides):
    s = dict(boundary_gt.DEFAULTS)
    s.update(overrides)
    return s


def _expected_png(mask, settings, raw_fn):
    """What the pipeline must have written, rebuilt from its primitives."""
    palette = boundary_gt.derive_palette(
        {"files": [{"mask": {"width": mask.shape[1], "height": mask.shape[0]},
                    "palette": {"top_colours": [
                        {"colour": c.tolist(), "pixels": int(n)} for c, n in zip(
                            *np.unique(mask.reshape(-1, 3), axis=0, return_counts=True))]}}]},
        settings)
    labels, _, _ = boundary_gt.snap_to_palette(mask, palette)
    out, *_ = boundary_gt.clean_boundary(raw_fn(labels, palette), settings)
    return out


def _read_png(path):
    from PIL import Image

    return np.asarray(Image.open(path))


def test_extract_folder_without_the_option_is_exactly_the_mode_b_pipeline(tmp_path):
    mask = _crossing_lines_mask()
    report = _make_folder(tmp_path, "DS", [mask])
    settings = _settings_with()
    result = boundary_gt.extract_folder(report, "B", tmp_path / "out", settings)
    assert result["mode_a_line_class"] is None and result["n_processed"] == 1

    expected = _expected_png(
        mask, settings, lambda labels, palette: boundary_gt.boundaries_from_labels(labels))
    assert np.array_equal(_read_png(tmp_path / "out" / "DS" / "t0.png"), expected)


def test_line_class_naming_another_dataset_leaves_this_one_byte_identical(tmp_path):
    """Steel1 must not move when Steel2 is switched to MODE A: the option is
    keyed by folder, and a folder it does not name is written exactly as before.
    """
    mask = _crossing_lines_mask()
    off = boundary_gt.extract_folder(_make_folder(tmp_path / "a", "Steel1", [mask]), "B",
                                     tmp_path / "out_off", _settings_with())
    other = boundary_gt.extract_folder(
        _make_folder(tmp_path / "b", "Steel1", [mask]), "B", tmp_path / "out_other",
        _settings_with(mode_a_line_class={"Steel2": 8}))
    assert off["mode_a_line_class"] is None and other["mode_a_line_class"] is None
    assert ((tmp_path / "out_off" / "Steel1" / "t0.png").read_bytes()
            == (tmp_path / "out_other" / "Steel1" / "t0.png").read_bytes())


def test_line_class_option_treats_the_darkest_class_as_the_line(tmp_path):
    mask = _crossing_lines_mask()
    report = _make_folder(tmp_path, "DS", [mask])
    settings = _settings_with(mode_a_line_class={"DS": 8})
    result = boundary_gt.extract_folder(report, "B", tmp_path / "out", settings)

    record = result["mode_a_line_class"]
    assert record == {"class_index": 1, "colour": [0, 0, 0], "blob_thickness_px": 8.0}

    expected = _expected_png(
        mask, settings,
        lambda labels, palette: boundary_gt.mode_a_raw_from_line_class(labels, 1, 8.0))
    written = _read_png(tmp_path / "out" / "DS" / "t0.png")
    assert np.array_equal(written, expected)

    mode_b = _expected_png(
        mask, settings, lambda labels, palette: boundary_gt.boundaries_from_labels(labels))
    assert not np.array_equal(written, mode_b), "the option must change the ground truth"


def test_line_class_fewer_regions_than_mode_b_on_a_line_network(tmp_path):
    """The reason for the option, on a synthetic mask: outlining both edges of
    a wide line traps the line's own pixels as a region of their own (the
    cross-shaped interior of the two bars), which drawing the line itself does
    not."""
    from skimage.measure import label

    mask = _crossing_lines_mask()
    a = boundary_gt.extract_folder(_make_folder(tmp_path / "a", "DS", [mask]), "B",
                                   tmp_path / "out_a", _settings_with(mode_a_line_class={"DS": 8}))
    b = boundary_gt.extract_folder(_make_folder(tmp_path / "b", "DS", [mask]), "B",
                                   tmp_path / "out_b", _settings_with())
    assert a["n_processed"] == b["n_processed"] == 1

    def n_regions(path):
        return int(label(_read_png(path) == 0, connectivity=1).max())

    assert n_regions(tmp_path / "out_a" / "DS" / "t0.png") < n_regions(
        tmp_path / "out_b" / "DS" / "t0.png")


def test_line_class_option_refuses_a_mode_a_dataset(tmp_path):
    report = _make_folder(tmp_path, "DS", [_crossing_lines_mask()])
    with pytest.raises(boundary_gt.ExtractionError, match="MODE A dataset"):
        boundary_gt.extract_folder(report, "A", tmp_path / "out",
                                   _settings_with(mode_a_line_class={"DS": 8}))


def test_line_class_option_refuses_a_non_positive_threshold(tmp_path):
    report = _make_folder(tmp_path, "DS", [_crossing_lines_mask()])
    with pytest.raises(boundary_gt.ExtractionError, match="must be > 0"):
        boundary_gt.extract_folder(report, "B", tmp_path / "out",
                                   _settings_with(mode_a_line_class={"DS": 0}))


def test_min_region_area_merges_phase_labels_BEFORE_the_line_class_is_drawn(tmp_path):
    """Interaction with min_region_area_px, stated as a test: both settings act
    on the palette-snapped LABEL map, in that order -- small components of any
    class merge into their surroundings first, then the line class is built
    from what is left. A 4 px white speckle inside the black blob therefore
    becomes black (part of the blob) and draws no loop; without the merge it
    would survive as a region of its own.
    """
    mask = _crossing_lines_mask()
    mask[13:15, 33:35] = 255                 # 2x2 white speckle inside the disk
    report = _make_folder(tmp_path, "DS", [mask])
    with_merge = _settings_with(mode_a_line_class={"DS": 8}, min_region_area_px={"DS": 20})
    without = _settings_with(mode_a_line_class={"DS": 8})
    merged = boundary_gt.extract_folder(report, "B", tmp_path / "m", with_merge)
    plain = boundary_gt.extract_folder(report, "B", tmp_path / "p", without)
    assert merged["region_merge"]["enabled"]
    assert merged["region_merge"]["n_components_merged_total"] >= 1

    def expected(settings):
        palette = boundary_gt.derive_palette(report, settings)
        labels, _, _ = boundary_gt.snap_to_palette(mask, palette)
        if settings["min_region_area_px"]:
            labels, _ = boundary_gt.merge_small_regions(
                labels, settings["min_region_area_px"]["DS"])
        raw = boundary_gt.mode_a_raw_from_line_class(
            labels, boundary_gt.line_class_index(palette), 8.0)
        return boundary_gt.clean_boundary(raw, settings)[0]

    assert np.array_equal(_read_png(tmp_path / "m" / "DS" / "t0.png"), expected(with_merge))
    assert np.array_equal(_read_png(tmp_path / "p" / "DS" / "t0.png"), expected(without))
    assert not np.array_equal(_read_png(tmp_path / "m" / "DS" / "t0.png"),
                              _read_png(tmp_path / "p" / "DS" / "t0.png"))


def test_line_class_index_is_the_darkest_palette_entry():
    assert boundary_gt.line_class_index([(255, 255, 255), (0, 0, 0)]) == 1
    assert boundary_gt.line_class_index([(0, 0, 0), (255, 255, 255)]) == 0
    assert boundary_gt.line_class_index([(120, 120, 120), (10, 10, 10), (200, 0, 0)]) == 1
    with pytest.raises(boundary_gt.ExtractionError):
        boundary_gt.line_class_index([])


# --------------------------------------------------------------------------
# skeleton endpoints: the open-contour measure
# --------------------------------------------------------------------------
def test_skeleton_endpoints_split_dangling_from_tile_edge_ends():
    m = np.zeros((40, 40), dtype=bool)
    m[20, 0:21] = True            # starts AT the left edge, stops mid-tile at col 20
    stats = boundary_gt.skeleton_endpoint_stats(m, border_margin_px=2)
    assert stats["n_endpoints"] == 2
    assert stats["n_interior_endpoints"] == 1, "only the mid-tile end is a dangling one"

    closed = np.zeros((40, 40), dtype=bool)
    closed[10, 10:30] = closed[29, 10:30] = closed[10:30, 10] = closed[10:30, 29] = True
    assert boundary_gt.skeleton_endpoint_stats(closed, 2)["n_endpoints"] == 0, (
        "a closed loop has no loose end")
    assert boundary_gt.skeleton_endpoint_stats(np.zeros((8, 8), bool))["n_endpoints"] == 0


# --------------------------------------------------------------------------
# merge_extraction_report: a partial re-run must not erase the other folders
# --------------------------------------------------------------------------
def _report(datasets, **settings):
    base = {"line_width_px": 4, "close_kernel": 3}
    base.update(settings)
    return {"generated_utc": "t0", "settings": base, "datasets": datasets,
            "failures": {}, "mode_overrides": {}}


def test_merge_keeps_untouched_folders_and_replaces_the_extracted_one():
    old = _report({"Steel1": {"n_processed": 907, "mode_a_line_class": None},
                   "Steel2": {"n_processed": 504, "mode_a_line_class": None}})
    record = {"class_index": 1, "colour": [0, 0, 0], "blob_thickness_px": 8.0}
    new = _report({"Steel2": {"n_processed": 504, "mode_a_line_class": record,
                              "files": [1, 2, 3]}},
                  mode_a_line_class={"Steel2": 8})
    new["generated_utc"] = "t1"
    merged = boundary_gt.merge_extraction_report(old, new)

    assert merged["datasets"]["Steel1"] == old["datasets"]["Steel1"]
    assert merged["datasets"]["Steel2"]["mode_a_line_class"] == record
    assert "files" not in merged["datasets"]["Steel2"]
    assert merged["settings"]["mode_a_line_class"] == {"Steel2": 8}
    assert merged["generated_utc"] == "t1"
    assert "mode_a_line_class" not in old["settings"], "the input must not be mutated"


def test_merge_refuses_settings_drift():
    old = _report({"Steel1": {}, "Steel2": {}})
    new = _report({"Steel2": {}}, line_width_px=2)
    with pytest.raises(boundary_gt.ExtractionError, match="line_width_px"):
        boundary_gt.merge_extraction_report(old, new)


def test_merge_refuses_a_line_class_naming_a_folder_it_did_not_extract():
    old = _report({"Steel1": {}, "Steel2": {}})
    new = _report({"Steel2": {}}, mode_a_line_class={"Steel1": 8})
    with pytest.raises(boundary_gt.ExtractionError, match="Steel1"):
        boundary_gt.merge_extraction_report(old, new)


def test_merge_accepts_an_empty_setting_the_old_report_predates():
    """A report written before min_region_area_px existed has no such key; the
    new run carrying it empty is the same ground truth, not drift."""
    old = _report({"Steel1": {}, "Steel2": {}})
    new = _report({"Steel2": {}}, min_region_area_px={})
    assert boundary_gt.merge_extraction_report(old, new)["datasets"].keys() == {"Steel1", "Steel2"}
    with pytest.raises(boundary_gt.ExtractionError, match="min_region_area_px"):
        boundary_gt.merge_extraction_report(
            old, _report({"Steel2": {}}, min_region_area_px={"Steel1": 50}))


def test_merge_refuses_a_report_with_failures():
    new = _report({"Steel2": {}})
    new["failures"] = {"Steel1": "boom"}
    with pytest.raises(boundary_gt.ExtractionError, match="failures"):
        boundary_gt.merge_extraction_report(_report({"Steel1": {}}), new)


# --------------------------------------------------------------------------
# profile summaries + generic payload report
# --------------------------------------------------------------------------
def test_region_profile_summary_is_the_five_standard_numbers():
    s = boundary_gt.region_profile_summary([2, 4], [10, 40, 60, 200, 100, 300])
    assert s["n_tiles"] == 2 and s["regions_per_tile"] == 3.0
    assert s["share_lt_50"] == pytest.approx(2 / 6)          # 10, 40
    assert s["share_lt_100"] == pytest.approx(3 / 6)         # 10, 40, 60 (100 is not < 100)
    assert s["area_percentiles"]["p50"] == pytest.approx(80.0)
    with pytest.raises(boundary_gt.ExtractionError):
        boundary_gt.region_profile_summary([], [])


def test_endpoint_profile_summary_counts_tiles_without_a_dangling_end():
    tiles = [{"n_endpoints": 2, "n_interior_endpoints": 0, "skeleton_px": 100},
             {"n_endpoints": 6, "n_interior_endpoints": 4, "skeleton_px": 300},
             {"n_endpoints": 0, "n_interior_endpoints": 0, "skeleton_px": 100}]
    s = boundary_gt.endpoint_profile_summary(tiles)
    assert s["share_tiles_without_interior_endpoint"] == pytest.approx(2 / 3)
    assert s["interior_endpoints_per_tile"]["max"] == 4.0
    assert s["endpoints_per_tile"]["mean"] == pytest.approx(8 / 3)
    assert s["interior_endpoints_per_1000_skeleton_px"] == pytest.approx(1000 * 4 / 500)
    with pytest.raises(boundary_gt.ExtractionError):
        boundary_gt.endpoint_profile_summary([])


def test_payload_report_writes_the_payload_verbatim_and_a_readable_twin(tmp_path):
    payload = {"pos_weight": {"old": 2.84, "new": 4.1}, "checks": ["a", "b"],
               "rows": [{"x": 1}], "note": "hello"}
    md, js = boundary_gt.write_payload_report("demo", "Demo", "why this exists",
                                              payload, tmp_path)
    import json

    assert json.loads(js.read_text()) == payload
    text = md.read_text()
    assert text.startswith("# Demo") and "why this exists" in text
    assert "## pos_weight" in text and "**old**: 2.84" in text and "hello" in text


# --------------------------------------------------------------------------
# structural: the regeneration and fold-rebuild cells do their safety steps in order
# --------------------------------------------------------------------------
def test_mode_a_adoption_notebook_cells_order_their_safety_steps():
    """Static, never executed: parses notebooks/06d_steel_combined.ipynb and checks
    the order of the steps that protect what must not change -- a backup before
    anything is overwritten, the cleanup forced off before extraction, Steel1
    hashed on BOTH sides of it, a MERGE (never a bare write) into the existing
    extraction record, and tile membership compared before any manifest is written.
    """
    import json
    from pathlib import Path

    nb_path = Path(__file__).resolve().parent.parent / "notebooks" / "06d_steel_combined.ipynb"
    if not nb_path.is_file():
        pytest.skip(f"{nb_path} not found")
    nb = json.loads(nb_path.read_text())

    def source(c):
        s = c["source"]
        return "".join(s) if isinstance(s, list) else s

    code = [source(c) for c in nb["cells"] if c["cell_type"] == "code"]
    regen = [c for c in code if "boundary_gt.extract_all(" in c]
    assert len(regen) == 1, f"expected exactly one regeneration cell, found {len(regen)}"
    regen = regen[0]
    run = regen.index("boundary_gt.extract_all(")
    assert regen.index("shutil.copytree(") < run, "back up before overwriting"
    assert regen.index('st["min_region_area_px"] = {}') < run, "cleanup forced off first"
    assert regen.index("steel1_before = ") < run < regen.index("steel1_after = ")
    assert "datasets=[\"Steel2\"]" in regen[run:run + 200], "Steel2 only"
    assert "boundary_gt.write_extraction_report(new_report" not in regen
    assert (regen.index("boundary_gt.merge_extraction_report(")
            < regen.index("boundary_gt.write_extraction_report("))
    assert regen.index("fingerprint_before = ") < run < regen.index("fingerprint_after = ")
    assert "Trainer(" not in regen and ".train(" not in regen, "regeneration must not retrain"

    fold = [c for c in code if "tiling.membership_diff(" in c]
    assert len(fold) == 1, f"expected exactly one fold-rebuild cell, found {len(fold)}"
    fold = fold[0]
    assert (fold.index("tiling.membership_diff(") < fold.index("tile membership changed")
            < fold.index("tiling.write_manifest(")), "compare membership before writing"
    assert fold.index("tiling.apply_gt_variant(") < fold.index("tiling.build_index(")
    assert "Trainer(" not in fold and ".train(" not in fold, "the fold rebuild must not retrain"

    import re

    cell2 = [c for c in code
             if re.search(r"^settings = tiling\.load_config\(\)", c, re.M) and "\nFOLD = " in c]
    assert len(cell2) == 1, f"expected exactly one Cell-2-style cell, found {len(cell2)}"
    assert (cell2[0].index("tiling.apply_gt_variant(")
            < cell2[0].index('steel_cfg = settings["steel_combined"]')), (
        "Cell 2 must resolve the fold from the recorded ground truth before reading its name")


# --------------------------------------------------------------------------
# refusal: a second Steel2 regeneration must not be able to overwrite the backup
# --------------------------------------------------------------------------
def _listing(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def test_regeneration_is_refused_when_the_extraction_already_records_the_line_class(tmp_path):
    record = {"class_index": 1, "colour": [0, 0, 0], "blob_thickness_px": 8.0}
    extraction = {"datasets": {"Steel2": {"mode_a_line_class": record}}}
    (tmp_path / "Steel2").mkdir()
    before = _listing(tmp_path)
    with pytest.raises(boundary_gt.ExtractionError, match="REFUSING.*already records"):
        boundary_gt.refuse_if_already_regenerated(extraction, tmp_path, "Steel2")
    assert _listing(tmp_path) == before, "a refusal must not write anything"


def test_regeneration_is_refused_when_the_backup_already_exists(tmp_path):
    backup = tmp_path / boundary_gt.MODE_B_BACKUP_SUBDIR / "Steel2"
    backup.mkdir(parents=True)
    (backup / "a.png").write_bytes(b"the only copy of the original")
    extraction = {"datasets": {"Steel2": {"mode_a_line_class": None}}}
    before = _listing(tmp_path)
    with pytest.raises(boundary_gt.ExtractionError, match="only copy of the original MODE B"):
        boundary_gt.refuse_if_already_regenerated(extraction, tmp_path, "Steel2")
    assert _listing(tmp_path) == before
    assert (backup / "a.png").read_bytes() == b"the only copy of the original"


def test_regeneration_refusal_names_every_problem_when_both_hold(tmp_path):
    (tmp_path / boundary_gt.MODE_B_BACKUP_SUBDIR / "Steel2").mkdir(parents=True)
    extraction = {"datasets": {"Steel2": {"mode_a_line_class": {"class_index": 1}}}}
    with pytest.raises(boundary_gt.ExtractionError) as exc:
        boundary_gt.refuse_if_already_regenerated(extraction, tmp_path, "Steel2")
    assert "already records" in str(exc.value) and "only copy" in str(exc.value)


def test_regeneration_is_allowed_on_a_clean_mode_b_state_and_for_other_datasets(tmp_path):
    extraction = {"datasets": {"Steel1": {"mode_a_line_class": None},
                               "Steel2": {"mode_a_line_class": None}}}
    boundary_gt.refuse_if_already_regenerated(extraction, tmp_path, "Steel2")   # no raise
    # another dataset's backup / record does not block this one
    (tmp_path / boundary_gt.MODE_B_BACKUP_SUBDIR / "Steel1").mkdir(parents=True)
    boundary_gt.refuse_if_already_regenerated(extraction, tmp_path, "Steel2")
