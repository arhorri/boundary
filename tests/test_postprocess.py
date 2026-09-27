"""Tests for src/postprocess.py -- binarizing a predicted boundary map.

Torch-free on purpose: postprocess_boundary is a pure numpy function and is
tested as one. Its use inside the evaluation functions (decompose_error,
evaluate_region_metrics, sweep_postprocess) is tested in tests/test_train.py
under the same ``postprocess`` keyword, and notebooks/06d_steel_combined.ipynb
runs both with ``-k postprocess`` before its sweep.

Every assertion here is exact by construction (a straight strip's width after
skeletonize + _dilate_to_width is a count of rows, not an estimate); nothing
depends on a number only a real checkpoint could produce.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import boundary_gt
from src import postprocess


def _strip_prob(height=40, width=60, row0=17, rows=5, on=0.9, off=0.1):
    """A probability map with one horizontal band ``rows`` px thick."""
    prob = np.full((height, width), off, dtype=np.float32)
    prob[row0:row0 + rows, :] = on
    return prob


def test_postprocess_none_is_byte_identical_to_a_plain_threshold():
    """mode="none" must be exactly ``prob >= threshold`` -- the expression every
    evaluation used before this module existed -- or the default changes
    numbers it promised not to change.
    """
    rng = np.random.default_rng(0)
    prob = rng.random((64, 64)).astype(np.float32)
    for threshold in (0.0, 0.3, 0.5, 0.7314, 0.9, 1.0):
        out = postprocess.postprocess_boundary(prob, threshold)
        expected = prob >= float(threshold)
        assert out.dtype == expected.dtype == np.bool_
        assert out.shape == expected.shape
        assert out.tobytes() == expected.tobytes(), threshold
        explicit = postprocess.postprocess_boundary(prob, threshold, "none", None)
        assert explicit.tobytes() == expected.tobytes(), threshold


@pytest.mark.parametrize("target", [2, 4])
def test_postprocess_skeleton_redilate_normalises_a_1px_too_fat_line(target):
    """A band one pixel fatter than the target comes back exactly ``target`` px.

    The band has an odd number of rows (target + 1 with target even), so its
    skeleton is its single middle row; _dilate_to_width then grows that row
    to exactly ``target`` rows. Checked two ways: the row count in every
    middle column (ends excluded, where skeletonize may shorten the line),
    and boundary_gt.measured_line_width -- the same measure step 2 uses on
    the ground truth itself.
    """
    prob = _strip_prob(rows=target + 1)
    out = postprocess.postprocess_boundary(prob, 0.5, "skeleton_redilate", target)
    assert out.dtype == np.bool_ and out.shape == prob.shape

    before = (prob >= 0.5)[:, 15:-15].sum(axis=0)
    after = out[:, 15:-15].sum(axis=0)
    assert (before == target + 1).all()
    assert (after == target).all(), f"column widths {sorted(set(after.tolist()))}"
    assert boundary_gt.measured_line_width(out, width_hint=target) == pytest.approx(
        target, abs=0.5)


def test_postprocess_skeleton_redilate_keeps_the_line_where_it_was():
    """Thinning moves no boundary: the redilated line sits inside the band."""
    prob = _strip_prob(row0=17, rows=5)
    out = postprocess.postprocess_boundary(prob, 0.5, "skeleton_redilate", 4)
    rows_hit = np.where(out[:, 15:-15].any(axis=1))[0]
    assert rows_hit.min() >= 17 and rows_hit.max() <= 17 + 5 - 1


@pytest.mark.parametrize("mode", postprocess.MODES)
def test_postprocess_empty_mask_stays_empty(mode):
    """Nothing above threshold -> nothing out. No centreline to grow, and
    nothing fabricated in its place."""
    prob = np.full((32, 32), 0.1, dtype=np.float32)
    out = postprocess.postprocess_boundary(prob, 0.5, mode, 4)
    assert out.dtype == np.bool_ and out.shape == prob.shape
    assert not out.any()


def test_postprocess_rejects_an_unknown_mode():
    with pytest.raises(postprocess.PostprocessError):
        postprocess.postprocess_boundary(np.zeros((4, 4)), 0.5, "thin_harder", 4)


def test_postprocess_skeleton_redilate_requires_a_target_width():
    prob = _strip_prob()
    with pytest.raises(postprocess.PostprocessError):
        postprocess.postprocess_boundary(prob, 0.5, "skeleton_redilate", None)
    with pytest.raises(postprocess.PostprocessError):
        postprocess.postprocess_boundary(prob, 0.5, "skeleton_redilate", 0)


def test_postprocess_skeleton_redilate_refuses_a_batch():
    """One 2-D map at a time: skeletonize on a 3-D stack would thin across
    tiles, which is not a boundary operation at all."""
    with pytest.raises(postprocess.PostprocessError):
        postprocess.postprocess_boundary(np.ones((2, 8, 8)), 0.5,
                                         "skeleton_redilate", 4)


def test_postprocess_default_mode_is_none():
    assert postprocess.DEFAULTS["mode"] == "none"
    assert "none" in postprocess.MODES and "skeleton_redilate" in postprocess.MODES


def test_postprocess_load_config_without_a_section_is_the_default(tmp_path):
    config_path = tmp_path / "default.yaml"
    config_path.write_text("train:\n  seed: 0\n")
    settings = postprocess.load_config(config_path)
    assert settings["mode"] == "none"
    assert settings["target_width_px"] is None


def test_postprocess_load_config_rejects_unknown_keys_and_modes(tmp_path):
    bad_key = tmp_path / "a.yaml"
    bad_key.write_text("postprocess:\n  moed: none\n")
    with pytest.raises(postprocess.PostprocessError):
        postprocess.load_config(bad_key)
    bad_mode = tmp_path / "b.yaml"
    bad_mode.write_text("postprocess:\n  mode: thin_harder\n")
    with pytest.raises(postprocess.PostprocessError):
        postprocess.load_config(bad_mode)


# --------------------------------------------------------------------------
# morph_close / skeleton_bridge -- gap-closing modes (step 6c gap-closing
# sweep). Region separation is checked with train_mod.true_regions_from_
# boundary -- the SAME boundary-to-region conversion the region metrics use,
# so "does this mode actually close the gap" is verified the way it matters,
# not by inspecting pixels the region metric never looks at.
# --------------------------------------------------------------------------
def _gapped_column(size=40, col=20, gap=slice(18, 21)):
    """A full-height 1 px boundary at ``col``, with rows ``gap`` removed."""
    true_b = np.zeros((size, size), dtype=bool)
    true_b[:, col] = True
    gapped = true_b.copy()
    gapped[gap, col] = False
    return gapped


def test_postprocess_morph_close_seals_a_one_gap_case():
    from src import train as train_mod

    gapped = _gapped_column()
    assert len(np.unique(train_mod.true_regions_from_boundary(gapped))) == 1, (
        "the gap must actually leak before closing, or this fixture proves nothing")

    prob = gapped.astype(np.float32)
    sealed = postprocess.postprocess_boundary(prob, 0.5, "morph_close", closing_radius=2)
    assert len(np.unique(train_mod.true_regions_from_boundary(sealed))) == 2

    # A 3-row gap needs radius >= ~2 (closing can bridge roughly 2*radius):
    # radius 1 must NOT be enough, so the sweep's radius actually matters.
    too_small = postprocess.postprocess_boundary(prob, 0.5, "morph_close", closing_radius=1)
    assert len(np.unique(train_mod.true_regions_from_boundary(too_small))) == 1


def test_postprocess_skeleton_bridge_seals_a_one_gap_case():
    from src import train as train_mod

    gapped = _gapped_column()   # endpoints at row 17 and row 21 -- distance 4
    assert len(np.unique(train_mod.true_regions_from_boundary(gapped))) == 1

    prob = gapped.astype(np.float32)
    bridged = postprocess.postprocess_boundary(prob, 0.5, "skeleton_bridge",
                                               target_width_px=1, bridge_px=4)
    assert len(np.unique(train_mod.true_regions_from_boundary(bridged))) == 2

    too_short = postprocess.postprocess_boundary(prob, 0.5, "skeleton_bridge",
                                                 target_width_px=1, bridge_px=2)
    assert len(np.unique(train_mod.true_regions_from_boundary(too_short))) == 1


def test_postprocess_morph_close_cannot_open_an_intact_diagonal_boundary():
    """Closing is EXTENSIVE (dilate then erode never removes an original
    foreground pixel: A subseteq closing(A)), so a diagonal boundary already
    proven to separate its two sides under connectivity=1 background labelling
    (the item-1 fix) cannot be made to leak by morph_close at any radius --
    it can only add material to the boundary, never remove it.
    """
    from src import train as train_mod

    size = 10
    true_b = np.zeros((size, size), dtype=bool)
    for i in range(size):
        true_b[i, i] = True
    for radius in (1, 2, 3):
        closed = postprocess.postprocess_boundary(true_b.astype(np.float32), 0.5,
                                                  "morph_close", closing_radius=radius)
        regions = train_mod.true_regions_from_boundary(closed)
        assert regions[2, 5] != regions[5, 2], f"leaked at closing_radius={radius}"


def test_postprocess_skeleton_bridge_is_a_no_op_on_a_single_intact_fragment():
    """A diagonal boundary from corner to corner is ONE connected fragment
    (8-connected skeleton), so it has exactly two endpoints and no OTHER
    fragment to bridge to -- skeleton_bridge must leave it untouched, at any
    bridge_px, rather than bridge an endpoint to itself or introduce a
    spurious connection across the tile.
    """
    from src import train as train_mod

    size = 10
    true_b = np.zeros((size, size), dtype=bool)
    for i in range(size):
        true_b[i, i] = True
    for bridge in (2, 5, 20):
        bridged = postprocess.postprocess_boundary(true_b.astype(np.float32), 0.5,
                                                    "skeleton_bridge", target_width_px=1,
                                                    bridge_px=bridge)
        assert bridged.tobytes() == true_b.tobytes(), f"changed at bridge_px={bridge}"
        regions = train_mod.true_regions_from_boundary(bridged)
        assert regions[2, 5] != regions[5, 2], f"leaked at bridge_px={bridge}"


def test_postprocess_morph_close_and_skeleton_bridge_need_their_own_parameter():
    prob = _gapped_column().astype(np.float32)
    with pytest.raises(postprocess.PostprocessError):
        postprocess.postprocess_boundary(prob, 0.5, "morph_close")
    with pytest.raises(postprocess.PostprocessError):
        postprocess.postprocess_boundary(prob, 0.5, "morph_close", closing_radius=0)
    with pytest.raises(postprocess.PostprocessError):
        postprocess.postprocess_boundary(prob, 0.5, "skeleton_bridge", target_width_px=4)
    with pytest.raises(postprocess.PostprocessError):
        postprocess.postprocess_boundary(prob, 0.5, "skeleton_bridge", bridge_px=4)
    with pytest.raises(postprocess.PostprocessError):
        postprocess.postprocess_boundary(prob, 0.5, "skeleton_bridge",
                                         target_width_px=4, bridge_px=0)


@pytest.mark.parametrize("mode", ["morph_close", "skeleton_bridge"])
def test_postprocess_new_modes_empty_mask_stays_empty(mode):
    prob = np.full((32, 32), 0.1, dtype=np.float32)
    out = postprocess.postprocess_boundary(prob, 0.5, mode, target_width_px=4,
                                           closing_radius=2, bridge_px=4)
    assert out.dtype == np.bool_ and not out.any()
