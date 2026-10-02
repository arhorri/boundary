"""Tests for src/losses.py and src/model.py. RUN THESE IN notebooks/05.

There is no local Python environment for this project, so nothing here is
executed on the machine that wrote it. The notebook runs pytest on the host,
where torch, smp and the mounted datasets exist, and prints the result as part
of its checks.

Tests that need the real data skip themselves when it is absent, so the file
still runs anywhere; the notebook's checks cell FAILS if they were skipped,
because a skip is not a pass.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from src import dataset as ds
from src import losses
from src.paths import REPO_ROOT

MANIFEST = REPO_ROOT / "reports" / "manifests" / "dev.csv"
FOLD = "dev"
PATCH = 256
MAGNITUDE = 10.0     # logit magnitude for a "confident" constructed prediction


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def settings():
    return losses.load_config()


@pytest.fixture(scope="module")
def fold_stats():
    return ds.load_fold_stats()


@pytest.fixture(scope="module")
def criterion(settings, fold_stats):
    return losses.BoundaryLoss.for_fold(FOLD, settings=settings,
                                        fold_stats=fold_stats)


@pytest.fixture(scope="module")
def roots():
    from src import paths as paths_mod

    resolved = paths_mod.resolve_paths()
    return {"data_root": resolved["data_root"],
            "gt_root": Path(resolved["gt_boundaries_root"])}


@pytest.fixture(scope="module")
def rows():
    if not MANIFEST.is_file():
        pytest.skip(f"{MANIFEST} not present; run step 3 first")
    return ds.load_manifest(MANIFEST)


@pytest.fixture(scope="module")
def dense_tile(rows, roots):
    """The densest MetalDam boundary tile in the dev fold, as a binary mask.

    MetalDam because its boundaries are the densest in the fold, so a
    perturbation is measured against plenty of signal rather than a handful of
    pixels. Real data, not a drawn grid: the whole point of the clDice claim is
    how it behaves on boundaries with real junctions and curvature.
    """
    return _real_mask(rows, roots, "MetalDam", densest=True)


def _real_mask(rows, roots, dataset_name, densest=True):
    """The densest tile of ``dataset_name`` (``dataset.densest_tile_mask``); a skip, not a
    failure, on a host that does not have the data."""
    assert densest, "only the densest tile is ever asked for"
    try:
        return ds.densest_tile_mask(rows, dataset_name, roots=roots)
    except ds.DatasetError as exc:
        pytest.skip(f"{exc} ({MANIFEST.name})")


@pytest.fixture(scope="module")
def line_width():
    """The width the ground truth on disk was generated at -- RECORDED in
    reports/gt_extraction.json, not whatever configs/default.yaml says now."""
    from src import tiling

    return float(tiling.load_extraction()["settings"]["line_width_px"])


def _synthetic_mask(size: int = 128) -> np.ndarray:
    """A 2 px grid, the same width the real ground truth is dilated to."""
    mask = np.zeros((size, size), dtype=np.uint8)
    for k in range(12, size - 2, 29):
        mask[k:k + 2, :] = 1
        mask[:, k:k + 2] = 1
    return mask


# --------------------------------------------------------------------------
# a perfect prediction costs nothing
# --------------------------------------------------------------------------
def test_perfect_prediction_is_zero_for_every_term(criterion):
    mask = _synthetic_mask()
    target = losses.as_target(mask)
    logits = losses.as_logits(mask, magnitude=20.0)

    terms = criterion.components(logits, target)
    for name in ("bce", "dice", "cldice", "total"):
        value = float(terms[name])
        assert np.isfinite(value), f"{name} is {value}"
        assert value >= 0.0, f"{name} is negative: {value}"
        assert value < 1e-3, f"{name} on a perfect prediction is {value:.6f}"


# --------------------------------------------------------------------------
# the loss falls as the prediction approaches the truth
# --------------------------------------------------------------------------
def test_loss_decreases_monotonically_toward_the_target(criterion):
    """Interpolate from "background everywhere" to the truth, in logit space.

    Off-boundary logits stay at -m throughout; on-boundary logits sweep from
    -m to +m. Every term must fall at every step -- a loss that is not
    monotone along this path has a term fighting the other two.
    """
    mask = _synthetic_mask()
    target = losses.as_target(mask)
    on_boundary = target > 0

    history = {"bce": [], "dice": [], "cldice": [], "total": []}
    for t in np.linspace(0.0, 1.0, 9):
        logits = torch.full_like(target, -MAGNITUDE)
        logits = torch.where(on_boundary,
                             torch.full_like(target, (2.0 * t - 1.0) * MAGNITUDE),
                             logits)
        terms = criterion.components(logits, target)
        for name in history:
            history[name].append(float(terms[name]))

    for name, values in history.items():
        for earlier, later in zip(values, values[1:]):
            assert later <= earlier + 1e-6, (
                f"{name} rose along the path to the target: {values}")
        assert values[0] - values[-1] > 0.1, (
            f"{name} barely moved between an empty prediction and the truth: "
            f"{values[0]:.4f} -> {values[-1]:.4f}")


# --------------------------------------------------------------------------
# clDice and the broken line
# --------------------------------------------------------------------------
def test_soft_skeleton_of_an_isolated_two_pixel_line_is_the_line(settings):
    """Half of what soft_skeletonize does to 2 px lines, measured.

    On a line with no junctions, one erosion removes it entirely -- every line
    pixel's 3x1 min-pool reaches a background row -- so ``open(x)`` is empty and
    ``skel = relu(x - 0) = x``. This half was reasoned about correctly the
    first time; the other half was not, and the test below is the one that
    caught it.
    """
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[64:66, 20:120] = 1
    target = losses.as_target(mask)
    skel = losses.soft_skeletonize(target, int(settings["cldice_iters"]))
    assert torch.allclose(skel, target, atol=1e-6), (
        "an isolated 2 px line no longer comes back unchanged; max |diff| "
        f"{float((skel - target).abs().max()):.4f}")


def test_soft_skeleton_carves_junctions_out_of_a_two_pixel_grid(settings):
    """The other half: a JUNCTION survives erosion, and the skeleton loses it.

    Where two 2 px lines cross, the shape is 3+ pixels thick in both
    directions, so erosion survives on the 2x2 core, ``open`` dilates that core
    back out to a 4x4 block, and ``relu(x - open(x))`` removes every line pixel
    inside that block. The skeleton of a boundary NETWORK is therefore the mask
    minus a patch at each junction -- which is why "skeleton(g) == g on this
    data" was wrong, and why the clDice claim resting on it had to be redone.

    Both halves are asserted: that the difference is real, and that it is
    confined to the junctions rather than being a general thinning.
    """
    size, offsets = 128, list(range(12, 126, 29))
    mask = np.zeros((size, size), dtype=np.uint8)
    for k in offsets:
        mask[k:k + 2, :] = 1
        mask[:, k:k + 2] = 1

    target = losses.as_target(mask)
    skel = losses.soft_skeletonize(target, int(settings["cldice_iters"]))
    diff = (target - skel).abs()[0, 0].numpy()
    differing = np.argwhere(diff > 1e-6)

    assert len(differing) > 0, (
        "the skeleton of a 2 px GRID came back identical to the grid; the "
        "junction behaviour this project's clDice reasoning depends on is gone")
    assert len(differing) < 0.25 * mask.sum(), (
        f"{len(differing)} of {int(mask.sum())} boundary pixels differ -- that "
        "is a general thinning, not junctions being carved out")

    # Every difference must sit at a junction. The opening can only reach one
    # pixel beyond the eroded 2x2 core, so 3 is a generous bound.
    junctions = np.array([(r, c) for r in offsets for c in offsets], dtype=int)
    for y, x in differing:
        chebyshev = np.max(np.abs(junctions - np.array([y, x])), axis=1).min()
        assert chebyshev <= 3, (
            f"pixel ({y}, {x}) differs but is {chebyshev} px from the nearest "
            "junction; the skeleton is being changed away from junctions too")


def test_cldice_is_far_less_sensitive_to_thickness_than_dice(criterion, dense_tile, line_width):
    """The claim clDice actually earns its place with, on a real tile.

    Two errors are applied to the ground truth itself:

    * **gap** -- the line severed periodically. Topology destroyed, ~3% of the
      pixels wrong.
    * **dilated** -- the line one pixel fatter all round. Topology intact, and
      roughly 100% MORE pixels wrong than in the gap case.

    Dice counts pixels, so it charges an order of magnitude more for the
    harmless error than for the destructive one. clDice compares skeletons, and
    a dilated line has the same centreline, so it charges almost nothing for
    it. That is the property: **clDice does not add gap sensitivity, it removes
    Dice's thickness bias.** Notebook 05 measures 0.335 / 0.390 for Dice on the
    dilated case against 0.007 / 0.000 for clDice.

    Thickness invariance is only meaningful if the skeletons are real, so that
    is checked too: a clDice of exactly 0 with an EMPTY predicted skeleton is
    ``smooth / smooth = 1`` -- degenerate, not invariant. The two are
    indistinguishable from the loss value alone.

    WIDTH-AWARE. Everything that depends on how wide the ground-truth line is is
    derived from ``line_width`` (the width recorded in gt_extraction.json), not fixed at
    the 2 px this was first written for:

    * the skeleton iterations come from ``losses.thickness_invariance_iters`` -- a line
      ``line_width + 2`` px wide (the dilated one) must be consumed by erosion before
      its skeleton exists, and at 4 px the default ``cldice_iters = 3`` does not get
      there on diagonal lines. ``loss.cldice_iters`` itself is NOT changed: this is the
      iteration count a MEASUREMENT needs, not a training setting;
    * the gap block must be wider than the line, or it only thins it (``cut_gaps``),
      and its spacing scales with the width so the same share of the boundary is cut.
    """
    truth = dense_tile
    iters = losses.thickness_invariance_iters(line_width)
    gapped = losses.cut_gaps(truth, spacing=int(90 * line_width),
                             gap=int(math.ceil(line_width)) + 1)
    dilated = losses.dilate_mask(truth, iterations=1)

    removed = 1.0 - gapped.sum() / max(1, truth.sum())
    added = dilated.sum() / max(1, truth.sum()) - 1.0
    assert 0.005 < removed < 0.25, (
        f"cut_gaps removed {removed:.1%} of the boundary; the comparison needs "
        "a break, not an erasure")
    assert added > 0.2, f"dilate_mask only added {added:.1%}"

    target = losses.as_target(truth)
    scores, parts = {}, {}
    for name, mask in (("gap", gapped), ("dilated", dilated)):
        logits = losses.as_logits(mask, MAGNITUDE)
        terms = criterion.components(logits, target)
        scores[name] = {
            "dice": float(terms["dice"]),
            # clDice at the width-derived iterations, not the training default
            "cldice": float(losses.cldice_term(logits, target, iters=iters,
                                               smooth=criterion.smooth,
                                               eps=criterion.eps))}
        parts[name] = losses.cldice_parts(
            logits, target, iters=iters, smooth=criterion.smooth, eps=criterion.eps)

    dice_gap, cl_gap = scores["gap"]["dice"], scores["gap"]["cldice"]
    dice_dil, cl_dil = scores["dilated"]["dice"], scores["dilated"]["cldice"]
    detail = (f"line_width={line_width:g} iters={iters} "
              f"(training default {criterion.cldice_iters}); "
              f"gap: dice={dice_gap:.5f} cldice={cl_gap:.5f} "
              f"(removed {removed:.1%}); dilated: dice={dice_dil:.5f} "
              f"cldice={cl_dil:.5f} (added {added:.1%}); dilated skeletons: "
              f"pred={parts['dilated']['skel_pred_sum']:.0f} px, "
              f"true={parts['dilated']['skel_true_sum']:.0f} px, "
              f"overlap={parts['dilated']['skel_pred_on_true']:.0f} px")

    # Dice is dominated by thickness.
    assert dice_dil > 3.0 * dice_gap, (
        "Dice was expected to charge far more for fattening the line than for "
        "cutting it -- " + detail)
    # clDice is not. This is the property being asserted.
    assert cl_dil < 0.25 * dice_dil, (
        "clDice charges nearly as much as Dice for a topology-preserving "
        "thickness change; its whole contribution here is not doing that -- "
        + detail)
    # ... and it is invariance, not degeneracy: both skeletons carry real
    # pixels and the predicted one lies on the true boundary. The width-independent
    # conditions are losses.invariance_checks, the same function Cell 33 measures with.
    checks = losses.invariance_checks(parts["dilated"], dice_dil, cl_dil, line_width)
    for name in losses.INVARIANCE_CHECKS:
        assert checks[name], f"{name}: {losses.INVARIANCE_CHECK_MEANING[name]} -- " + detail
    # Consequence of the two above, stated as the ranking it produces: relative
    # to the thickness error, clDice puts the break far higher than Dice does.
    # Cross-multiplied, so a clDice of exactly 0 on the dilated case cannot
    # blow the ratio up.
    assert cl_gap > 0.0, "clDice charged nothing at all for a severed line"
    assert cl_gap * dice_dil > 3.0 * cl_dil * dice_gap, (
        "clDice does not rank the break above the thickness error more "
        "strongly than Dice does -- " + detail)


def test_cut_gaps_actually_severs_the_line():
    """A gap narrower than the line does not break it -- which is why gap=3.

    One isolated 2 px line, so the component count is unambiguous. Erasing
    single pixels leaves it connected, because the other row of the line bridges
    every hole; erasing a block as wide as the line cuts it into pieces.
    """
    from skimage.measure import label

    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[64:66, 20:120] = 1
    assert label(mask, connectivity=2).max() == 1

    single = losses.cut_gaps(mask, spacing=40, gap=1)
    proper = losses.cut_gaps(mask, spacing=40, gap=3)
    assert single.sum() < mask.sum(), "cut_gaps removed nothing"
    assert label(single, connectivity=2).max() == 1, (
        "a 1 px gap severed a 2 px line; the reasoning behind gap=3 is wrong")
    assert label(proper, connectivity=2).max() > 1, (
        "a 3 px gap did not sever a 2 px line")


# --------------------------------------------------------------------------
# empty against empty
# --------------------------------------------------------------------------
def test_empty_prediction_on_empty_target_is_finite_and_zero(criterion):
    """Some val tiles are legitimately near-empty. They must not score ~1.

    This is what an ``eps``-sized smoothing constant gets wrong: sigmoid(-10)
    summed over 65k pixels is far larger than 1e-6, so the ratio collapses and
    a correct prediction is scored as a total failure.
    """
    empty = np.zeros((PATCH, PATCH), dtype=np.uint8)
    target = losses.as_target(empty)
    # magnitude 30, not 10: "all-zero" has to mean a probability that really is
    # zero. sigmoid(-10) is 4.5e-5, and over 65,536 pixels that sums to 3.0 --
    # three pixels' worth of predicted boundary. The smoothing constant is
    # there to stop 0/0, not to absorb three pixels of leakage.
    logits = losses.as_logits(empty, magnitude=30.0)

    terms = criterion.components(logits, target)
    for name, value in terms.items():
        value = float(value)
        assert np.isfinite(value), f"{name} is {value}"
        assert value >= 0.0, f"{name} is negative: {value}"
    for name in ("dice", "cldice", "total"):
        assert float(terms[name]) < 1e-2, (
            f"{name} on an empty prediction against an empty target is "
            f"{float(terms[name]):.6f}, not ~0")


def test_a_near_empty_tile_is_not_scored_as_a_failure(criterion):
    """One short line, predicted correctly, on an otherwise empty tile."""
    # tiling.min_boundary_frac is 0.005, so the sparsest tile that survives
    # into a manifest carries ~330 boundary pixels. That is what "near-empty"
    # means here -- not zero, which only happens in a synthetic test.
    mask = np.zeros((PATCH, PATCH), dtype=np.uint8)
    mask[100:102, 28:228] = 1
    assert mask.sum() / mask.size > 0.005
    target = losses.as_target(mask)
    terms = criterion.components(losses.as_logits(mask, MAGNITUDE), target)
    assert float(terms["total"]) < 0.05, (
        f"a correct prediction on a near-empty tile scored "
        f"{float(terms['total']):.6f}")


# --------------------------------------------------------------------------
# pos_weight is per fold, and unknown folds are an error
# --------------------------------------------------------------------------
def test_pos_weight_is_read_per_fold_and_differs_between_folds(fold_stats):
    weights = {name: losses.fold_pos_weight(name, fold_stats=fold_stats)
               for name in ("fold_MetalDam", "fold_uhcs1", "fold_uhcs2")}
    for name, value in weights.items():
        assert value == pytest.approx(
            float(fold_stats["folds"][name]["pos_weight"])), name
        assert value > 1.0, f"{name} pos_weight {value} does not up-weight boundary"

    spread = max(weights.values()) / min(weights.values())
    assert spread > 2.0, (
        f"the folds' pos_weights are within {spread:.2f}x of each other "
        f"({weights}); the per-fold lookup would be pointless if so")
    assert weights["fold_MetalDam"] == max(weights.values()), (
        "fold_MetalDam trains on the sparsest boundaries and should carry the "
        f"largest pos_weight; got {weights}")


def test_criterion_carries_the_folds_pos_weight(fold_stats):
    for name in ("fold_MetalDam", "fold_uhcs1"):
        criterion = losses.BoundaryLoss.for_fold(name, fold_stats=fold_stats)
        assert float(criterion.pos_weight) == pytest.approx(
            float(fold_stats["folds"][name]["pos_weight"]))


def test_unknown_fold_raises_rather_than_falling_back(fold_stats):
    with pytest.raises(losses.LossError) as exc:
        losses.fold_pos_weight("fold_does_not_exist", fold_stats=fold_stats)
    assert "fold_does_not_exist" in str(exc.value)
    with pytest.raises(losses.LossError):
        losses.BoundaryLoss.for_fold("fold_does_not_exist", fold_stats=fold_stats)


def test_pos_weight_must_be_positive_and_finite():
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(losses.LossError):
            losses.BoundaryLoss(bad)


# --------------------------------------------------------------------------
# gradients
# --------------------------------------------------------------------------
def test_gradients_flow_to_the_logits_through_every_term(criterion):
    mask = _synthetic_mask()
    target = losses.as_target(mask)
    for name in ("bce", "dice", "cldice", "total"):
        logits = losses.as_logits(mask * 0, magnitude=1.0).clone()
        logits.requires_grad_(True)
        criterion.components(logits, target)[name].backward()
        grad = logits.grad
        assert grad is not None, f"{name} produced no gradient"
        assert torch.isfinite(grad).all(), f"{name} produced a non-finite gradient"
        assert float(grad.abs().sum()) > 0, f"{name} produced an all-zero gradient"


def test_gradients_reach_the_model_input_through_every_term(criterion):
    """End to end: input -> U-Net -> each term -> back to the input tensor.

    Built with random weights, not ImageNet: this is about the graph, and a
    weight download inside a unit test is a network dependency nobody asked
    for.
    """
    from src import model as model_mod

    settings = dict(model_mod.load_config())
    settings["encoder_weights"] = None
    net = model_mod.build_model(settings=settings)
    net.eval()

    mask = _synthetic_mask(64)
    target = losses.as_target(mask).repeat(2, 1, 1, 1)
    for name in ("bce", "dice", "cldice", "total"):
        x = torch.randn(2, 1, 64, 64, requires_grad=True)
        logits = net(x)
        assert logits.shape == target.shape, (
            f"model returned {tuple(logits.shape)} for a "
            f"{tuple(x.shape)} input")
        criterion.components(logits, target)[name].backward()
        assert x.grad is not None, f"{name}: no gradient reached the input"
        assert torch.isfinite(x.grad).all(), f"{name}: non-finite input gradient"
        assert float(x.grad.abs().sum()) > 0, f"{name}: zero input gradient"


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
def test_shape_and_value_violations_raise(criterion):
    target = losses.as_target(_synthetic_mask(64))
    with pytest.raises(losses.LossError):
        criterion.components(torch.zeros(1, 1, 32, 32), target)
    with pytest.raises(losses.LossError):
        criterion.components(torch.zeros(1, 64, 64), target[0])
    with pytest.raises(losses.LossError):
        criterion.components(torch.zeros_like(target), target * 1.5)


def test_weights_are_the_configured_ones(criterion, settings):
    mask = _synthetic_mask()
    target = losses.as_target(mask)
    logits = losses.as_logits(losses.cut_gaps(mask, spacing=60, gap=3), MAGNITUDE)
    terms = criterion.components(logits, target)
    expected = (float(settings["w_bce"]) * float(terms["bce"])
                + float(settings["w_dice"]) * float(terms["dice"])
                + float(settings["w_cldice"]) * float(terms["cldice"]))
    assert float(terms["total"]) == pytest.approx(expected, rel=1e-5)


# --------------------------------------------------------------------------
# soft skeleton vs line width: what the iterations do on a straight line, measured
# by running the real code (the derivation is in thickness_invariance_iters)
# --------------------------------------------------------------------------
def _horizontal_band(rows, size=64, c0=4, c1=60):
    """A ``rows``-row band, columns c0..c1 inclusive, centred vertically."""
    mask = np.zeros((size, size), dtype=np.uint8)
    top = (size - rows) // 2
    mask[top:top + rows, c0:c1 + 1] = 1
    return mask


def test_thickness_invariance_iters_is_ceil_of_the_dilated_width_over_root_two():
    assert losses.thickness_invariance_iters(2) == 3      # the default, and where the test passed
    assert losses.thickness_invariance_iters(4) == 5
    assert losses.thickness_invariance_iters(6) == 6
    assert losses.thickness_invariance_iters(1) == 3
    assert losses.thickness_invariance_iters(2.5) == 4
    for bad in (0, 0.5, -2, float("nan")):
        with pytest.raises(losses.LossError):
            losses.thickness_invariance_iters(bad)


@pytest.mark.parametrize("rows", [2, 4, 6])
@pytest.mark.parametrize("iters", [3, 4, 5])
def test_soft_skeleton_of_a_straight_even_width_line_is_a_two_pixel_central_band(rows, iters):
    """A straight line of any even width W has a 2 px central band as its skeleton, at
    every iteration count once W has been consumed -- never a 1 px centreline. (W = 2 comes
    back whole; W = 4 loses its outer rows at the first erosion; W = 6 at the second.)"""
    mask = _horizontal_band(rows)
    skel = losses.soft_skeletonize(losses.as_target(mask), iters)[0, 0].numpy()
    column = skel[:, 32]
    centre = 32                                   # band rows are 32 - rows/2 .. 31 + rows/2
    assert column.sum() == pytest.approx(2.0)
    assert column[centre - 1] == pytest.approx(1.0) and column[centre] == pytest.approx(1.0)


def test_skeleton_measurements_report_thickness_two_and_a_length_for_straight_lines():
    for rows in (2, 4):
        got = losses.skeleton_measurements(_horizontal_band(rows), iters_list=(3, 4, 5))
        assert set(got) == {3, 4, 5}
        for iters, row in got.items():
            assert row["mask_px"] == rows * 57
            assert row["thickness_px"] == pytest.approx(2.0, abs=0.3), (rows, iters, row)
            assert 40 <= row["centreline_px"] <= 57, (rows, iters, row)
            assert row["skeleton_px"] == pytest.approx(2 * row["centreline_px"], rel=0.15)
    # an empty skeleton reports None, not a division by zero
    assert losses.skeleton_measurements(np.zeros((16, 16), dtype=np.uint8), (3,))[3][
        "thickness_px"] is None


@pytest.mark.parametrize("rows, first_iters", [(2, 1), (4, 2)])
def test_thickness_invariance_on_a_straight_line_needs_iterations_of_about_half_the_width(
        rows, first_iters):
    """The straight-line case of the property the dense-tile test asserts. Below
    ``rows / 2`` iterations the dilated (rows + 2 px) line is not yet consumed, its skeleton
    is EMPTY, and clDice comes out ~0 for the wrong reason -- ``pred_skeleton_real`` is the
    check that catches it. From ``rows / 2`` up it holds at every count."""
    profile = losses.thickness_invariance_profile(_horizontal_band(rows), [1, 2, 3, 4, 5],
                                                  line_width=rows)
    for iters, row in profile.items():
        if iters < first_iters:
            # an EMPTY predicted skeleton also has nothing "on" the truth (0 > 0 is false), so
            # the on-truth check fails with it; the real-skeleton check is the one that names it
            assert not row["holds"] and "pred_skeleton_real" in row["failed"], (iters, row)
            assert row["cldice_dilated"] < 0.25 * row["dice_dilated"], (
                "the degenerate case LOOKS invariant, which is the point", iters, row)
            assert row["skel_pred_sum"] == pytest.approx(0.0, abs=1e-2)
        else:
            assert row["holds"], (iters, row)
            assert row["skel_pred_on_true"] > 0.99 * row["skel_pred_sum"]


def test_thickness_invariance_profile_names_the_checks_it_applies():
    profile = losses.thickness_invariance_profile(_horizontal_band(4), [3], line_width=4)
    assert set(profile[3]["checks"]) == set(losses.INVARIANCE_CHECKS)
    assert profile[3]["failed"] == [] and profile[3]["holds"] is True


# --------------------------------------------------------------------------
# the width-independent invariance checks
# --------------------------------------------------------------------------
def _old_width2_checks(parts, dice, cldice):
    """The checks as first written (calibrated at 2 px): both skeleton thresholds were a
    fraction of the MASK AREA. Kept here only to show the new ones agree at width 2."""
    return {
        "cldice_below_quarter_of_dice": cldice < 0.25 * dice,
        "pred_skeleton_real": parts["skel_pred_sum"] > 0.25 * parts["true_sum"],
        "true_skeleton_real": parts["skel_true_sum"] > 0.25 * parts["true_sum"],
        "pred_skeleton_on_true": parts["skel_pred_on_true"] > 0.8 * parts["skel_pred_sum"],
    }


def _grid_of_lines(width, size=64, spacing=16):
    mask = np.zeros((size, size), dtype=np.uint8)
    for k in range(spacing // 2, size, spacing):
        mask[k:k + width, 4:size - 4] = 1
        mask[4:size - 4, k:k + width] = 1
    return mask


@pytest.mark.parametrize("fixture", ["band", "grid"])
@pytest.mark.parametrize("iters", [1, 2, 3, 4, 5, 6])
def test_new_invariance_checks_agree_with_the_old_ones_at_width_two(fixture, iters):
    mask = _horizontal_band(2) if fixture == "band" else _grid_of_lines(2)
    target = losses.as_target(mask)
    logits = losses.as_logits(losses.dilate_mask(mask, iterations=1), 10.0)
    dice = float(losses.dice_term(logits, target, smooth=1.0))
    cl = float(losses.cldice_term(logits, target, iters=iters, smooth=1.0, eps=1e-6))
    parts = losses.cldice_parts(logits, target, iters=iters, smooth=1.0, eps=1e-6)
    new = losses.invariance_checks(parts, dice, cl, line_width=2)
    assert new == _old_width2_checks(parts, dice, cl), (fixture, iters, parts, new)


def test_invariance_checks_flag_the_four_pixel_dense_tile_failures_from_colab():
    """The numbers reported by the Colab run of the dense MetalDam tile at 4 px (the data is
    not available here, so they are entered as given): iters=3 left a 904 px predicted
    skeleton against a 7069 px true one -- NOT real -- while iters=5 (skeleton 3385.7 against
    7190, mask area 39486, overlap 3225) is, which the old area-based threshold (0.25 * 39486
    = 9871 > 3385.7) wrongly refused."""
    dice = 0.15544
    iters3 = {"skel_pred_sum": 904.0, "skel_true_sum": 7069.0, "true_sum": 39486.0,
              "skel_pred_on_true": 850.0}
    got3 = losses.invariance_checks(iters3, dice, 0.04205, line_width=4)
    assert not got3["pred_skeleton_real"]                 # 904 < 0.25 * 7069
    iters5 = {"skel_pred_sum": 3385.7, "skel_true_sum": 7190.0, "true_sum": 39486.0,
              "skel_pred_on_true": 3225.0}
    got5 = losses.invariance_checks(iters5, dice, 0.02437, line_width=4)
    assert all(got5.values()), got5
    assert not _old_width2_checks(iters5, dice, 0.02437)["pred_skeleton_real"]


def test_invariance_checks_refuse_a_nonsense_width():
    parts = {"skel_pred_sum": 1.0, "skel_true_sum": 1.0, "true_sum": 1.0, "skel_pred_on_true": 1.0}
    for bad in (0, 0.5, float("nan")):
        with pytest.raises(losses.LossError):
            losses.invariance_checks(parts, 0.1, 0.01, bad)
