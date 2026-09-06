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
    candidates = [r for r in rows if r["dataset"] == dataset_name]
    if not candidates:
        pytest.skip(f"no {dataset_name} tiles in {MANIFEST.name}")
    candidates.sort(key=lambda r: r["boundary_fraction"], reverse=densest)
    crops = ds.load_crops()
    for row in candidates[:20]:
        try:
            _, gt_full = ds.read_pair(row, crops=crops, roots=roots)
        except Exception:
            continue
        mask = (ds.crop_tile(gt_full, row["x"], row["y"], int(row["patch"])) > 0)
        if mask.sum() > 200:
            return mask.astype(np.uint8)
    pytest.skip(f"{dataset_name} source images are not reachable on this host")


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
def test_cldice_weights_a_gap_more_than_dice_does(criterion, dense_tile):
    """The claim clDice is in the loss for, measured on a real boundary tile.

    Two errors are compared, both applied to the ground truth itself:

    * **gap** -- the line is severed periodically. Topology destroyed.
    * **dilated** -- the line is one pixel fatter all round. Topology intact,
      and far MORE pixels are wrong than in the gap case.

    Dice counts pixels, so it charges much more for the harmless error than for
    the destructive one. clDice compares skeletons, so dilation is nearly free
    to it while the break is not. The assertion is the cross-ratio, written
    without division so a near-zero clDice penalty on the dilated case cannot
    blow it up.

    On the gap case the two terms come out ALMOST EQUAL rather than clDice
    coming out larger, and that is a real property of this data, not a weak
    test: ``boundary_gt.line_width_px`` is 2, and one erosion removes a 2 px
    line entirely, so the soft skeleton of the ground truth IS the ground
    truth and clDice reduces to a harmonic mean of precision and recall. What
    clDice contributes here is therefore the RELATIVE weighting -- it refuses
    to be dominated by thickness the way Dice is -- plus real skeletonization
    of the model's own soft, over-thick predictions during training.
    """
    truth = dense_tile
    gapped = losses.cut_gaps(truth, spacing=180, gap=3)
    dilated = losses.dilate_mask(truth, iterations=1)

    removed = 1.0 - gapped.sum() / max(1, truth.sum())
    added = dilated.sum() / max(1, truth.sum()) - 1.0
    assert 0.005 < removed < 0.25, (
        f"cut_gaps removed {removed:.1%} of the boundary; the comparison needs "
        "a break, not an erasure")
    assert added > 0.2, f"dilate_mask only added {added:.1%}"

    target = losses.as_target(truth)
    scores = {}
    for name, mask in (("gap", gapped), ("dilated", dilated)):
        terms = criterion.components(losses.as_logits(mask, MAGNITUDE), target)
        scores[name] = {k: float(terms[k]) for k in ("dice", "cldice", "total")}

    dice_gap, cl_gap = scores["gap"]["dice"], scores["gap"]["cldice"]
    dice_dil, cl_dil = scores["dilated"]["dice"], scores["dilated"]["cldice"]
    detail = (f"gap: dice={dice_gap:.5f} cldice={cl_gap:.5f} "
              f"(removed {removed:.1%}); dilated: dice={dice_dil:.5f} "
              f"cldice={cl_dil:.5f} (added {added:.1%})")

    assert cl_gap >= 0.9 * dice_gap, (
        "clDice charges LESS than Dice for a severed line -- " + detail)
    assert dice_dil > 3.0 * dice_gap, (
        "Dice was expected to be dominated by the thickness error -- " + detail)
    # Cross-multiplied: cl_gap / cl_dil  >  1.5 * (dice_gap / dice_dil)
    assert cl_gap * dice_dil > 1.5 * cl_dil * dice_gap, (
        "clDice does not weight the break above the thickness error more than "
        "Dice does -- " + detail)


def test_soft_skeleton_of_a_two_pixel_line_is_the_line(settings):
    """The property the test above rests on, isolated and stated outright."""
    mask = _synthetic_mask()
    target = losses.as_target(mask)
    skel = losses.soft_skeletonize(target, int(settings["cldice_iters"]))
    assert torch.allclose(skel, target, atol=1e-6), (
        "the soft skeleton of a 2 px line is no longer the line itself; the "
        "clDice reasoning in notebook 05 needs re-deriving")


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
