"""Tests for src/train.py's metric sweep. RUN THESE IN notebooks/06.

The threshold sweep evaluates 19 operating points from one pass over each
tile, using ``searchsorted`` cumulants and two morphological dilations instead
of 19 thresholdings and 19 distance transforms. That is a worthwhile speedup
and an easy thing to get subtly wrong, so every count it produces is checked
against the obvious slow implementation on random data.

Nothing here needs a GPU or the datasets, so nothing here can skip.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import train as train_mod


THRESHOLDS = np.round(np.linspace(0.05, 0.95, 19), 6)


def _settings(**overrides):
    settings = dict(train_mod.DEFAULTS)
    settings.update(overrides)
    return settings


def _at(values, threshold):
    """``threshold`` as the data sees it -- the reference rule, in one place.

    The grid is decimal and the probabilities are float32, and none of 0.05,
    0.95 ... survives that round trip exactly. Comparing a float32 array
    against a float64 constant therefore counts a pixel sitting exactly on its
    threshold or not depending on which way that particular constant rounded:
    ``float32(0.05)`` lands above the float64 0.05, ``float32(0.95)`` lands
    below the float64 0.95. ``src.train.snap_thresholds`` fixes that rule at
    the data's precision, and this is the same rule written the obvious way.
    """
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.floating):
        return values.dtype.type(threshold)
    return threshold


def _brute_force(prob, true, threshold, tolerance):
    """The obvious implementation: threshold, then measure, one point at a time."""
    import cv2

    pred = prob >= _at(prob, threshold)
    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    n_pred, n_true = int(pred.sum()), int(true.sum())

    # Boundary hits from an explicit Euclidean distance transform, which is a
    # completely different route to the same number than the dilation the
    # accumulator uses.
    # DIST_MASK_PRECISE, not the 3x3 or 5x5 chamfer approximations: those are
    # off by a couple of percent, which is exactly enough to disagree with an
    # exact disk at a distance of exactly 2 and turn this into a flaky test.
    if n_true:
        dist_to_true = cv2.distanceTransform((~true).astype(np.uint8),
                                             cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        hits_p = int((pred & (dist_to_true <= tolerance + 1e-6)).sum())
    else:
        hits_p = 0
    if n_pred:
        dist_to_pred = cv2.distanceTransform((~pred).astype(np.uint8),
                                             cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        hits_r = int((true & (dist_to_pred <= tolerance + 1e-6)).sum())
    else:
        hits_r = 0
    return {"tp": tp, "fp": fp, "fn": fn, "n_pred": n_pred, "n_true": n_true,
            "hits_p": hits_p, "hits_r": hits_r}


def _random_tile(seed=0, size=64, boundary_fraction=0.08):
    """A blurry probability map over a thin, structured target."""
    import cv2

    rng = np.random.default_rng(seed)
    true = np.zeros((size, size), dtype=bool)
    for k in range(7, size - 2, 17):
        true[k:k + 2, :] = True
        true[:, k:k + 2] = True
    # A prediction correlated with the truth but wrong in both directions.
    prob = cv2.GaussianBlur(true.astype(np.float32), (7, 7), 2.0)
    prob = np.clip(prob * 1.4 + rng.normal(0, 0.15, true.shape), 0.0, 1.0)
    return prob.astype(np.float32), true


# --------------------------------------------------------------------------
# ge_counts is exactly "how many values clear this threshold"
# --------------------------------------------------------------------------
def test_ge_counts_matches_the_obvious_comparison():
    rng = np.random.default_rng(0)
    values = rng.random(5000).astype(np.float32)
    got = train_mod.ge_counts(values, THRESHOLDS)
    for k, threshold in enumerate(THRESHOLDS):
        assert got[k] == int((values >= _at(values, threshold)).sum()), (
            f"threshold {threshold}")


def test_ge_counts_handles_exact_hits_and_empty_input():
    """The case that caught the bug: a value sitting exactly on 0.95.

    Random data never lands on a threshold, so it cannot exercise this at all.
    ``float32(0.95)`` is 0.9499999880790710, just BELOW the float64 0.95, so
    comparing across dtypes dropped it -- while ``float32(0.05)`` is just ABOVE
    the float64 0.05 and was kept. Same situation, opposite answers, decided by
    nothing a reader could predict.
    """
    values = np.array([0.05, 0.5, 0.95, 0.5], dtype=np.float32)
    got = train_mod.ge_counts(values, THRESHOLDS)
    assert got[0] == 4, "a value exactly on the threshold must count as >= it"
    assert got[int(np.nonzero(np.isclose(THRESHOLDS, 0.5))[0][0])] == 3
    assert got[-1] == 1, (
        "float32(0.95) sits exactly on the 0.95 threshold at the data's "
        "precision and must clear it, exactly as float32(0.05) clears 0.05")
    assert train_mod.ge_counts(np.array([], dtype=np.float32),
                               THRESHOLDS).tolist() == [0] * len(THRESHOLDS)


def test_ge_counts_matches_brute_force_at_every_exact_threshold_hit():
    """Every grid point, in both dtypes, with values sitting exactly on it.

    This is the check the random-data test cannot make: random floats never
    land on a threshold, so an exact-hit off-by-one is invisible to it. Here
    each grid value is fed in exactly, twice over, with neighbours just below
    and just above, and every count is compared against the obvious
    implementation at the same precision.
    """
    for dtype in (np.float32, np.float64):
        for threshold in THRESHOLDS:
            values = np.array(
                [threshold, threshold, threshold - 0.01, threshold + 0.01,
                 0.0, 1.0], dtype=dtype)
            got = train_mod.ge_counts(values, THRESHOLDS)
            for k, other in enumerate(THRESHOLDS):
                expected = int((values >= _at(values, other)).sum())
                assert got[k] == expected, (
                    f"{np.dtype(dtype).name} values sitting on {threshold}: "
                    f"count at threshold {other} was {got[k]}, brute force "
                    f"says {expected}")
            index = int(np.nonzero(np.isclose(THRESHOLDS, threshold))[0][0])
            assert got[index] >= 2, (
                f"the two values placed exactly on {threshold} must clear it")


def test_snap_thresholds_is_uniform_across_the_grid_and_safe():
    """The rule holds at every grid point, and never breaks the grid itself."""
    snapped = train_mod.snap_thresholds(THRESHOLDS, np.float32)
    assert np.all(np.diff(snapped) > 0), "searchsorted needs an increasing grid"
    for threshold, point in zip(THRESHOLDS, snapped):
        assert np.float32(threshold) >= point, (
            f"a float32 value equal to {threshold} must clear its own "
            "threshold")
    assert train_mod.snap_thresholds(THRESHOLDS, np.float64).tolist() == \
        THRESHOLDS.tolist(), "float64 data needs no snapping"

    # A grid so fine that snapping would collapse two points keeps float64
    # rather than handing searchsorted a non-increasing array.
    fine = np.array([0.5, 0.5 + 1e-9, 0.6])
    assert train_mod.snap_thresholds(fine, np.float32).tolist() == fine.tolist()


# --------------------------------------------------------------------------
# the disk is the tolerance ball, not an approximation of it
# --------------------------------------------------------------------------
def test_euclidean_disk_is_the_exact_tolerance_ball():
    disk = train_mod.euclidean_disk(2)
    assert disk.shape == (5, 5)
    # Corners are at distance sqrt(8) > 2 and must be excluded; (2, 0) is at
    # exactly 2 and must be included.
    assert disk[0, 0] == 0 and disk[4, 4] == 0
    assert disk[0, 2] == 1 and disk[2, 0] == 1
    assert int(disk.sum()) == 13
    assert train_mod.euclidean_disk(0).shape == (1, 1)


# --------------------------------------------------------------------------
# every count the sweep produces, against the slow implementation
# --------------------------------------------------------------------------
def test_sweep_counts_match_brute_force_at_every_threshold():
    tolerance = 2
    prob, true = _random_tile(seed=1)
    accumulator = train_mod.MetricAccumulator(
        THRESHOLDS, fixed_threshold=0.5, tolerance_px=tolerance)
    accumulator.update(prob[None, None], true[None, None].astype(np.float32),
                       ["probe"])
    slot = accumulator.counts["probe"]

    for k, threshold in enumerate(THRESHOLDS):
        expected = _brute_force(prob, true, threshold, tolerance)
        counts = accumulator._counts_at(slot, k)
        assert counts["tp"] == expected["tp"], f"tp at {threshold}"
        assert counts["fp"] == expected["fp"], f"fp at {threshold}"
        assert counts["fn"] == expected["fn"], f"fn at {threshold}"
        assert counts["pred_px"] == expected["n_pred"], f"n_pred at {threshold}"
        assert counts["bf_hits_p"] == expected["hits_p"], (
            f"boundary precision hits at {threshold}")
        assert counts["bf_hits_r"] == expected["hits_r"], (
            f"boundary recall hits at {threshold}")


def test_sweep_accumulates_across_tiles_and_keeps_datasets_apart():
    tolerance = 2
    tiles = [_random_tile(seed=s) for s in (2, 3, 4)]
    names = ["alpha", "beta", "alpha"]
    accumulator = train_mod.MetricAccumulator(
        THRESHOLDS, fixed_threshold=0.5, tolerance_px=tolerance)
    for (prob, true), name in zip(tiles, names):
        accumulator.update(prob[None, None], true[None, None].astype(np.float32),
                           [name])

    result = accumulator.result()
    assert set(result["per_dataset"]) == {"alpha", "beta"}
    assert result["per_dataset"]["alpha"]["fixed"]["tiles"] == 2
    assert result["per_dataset"]["beta"]["fixed"]["tiles"] == 1
    assert result["pooled"]["fixed"]["tiles"] == 3

    # Pooled counts are the sum of the per-dataset ones, at every threshold.
    for k in range(len(THRESHOLDS)):
        pooled = accumulator._counts_at(accumulator.counts["__pooled__"], k)
        parts = [accumulator._counts_at(accumulator.counts[n], k)
                 for n in ("alpha", "beta")]
        for field in ("tp", "fp", "fn", "pred_px", "bf_hits_p", "bf_hits_r"):
            assert pooled[field] == sum(p[field] for p in parts), field


# --------------------------------------------------------------------------
# the reported rows mean what they say
# --------------------------------------------------------------------------
def test_fixed_row_is_the_configured_threshold_and_best_is_at_least_as_good():
    prob, true = _random_tile(seed=5)
    accumulator = train_mod.MetricAccumulator(
        THRESHOLDS, fixed_threshold=0.5, tolerance_px=2)
    accumulator.update(prob[None, None], true[None, None].astype(np.float32),
                       ["probe"])
    entry = accumulator.result()["per_dataset"]["probe"]

    assert entry["fixed"]["threshold"] == pytest.approx(0.5)
    assert entry["fixed_threshold"] == pytest.approx(0.5)
    assert entry["best"]["threshold"] == pytest.approx(entry["best_threshold"])
    assert entry["best"]["dice"] >= entry["fixed"]["dice"], (
        "the swept best must be at least as good as the fixed point, since the "
        "fixed point is on the grid")
    assert entry["best"]["dice"] == pytest.approx(
        max(row["dice"] for row in entry["sweep"]))
    assert len(entry["sweep"]) == len(THRESHOLDS)


def test_best_threshold_is_found_when_the_fixed_one_is_badly_wrong():
    """A model whose probabilities are shifted high: 0.5 over-paints massively.

    This is the situation that prompted the sweep -- pred_frac 0.312 against a
    true 0.054 at a fixed 0.5 -- reproduced deliberately.
    """
    size = 64
    true = np.zeros((size, size), dtype=bool)
    true[20:22, :] = True
    prob = np.full((size, size), 0.62, dtype=np.float32)   # confident everywhere
    prob[true] = 0.93                                      # more so on boundary

    accumulator = train_mod.MetricAccumulator(
        THRESHOLDS, fixed_threshold=0.5, tolerance_px=2)
    accumulator.update(prob[None, None], true[None, None].astype(np.float32),
                       ["probe"])
    entry = accumulator.result()["per_dataset"]["probe"]

    assert entry["fixed"]["pred_fraction"] == pytest.approx(1.0), (
        "at 0.5 this model paints the whole tile, which is the point")
    assert entry["fixed"]["precision"] < 0.1
    assert entry["best_threshold"] > 0.62, (
        f"the sweep should have moved above the background level, got "
        f"{entry['best_threshold']}")
    assert entry["best"]["dice"] > 0.9 * entry["fixed"]["dice"] + 0.5, (
        f"fixed dice {entry['fixed']['dice']:.4f} -> best "
        f"{entry['best']['dice']:.4f}; the sweep barely helped")
    assert entry["best"]["precision"] > entry["fixed"]["precision"]


def test_ties_break_towards_the_conservative_threshold():
    """A false boundary splits a region permanently; a missed one may not."""
    size = 32
    true = np.zeros((size, size), dtype=bool)
    true[10:12, :] = True
    # Identical Dice at every threshold between 0.3 and 0.7, because nothing
    # in the map lies in that band.
    prob = np.zeros((size, size), dtype=np.float32)
    prob[true] = 0.8
    accumulator = train_mod.MetricAccumulator(
        THRESHOLDS, fixed_threshold=0.5, tolerance_px=2)
    accumulator.update(prob[None, None], true[None, None].astype(np.float32),
                       ["probe"])
    entry = accumulator.result()["per_dataset"]["probe"]
    assert entry["best_threshold"] == pytest.approx(0.8), (
        f"expected the highest tying threshold, got {entry['best_threshold']}")


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
def test_threshold_grid_contains_the_fixed_threshold():
    grid = train_mod.threshold_grid(_settings())
    assert np.isclose(grid, 0.5).sum() == 1
    assert grid[0] == pytest.approx(0.05) and grid[-1] == pytest.approx(0.95)
    assert len(grid) == 19

    odd = train_mod.threshold_grid(_settings(threshold=0.42))
    assert np.isclose(odd, 0.42).sum() == 1, (
        "a fixed threshold off the grid must be inserted, or the two reported "
        "rows would not be comparable")
    assert len(odd) == 20


def test_a_fixed_threshold_off_the_grid_is_rejected_by_the_accumulator():
    with pytest.raises(train_mod.TrainError):
        train_mod.MetricAccumulator(THRESHOLDS, fixed_threshold=0.42,
                                    tolerance_px=2)


def test_bad_sweep_configuration_raises():
    for bad in ({"threshold_sweep_min": 0.9, "threshold_sweep_max": 0.1},
                {"threshold_sweep_min": 0.0},
                {"threshold_sweep_max": 1.0},
                {"threshold_sweep_step": 0.0},
                {"threshold_sweep_step": 5.0},
                {"threshold": 0.0},
                {"threshold": 1.0}):
        with pytest.raises(train_mod.TrainError):
            train_mod.threshold_grid(_settings(**bad))


def test_the_sweep_grid_is_part_of_the_config_hash():
    """Changing it changes which checkpoint is `best`, so a resume must refuse."""
    base = _settings()
    changed = _settings(threshold_sweep_step=0.1)
    model = {"encoder": "resnet34"}
    loss = {"w_cldice": 0.5}
    dataset = {"patch_size": 256}
    assert (train_mod.config_hash(model, loss, base, dataset)[0]
            != train_mod.config_hash(model, loss, changed, dataset)[0])


# --------------------------------------------------------------------------
# what is NOT reported
# --------------------------------------------------------------------------
def test_no_metric_is_called_accuracy():
    prob, true = _random_tile(seed=6)
    accumulator = train_mod.MetricAccumulator(
        THRESHOLDS, fixed_threshold=0.5, tolerance_px=2)
    accumulator.update(prob[None, None], true[None, None].astype(np.float32),
                       ["probe"])
    result = accumulator.result()
    for entry in list(result["per_dataset"].values()) + [result["pooled"]]:
        for row in ("fixed", "best"):
            assert not any("accuracy" in key for key in entry[row]), (
                "boundaries are 5-15% of pixels; accuracy reads 85-95% for a "
                "model that predicts nothing")
