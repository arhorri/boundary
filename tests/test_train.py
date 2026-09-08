"""Tests for src/train.py's metric sweep. RUN THESE IN notebooks/06.

The threshold sweep evaluates 19 operating points from one pass over each
tile, using ``searchsorted`` cumulants and two morphological dilations instead
of 19 thresholdings and 19 distance transforms. That is a worthwhile speedup
and an easy thing to get subtly wrong, so every count it produces is checked
against the obvious slow implementation on random data.

Nothing here needs a GPU or the datasets, so nothing here can skip.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

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


# --------------------------------------------------------------------------
# rank_tiles_by_dice -- best / median / two worst, deterministically
# --------------------------------------------------------------------------
def _record(row_index, dataset, dice):
    return {"row_index": row_index, "dataset": dataset, "tile_id": f"t{row_index}",
            "dice": dice, "threshold": 0.5}


def test_rank_tiles_by_dice_picks_best_median_and_two_worst():
    records = [_record(0, "A", 0.10), _record(1, "A", 0.90),
               _record(2, "A", 0.50), _record(3, "A", 0.20),
               _record(4, "A", 0.70), _record(5, "A", 0.30),
               _record(6, "B", 0.99)]
    picks = train_mod.rank_tiles_by_dice(records, "A")
    assert picks["n_tiles"] == 6
    assert picks["best"]["row_index"] == 1 and picks["best"]["dice"] == 0.90
    assert picks["worst"]["row_index"] == 0 and picks["worst"]["dice"] == 0.10
    assert picks["worst2"]["row_index"] == 3 and picks["worst2"]["dice"] == 0.20
    # sorted ascending [0.10, 0.20, 0.30, 0.50, 0.70, 0.90] -> index 6//2=3 -> 0.50
    assert picks["median"]["dice"] == 0.50
    assert all(r["dataset"] == "A" for r in picks.values() if isinstance(r, dict))


def test_rank_tiles_by_dice_ties_break_by_row_index_deterministically():
    records = [_record(5, "A", 0.5), _record(2, "A", 0.5), _record(9, "A", 0.5)]
    picks = train_mod.rank_tiles_by_dice(records, "A")
    # sort key is (dice, -row_index): equal dice orders HIGHEST row_index first,
    # so ascending order is [9, 5, 2] and "worst" (position 0) is row 9.
    assert picks["worst"]["row_index"] == 9
    assert picks["best"]["row_index"] == 2


def test_rank_tiles_by_dice_degenerate_single_tile_fills_every_slot():
    records = [_record(0, "A", 0.42)]
    picks = train_mod.rank_tiles_by_dice(records, "A")
    assert picks["n_tiles"] == 1
    assert picks["best"] is picks["median"] is picks["worst"] is picks["worst2"]


def test_rank_tiles_by_dice_missing_dataset_raises():
    with pytest.raises(train_mod.TrainError):
        train_mod.rank_tiles_by_dice([_record(0, "A", 0.5)], "B")


# --------------------------------------------------------------------------
# evaluate_checkpoint -- per-tile bookkeeping against a synthetic model
# --------------------------------------------------------------------------
class _FakeValDataset(torch.utils.data.Dataset):
    """A minimal val_ds stand-in: fixed masks, no images, no augmentation.

    Just enough of the TileDataset contract for evaluate_checkpoint to run
    against: ``rows`` (dataset name + tile_id per index, in order) and
    ``__getitem__`` returning ``{"image", "mask"}`` as CHW float32 arrays.
    """

    def __init__(self, masks, datasets):
        assert len(masks) == len(datasets)
        self.masks = [np.asarray(m, dtype=np.float32) for m in masks]
        self.rows = [{"dataset": d, "tile_id": f"tile{i}"}
                    for i, d in enumerate(datasets)]

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        mask = self.masks[idx][None, ...]
        return {"image": np.zeros_like(mask), "mask": mask}


class _ConstantLogits(torch.nn.Module):
    """Returns pre-baked logits per call, in the order batches are requested.

    Requires ``shuffle=False`` in the caller's DataLoader -- exactly what
    evaluate_checkpoint uses -- so the Nth forward call corresponds to the Nth
    dataset row.
    """

    def __init__(self, logits_by_index):
        super().__init__()
        self.logits_by_index = logits_by_index
        self._next = 0

    def forward(self, x):
        batch = x.shape[0]
        out = torch.stack(
            [self.logits_by_index[self._next + i] for i in range(batch)])
        self._next += batch
        return out


def test_evaluate_checkpoint_matches_hand_computed_dice_and_preserves_order():
    size = 16
    true_a = np.zeros((size, size), dtype=np.float32)
    true_a[4:6, :] = 1.0                       # a clean 2-row band, 32 px
    true_b = np.zeros((size, size), dtype=np.float32)
    true_b[10:12, :] = 1.0

    # Tile 0 (dataset A): predict the band exactly -> Dice 1.0.
    logit_exact = torch.where(torch.from_numpy(true_a) > 0,
                              torch.tensor(10.0), torch.tensor(-10.0))
    # Tile 1 (dataset A): predict nothing -> Dice 0.0 (true has boundary).
    logit_empty = torch.full((size, size), -10.0)
    # Tile 2 (dataset B): predict everything -> known tp/fp/fn by hand.
    logit_full = torch.full((size, size), 10.0)

    val_ds = _FakeValDataset([true_a, true_a, true_b],
                             datasets=["A", "A", "B"])
    model = _ConstantLogits([logit_exact[None], logit_empty[None],
                            logit_full[None]])
    thresholds = {"A": 0.5, "B": 0.5}

    records = train_mod.evaluate_checkpoint(
        model, val_ds, thresholds, device=torch.device("cpu"),
        batch_size=2, num_workers=0)

    assert [r["row_index"] for r in records] == [0, 1, 2]
    assert [r["dataset"] for r in records] == ["A", "A", "B"]
    assert [r["tile_id"] for r in records] == ["tile0", "tile1", "tile2"]

    assert records[0]["dice"] == pytest.approx(1.0)
    assert records[1]["dice"] == pytest.approx(0.0)

    # tile 2: true_b has 32 true px out of 256; predicting everything gives
    # tp=32, fp=224, fn=0 -> dice = 64 / (64 + 224) = 0.2222...
    assert records[2]["dice"] == pytest.approx(64 / 288, rel=1e-6)
    assert records[2]["pred_fraction"] == pytest.approx(1.0)
    assert records[2]["true_fraction"] == pytest.approx(32 / 256)

    # Width: an unbroken 2-row band has a measurable width close to 2 px.
    assert records[0]["true_width_px"] is not None
    assert records[0]["true_width_px"] == pytest.approx(2.0, abs=0.5)
    # tile 1 predicts nothing, so its predicted width is undefined (None),
    # not zero -- an empty prediction has no skeleton to measure.
    assert records[1]["pred_width_px"] is None


def test_evaluate_checkpoint_requires_a_threshold_for_every_dataset():
    val_ds = _FakeValDataset([np.zeros((8, 8), dtype=np.float32)],
                             datasets=["A"])
    model = _ConstantLogits([torch.full((1, 8, 8), -10.0)])
    with pytest.raises(train_mod.TrainError):
        train_mod.evaluate_checkpoint(model, val_ds, {}, device=torch.device("cpu"))


# --------------------------------------------------------------------------
# best_epoch_thresholds -- reads the history record for the SAVED best epoch
# --------------------------------------------------------------------------
def test_best_epoch_thresholds_reads_the_matching_history_record():
    state = {
        "best": {"epoch": 2},
        "history": [
            {"epoch": 0, "best_thresholds": {"A": 0.3}},
            {"epoch": 1, "best_thresholds": {"A": 0.4}},
            {"epoch": 2, "best_thresholds": {"A": 0.6, "B": 0.8}},
        ],
    }
    assert train_mod.best_epoch_thresholds(state) == {"A": 0.6, "B": 0.8}


def test_best_epoch_thresholds_missing_best_epoch_raises():
    with pytest.raises(train_mod.TrainError):
        train_mod.best_epoch_thresholds({"history": []})


def test_best_epoch_thresholds_no_matching_history_record_raises():
    state = {"best": {"epoch": 5}, "history": [{"epoch": 0, "best_thresholds": {}}]}
    with pytest.raises(train_mod.TrainError):
        train_mod.best_epoch_thresholds(state)


# --------------------------------------------------------------------------
# load_checkpoint_model -- the same fold/hash guards maybe_resume applies
# --------------------------------------------------------------------------
def _model_settings_without_pretrained_download():
    from src import model as model_mod

    settings = dict(model_mod.load_config())
    settings["encoder_weights"] = None   # no network access in a unit test
    return settings


def test_load_checkpoint_model_refuses_a_checkpoint_from_another_fold(tmp_path):
    from src import model as model_mod

    settings = _model_settings_without_pretrained_download()
    dummy = model_mod.build_model(settings=settings)
    path = tmp_path / "other_fold.pt"
    train_mod.atomic_save(
        {"fold": "fold_not_this_one", "config_hash": "abc", "model": dummy.state_dict()},
        path)
    with pytest.raises(train_mod.TrainError):
        train_mod.load_checkpoint_model(path, settings, fold="dev")


def test_load_checkpoint_model_refuses_a_checkpoint_under_a_different_hash(tmp_path):
    from src import model as model_mod

    settings = _model_settings_without_pretrained_download()
    dummy = model_mod.build_model(settings=settings)
    path = tmp_path / "wrong_hash.pt"
    train_mod.atomic_save(
        {"fold": "dev", "config_hash": "abc", "model": dummy.state_dict()}, path)
    with pytest.raises(train_mod.TrainError):
        train_mod.load_checkpoint_model(path, settings, fold="dev",
                                        expected_hash="different")


def test_load_checkpoint_model_loads_a_matching_checkpoint(tmp_path):
    from src import model as model_mod

    settings = _model_settings_without_pretrained_download()
    dummy = model_mod.build_model(settings=settings)
    path = tmp_path / "ok.pt"
    train_mod.atomic_save(
        {"fold": "dev", "config_hash": "abc", "model": dummy.state_dict()}, path)
    model, state = train_mod.load_checkpoint_model(
        path, settings, fold="dev", expected_hash="abc")
    assert state["fold"] == "dev"
    assert not model.training, "a model returned for evaluation must be in eval mode"


# --------------------------------------------------------------------------
# placement vs thickness -- the decomposition must tell the two apart
# --------------------------------------------------------------------------
RADII = (0, 1, 2, 3)
DISTS = (0, 1, 2, 3, 5)


def _grid(size=96):
    """A 2-px grid, the width boundary_gt.line_width_px actually produces."""
    true = np.zeros((size, size), dtype=bool)
    for offset in range(size // 8, size - 2, size // 4):
        true[offset:offset + 2, :] = True
        true[:, offset:offset + 2] = True
    return true


def _summarise_one(pred, true):
    slot = train_mod._decomposition_slot(RADII, DISTS)
    train_mod._accumulate_decomposition(
        slot, train_mod.decomposition_counts(pred, true, RADII, DISTS))
    return train_mod.summarise_decomposition(slot, RADII, DISTS)


def test_decomposition_k0_reproduces_the_plain_pixel_dice():
    """The k=0 row is the anchor: it must equal the ordinary Dice exactly."""
    import cv2

    true = _grid()
    pred = cv2.dilate(true.astype(np.uint8),
                      train_mod.euclidean_disk(1)).astype(bool)
    entry = _summarise_one(pred, true)

    tp = int((pred & true).sum())
    expected = 2 * tp / (int(pred.sum()) + int(true.sum()))
    assert entry["pixel_dice"] == pytest.approx(expected)
    assert entry["dilated_dice"][0] == pytest.approx(expected), (
        "dilating by 0 must be the identity, or the sweep has no anchor")


def test_decomposition_of_a_perfect_prediction_is_one_everywhere():
    true = _grid()
    entry = _summarise_one(true.copy(), true)
    assert entry["pixel_dice"] == pytest.approx(1.0)
    assert entry["skeleton_dice"] == pytest.approx(1.0)
    for d in DISTS:
        assert entry["pred_within"][d] == pytest.approx(1.0)
        assert entry["skeleton_pred_within"][d] == pytest.approx(1.0)
    assert entry["placement_ok"]


def test_a_fat_but_correctly_placed_prediction_reads_as_a_thickness_problem():
    """The case the whole diagnostic exists to identify.

    Pixel Dice collapses while the skeletons stay on top of each other. If this
    ever stops holding, the notebook's verdict is worthless.
    """
    import cv2

    true = _grid()
    for k in (1, 2, 3):
        pred = cv2.dilate(true.astype(np.uint8),
                          train_mod.euclidean_disk(k)).astype(bool)
        entry = _summarise_one(pred, true)
        assert entry["pixel_dice"] < 0.75, f"k={k} should hurt the pixel Dice"
        assert entry["skeleton_dice"] > 0.90, (
            f"k={k}: thickness must not move the skeletons "
            f"(got {entry['skeleton_dice']:.3f})")
        assert entry["skeleton_lift"] > 1.2, f"k={k}"
        assert entry["skeleton_pred_within"][1] > 0.95, f"k={k}"
        assert entry["placement_ok"], f"k={k} verdict: {entry['verdict']}"
        # width is measured as pooled area / skeleton length, and must rise
        assert entry["width_pred"] > entry["width_true"], f"k={k}"


def test_a_misplaced_prediction_reads_as_a_placement_problem():
    true = _grid()
    # Diagonal, so nothing lands back on itself: a purely axial shift leaves
    # the full-width lines overlapping themselves and measures as half-placed.
    pred = np.zeros_like(true)
    pred[4:, 4:] = true[:-4, :-4]
    entry = _summarise_one(pred, true)

    assert entry["skeleton_dice"] < 0.5
    assert entry["skeleton_lift"] < 1.2, (
        "a misplaced prediction's skeleton Dice must track its pixel Dice "
        "rather than lifting above it")
    assert entry["label"] == "MISPLACED", entry["verdict"]
    assert entry["found"] < train_mod.PLACEMENT_FOUND_OK, (
        "the defining feature of misplacement is that the TRUE curves were "
        "not found -- which is what separates it from over-detection")
    assert not entry["placement_ok"]
    assert entry["width_pred"] == pytest.approx(entry["width_true"], rel=0.1), (
        "a shifted copy has the same thickness as the truth, so the width "
        "columns must NOT be what distinguishes it")


def test_dilated_dice_alone_cannot_tell_the_two_apart():
    """Why measurement (b) is reported with a caveat instead of on its own.

    Dilating both masks by 3 px makes almost anything overlap at this boundary
    density, to the point that a correctly-placed-but-fat prediction and a
    misplaced one land within a hundredth of each other -- close enough that
    which of the two comes out ahead is not stable. What is asserted here is
    that (b) FAILS TO SEPARATE them, which is the claim the notebook makes;
    asserting a particular ordering would be pinning a coin flip.
    """
    import cv2

    true = _grid()
    fat = cv2.dilate(true.astype(np.uint8),
                     train_mod.euclidean_disk(3)).astype(bool)
    shifted = np.zeros_like(true)
    shifted[2:, 2:] = true[:-2, :-2]

    fat_entry = _summarise_one(fat, true)
    shifted_entry = _summarise_one(shifted, true)

    gap = abs(shifted_entry["dilated_dice"][3] - fat_entry["dilated_dice"][3])
    assert gap < 0.10, (
        f"measurement (b) now separates placed from misplaced by {gap:.3f} at "
        "k=3; the notebook's warning that it cannot needs rewriting from the "
        "new numbers")
    # ... while the instruments that DO work separate them by a mile.
    assert fat_entry["skeleton_dice"] - shifted_entry["skeleton_dice"] > 0.5, (
        "skeleton Dice must separate placed from misplaced by far more than "
        "the dilated Dice does, or the diagnostic has no instrument at all")
    assert (fat_entry["skeleton_pred_within"][1]
            - shifted_entry["skeleton_pred_within"][1] > 0.5)


def test_noise_is_misplaced_at_every_boundary_density():
    """Coincidental coverage must not be mistaken for over-detection.

    At high boundary density a random pixel lands within tolerance of almost
    anything, so "did we find the true curves" alone rises to 0.90 for pure
    noise and would read as "found them and drew more besides". Skeleton Dice
    is what separates coincidence from structure, and it does so independently
    of density -- which is why the rule consults it and why this test sweeps
    densities rather than checking one grid.
    """
    for size, spacing in ((96, 4), (128, 4), (192, 5), (256, 5)):
        true = np.zeros((size, size), dtype=bool)
        for offset in range(size // 8, size - 2, size // spacing):
            true[offset:offset + 2, :] = True
            true[:, offset:offset + 2] = True
        pred = np.random.default_rng(0).random(true.shape) < true.mean()
        entry = _summarise_one(pred, true)
        assert entry["label"] == "MISPLACED", (
            f"grid {size} (density {true.mean():.3f}): noise was called "
            f"{entry['label']} -- found={entry['found']:.3f}, "
            f"skeleton Dice={entry['skeleton_dice']:.3f}")
        assert entry["skeleton_dice"] < train_mod.PLACEMENT_STRUCTURE_OK
        assert not entry["placement_ok"]


#: Every calibration case and the label it must receive. Written out rather
#: than derived from the name, so that adding a case forces a deliberate
#: decision about what it is supposed to demonstrate.
EXPECTED_LABELS = {
    "perfect": "GOOD",
    "placed, 1 px too fat": "THICKNESS",
    "placed, 2 px too fat": "THICKNESS",
    "placed, 3 px too fat": "THICKNESS",
    "over-detected 1x": "OVER-DETECTION",
    "over-detected 2x": "OVER-DETECTION",
    "over-detected 3x": "OVER-DETECTION",
    "over-detected 1x, 2x too fat": "OVER-DETECTION + THICKNESS",
    "over-detected 2x, 2x too fat": "OVER-DETECTION + THICKNESS",
    "over-detected 3x, 2x too fat": "OVER-DETECTION + THICKNESS",
    "displaced 2 px (in tolerance)": "OFFSET",
    "misplaced by 5 px": "MISPLACED",
    "misplaced by 8 px": "MISPLACED",
    "noise at the same density": "MISPLACED",
}


def test_every_calibration_case_receives_its_own_label():
    """The calibration table the notebook prints must actually calibrate.

    This is the test that would have caught the original bug: over-detection
    was absent from the reference set, so nothing checked that the rule could
    name it, and it was silently reported as misplacement instead.
    """
    reference = train_mod.reference_decomposition(size=256, seed=0)
    assert set(reference) == set(EXPECTED_LABELS), (
        f"calibration cases changed: {sorted(set(reference) ^ set(EXPECTED_LABELS))}")
    wrong = {name: (entry["label"], EXPECTED_LABELS[name])
             for name, entry in reference.items()
             if entry["label"] != EXPECTED_LABELS[name]}
    assert not wrong, f"mislabelled (got, expected): {wrong}"
    # All five distinct outcomes must be exercised, or a rule branch is untested.
    assert set(EXPECTED_LABELS.values()) <= {e["label"] for e in reference.values()}


def test_over_detection_is_not_reported_as_misplacement():
    """The exact regression: true curves found, plus extra curves besides.

    Skeletonizing an over-detected prediction multiplies the skeleton length,
    which drags skeleton Dice down to the range a misplaced prediction sits in.
    Only the second proximity direction separates them, so both are asserted.
    """
    reference = train_mod.reference_decomposition(size=256, seed=0)
    for name in ("over-detected 1x", "over-detected 2x", "over-detected 3x"):
        entry = reference[name]
        assert entry["label"] == "OVER-DETECTION", entry["verdict"]
        assert entry["found"] > 0.95, (
            f"{name}: every true curve is present by construction, so 'found' "
            f"must be high -- got {entry['found']:.3f}")
        assert entry["real"] < train_mod.PLACEMENT_REAL_OK, (
            f"{name}: most of what was drawn is not on a true boundary")
        assert entry["curve_length_ratio"] > 1.5, (
            f"{name}: the extra curves must show up as curve length")
        assert entry["width_ratio"] < train_mod.PLACEMENT_WIDTH_HIGH, (
            f"{name}: nothing was fattened, so width must NOT be blamed")
        # And the old single-direction rule would have got it wrong.
        assert entry["skeleton_dice"] < 0.75, (
            "skeleton Dice alone lands in misplacement territory here, which "
            "is precisely why the verdict may not be read off it")


def test_over_detection_and_thickness_are_named_separately():
    """uhcs2's real failure is both at once, and the verdict must say both."""
    reference = train_mod.reference_decomposition(size=256, seed=0)
    entry = reference["over-detected 3x, 2x too fat"]
    assert entry["label"] == "OVER-DETECTION + THICKNESS"
    assert entry["curve_length_ratio"] > 1.5
    assert entry["width_ratio"] >= train_mod.PLACEMENT_WIDTH_HIGH
    assert "too many curves" in entry["verdict"]
    assert "too fat" in entry["verdict"]


def test_a_sub_tolerance_offset_is_not_called_misplaced():
    """2 px is INSIDE train.boundary_tolerance_px, so it is not misplacement.

    Its pixel Dice is destroyed and its exact skeleton overlap is ~0, but every
    curve is within the tolerance the ground truth's own placement is accurate
    to. Calling that MISPLACED while measuring at 2 px tolerance would be
    self-contradictory, so it gets its own label.
    """
    entry = train_mod.reference_decomposition(size=256, seed=0)[
        "displaced 2 px (in tolerance)"]
    assert entry["label"] == "OFFSET"
    assert entry["pixel_dice"] < 0.2, "the pixel Dice really is destroyed"
    assert entry["found"] > 0.9 and entry["real"] > 0.9
    assert entry["width_ratio"] == pytest.approx(1.0, abs=0.15)
    assert entry["curve_length_ratio"] == pytest.approx(1.0, abs=0.15)


def test_the_curve_length_ratio_is_the_area_ratio_divided_by_the_width_ratio():
    """The identity the notebook states: area = width x curves.

    Reported from skeleton lengths directly rather than by dividing three
    derived numbers, so this pins that the two definitions agree.
    """
    reference = train_mod.reference_decomposition(size=256, seed=0)
    for name, entry in reference.items():
        if entry["true_px"] == 0 or entry["width_ratio"] == 0:
            continue
        area_ratio = entry["pred_px"] / entry["true_px"]
        assert entry["curve_length_ratio"] == pytest.approx(
            area_ratio / entry["width_ratio"], rel=1e-6), name


def test_decompose_error_requires_a_threshold_for_every_dataset():
    val_ds = _FakeValDataset([np.zeros((8, 8), dtype=np.float32)],
                             datasets=["A"])
    model = _ConstantLogits([torch.full((1, 8, 8), -10.0)])
    with pytest.raises(train_mod.TrainError):
        train_mod.decompose_error(model, val_ds, {},
                                  device=torch.device("cpu"))


def test_decompose_error_groups_by_dataset_and_keeps_row_order():
    size = 24
    band = np.zeros((size, size), dtype=np.float32)
    band[8:10, :] = 1.0
    exact = torch.where(torch.from_numpy(band) > 0,
                        torch.tensor(10.0), torch.tensor(-10.0))
    empty = torch.full((size, size), -10.0)

    val_ds = _FakeValDataset([band, band, band], datasets=["A", "B", "B"])
    model = _ConstantLogits([exact[None], exact[None], empty[None]])

    out = train_mod.decompose_error(
        model, val_ds, {"A": 0.5, "B": 0.5}, device=torch.device("cpu"),
        dilations=(1,), distances=(1,), batch_size=2)

    assert set(out) == {"A", "B"}
    assert out["A"]["tiles"] == 1 and out["B"]["tiles"] == 2
    # A predicted its one tile exactly; B got one exact and one empty.
    assert out["A"]["pixel_dice"] == pytest.approx(1.0)
    assert 0.0 < out["B"]["pixel_dice"] < 1.0


# --------------------------------------------------------------------------
# per-fold training-set exclusion
# --------------------------------------------------------------------------
DEV_ENTRY = {"train_datasets": ["MetalDam", "Steel1", "uhcs1"],
             "val_datasets": ["Steel1", "uhcs2"], "held_out": "uhcs2"}


def test_no_exclusion_by_default():
    assert train_mod.DEFAULTS["exclude_datasets"] == {}
    assert train_mod.resolve_exclusions("dev", _settings(), DEV_ENTRY) == []


def test_exclusion_is_scoped_to_its_own_fold():
    """A mapping, not a flat list: excluding for one fold must not touch others."""
    settings = _settings(exclude_datasets={"dev": ["uhcs1"]})
    assert train_mod.resolve_exclusions("dev", settings, DEV_ENTRY) == ["uhcs1"]
    other = {"train_datasets": ["Steel1", "uhcs1", "uhcs2"]}
    assert train_mod.resolve_exclusions("fold_MetalDam", settings, other) == []


def test_excluding_a_dataset_the_fold_never_trained_on_raises():
    """Silently doing nothing would leave the header claiming a false exclusion."""
    settings = _settings(exclude_datasets={"dev": ["Steel2"]})
    with pytest.raises(train_mod.TrainError) as exc:
        train_mod.resolve_exclusions("dev", settings, DEV_ENTRY)
    assert "Steel2" in str(exc.value)


def test_excluding_every_training_dataset_raises():
    settings = _settings(exclude_datasets={"dev": ["MetalDam", "Steel1", "uhcs1"]})
    with pytest.raises(train_mod.TrainError):
        train_mod.resolve_exclusions("dev", settings, DEV_ENTRY)


def _config_with_exclusions(tmp_path, value):
    """A copy of the real default.yaml with train.exclude_datasets replaced.

    Written into tmp_path so that load_config's platform overlay lookup --
    which sits beside the config file -- cannot pick up the repo's own
    overlays and turn this into a test of something else.
    """
    import yaml

    base = yaml.safe_load(
        (Path(train_mod.REPO_ROOT) / "configs" / "default.yaml").read_text())
    base["train"] = dict(base["train"], exclude_datasets=value)
    path = tmp_path / "default.yaml"
    path.write_text(yaml.safe_dump(base))
    return path


def test_exclusion_is_normalised_so_the_hash_ignores_typing_order(tmp_path):
    path = _config_with_exclusions(
        tmp_path, {"dev": ["uhcs1", "MetalDam", "uhcs1"]})
    settings = train_mod.load_config(path)
    assert settings["exclude_datasets"]["dev"] == ["MetalDam", "uhcs1"], (
        "the list must be sorted and de-duplicated, or the config hash would "
        "depend on the order someone typed the names in")


def test_a_bare_list_or_string_exclusion_is_rejected(tmp_path):
    for index, bad in enumerate((["uhcs1"], {"dev": "uhcs1"})):
        directory = tmp_path / str(index)
        directory.mkdir()
        with pytest.raises(train_mod.TrainError):
            train_mod.load_config(_config_with_exclusions(directory, bad))


def test_the_exclusion_is_in_the_config_hash():
    """An excluded-set run must never resume from a full-set checkpoint.

    What is hashed is the exclusion RESOLVED FOR ONE FOLD -- a list -- not the
    whole ``{fold: [...]}`` mapping. Hashing the mapping would invalidate every
    fold's checkpoints whenever any fold's exclusion changed, which is why
    Trainer substitutes the resolved list before hashing.
    """
    model, loss, dataset = ({"encoder": "resnet34"}, {"w_cldice": 0.5},
                            {"patch_size": 256})
    assert "exclude_datasets" in train_mod.HASHED_TRAIN_KEYS
    full = _settings(exclude_datasets=[])
    reduced = _settings(exclude_datasets=["uhcs1"])
    assert (train_mod.config_hash(model, loss, full, dataset)[0]
            != train_mod.config_hash(model, loss, reduced, dataset)[0])


def test_another_folds_exclusion_does_not_disturb_this_folds_hash():
    """The reason the RESOLVED list is hashed rather than the whole mapping.

    Two configs that differ only in what some OTHER fold excludes must produce
    the same hash for this fold, or adding an experiment on fold_MetalDam would
    silently invalidate every dev checkpoint.
    """
    model, loss, dataset = ({"encoder": "resnet34"}, {"w_cldice": 0.5},
                            {"patch_size": 256})
    entry = DEV_ENTRY

    def hash_for(mapping):
        settings = _settings(exclude_datasets=mapping)
        # Exactly what Trainer.__init__ does before hashing.
        resolved = train_mod.resolve_exclusions("dev", settings, entry)
        hashed = dict(settings)
        hashed["exclude_datasets"] = resolved
        return train_mod.config_hash(model, loss, hashed, dataset)[0]

    assert hash_for({"dev": ["uhcs1"]}) == hash_for(
        {"dev": ["uhcs1"], "fold_MetalDam": ["uhcs2"]})
    assert hash_for({"dev": ["uhcs1"]}) != hash_for({})


def test_the_hash_ignores_the_order_names_were_written_in():
    """Order-independence comes from normalisation, not from config_hash.

    ``config_hash`` serialises with ``json.dumps(sort_keys=True)``, which sorts
    dict KEYS and leaves list ELEMENTS alone -- so an unsorted list would hash
    differently. Both entry points sort: ``load_config`` normalises the mapping
    and ``resolve_exclusions`` returns ``sorted(set(...))``. This tests the
    guarantee where it is actually made rather than assuming config_hash
    provides it.
    """
    model, loss, dataset = ({"encoder": "resnet34"}, {"w_cldice": 0.5},
                            {"patch_size": 256})
    entry = {"train_datasets": ["MetalDam", "Steel1", "uhcs1"]}

    def hash_for(order):
        settings = _settings(exclude_datasets={"dev": order})
        hashed = dict(settings)
        hashed["exclude_datasets"] = train_mod.resolve_exclusions(
            "dev", settings, entry)
        return train_mod.config_hash(model, loss, hashed, dataset)[0]

    assert hash_for(["MetalDam", "uhcs1"]) == hash_for(["uhcs1", "MetalDam"])
    # And the normaliser really is what sorts them.
    assert train_mod.resolve_exclusions(
        "dev", _settings(exclude_datasets={"dev": ["uhcs1", "MetalDam"]}),
        entry) == ["MetalDam", "uhcs1"]


# --------------------------------------------------------------------------
# reports are keyed by RUN, and the two arms can be compared
# --------------------------------------------------------------------------
def test_report_paths_are_keyed_by_run_not_just_fold():
    plain_md, plain_json = train_mod.report_paths("dev", "colab", Path("/tmp/r"))
    excl_md, excl_json = train_mod.report_paths("dev-no-uhcs1", "colab",
                                                Path("/tmp/r"))
    assert plain_md.name == "train_dev_colab.md", (
        "a run with no exclusions must keep the filename it already has")
    assert excl_md.name == "train_dev-no-uhcs1_colab.md"
    assert plain_json != excl_json, "the two arms must not overwrite each other"


def _write_report_stub(directory, run_name, host, dataset, dice, tiles=265,
                       excluded=(), epoch=3, config_hash="h", hashed=None):
    payload = {
        "fold": "dev", "run_name": run_name, "platform": host,
        "excluded_datasets": list(excluded), "config_hash": config_hash,
        "hashed_config": hashed,
        "best": {"epoch": epoch},
        "history": [{
            "epoch": epoch,
            "metrics": {"per_dataset": {dataset: {
                "best": {"threshold": 0.4, "iou": dice / 2, "dice": dice,
                         "precision": 0.1, "recall": 0.5, "boundary_f": 0.2,
                         "pred_fraction": 0.3, "true_fraction": 0.05,
                         "tiles": tiles},
                "fixed": {"threshold": 0.5, "iou": dice / 3, "dice": dice / 1.5,
                          "precision": 0.08, "recall": 0.6, "boundary_f": 0.15,
                          "pred_fraction": 0.4, "true_fraction": 0.05,
                          "tiles": tiles}}}},
        }],
    }
    (directory / f"train_{run_name}_{host}.json").write_text(json.dumps(payload))


def test_load_run_reports_finds_both_arms_and_ignores_other_folds(tmp_path):
    _write_report_stub(tmp_path, "dev", "colab", "uhcs2", 0.149)
    _write_report_stub(tmp_path, "dev-no-uhcs1", "colab", "uhcs2", 0.210,
                       excluded=["uhcs1"])
    _write_report_stub(tmp_path, "fold_MetalDam", "colab", "MetalDam", 0.4)
    # A fold whose name merely starts the same way must not be swept in.
    _write_report_stub(tmp_path, "development", "colab", "uhcs2", 0.9)

    found = train_mod.load_run_reports("dev", reports_dir=tmp_path)
    assert set(found) == {("dev", "colab"), ("dev-no-uhcs1", "colab")}, (
        f"got {sorted(found)}")


def test_compare_runs_puts_the_arms_side_by_side(tmp_path):
    _write_report_stub(tmp_path, "dev", "colab", "uhcs2", 0.149)
    _write_report_stub(tmp_path, "dev-no-uhcs1", "colab", "uhcs2", 0.210,
                       excluded=["uhcs1"])
    reports = train_mod.load_run_reports("dev", reports_dir=tmp_path)
    comparison = train_mod.compare_runs(reports, "uhcs2", row="best")

    assert set(comparison["runs"]) == {"dev (colab)", "dev-no-uhcs1 (colab)"}
    assert comparison["runs"]["dev (colab)"]["excluded"] == "-"
    assert comparison["runs"]["dev-no-uhcs1 (colab)"]["excluded"] == "uhcs1"
    assert comparison["runs"]["dev-no-uhcs1 (colab)"]["dice"] == pytest.approx(0.210)
    assert comparison["comparable"], comparison["note"]


def test_compare_runs_flags_arms_scored_on_different_tiles(tmp_path):
    """If validation differed between arms the comparison means nothing."""
    _write_report_stub(tmp_path, "dev", "colab", "uhcs2", 0.149, tiles=265)
    _write_report_stub(tmp_path, "dev-no-uhcs1", "colab", "uhcs2", 0.210,
                       tiles=200, excluded=["uhcs1"])
    reports = train_mod.load_run_reports("dev", reports_dir=tmp_path)
    comparison = train_mod.compare_runs(reports, "uhcs2", row="best")
    assert not comparison["comparable"]
    assert "NOT COMPARABLE" in comparison["note"]


def test_diff_runs_reports_what_actually_differed(tmp_path):
    _write_report_stub(tmp_path, "dev", "colab", "uhcs2", 0.149, config_hash="a",
                       hashed={"train": {"exclude_datasets": [], "lr": 0.0003}})
    _write_report_stub(tmp_path, "dev-no-uhcs1", "colab", "uhcs2", 0.21,
                       excluded=["uhcs1"], config_hash="b",
                       hashed={"train": {"exclude_datasets": ["uhcs1"],
                                         "lr": 0.0003}})
    reports = train_mod.load_run_reports("dev", reports_dir=tmp_path)
    diff = train_mod.diff_runs(reports, ("dev", "colab"), ("dev-no-uhcs1", "colab"))
    assert len(diff) == 1, f"exactly one key should differ, got {diff}"
    assert "exclude_datasets" in diff[0]


def test_diff_runs_says_so_when_a_report_predates_hashed_config(tmp_path):
    _write_report_stub(tmp_path, "dev", "colab", "uhcs2", 0.149, hashed=None)
    _write_report_stub(tmp_path, "dev-no-uhcs1", "colab", "uhcs2", 0.21,
                       excluded=["uhcs1"],
                       hashed={"train": {"exclude_datasets": ["uhcs1"]}})
    reports = train_mod.load_run_reports("dev", reports_dir=tmp_path)
    diff = train_mod.diff_runs(reports, ("dev", "colab"), ("dev-no-uhcs1", "colab"))
    assert any("hashed_config absent" in line for line in diff)
