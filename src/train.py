"""Training loop for the boundary U-Net: fold-aware, crash-safe, honestly measured.

Three things in here are not boilerplate and are the reason this file is as
long as it is.

**Validation is a MIXTURE and is never reported pooled alone.** ``fold_uhcs2``
validates on 265 uhcs2 tiles and 189 Steel1 tiles. Steel1's boundaries are
sparse, blob-like outlines that any model gets right early; uhcs2 is the
held-out microscope and the only thing the fold is actually measuring. A pooled
Dice averages the two and lets the easy half carry the hard half. So every
metric is computed per dataset AND pooled, the per-dataset breakdown is the
headline, and `best` checkpoints are selected on the HELD-OUT dataset's Dice --
not on the pooled number, and not on the loss.

**The three loss terms are logged separately.** They differ by orders of
magnitude -- on an empty prediction BCE is ~25 and Dice is ~1 -- so a falling
total says almost nothing about which term moved. Step 5 measured that; step 6
does not throw it away.

**The clDice term is on probation.** Step 5 established what clDice does to
PERTURBED GROUND TRUTH: it is nearly blind to thickness, which is why it is in
the loss. It did not establish what it does to real model output, which is soft
and thicker than the target. So each epoch logs, on a fixed sample of
validation tiles, the skeleton sums and ``mean |soft_skeleton(p) - p|``. If
that difference stays near zero across epochs, the soft skeleton is returning
its input, clDice is degenerate on real predictions, and the term is dead
weight. That finding gets RECORDED, in the report and in the notebook. It does
not get silently fixed by dropping the term or retuning its weight.

Everything is resumable because this runs on free tiers that are killed without
warning: a full checkpoint is written atomically after every epoch, and a
resume refuses to continue a run whose config hash or fold name does not match.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Callable, Optional, Sequence


class TrainError(RuntimeError):
    """Raised when a run cannot start, continue or be resumed. Never silent."""


try:
    import numpy as np
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - hosts ship both
    raise TrainError(
        "torch/numpy are not importable. Colab and Kaggle ship them; this "
        "module is meant to run on a host, never on a local machine."
    ) from exc

from src.paths import REPO_ROOT


# --------------------------------------------------------------------------
# defaults -- overridable from configs/default.yaml under ``train:``
# --------------------------------------------------------------------------
DEFAULTS = {
    "epochs": 40,
    # Measured by notebooks/05: the largest batch that fits at patch 256 with
    # the optimizer state. Never a literal here.
    "batch_size": 64,
    "val_batch_size": None,        # None -> batch_size
    "optimizer": "adamw",
    "lr": 3.0e-4,
    # The encoder is pretrained and the decoder is random. One learning rate
    # for both either moves the decoder too slowly or walks the pretrained
    # filters off before the decoder is producing a useful gradient.
    "encoder_lr_scale": 0.1,
    "weight_decay": 1.0e-4,
    "scheduler": "cosine",
    "warmup_epochs": 2,
    "min_lr_scale": 0.01,          # final LR = lr * this
    "grad_clip": 1.0,
    "amp": True,
    "seed": 0,
    # cudnn.benchmark picks the fastest conv algorithm per shape and is worth
    # 20-30%; it also makes runs non-bitwise-reproducible. Seeding still makes
    # them statistically reproducible. Set true to trade the speed for exactness.
    "deterministic": False,
    # None -> configs/dataloader.yaml, hosts.<this platform>.num_workers.
    # Never a literal and never the other host's value.
    "num_workers": None,
    "sampler": True,               # WeightedRandomSampler from fold_stats
    # Datasets to drop from a fold's TRAIN split, as {fold: [dataset, ...]}.
    # Per fold, because an exclusion is a claim about one fold's training
    # mixture and not a global setting. Validation is NEVER filtered: the whole
    # point of excluding something is to measure the effect on an unchanged
    # validation split, and a run that quietly changed both could not be
    # compared with anything.
    "exclude_datasets": {},
    # The fixed operating point every epoch is also reported at, and the
    # reference the sweep is compared against. Must fall on the sweep grid.
    "threshold": 0.5,
    # Validation is ALSO evaluated across this grid, per dataset, and the
    # best-Dice point is reported alongside the fixed one. A single fixed
    # threshold measures the model and the operating point together and
    # reports the sum as if it were the model.
    "threshold_sweep_min": 0.05,
    "threshold_sweep_max": 0.95,
    "threshold_sweep_step": 0.05,
    "boundary_tolerance_px": 2,    # for the boundary F-score
    "cldice_probe_tiles": 24,      # fixed val tiles for the clDice diagnostic
    "log_every": 20,               # train steps between progress updates
    "checkpoint_subdir": "checkpoints",
    "log_subdir": "logs",
    # last.pt must be resumable, so it carries optimizer + scaler + RNG (~3x
    # the model). best.pt is a deliverable for step 7 inference, not a resume
    # point, so by default it carries weights only and stays ~93 MB.
    "best_includes_optimizer": False,
}

#: ``train:`` keys that change what the model becomes. A resume whose hash of
#: these differs is refused. Operational keys (num_workers, amp, log_every,
#: subdirs, probe size) are deliberately excluded: changing them mid-run after
#: an OOM is legitimate, and they are printed on resume so a change is visible.
HASHED_TRAIN_KEYS = (
    "epochs", "batch_size", "optimizer", "lr", "encoder_lr_scale",
    "weight_decay", "scheduler", "warmup_epochs", "min_lr_scale", "grad_clip",
    "seed", "deterministic", "sampler", "threshold", "boundary_tolerance_px",
    # In the hash because they change WHICH CHECKPOINT is selected: best.pt is
    # chosen on best-threshold Dice, so a run started under a different sweep
    # grid produced a different `best`, and resuming into it would leave two
    # incompatible selection criteria in one curve.
    "threshold_sweep_min", "threshold_sweep_max", "threshold_sweep_step",
    # In the hash so an excluded-set run can NEVER resume from a full-set
    # checkpoint, or vice versa: the two saw different data and averaging their
    # curves together would be meaningless. What is hashed is the exclusion
    # RESOLVED FOR THIS FOLD (see Trainer.__init__), not the whole mapping --
    # changing fold_MetalDam's exclusion must not invalidate a dev checkpoint.
    "exclude_datasets",
)


def load_config(config_path: Optional[Path] = None) -> dict:
    """Merge ``train:`` from configs/default.yaml over DEFAULTS."""
    from src import paths as paths_mod

    cfg = paths_mod.load_config(config_path)
    settings = dict(DEFAULTS)
    section = cfg.get("train") or {}
    if not isinstance(section, dict):
        raise TrainError(
            f"configs/default.yaml: train must be a mapping, got "
            f"{type(section).__name__}")
    unknown = set(section) - set(DEFAULTS)
    if unknown:
        raise TrainError(
            f"configs/default.yaml: unknown train keys {sorted(unknown)}; "
            f"known keys are {sorted(DEFAULTS)}")
    for key, value in section.items():
        if value is not None:
            settings[key] = value
    if int(settings["epochs"]) < 1:
        raise TrainError(f"train.epochs must be >= 1, got {settings['epochs']}")
    if int(settings["batch_size"]) < 1:
        raise TrainError(
            f"train.batch_size must be >= 1, got {settings['batch_size']}")
    if float(settings["lr"]) <= 0:
        raise TrainError(f"train.lr must be > 0, got {settings['lr']}")
    if str(settings["optimizer"]).lower() != "adamw":
        raise TrainError(
            f"train.optimizer is {settings['optimizer']!r}; only 'adamw' is "
            "implemented. Add the optimizer here deliberately.")
    if str(settings["scheduler"]).lower() != "cosine":
        raise TrainError(
            f"train.scheduler is {settings['scheduler']!r}; only 'cosine' "
            "(with linear warmup) is implemented.")
    excludes = settings["exclude_datasets"] or {}
    if not isinstance(excludes, dict):
        raise TrainError(
            f"train.exclude_datasets must be a mapping of fold -> [dataset, "
            f"...], got {type(excludes).__name__}. A bare list would apply to "
            "every fold, which is never what an exclusion means.")
    normalised = {}
    for fold_name, names in excludes.items():
        if isinstance(names, str):
            raise TrainError(
                f"train.exclude_datasets[{fold_name!r}] is the string "
                f"{names!r}; it must be a LIST of dataset names, or a "
                "one-character dataset would be excluded letter by letter.")
        # Sorted and de-duplicated so the config hash does not depend on the
        # order someone happened to type them in.
        normalised[str(fold_name)] = sorted({str(n) for n in (names or [])})
    settings["exclude_datasets"] = normalised

    if int(settings["warmup_epochs"]) >= int(settings["epochs"]):
        raise TrainError(
            f"train.warmup_epochs ({settings['warmup_epochs']}) must be less "
            f"than train.epochs ({settings['epochs']}); otherwise the cosine "
            "phase never runs.")
    return settings


def resolve_exclusions(fold: str, settings: dict, fold_entry: dict) -> list:
    """The datasets to drop from THIS fold's train split, validated.

    A name that is not in the fold's training mixture is an error, not a no-op.
    The failure this prevents is a typo or a stale fold name silently training
    on everything while the run header, the report and the config hash all
    claim an exclusion was applied -- which would be worse than not having the
    feature, because the resulting comparison would look valid.
    """
    requested = list((settings.get("exclude_datasets") or {}).get(fold, []))
    if not requested:
        return []

    available = list(fold_entry.get("train_datasets") or [])
    if not available:
        raise TrainError(
            f"configs/fold_stats.yaml has no train_datasets for fold {fold!r}, "
            "so an exclusion cannot be checked against anything. Re-run step 3.")
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise TrainError(
            f"train.exclude_datasets[{fold!r}] names {unknown}, which "
            f"{'is' if len(unknown) == 1 else 'are'} not in that fold's "
            f"training mixture {sorted(available)}. Excluding something that "
            "was never there would leave the run header claiming an exclusion "
            "that changed nothing.")
    keep = [d for d in available if d not in requested]
    if not keep:
        raise TrainError(
            f"train.exclude_datasets[{fold!r}] excludes every training "
            f"dataset {sorted(available)}; there would be nothing to train on.")
    return sorted(set(requested))


# --------------------------------------------------------------------------
# seeding
# --------------------------------------------------------------------------
def seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker its own reproducible numpy/random stream.

    Module level and argument-free by design: it has to survive being sent to a
    worker process, so it cannot close over anything.
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed: int, deterministic: bool = False) -> dict:
    """Seed every generator that touches this run, and say what was set."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    return {
        "seed": seed,
        "deterministic": bool(deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "note": ("bitwise reproducible" if deterministic else
                 "statistically reproducible; cudnn.benchmark picks algorithms "
                 "by timing, so bitwise equality across runs is not claimed"),
    }


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (torch.cuda.get_rng_state_all()
                 if torch.cuda.is_available() else None),
    }


def load_rng_state(state: Optional[dict]) -> bool:
    if not state:
        return False
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except Exception:
            # A different GPU count than the run that saved it. Not fatal, but
            # not silent either.
            return False
    return True


# --------------------------------------------------------------------------
# num_workers -- from THIS host's measured entry, never a literal
# --------------------------------------------------------------------------
def dataloader_settings(platform: str,
                        configs_dir: Optional[Path] = None) -> dict:
    """``hosts.<platform>`` from configs/dataloader.yaml, written by step 4.

    Colab has 2-4 CPUs and Kaggle has 2, and step 4 measured tiles/s per worker
    count on each. Using the other host's number is how a run ends up
    I/O-bound on hardware that was never measured, so a missing entry raises
    instead of falling back to anything.
    """
    import yaml

    path = Path(configs_dir or (REPO_ROOT / "configs")) / "dataloader.yaml"
    if not path.is_file():
        raise TrainError(
            f"{path} does not exist. Run notebooks/04_dataset.ipynb on this "
            "host: num_workers is measured there, per platform.")
    doc = yaml.safe_load(path.read_text()) or {}
    hosts = doc.get("hosts") or {}
    if platform not in hosts:
        raise TrainError(
            f"configs/dataloader.yaml has no entry for platform "
            f"{platform!r}; it has {sorted(hosts)}. Run "
            "notebooks/04_dataset.ipynb HERE -- another host's worker count "
            "is not a substitute for this one's.")
    entry = dict(hosts[platform])
    if entry.get("num_workers") is None:
        raise TrainError(
            f"configs/dataloader.yaml hosts.{platform}.num_workers is null; "
            "re-run notebooks/04_dataset.ipynb on this host.")
    entry["source"] = f"configs/dataloader.yaml hosts.{platform}"
    return entry


# --------------------------------------------------------------------------
# metrics -- never pixel accuracy
# --------------------------------------------------------------------------
def _safe_div(numerator: float, denominator: float) -> float:
    """0/0 is reported as 0.0, not NaN, and never silently as 1.0."""
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def euclidean_disk(radius: int) -> np.ndarray:
    """The exact set of offsets within Euclidean distance ``radius``.

    Dilating with this is identical to thresholding a Euclidean distance
    transform at ``radius``, and it is what lets the boundary F-score be
    evaluated at every threshold in one pass -- see :class:`MetricAccumulator`.
    """
    r = int(radius)
    if r < 0:
        raise TrainError(f"tolerance must be >= 0, got {radius}")
    if r == 0:
        return np.ones((1, 1), dtype=np.uint8)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return ((yy ** 2 + xx ** 2) <= r * r).astype(np.uint8)


def threshold_grid(settings: dict) -> np.ndarray:
    """The sweep grid, with ``train.threshold`` guaranteed to be in it.

    The fixed threshold has to be a grid point so that the "at 0.5" row and the
    "at the best threshold" row are computed from the same counts and are
    genuinely comparable.
    """
    lo = float(settings["threshold_sweep_min"])
    hi = float(settings["threshold_sweep_max"])
    step = float(settings["threshold_sweep_step"])
    if not (0.0 < lo < hi < 1.0):
        raise TrainError(
            f"threshold sweep must satisfy 0 < min < max < 1, got {lo}..{hi}")
    if step <= 0 or step > (hi - lo):
        raise TrainError(
            f"train.threshold_sweep_step is {step}, which cannot span "
            f"{lo}..{hi}")
    n = int(round((hi - lo) / step)) + 1
    grid = np.round(np.linspace(lo, hi, n), 6)
    fixed = round(float(settings["threshold"]), 6)
    if not (0.0 < fixed < 1.0):
        raise TrainError(f"train.threshold must be in (0, 1), got {fixed}")
    return np.unique(np.append(grid, fixed))


def snap_thresholds(thresholds: np.ndarray, dtype) -> np.ndarray:
    """The grid as the DATA sees it, so "exactly on the threshold" is decidable.

    None of 0.05, 0.15, 0.95 ... is exactly representable in binary, and the
    probabilities being compared against them are float32 out of a sigmoid. The
    two do not round the same way, and -- worse -- they do not round the same
    *direction* per constant: ``float32(0.05)`` lands just above the float64
    0.05 and counts as clearing it, while ``float32(0.95)`` lands just below
    the float64 0.95 and does not. So whether a pixel sitting exactly on its
    threshold is counted depended on which way that particular decimal
    constant happened to round, which is not a rule anyone can reason about.

    Rounding the grid to the precision of the values fixes the rule at "a value
    equal to the threshold AT THE DATA'S PRECISION clears it", uniformly at
    every grid point. The float64 grid is kept if rounding would collapse two
    grid points into one, since a non-increasing grid would break
    ``searchsorted`` -- a far worse failure than the ambiguity being fixed.
    """
    grid = np.asarray(thresholds, dtype=np.float64)
    if not np.issubdtype(np.dtype(dtype), np.floating):
        return grid
    if np.dtype(dtype) == np.float64:
        return grid
    snapped = grid.astype(dtype).astype(np.float64)
    if grid.size > 1 and not np.all(np.diff(snapped) > 0):
        return grid
    return snapped


def ge_counts(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """``#{v >= t}`` for every ``t``, in one pass rather than one pass each.

    A sweep over 19 thresholds on 454 validation tiles is 19 comparisons over
    65,536 pixels per tile if written naively. ``searchsorted`` gives, for each
    value, how many thresholds it clears; the reverse cumulative sum of those
    counts is the answer for all thresholds at once. Exact, not approximate:
    it returns what ``(values >= t).sum()`` returns, for every ``t``, including
    at values sitting exactly on a threshold -- see :func:`snap_thresholds` for
    what "exactly" has to mean when the grid is decimal and the data is
    float32.
    """
    values = np.asarray(values).ravel()
    grid = snap_thresholds(thresholds, values.dtype)
    if values.size == 0:
        return np.zeros(grid.size, dtype=np.int64)
    # side="right" so that `cleared` is the number of thresholds <= v, which is
    # the count that "v >= t" asks for. v clears thresholds[k] exactly when
    # cleared >= k + 1, hence the tail sums below being read from index 1.
    cleared = np.searchsorted(grid, values.astype(np.float64), side="right")
    counts = np.bincount(cleared, minlength=grid.size + 1)
    tail = np.cumsum(counts[::-1])[::-1]
    return tail[1:].astype(np.int64)


def metrics_from_counts(counts: dict) -> dict:
    """IoU, Dice, Precision, Recall, boundary-F. No pixel accuracy, ever.

    Pixel accuracy is excluded on purpose and not as an oversight: boundaries
    are 5-15% of pixels, so predicting none of them scores 85-95% and that
    number is worse than useless -- it is actively misleading.
    """
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    b_precision = _safe_div(counts["bf_hits_p"], counts["bf_n_pred"])
    b_recall = _safe_div(counts["bf_hits_r"], counts["bf_n_true"])
    return {
        "iou": _safe_div(tp, tp + fp + fn),
        "dice": _safe_div(2 * tp, 2 * tp + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "boundary_precision": b_precision,
        "boundary_recall": b_recall,
        "boundary_f": _safe_div(2 * b_precision * b_recall,
                                b_precision + b_recall),
        "tiles": counts["tiles"],
        "true_fraction": _safe_div(counts["true_px"], counts["total_px"]),
        "pred_fraction": _safe_div(counts["pred_px"], counts["total_px"]),
        "prob_min": counts["prob_min"],
        "prob_max": counts["prob_max"],
        "prob_mean": _safe_div(counts["prob_sum"], counts["total_px"]),
    }


class MetricAccumulator:
    """Counts at EVERY threshold, kept per dataset and pooled.

    A single fixed threshold measures the model and the threshold together and
    reports the sum as if it were the model. On uhcs2 at 0.5 the model paints
    0.312 of the tile against a true 0.054 and precision reads 0.087 -- most of
    which is the threshold. The operating point is a free parameter that step 7
    will choose anyway, so it is swept here instead of guessed: every metric is
    reported twice, once at ``train.threshold`` and once at the threshold that
    maximised Dice, with that threshold named.

    Both are needed. The fixed row is what the loss is actually optimising and
    is comparable across epochs and folds without qualification; the best row
    is what the model can do once its operating point is set, which is the
    thing step 7 inherits.

    **The chosen threshold is itself a measurement.** If the held-out dataset
    wants 0.85 and Steel1 wants 0.35, that gap IS the domain shift, stated in
    the units of the decision the downstream watershed has to make -- and a
    single global threshold will not serve both.

    Counts are accumulated over pixels and the metrics computed once at the
    end, rather than averaging per-tile metrics: per-tile averaging gives a
    nearly empty tile the same weight as a dense one.
    """

    POOLED = "__pooled__"

    def __init__(self, thresholds, fixed_threshold: float = 0.5,
                 tolerance_px: int = 2):
        self.thresholds = np.asarray(thresholds, dtype=np.float64)
        if self.thresholds.ndim != 1 or self.thresholds.size < 2:
            raise TrainError(
                f"the threshold sweep needs at least 2 points, got "
                f"{self.thresholds}")
        self.fixed_threshold = float(fixed_threshold)
        matches = np.nonzero(np.isclose(self.thresholds,
                                        self.fixed_threshold))[0]
        if matches.size != 1:
            raise TrainError(
                f"the fixed threshold {self.fixed_threshold} is not a unique "
                f"point of the sweep grid {self.thresholds}; the two rows "
                "would not be comparable.")
        self.fixed_index = int(matches[0])
        self.tolerance_px = int(tolerance_px)
        self._disk = euclidean_disk(self.tolerance_px)
        self.counts = {}

    def _slot(self, key: str) -> dict:
        k = self.thresholds.size
        return self.counts.setdefault(key, {
            "tiles": 0, "total_px": 0, "true_px": 0, "prob_sum": 0.0,
            "prob_min": float("inf"), "prob_max": float("-inf"),
            "tp": np.zeros(k, dtype=np.int64),
            "n_pred": np.zeros(k, dtype=np.int64),
            "bf_hits_p": np.zeros(k, dtype=np.int64),
            "bf_hits_r": np.zeros(k, dtype=np.int64),
        })

    def update(self, probs, targets, dataset_names: Sequence[str]) -> None:
        """One batch. ``probs`` are probabilities, ``targets`` are 0/1."""
        import cv2

        probs = np.asarray(probs, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        if probs.shape != targets.shape:
            raise TrainError(
                f"probs {probs.shape} and targets {targets.shape} disagree")
        if len(dataset_names) != probs.shape[0]:
            raise TrainError(
                f"{len(dataset_names)} dataset names for {probs.shape[0]} tiles")

        height, width = probs.shape[-2], probs.shape[-1]
        for i, name in enumerate(dataset_names):
            prob = np.ascontiguousarray(probs[i].reshape(height, width))
            true = targets[i].reshape(height, width) > 0.5

            # Threshold-independent, computed once per tile:
            #   true_dilated -- a predicted pixel here is within tolerance of a
            #     true boundary, so counting predictions inside it gives the
            #     boundary-precision hits at every threshold at once;
            #   prob_dilated -- the max probability within the tolerance disk,
            #     so a true pixel is within tolerance of SOME prediction at
            #     threshold t exactly when prob_dilated >= t.
            true_dilated = cv2.dilate(true.astype(np.uint8), self._disk) > 0
            prob_dilated = cv2.dilate(prob, self._disk)

            tp = ge_counts(prob[true], self.thresholds)
            n_pred = ge_counts(prob, self.thresholds)
            hits_p = ge_counts(prob[true_dilated], self.thresholds)
            hits_r = ge_counts(prob_dilated[true], self.thresholds)

            for key in (name, self.POOLED):
                slot = self._slot(key)
                slot["tiles"] += 1
                slot["total_px"] += int(prob.size)
                slot["true_px"] += int(true.sum())
                slot["prob_sum"] += float(prob.sum())
                slot["prob_min"] = min(slot["prob_min"], float(prob.min()))
                slot["prob_max"] = max(slot["prob_max"], float(prob.max()))
                slot["tp"] += tp
                slot["n_pred"] += n_pred
                slot["bf_hits_p"] += hits_p
                slot["bf_hits_r"] += hits_r

    def _counts_at(self, slot: dict, index: int) -> dict:
        tp = int(slot["tp"][index])
        n_pred = int(slot["n_pred"][index])
        return {
            "tp": tp,
            "fp": n_pred - tp,
            "fn": int(slot["true_px"]) - tp,
            "tiles": slot["tiles"],
            "total_px": slot["total_px"],
            "true_px": slot["true_px"],
            "pred_px": n_pred,
            "prob_sum": slot["prob_sum"],
            "prob_min": slot["prob_min"],
            "prob_max": slot["prob_max"],
            "bf_hits_p": int(slot["bf_hits_p"][index]),
            "bf_n_pred": n_pred,
            "bf_hits_r": int(slot["bf_hits_r"][index]),
            "bf_n_true": int(slot["true_px"]),
        }

    def _sweep(self, slot: dict) -> list:
        return [dict(metrics_from_counts(self._counts_at(slot, k)),
                     threshold=float(self.thresholds[k]))
                for k in range(self.thresholds.size)]

    def _summarise(self, slot: dict) -> dict:
        sweep = self._sweep(slot)
        dice = np.array([row["dice"] for row in sweep])
        # Ties go to the HIGHER threshold. This model is a prior for a
        # watershed, where a false boundary splits a region permanently and a
        # missed one can still be recovered downstream, so when two operating
        # points score the same, take the conservative one.
        best_index = int(dice.size - 1 - np.argmax(dice[::-1]))
        fixed = dict(sweep[self.fixed_index])
        best = dict(sweep[best_index])
        return {
            "fixed": fixed,
            "best": best,
            "fixed_threshold": float(self.thresholds[self.fixed_index]),
            "best_threshold": float(self.thresholds[best_index]),
            "sweep": sweep,
        }

    def result(self) -> dict:
        """Per dataset and pooled, each with ``fixed``, ``best`` and ``sweep``.

        The per-dataset breakdown is the headline. The pooled entry is a
        footnote and is labelled as one everywhere it is printed.
        """
        if not self.counts:
            raise TrainError("no batches were accumulated; validation ran on "
                             "an empty loader.")
        return {
            "thresholds": [float(t) for t in self.thresholds],
            "fixed_threshold": float(self.thresholds[self.fixed_index]),
            "pooled": self._summarise(self.counts[self.POOLED]),
            "per_dataset": {name: self._summarise(slot)
                            for name, slot in sorted(self.counts.items())
                            if name != self.POOLED},
        }


# --------------------------------------------------------------------------
# schedule
# --------------------------------------------------------------------------
class WarmupCosine:
    """Linear warmup then cosine decay, stepped per optimizer step.

    Written out rather than wrapped around ``torch.optim.lr_scheduler`` for one
    reason: the optimizer gains a parameter group at the unfreeze epoch, and a
    stock scheduler's state is tied to the group list it was built with. This
    one holds only a step count and reads ``base_lr`` off each group, so adding
    a group mid-run cannot desynchronise it.
    """

    def __init__(self, total_steps: int, warmup_steps: int,
                 min_lr_scale: float = 0.01):
        if total_steps < 1:
            raise TrainError(f"total_steps must be >= 1, got {total_steps}")
        if warmup_steps >= total_steps:
            raise TrainError(
                f"warmup_steps ({warmup_steps}) must be < total_steps "
                f"({total_steps})")
        self.total_steps = int(total_steps)
        self.warmup_steps = max(0, int(warmup_steps))
        self.min_lr_scale = float(min_lr_scale)
        self.step_count = 0

    def factor(self, step: Optional[int] = None) -> float:
        step = self.step_count if step is None else int(step)
        if self.warmup_steps and step < self.warmup_steps:
            return (step + 1) / float(self.warmup_steps)
        progress = ((step - self.warmup_steps)
                    / max(1, self.total_steps - self.warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + float(np.cos(np.pi * progress)))
        return self.min_lr_scale + (1.0 - self.min_lr_scale) * cosine

    def apply(self, optimizer) -> float:
        factor = self.factor()
        for group in optimizer.param_groups:
            if "base_lr" not in group:
                raise TrainError(
                    "an optimizer param group has no base_lr; groups must be "
                    "created through build_optimizer or add_encoder_group.")
            group["lr"] = group["base_lr"] * factor
        return float(optimizer.param_groups[0]["lr"])

    def step(self) -> None:
        self.step_count += 1

    def state_dict(self) -> dict:
        return {"step_count": self.step_count, "total_steps": self.total_steps,
                "warmup_steps": self.warmup_steps,
                "min_lr_scale": self.min_lr_scale}

    def load_state_dict(self, state: dict) -> None:
        self.step_count = int(state["step_count"])
        self.total_steps = int(state["total_steps"])
        self.warmup_steps = int(state["warmup_steps"])
        self.min_lr_scale = float(state["min_lr_scale"])


# --------------------------------------------------------------------------
# optimizer
# --------------------------------------------------------------------------
def _encoder_param_ids(model: "nn.Module") -> set:
    return {id(p) for p in model.encoder.parameters()}


def build_optimizer(model: "nn.Module", settings: dict):
    """AdamW over exactly the parameters that are trainable RIGHT NOW.

    Two groups when the encoder is trainable: the pretrained encoder at
    ``lr * encoder_lr_scale`` and the random decoder at ``lr``. One group while
    the encoder is frozen -- which is why :func:`sync_optimizer_groups` exists.
    """
    lr = float(settings["lr"])
    scale = float(settings["encoder_lr_scale"])
    wd = float(settings["weight_decay"])
    encoder_ids = _encoder_param_ids(model)

    encoder, decoder = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (encoder if id(param) in encoder_ids else decoder).append(param)
    if not decoder and not encoder:
        raise TrainError("no trainable parameters; nothing to optimize.")

    groups = []
    if decoder:
        groups.append({"params": decoder, "name": "decoder",
                       "lr": lr, "base_lr": lr, "weight_decay": wd})
    if encoder:
        groups.append({"params": encoder, "name": "encoder",
                       "lr": lr * scale, "base_lr": lr * scale,
                       "weight_decay": wd})
    return torch.optim.AdamW(groups, lr=lr, weight_decay=wd)


def sync_optimizer_groups(optimizer, model: "nn.Module",
                          settings: dict) -> dict:
    """Give newly trainable parameters a param group. THIS IS LOAD-BEARING.

    ``apply_freeze_schedule`` flips ``requires_grad`` back on at the unfreeze
    epoch, but an optimizer built while the encoder was frozen has no group
    holding those tensors. Their gradients would be computed, and then
    discarded, every step for the rest of the run -- no error, no warning, just
    an encoder that never moves. This adds the missing group, preserving the
    decoder's Adam moments (which rebuilding the optimizer would throw away).
    """
    known = {id(p) for group in optimizer.param_groups for p in group["params"]}
    missing = [p for p in model.parameters()
               if p.requires_grad and id(p) not in known]
    before = sum(len(g["params"]) for g in optimizer.param_groups)
    if not missing:
        return {"added_tensors": 0, "added_params": 0, "groups": before,
                "changed": False}

    encoder_ids = _encoder_param_ids(model)
    is_encoder = all(id(p) in encoder_ids for p in missing)
    lr = float(settings["lr"]) * (float(settings["encoder_lr_scale"])
                                  if is_encoder else 1.0)
    optimizer.add_param_group({
        "params": missing,
        "name": "encoder" if is_encoder else "late",
        "lr": lr, "base_lr": lr,
        "weight_decay": float(settings["weight_decay"]),
    })
    return {
        "added_tensors": len(missing),
        "added_params": int(sum(p.numel() for p in missing)),
        "groups": len(optimizer.param_groups),
        "changed": True,
    }


def optimizer_param_count(optimizer) -> int:
    return int(sum(p.numel() for g in optimizer.param_groups
                   for p in g["params"]))


# --------------------------------------------------------------------------
# AMP helpers -- API moved between torch versions
# --------------------------------------------------------------------------
def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=bool(enabled))
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=bool(enabled))


def autocast(device_type: str, enabled: bool):
    return torch.autocast(device_type=device_type, dtype=torch.float16,
                          enabled=bool(enabled))


# --------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------
def config_hash(model_settings: dict, loss_settings: dict,
                train_settings: dict, dataset_settings: dict) -> tuple:
    """(hash, the dict that was hashed) for everything that changes the model.

    Operational keys are excluded (see HASHED_TRAIN_KEYS) so that resuming
    after an OOM with fewer workers or AMP off is allowed; anything that
    changes what the network becomes is in, including the augmentation
    settings, because resuming a run under different augmentation is a
    different experiment wearing the same name.
    """
    payload = {
        "model": {k: model_settings[k] for k in sorted(model_settings)},
        "loss": {k: loss_settings[k] for k in sorted(loss_settings)},
        "train": {k: train_settings[k] for k in sorted(HASHED_TRAIN_KEYS)},
        "dataset": {k: dataset_settings[k] for k in sorted(dataset_settings)
                    if not k.startswith("cache") and k != "persist_cache"},
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16], payload


def atomic_save(state: dict, path: Path) -> Path:
    """Write, flush, fsync, THEN rename. A killed session cannot half-write it.

    ``torch.save`` straight to the destination is how a session that dies
    mid-write leaves a truncated file that looks like a checkpoint and fails to
    load hours later. The temporary file is in the same directory so the rename
    stays on one filesystem and therefore atomic.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        torch.save(state, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def load_checkpoint(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise TrainError(f"no checkpoint at {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:                     # torch < 2.0 has no weights_only
        return torch.load(path, map_location="cpu")


def state_dict_checksum(state_dict: dict) -> str:
    """A deterministic checksum of a model state dict's actual VALUES.

    Exists to prove weights did not move across an operation that is only
    supposed to touch metadata -- migrating a checkpoint's recorded config
    hash, specifically. Sorted by key, so dict ordering cannot change the
    result; each tensor's shape is hashed alongside its bytes, so a reshape or
    a dropped/renamed parameter changes the checksum even where the raw bytes
    happen to coincide.
    """
    hasher = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key]
        hasher.update(key.encode())
        hasher.update(str(tuple(tensor.shape)).encode())
        hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def migrate_config_hash(path: Path, new_hash: str, new_hashed_config: dict,
                        expected_fold: str) -> dict:
    """Re-stamp a checkpoint's recorded config hash WITHOUT touching weights.

    Used only once a human has explicitly accepted a config change via
    ``Trainer.maybe_resume(allow_config_change=True)``: the training data and
    model this checkpoint holds have not changed, but its stated provenance
    now disagrees with the config this run continues under. Left unmigrated,
    every later check compares the checkpoint's OLD hash against the run's NEW
    one and reports a "mismatch" that is not actually a problem -- which is
    what once led a session to "fix" it by hand with a raw
    ``save(is_best=True)`` call, silently overwriting best.pt's real
    best-epoch weights with whatever the last epoch happened to be. This
    function is the correct way to do what that call was reaching for:
    relabel, never reweight.

    Only ``config_hash`` and ``hashed_config`` are replaced; every other key
    -- above all ``model`` -- is carried through byte-for-byte. Verified by
    re-reading the file afterward and comparing the model state dict's
    checksum against the one computed before the rewrite; a mismatch raises
    rather than leaving a file whose weights cannot be trusted in place.
    """
    path = Path(path)
    if not path.is_file():
        raise TrainError(f"cannot migrate a hash: no checkpoint at {path}")
    state = load_checkpoint(path)
    if state.get("fold") != str(expected_fold):
        raise TrainError(
            f"{path} belongs to fold {state.get('fold')!r}, not "
            f"{expected_fold!r}; refusing to migrate a checkpoint that is not "
            "this run's own.")

    old_hash = state.get("config_hash")
    if old_hash == new_hash:
        return {"path": str(path), "migrated": False,
                "reason": "already at the new hash", "from_hash": old_hash,
                "to_hash": new_hash, "state": state}

    before_checksum = state_dict_checksum(state["model"])
    state["config_hash"] = new_hash
    state["hashed_config"] = new_hashed_config
    atomic_save(state, path)

    verify = load_checkpoint(path)
    after_checksum = state_dict_checksum(verify["model"])
    if after_checksum != before_checksum:
        raise TrainError(
            f"hash migration of {path} changed the model weights' checksum "
            f"({before_checksum[:12]} -> {after_checksum[:12]}). That must "
            "never happen for a metadata-only rewrite; treat this file as "
            "corrupted and restore it from a backup rather than trusting it.")
    if verify.get("config_hash") != new_hash:
        raise TrainError(
            f"hash migration of {path} did not take: recorded hash is still "
            f"{verify.get('config_hash')} after the rewrite.")

    return {"path": str(path), "migrated": True, "from_hash": old_hash,
            "to_hash": new_hash, "weight_checksum": after_checksum,
            "state": verify}


# --------------------------------------------------------------------------
# the trainer
# --------------------------------------------------------------------------
class Trainer:
    """One fold, end to end, resumable.

    Usage from a notebook::

        trainer = Trainer(fold="dev", resolved=PATHS)
        trainer.setup(progress=tqdm)
        status = trainer.maybe_resume()
        history = trainer.fit(on_epoch_end=print_table)
    """

    def __init__(self, fold: str, resolved: Optional[dict] = None,
                 settings: Optional[dict] = None,
                 model_settings: Optional[dict] = None,
                 loss_settings: Optional[dict] = None,
                 dataset_settings: Optional[dict] = None,
                 configs_dir: Optional[Path] = None,
                 run_name: Optional[str] = None):
        from src import dataset as ds
        from src import losses as losses_mod
        from src import model as model_mod
        from src import paths as paths_mod

        self._ds = ds
        self._losses = losses_mod
        self._model_mod = model_mod

        self.fold = str(fold)
        self.resolved = resolved or paths_mod.resolve_paths()
        self.settings = settings or load_config()
        self.model_settings = model_settings or model_mod.load_config()
        self.loss_settings = loss_settings or losses_mod.load_config()
        self.dataset_settings = dataset_settings or ds.load_config()
        self.configs_dir = Path(configs_dir or (REPO_ROOT / "configs"))

        self.fold_stats = ds.load_fold_stats(self.configs_dir)
        if self.fold not in self.fold_stats["folds"]:
            raise TrainError(
                f"fold {self.fold!r} is not in configs/fold_stats.yaml; "
                f"available: {sorted(self.fold_stats['folds'])}")
        self.fold_entry = self.fold_stats["folds"][self.fold]
        self.held_out = self.fold_entry.get("held_out")

        self.platform = str(self.resolved["platform"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Datasets dropped from THIS fold's train split. Resolved once, here,
        # so that the run name, the config hash, the header, the report and the
        # loaders all describe the same experiment.
        self.excluded = resolve_exclusions(self.fold, self.settings,
                                           self.fold_entry)
        # A distinct run name, so the two arms of an exclusion experiment keep
        # separate checkpoints, logs and reports instead of overwriting each
        # other. Without this the comparison the exclusion exists to enable
        # could not be made without deleting one arm first.
        self.run_name = run_name or (
            f"{self.fold}-no-{'-'.join(self.excluded)}" if self.excluded
            else self.fold)

        # The hash carries the exclusion RESOLVED FOR THIS FOLD rather than the
        # whole mapping: adding an exclusion for another fold must not
        # invalidate this fold's checkpoints.
        hashed_train = dict(self.settings)
        hashed_train["exclude_datasets"] = list(self.excluded)
        self.hash, self.hashed_config = config_hash(
            self.model_settings, self.loss_settings, hashed_train,
            self.dataset_settings)

        checkpoint_dir = (Path(self.resolved["persistent_dir"])
                          / self.settings["checkpoint_subdir"] / self.run_name)
        self.checkpoint_dir = checkpoint_dir
        self.last_path = checkpoint_dir / "last.pt"
        self.best_path = checkpoint_dir / "best.pt"
        self.log_dir = (Path(self.resolved["persistent_dir"])
                        / self.settings["log_subdir"] / self.run_name)

        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.writer = None
        self.train_ds = None
        self.val_ds = None
        self.train_loader = None
        self.val_loader = None
        self.probe_loader = None
        self.history = []
        self.start_epoch = 0
        self.best = {"metric": float("-inf"), "epoch": None, "key": None,
                     "threshold": None,
                     "criterion": "best-threshold Dice on the held-out dataset"}
        self.seed_report = None
        self.num_workers = None
        self.sources = {}
        self.cache_report = {}
        self.freeze_transitions = []

    # -- setup ------------------------------------------------------------
    def resolve_num_workers(self) -> int:
        override = self.settings.get("num_workers")
        if override is not None:
            self.sources["num_workers"] = (
                f"configs/default.yaml train.num_workers (explicit override "
                f"of this host's measured value)")
            return int(override)
        entry = dataloader_settings(self.platform, self.configs_dir)
        measured = entry.get("measured_utc")
        self.sources["num_workers"] = (
            f"{entry['source']}.num_workers"
            + (f", measured {measured}" if measured else ", NOT YET MEASURED "
               "on this host -- run notebooks/04 here"))
        return int(entry["num_workers"])

    def setup(self, progress: Optional[Callable] = None,
              tensorboard: bool = True) -> dict:
        """Datasets, loaders, model, loss, optimizer. Nothing trains yet."""
        ds = self._ds
        self.seed_report = set_seed(self.settings["seed"],
                                    self.settings["deterministic"])

        reports_dir = Path(self.resolved["reports_dir"])
        manifest = (reports_dir / self.dataset_settings["manifest_subdir"]
                    / f"{self.fold}.csv")
        crops = ds.load_crops(reports_dir)
        roots = {"data_root": Path(self.resolved["data_root"]),
                 "gt_root": Path(self.resolved["gt_boundaries_root"])}

        # The exclusion is applied HERE, to the train split only, by naming the
        # datasets to keep. TileDataset.from_manifest filters rows through
        # load_manifest, so an excluded dataset's tiles are never indexed at
        # all -- they are not loaded and then skipped.
        keep = None
        if self.excluded:
            keep = [d for d in self.fold_entry["train_datasets"]
                    if d not in self.excluded]
        self.train_ds = ds.TileDataset.from_manifest(
            manifest, "train", settings=self.dataset_settings, crops=crops,
            roots=roots, datasets=keep)
        # Validation is NOT filtered, deliberately and load-bearingly: the
        # excluded and full-set arms have to be scored on exactly the same
        # tiles or the comparison measures two changes at once.
        self.val_ds = ds.TileDataset.from_manifest(
            manifest, "val", settings=self.dataset_settings, crops=crops,
            roots=roots)

        if self.excluded:
            # Verified against the rows that were actually indexed, not assumed
            # from the argument passed in.
            present = sorted({r["dataset"] for r in self.train_ds.rows})
            leaked = sorted(set(present) & set(self.excluded))
            if leaked:
                raise TrainError(
                    f"{leaked} was excluded from fold {self.fold!r} but its "
                    "tiles are still in the train split; the manifest filter "
                    "did not take effect.")
            in_val = sorted({r["dataset"] for r in self.val_ds.rows}
                            & set(self.excluded))
            if in_val:
                raise TrainError(
                    f"{in_val} is excluded from training but also appears in "
                    f"the VALIDATION split of fold {self.fold!r}. Excluding a "
                    "dataset that is validated on would train on nothing and "
                    "score on it anyway; that is not an experiment, it is a "
                    "mistake.")
        if self.val_ds.augment:
            raise TrainError(
                "the validation dataset came back with augmentation enabled; "
                "a validation number that changes between epochs for reasons "
                "other than the model is not a measurement.")

        # Warm the persistent tile cache. A cold build is ~1,000 Drive reads;
        # a resume must not pay that again, which is why it is reported.
        cache_dir = (Path(self.resolved["persistent_dir"])
                     / self.dataset_settings["cache_subdir"])
        for name, split in (("train", self.train_ds), ("val", self.val_ds)):
            self.cache_report[name] = split.ensure_cache(
                cache_dir, name=f"{self.fold}-{name}",
                progress=(lambda seq, d=name: progress(seq, desc=f"decoding {d}"))
                if progress else None)

        self.num_workers = self.resolve_num_workers()
        self.sources["batch_size"] = "configs/default.yaml train.batch_size"
        self.sources["pos_weight"] = (
            f"configs/fold_stats.yaml folds.{self.fold}.pos_weight")
        self.sources["sampler_weights"] = (
            f"configs/fold_stats.yaml folds.{self.fold}.sampling.weights")

        self._build_loaders()

        self.model = self._model_mod.build_model(
            settings=self.model_settings).to(self.device)
        self.criterion = self._losses.BoundaryLoss.for_fold(
            self.fold, settings=self.loss_settings,
            fold_stats=self.fold_stats).to(self.device)

        # The freeze state for epoch 0 must be applied BEFORE the optimizer is
        # built, or the optimizer holds parameters that are frozen and the
        # unfreeze transition has nothing to add.
        self._model_mod.apply_freeze_schedule(self.model, 0, self.model_settings)
        self.optimizer = build_optimizer(self.model, self.settings)

        steps_per_epoch = max(1, len(self.train_loader))
        self.scheduler = WarmupCosine(
            total_steps=steps_per_epoch * int(self.settings["epochs"]),
            warmup_steps=steps_per_epoch * int(self.settings["warmup_epochs"]),
            min_lr_scale=float(self.settings["min_lr_scale"]))
        self.scaler = make_scaler(self.amp_enabled)

        if tensorboard:
            self._open_writer()
        return self.summary()

    @property
    def amp_enabled(self) -> bool:
        return bool(self.settings["amp"]) and self.device.type == "cuda"

    def _build_loaders(self) -> None:
        from torch.utils.data import DataLoader, Subset

        ds = self._ds
        generator = torch.Generator()
        generator.manual_seed(int(self.settings["seed"]))
        batch_size = int(self.settings["batch_size"])
        val_batch = int(self.settings["val_batch_size"] or batch_size)

        sampler = None
        if self.settings["sampler"]:
            sampler = ds.build_sampler(self.train_ds, self.fold,
                                       fold_stats=self.fold_stats,
                                       generator=generator)
        self.train_loader = DataLoader(
            self.train_ds, batch_size=batch_size, sampler=sampler,
            shuffle=(sampler is None), num_workers=self.num_workers,
            drop_last=len(self.train_ds) > batch_size, pin_memory=True,
            persistent_workers=bool(self.num_workers),
            worker_init_fn=seed_worker, generator=generator)
        self.val_loader = DataLoader(
            self.val_ds, batch_size=val_batch, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=bool(self.num_workers),
            worker_init_fn=seed_worker)

        # A FIXED sample of val tiles for the clDice diagnostic. Fixed because
        # the question is how the skeleton changes across epochs, and a moving
        # sample cannot answer it.
        n_probe = min(int(self.settings["cldice_probe_tiles"]), len(self.val_ds))
        rng = np.random.default_rng(int(self.settings["seed"]))
        self.probe_indices = sorted(int(i) for i in rng.choice(
            len(self.val_ds), size=n_probe, replace=False))
        self.probe_loader = DataLoader(
            Subset(self.val_ds, self.probe_indices),
            batch_size=min(val_batch, n_probe), shuffle=False, num_workers=0)

    def _open_writer(self) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        except Exception as exc:                      # noqa: BLE001
            print(f"tensorboard unavailable ({type(exc).__name__}: {exc}); "
                  "training continues without it")
            self.writer = None

    def pos_weight_drift(self) -> dict:
        """What the fold's recorded pos_weight implies against the ACTUAL split.

        ``pos_weight`` is ``n_negative / n_positive``, which for equal-sized
        tiles is exactly ``(1 - mean boundary fraction) / mean boundary
        fraction`` -- and that identity reproduces the value step 3 recorded, so
        the same arithmetic over the rows actually being trained on says what
        the reduced split would have wanted.

        The recorded value is still the one used, deliberately. Holding the loss
        identical across both arms is what makes the comparison attributable: if
        the exclusion changed the training data AND the class weighting, a
        difference in the result could not be assigned to either. So the drift
        is measured and printed rather than silently corrected.
        """
        recorded = float(self.fold_entry["pos_weight"])
        fractions = [float(r["boundary_fraction"]) for r in self.train_ds.rows]
        mean_fraction = sum(fractions) / len(fractions) if fractions else 0.0
        implied = ((1.0 - mean_fraction) / mean_fraction
                   if mean_fraction > 0 else float("inf"))
        return {
            "recorded": recorded,
            "recorded_source": f"configs/fold_stats.yaml folds.{self.fold}",
            "implied_by_this_split": implied,
            "mean_boundary_fraction": mean_fraction,
            "ratio": (implied / recorded) if recorded else float("inf"),
            "in_use": recorded,
            "note": ("the recorded value is used in BOTH arms on purpose, so "
                     "that the training data is the only thing that differs"),
        }

    def summary(self) -> dict:
        counts = self._model_mod.summarize(self.model) if self.model else {}
        return {
            "fold": self.fold,
            "run_name": self.run_name,
            "excluded_datasets": list(self.excluded),
            "pos_weight_drift": (self.pos_weight_drift()
                                 if self.train_ds else None),
            "held_out": self.held_out,
            "platform": self.platform,
            "device": str(self.device),
            "gpu": (torch.cuda.get_device_name(0)
                    if self.device.type == "cuda" else None),
            "config_hash": self.hash,
            "seed": self.seed_report,
            "amp": self.amp_enabled,
            "num_workers": self.num_workers,
            "batch_size": int(self.settings["batch_size"]),
            "epochs": int(self.settings["epochs"]),
            "steps_per_epoch": len(self.train_loader) if self.train_loader else 0,
            "lr": float(self.settings["lr"]),
            "encoder_lr": float(self.settings["lr"])
                          * float(self.settings["encoder_lr_scale"]),
            "weight_decay": float(self.settings["weight_decay"]),
            "warmup_epochs": int(self.settings["warmup_epochs"]),
            "grad_clip": float(self.settings["grad_clip"]),
            "pos_weight": float(self.criterion.pos_weight) if self.criterion else None,
            "freeze_encoder_epochs": int(self.model_settings["freeze_encoder_epochs"]),
            "train_tiles": len(self.train_ds) if self.train_ds else 0,
            "val_tiles": len(self.val_ds) if self.val_ds else 0,
            "val_composition": self.val_ds.composition() if self.val_ds else {},
            "train_composition": self.train_ds.composition() if self.train_ds else {},
            "parameters": counts,
            "checkpoint_dir": str(self.checkpoint_dir),
            "log_dir": str(self.log_dir),
            "cache": {k: {"source": v["source"], "seconds": v["seconds"],
                          "images": v["images"]}
                      for k, v in self.cache_report.items()},
            "sources": dict(self.sources),
            "probe_tiles": len(getattr(self, "probe_indices", []) or []),
        }

    # -- resume -----------------------------------------------------------
    def maybe_resume(self, path: Optional[Path] = None,
                     allow_config_change: bool = False) -> dict:
        """Continue ``last.pt`` if it is genuinely this run. Otherwise refuse.

        Two things are verified and neither is advisory. The **fold** must
        match: resuming a fold_uhcs1 checkpoint into a fold_MetalDam run would
        train with pos_weight 15.974 on weights fitted under 7.111 and validate
        against the wrong held-out dataset, and nothing about that raises on
        its own. The **config hash** must match: a resumed run that quietly
        adopts a new learning rate is not the run whose curve is in the report.
        """
        path = Path(path or self.last_path)
        if not path.is_file():
            return {"resumed": False, "reason": f"no checkpoint at {path}",
                    "path": str(path), "start_epoch": 0}

        state = load_checkpoint(path)
        if state.get("fold") != self.fold:
            raise TrainError(
                f"{path} was written by fold {state.get('fold')!r} but this "
                f"run is fold {self.fold!r}. Resuming would train under the "
                f"wrong pos_weight and validate on the wrong held-out dataset. "
                f"Move or delete that checkpoint, or run the fold it belongs to.")
        hash_migrated = False
        if state.get("config_hash") != self.hash:
            diff = _diff_config(state.get("hashed_config") or {},
                                self.hashed_config)
            if not allow_config_change:
                raise TrainError(
                    f"{path} was written under config hash "
                    f"{state.get('config_hash')} and this run is {self.hash}. "
                    "Continuing would produce a curve whose first half and "
                    "second half are different experiments.\n"
                    "  what changed:\n" + "\n".join(f"    {d}" for d in diff)
                    + "\n  Start a clean run, or pass "
                    "allow_config_change=True if you have decided that this "
                    "difference does not matter and will say so in the report.")
            print(f"! resuming across a config change ({len(diff)} keys) "
                  "because allow_config_change=True:")
            for line in diff:
                print(f"    {line}")
            # The checkpoint's recorded hash is stale the moment this is
            # accepted: it still says the OLD config produced it, while this
            # run now continues it under the NEW one. Migrate BOTH files'
            # labels now -- verified, weights untouched -- rather than leave a
            # "mismatch" for every later check (and human) to rediscover. See
            # migrate_config_hash's docstring for the incident this replaces.
            migration = migrate_config_hash(path, self.hash,
                                            self.hashed_config, self.fold)
            state = migration["state"]
            hash_migrated = migration["migrated"]
            print(f"    relabelled {migration['path']}: "
                  f"{migration['from_hash']} -> {migration['to_hash']} "
                  f"(weights unchanged{': ' + migration['weight_checksum'][:12] if migration['migrated'] else ''})")
            if self.best_path.is_file():
                best_migration = migrate_config_hash(
                    self.best_path, self.hash, self.hashed_config, self.fold)
                hash_migrated = hash_migrated or best_migration["migrated"]
                print(f"    relabelled {best_migration['path']}: "
                      f"{best_migration['from_hash']} -> "
                      f"{best_migration['to_hash']} (weights unchanged"
                      f"{': ' + best_migration['weight_checksum'][:12] if best_migration['migrated'] else ''})")

        self.model.load_state_dict(state["model"])
        saved_epoch = int(state["epoch"])
        self.start_epoch = saved_epoch + 1
        # Rebuild the optimizer for the freeze state of the epoch that was
        # SAVED, not the one about to run. Those differ by exactly one param
        # group when the crash landed on the epoch before the unfreeze, and
        # loading a one-group state into a two-group optimizer raises. The
        # transition itself is then made by fit()'s own _apply_freeze on the
        # first epoch it runs, which is where it belongs and where it is
        # asserted.
        self._model_mod.apply_freeze_schedule(self.model, saved_epoch,
                                              self.model_settings)
        self.optimizer = build_optimizer(self.model, self.settings)
        try:
            self.optimizer.load_state_dict(state["optimizer"])
        except (ValueError, KeyError) as exc:
            raise TrainError(
                f"the saved optimizer state does not fit the optimizer rebuilt "
                f"for epoch {saved_epoch} ({exc}). This usually means "
                "model.freeze_encoder_epochs changed between runs; start a "
                "clean run."
            ) from exc
        self.scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler") is not None:
            try:
                self.scaler.load_state_dict(state["scaler"])
            except Exception:                          # noqa: BLE001
                print("! AMP scaler state could not be restored (AMP toggled "
                      "between runs?); continuing with a fresh scaler")
        rng_ok = load_rng_state(state.get("rng"))
        self.history = list(state.get("history") or [])
        self.best = dict(state.get("best") or self.best)
        self.freeze_transitions = list(state.get("freeze_transitions") or [])

        return {
            "resumed": True,
            "path": str(path),
            "start_epoch": self.start_epoch,
            "epochs_done": self.start_epoch,
            "reason": (f"checkpoint at epoch {state['epoch']} matches fold "
                       f"{self.fold!r} and config hash {self.hash}"),
            "rng_restored": rng_ok,
            "best": dict(self.best),
            "saved_amp": state.get("amp"),
            "saved_utc": state.get("saved_utc"),
            "hash_migrated": hash_migrated,
        }

    def save(self, epoch: int, is_best: bool = False) -> dict:
        """Write last.pt, and best.pt ONLY when this epoch actually set the record.

        ``is_best`` is not trusted as a bare flag. Writing best.pt is gated on
        ``epoch == self.best["epoch"]``, which ``fit()`` sets in the SAME
        iteration, immediately before calling this -- so the two can never
        disagree in the one caller meant to use them. A caller that has not
        just updated ``self.best`` to say ``epoch`` is the new record is
        refused, loudly, rather than allowed to overwrite the real best
        weights with whatever happens to be sitting in ``self.model`` at the
        moment.

        This guard exists because of a real incident, not a hypothetical one.
        A resume that trained ZERO further epochs (the checkpoint was already
        at the run's final epoch) left ``self.model`` holding the LAST epoch's
        weights, not the BEST epoch's -- ``maybe_resume`` loads from
        ``last.pt``, never from ``best.pt``. A notebook cell then called
        ``trainer.save(last_epoch, is_best=True)`` by hand to "resync" an
        unrelated checkpoint-provenance mismatch, and that single call
        silently overwrote best.pt's epoch-22 selected weights with epoch 39's
        -- epoch 22 is gone and the fold has to be retrained to get it back.
        Never call ``save(is_best=True)`` from outside ``fit()``; if a
        checkpoint's recorded config hash needs updating without retraining,
        that is :func:`migrate_config_hash`, which touches metadata only and
        is verified never to touch a weight.
        """
        if is_best and int(epoch) != self.best.get("epoch"):
            raise TrainError(
                f"save(epoch={epoch}, is_best=True) was requested, but "
                f"self.best['epoch'] is {self.best.get('epoch')!r} -- epoch "
                f"{epoch} is not the epoch that set the record. Writing "
                "best.pt here would overwrite the real best weights with "
                "whatever is currently loaded in self.model. This is exactly "
                "how a prior run lost its epoch-22 best.pt to a resume that "
                "trained nothing (see reports/train_dev_colab.md). Let "
                "fit() call save() -- it keeps epoch and self.best in sync by "
                "construction -- and use migrate_config_hash() if what you "
                "actually want is to relabel a checkpoint's config hash "
                "without retraining.")
        state = {
            "fold": self.fold,
            "held_out": self.held_out,
            "epoch": int(epoch),
            "config_hash": self.hash,
            "hashed_config": self.hashed_config,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "rng": rng_state(),
            "history": self.history,
            "best": self.best,
            "freeze_transitions": self.freeze_transitions,
            "amp": self.amp_enabled,
            "platform": self.platform,
            "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        written = {"last": str(atomic_save(state, self.last_path))}
        if is_best:
            if self.settings["best_includes_optimizer"]:
                best_state = state
            else:
                # best.pt is a deliverable for step 7, not a resume point.
                # Dropping the optimizer moments takes it from ~280 MB to
                # ~93 MB, which matters when both files live on a free Drive.
                best_state = {k: v for k, v in state.items()
                              if k not in ("optimizer", "scaler", "rng")}
                best_state["resumable"] = False
            written["best"] = str(atomic_save(best_state, self.best_path))
        return written

    # -- one epoch --------------------------------------------------------
    def _apply_freeze(self, epoch: int) -> dict:
        """Freeze/unfreeze, and make the optimizer follow. Asserted, not hoped.

        At the unfreeze epoch the trainable parameter count MUST rise and the
        optimizer MUST gain those tensors. If either fails to happen the run
        would continue happily with a permanently frozen encoder, so both are
        checked and a mismatch raises.
        """
        before_trainable = sum(p.numel() for p in self.model.parameters()
                               if p.requires_grad)
        before_optimized = optimizer_param_count(self.optimizer)
        state = self._model_mod.apply_freeze_schedule(self.model, epoch,
                                                      self.model_settings)
        after_trainable = sum(p.numel() for p in self.model.parameters()
                              if p.requires_grad)
        synced = sync_optimizer_groups(self.optimizer, self.model, self.settings)
        after_optimized = optimizer_param_count(self.optimizer)

        if state["changed"] and not state["frozen"]:
            if after_trainable <= before_trainable:
                raise TrainError(
                    f"epoch {epoch} is the unfreeze epoch but the trainable "
                    f"parameter count did not rise ({before_trainable:,} -> "
                    f"{after_trainable:,}). The encoder is not actually being "
                    "unfrozen.")
            if after_optimized <= before_optimized:
                raise TrainError(
                    f"epoch {epoch} unfroze {after_trainable - before_trainable:,} "
                    "parameters but the optimizer still holds "
                    f"{after_optimized:,}. Their gradients would be computed "
                    "and thrown away every step for the rest of the run.")
            self.freeze_transitions.append({
                "epoch": int(epoch),
                "trainable_before": int(before_trainable),
                "trainable_after": int(after_trainable),
                "optimized_before": int(before_optimized),
                "optimized_after": int(after_optimized),
                "groups": synced["groups"],
            })
        return {
            "frozen": state["frozen"],
            "changed": state["changed"],
            "trainable": after_trainable,
            "optimized": after_optimized,
            "groups": len(self.optimizer.param_groups),
            "synced": synced,
        }

    def train_one_epoch(self, epoch: int,
                        progress: Optional[Callable] = None) -> dict:
        self.model.train()
        freeze = self._apply_freeze(epoch)      # AFTER .train(): it resets BN

        totals = {"total": 0.0, "bce": 0.0, "dice": 0.0, "cldice": 0.0}
        seen, last_lr = 0, 0.0
        iterator = self.train_loader
        if progress is not None:
            iterator = progress(iterator, desc=f"epoch {epoch} train")

        for step, batch in enumerate(iterator):
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            last_lr = self.scheduler.apply(self.optimizer)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(self.device.type, self.amp_enabled):
                logits = self.model(images)
            # The LOSS runs in fp32, outside autocast, on purpose. Dice and
            # clDice sum over 65,536 pixels per tile; in fp16 that sum can
            # exceed 65,504 and become inf, and the NaN that follows would look
            # like a diverging model rather than an overflow.
            terms = self.criterion.components(logits.float(), masks)
            loss = terms["total"]

            self.scaler.scale(loss).backward()
            if float(self.settings["grad_clip"]) > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    [p for g in self.optimizer.param_groups for p in g["params"]],
                    float(self.settings["grad_clip"]))
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            batch_n = images.shape[0]
            seen += batch_n
            for key in totals:
                totals[key] += float(terms[key]) * batch_n
            if self.writer is not None and step % int(self.settings["log_every"]) == 0:
                global_step = self.scheduler.step_count
                for key in totals:
                    self.writer.add_scalar(f"train_step/{key}",
                                           float(terms[key]), global_step)
                self.writer.add_scalar("train_step/lr", last_lr, global_step)

        if seen == 0:
            raise TrainError("the training loader produced no samples.")
        out = {key: value / seen for key, value in totals.items()}
        out.update({"lr": last_lr, "samples": seen, **freeze})
        return out

    @torch.no_grad()
    def validate(self, progress: Optional[Callable] = None) -> dict:
        self.model.eval()
        accumulator = MetricAccumulator(
            threshold_grid(self.settings),
            fixed_threshold=float(self.settings["threshold"]),
            tolerance_px=int(self.settings["boundary_tolerance_px"]))
        totals = {"total": 0.0, "bce": 0.0, "dice": 0.0, "cldice": 0.0}
        seen = 0

        iterator = self.val_loader
        if progress is not None:
            iterator = progress(iterator, desc="validate")
        for batch in iterator:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            with autocast(self.device.type, self.amp_enabled):
                logits = self.model(images)
            logits = logits.float()
            terms = self.criterion.components(logits, masks)
            batch_n = images.shape[0]
            seen += batch_n
            for key in totals:
                totals[key] += float(terms[key]) * batch_n
            accumulator.update(torch.sigmoid(logits).cpu().numpy(),
                               masks.cpu().numpy(), batch["dataset"])

        if seen == 0:
            raise TrainError("the validation loader produced no samples.")
        loss = {key: value / seen for key, value in totals.items()}
        return {"loss": loss, "metrics": accumulator.result()}

    @torch.no_grad()
    def cldice_probe(self) -> dict:
        """Is clDice measuring anything on REAL predictions, or is it inert?

        Step 5 measured clDice against perturbed ground truth, which is hard
        and binary. A model's output is soft and over-thick, and the soft
        skeleton might behave completely differently on it -- or not at all.

        ``skeleton_delta`` is the number that answers it:
        ``mean |soft_skeleton(sigmoid(logits)) - sigmoid(logits)|``. The soft
        skeleton returns its input unchanged whenever erosion cannot bite, so a
        delta pinned near zero across every epoch means the term is computing
        Dice on a slightly different denominator and contributing nothing.
        That is a finding to record, not a reason to quietly retune w_cldice.
        """
        self.model.eval()
        iters = int(self.criterion.cldice_iters)
        agg = {"skel_pred_sum": 0.0, "skel_true_sum": 0.0,
               "skel_pred_on_true": 0.0, "t_prec": 0.0, "t_rec": 0.0,
               "skeleton_delta": 0.0, "skeleton_delta_max": 0.0,
               "pred_sum": 0.0, "true_sum": 0.0,
               "dice": 0.0, "cldice": 0.0}
        batches = 0
        for batch in self.probe_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            with autocast(self.device.type, self.amp_enabled):
                logits = self.model(images)
            logits = logits.float()

            parts = self._losses.cldice_parts(
                logits, masks, iters=iters, smooth=self.criterion.smooth,
                eps=self.criterion.eps)
            probs = torch.sigmoid(logits)
            skeleton = self._losses.soft_skeletonize(probs, iters)
            delta = (skeleton - probs).abs()

            for key in ("skel_pred_sum", "skel_true_sum", "skel_pred_on_true",
                        "t_prec", "t_rec", "pred_sum", "true_sum"):
                agg[key] += float(parts[key])
            agg["skeleton_delta"] += float(delta.mean())
            agg["skeleton_delta_max"] = max(agg["skeleton_delta_max"],
                                            float(delta.max()))
            agg["dice"] += float(self._losses.dice_term(
                logits, masks, smooth=self.criterion.smooth))
            agg["cldice"] += float(parts["loss"])
            batches += 1

        if batches == 0:
            raise TrainError("the clDice probe loader produced no batches.")
        out = {key: (value / batches if key != "skeleton_delta_max" else value)
               for key, value in agg.items()}
        out["tiles"] = len(self.probe_indices)
        out["degenerate"] = bool(out["skeleton_delta"] < 1e-4)
        out["verdict"] = (
            "soft skeleton is returning its input -- clDice is degenerate on "
            "these predictions and is contributing nothing beyond Dice"
            if out["degenerate"] else
            "soft skeleton differs from the prediction -- clDice is measuring "
            "a real centreline")
        return out

    # -- the loop ---------------------------------------------------------
    def best_key(self) -> str:
        """Dice on the HELD-OUT dataset, not pooled and not the loss.

        The fold exists to measure one thing: performance on the microscope
        that was held out. Selecting on the pooled number would let Steel1's
        easy validation tiles pick the checkpoint.
        """
        return self.held_out or MetricAccumulator.POOLED

    def _score(self, metrics: dict) -> tuple:
        """Best-THRESHOLD Dice on the held-out dataset.

        Not the value at 0.5. A checkpoint selected at a fixed operating point
        is selected partly on how well 0.5 happens to suit it that epoch, which
        moves as the model's confidence calibrates; the best-threshold value
        asks what the model can do once step 7 sets the threshold it is going
        to set anyway. The threshold that achieved it travels with the score.
        """
        key = self.best_key()
        per_dataset = metrics["per_dataset"]
        entry = per_dataset.get(key)
        if entry is None:
            entry = metrics["pooled"]
            key = "pooled (held-out dataset absent)"
        return float(entry["best"]["dice"]), key

    def _score_threshold(self, metrics: dict, key: str) -> float:
        entry = metrics["per_dataset"].get(key) or metrics["pooled"]
        return float(entry["best_threshold"])

    def fit(self, epochs: Optional[int] = None,
            on_epoch_end: Optional[Callable] = None,
            progress: Optional[Callable] = None) -> list:
        """Run to ``train.epochs``, checkpointing after every single epoch."""
        if self.model is None:
            raise TrainError("call setup() before fit().")
        total_epochs = int(epochs or self.settings["epochs"])
        started = time.perf_counter()

        for epoch in range(self.start_epoch, total_epochs):
            epoch_started = time.perf_counter()
            train_stats = self.train_one_epoch(epoch, progress=progress)
            val_stats = self.validate(progress=progress)
            probe = self.cldice_probe()
            score, score_key = self._score(val_stats["metrics"])

            is_best = score > self.best["metric"]
            if is_best:
                self.best = {
                    "metric": score, "epoch": epoch, "key": score_key,
                    "threshold": self._score_threshold(val_stats["metrics"],
                                                       score_key),
                    "criterion": "best-threshold Dice on the held-out dataset",
                }

            elapsed = time.perf_counter() - epoch_started
            done = epoch - self.start_epoch + 1
            record = {
                "epoch": epoch,
                "seconds": elapsed,
                "eta_seconds": (time.perf_counter() - started) / done
                               * (total_epochs - epoch - 1),
                "lr": train_stats["lr"],
                "trainable_params": train_stats["trainable"],
                "optimized_params": train_stats["optimized"],
                "encoder_frozen": train_stats["frozen"],
                "freeze_changed": train_stats["changed"],
                "train": {k: train_stats[k]
                          for k in ("total", "bce", "dice", "cldice")},
                "val": val_stats["loss"],
                "metrics": val_stats["metrics"],
                "cldice_probe": probe,
                "score": score,
                "score_key": score_key,
                "score_threshold": self._score_threshold(val_stats["metrics"],
                                                         score_key),
                "best_thresholds": {
                    name: entry["best_threshold"] for name, entry
                    in val_stats["metrics"]["per_dataset"].items()},
                "is_best": is_best,
            }
            self.history.append(record)
            self._log_tensorboard(record)
            record["checkpoints"] = self.save(epoch, is_best=is_best)

            if on_epoch_end is not None:
                on_epoch_end(record, self)

        if self.writer is not None:
            self.writer.flush()
        return self.history

    def _log_tensorboard(self, record: dict) -> None:
        if self.writer is None:
            return
        epoch = record["epoch"]
        for split in ("train", "val"):
            for key, value in record[split].items():
                self.writer.add_scalar(f"loss_{split}/{key}", value, epoch)
        for label, entry in ([("pooled", record["metrics"]["pooled"])]
                             + list(record["metrics"]["per_dataset"].items())):
            for row in ("fixed", "best"):
                for key, value in entry[row].items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f"val_{label}_{row}/{key}",
                                               value, epoch)
            # The chosen threshold is a measurement in its own right: a
            # held-out dataset that wants a very different operating point from
            # the training-side datasets IS the domain shift, in the units of
            # the decision step 7 has to make.
            self.writer.add_scalar(f"val_{label}/best_threshold",
                                   entry["best_threshold"], epoch)
        for key in ("skeleton_delta", "skel_pred_sum", "skel_true_sum",
                    "t_prec", "t_rec", "cldice", "dice"):
            self.writer.add_scalar(f"cldice_probe/{key}",
                                   record["cldice_probe"][key], epoch)
        self.writer.add_scalar("schedule/lr", record["lr"], epoch)
        self.writer.add_scalar("schedule/trainable_params",
                               record["trainable_params"], epoch)

    # -- figures ----------------------------------------------------------
    @torch.no_grad()
    def example_predictions(self, per_dataset: int = 1,
                            thresholds: Optional[dict] = None) -> list:
        """One representative val tile per dataset: raw, ground truth, prediction.

        The tile is the one whose boundary fraction is the MEDIAN for its
        dataset, chosen once and never re-drawn, so the panels across epochs
        show the model changing rather than the sample changing.
        """
        self.model.eval()
        by_dataset = {}
        for index, row in enumerate(self.val_ds.rows):
            by_dataset.setdefault(row["dataset"], []).append(
                (float(row["boundary_fraction"]), index))

        picks = []
        for name, entries in sorted(by_dataset.items()):
            entries.sort()
            for k in range(min(per_dataset, len(entries))):
                position = int(len(entries) * (k + 1) / (per_dataset + 1))
                picks.append((name, entries[min(position, len(entries) - 1)][1]))

        out = []
        for name, index in picks:
            sample = self.val_ds[index]
            image = torch.from_numpy(sample["image"])[None].to(self.device)
            with autocast(self.device.type, self.amp_enabled):
                logits = self.model(image)
            prob = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
            fixed = float(self.settings["threshold"])
            tuned = float((thresholds or {}).get(name, fixed))
            out.append({
                "dataset": name,
                "tile_id": sample["tile_id"],
                "image": sample["image"][0],
                "truth": sample["mask"][0],
                "prob": prob,
                "threshold": fixed,
                "best_threshold": tuned,
                "pred": (prob >= fixed),
                "pred_best": (prob >= tuned),
            })
        return out


# --------------------------------------------------------------------------
# validation-only evaluation of an ARBITRARY checkpoint -- no Trainer state
# touched, nothing retrained. Used by notebook 06's prediction gallery to
# inspect best.pt after the fact, independent of whatever trainer.model
# currently holds (the last epoch's weights, not necessarily the best one).
# --------------------------------------------------------------------------
def load_checkpoint_model(path: Path, model_settings: dict, fold: str,
                          expected_hash: Optional[str] = None,
                          device=None) -> tuple:
    """Build a fresh model and load ``path`` into it. Verifies fold and hash.

    Returns ``(model, state)``. The same two guards ``Trainer.maybe_resume``
    applies: a checkpoint from another fold would be evaluated under the wrong
    ``pos_weight``'s worth of training and the wrong held-out dataset, and one
    under a different config hash is not the run this notebook just described.
    Both are refused rather than silently loaded.
    """
    from src import model as model_mod

    state = load_checkpoint(path)
    if state.get("fold") != str(fold):
        raise TrainError(
            f"{path} was written by fold {state.get('fold')!r}, not "
            f"{fold!r}. Evaluating it here would describe the wrong fold's "
            "model as this one's.")
    if expected_hash is not None and state.get("config_hash") != expected_hash:
        raise TrainError(
            f"{path} was written under config hash {state.get('config_hash')}, "
            f"not {expected_hash!r}. It is not the checkpoint this run just "
            "produced; load it deliberately with expected_hash=None if that "
            "is intended.")

    model = model_mod.build_model(settings=model_settings)
    model.load_state_dict(state["model"])
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, state


def best_epoch_thresholds(state: dict) -> dict:
    """The per-dataset tuned thresholds recorded AT THE BEST EPOCH.

    Not the fixed 0.5, and not a threshold re-swept now: the operating point
    step 6 already chose when it selected this checkpoint as best, read back
    from the history entry for that exact epoch. A checkpoint saved before the
    threshold sweep existed has no such entry, and that is reported rather
    than papered over with a default.
    """
    best = state.get("best") or {}
    epoch = best.get("epoch")
    if epoch is None:
        raise TrainError(
            "this checkpoint carries no best['epoch']; it predates the "
            "threshold sweep. Re-train or re-select best.pt under the current "
            "code before using this cell.")
    record = next((r for r in (state.get("history") or [])
                   if r.get("epoch") == epoch), None)
    if record is None or "best_thresholds" not in record:
        raise TrainError(
            f"no history record with best_thresholds for epoch {epoch}; this "
            "checkpoint predates the per-dataset threshold sweep. Re-train "
            "under the current code.")
    return dict(record["best_thresholds"])


@torch.no_grad()
def evaluate_checkpoint(model: "nn.Module", val_ds, thresholds: dict,
                        device, amp_enabled: bool = False,
                        batch_size: int = 32, num_workers: int = 0) -> list:
    """Per-tile Dice AND measured boundary width, one pass over ``val_ds``.

    Validation only: no optimizer, no loss term, no backward pass. This exists
    because :class:`MetricAccumulator` only ever keeps running pixel counts --
    it can report a dataset's pooled Dice but not whether that number is one
    uniform mediocre score or a mix of tiles that work and tiles that fail
    outright, and it never touches boundary WIDTH at all. Both come from the
    same per-tile pass over the model's output, so they are computed together
    here instead of twice.

    ``thresholds`` must carry an entry for every dataset present in
    ``val_ds`` -- the operating point already chosen for it, not a threshold
    guessed fresh here. Width is measured exactly as
    ``src.boundary_gt.measured_line_width`` measures the ground truth itself:
    skeletonized area divided by skeleton length, so the two numbers this cell
    prints are directly comparable to the numbers notebook 02 reported.
    """
    from torch.utils.data import DataLoader

    from src import boundary_gt

    present = sorted({r["dataset"] for r in val_ds.rows})
    missing = sorted(set(present) - set(thresholds))
    if missing:
        raise TrainError(
            f"evaluate_checkpoint has no threshold for {missing}; pass one "
            "for every dataset in the validation split (see "
            "best_epoch_thresholds).")

    model.eval()
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    records = []
    row_index = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].numpy()
        with autocast(device.type, amp_enabled):
            logits = model(images)
        probs = torch.sigmoid(logits.float()).cpu().numpy()

        for i in range(probs.shape[0]):
            row = val_ds.rows[row_index]
            dataset = row["dataset"]
            threshold = float(thresholds[dataset])
            prob = probs[i, 0]
            true = masks[i, 0] > 0.5
            pred = prob >= threshold

            tp = int((pred & true).sum())
            fp = int((pred & ~true).sum())
            fn = int((~pred & true).sum())
            denom = 2 * tp + fp + fn
            dice = (2.0 * tp / denom) if denom > 0 else 0.0

            records.append({
                "row_index": row_index,
                "dataset": dataset,
                "tile_id": row["tile_id"],
                "threshold": threshold,
                "dice": float(dice),
                "true_fraction": float(true.mean()),
                "pred_fraction": float(pred.mean()),
                "pred_width_px": boundary_gt.measured_line_width(pred),
                "true_width_px": boundary_gt.measured_line_width(true),
            })
            row_index += 1

    if row_index != len(val_ds):
        raise TrainError(
            f"evaluated {row_index} tiles but val_ds has {len(val_ds)}; the "
            "loader dropped or duplicated rows.")
    return records


def rank_tiles_by_dice(records: list, dataset: str) -> dict:
    """Best, median, and two worst tiles of one dataset, by per-tile Dice.

    Ties are broken by ``row_index`` so the choice is deterministic rather
    than depending on Python's sort stability lining up with load order by
    accident. With fewer than 4 tiles, the same tile fills more than one slot
    -- reported, not hidden, since a dataset that small is itself worth
    knowing about.
    """
    subset = sorted((r for r in records if r["dataset"] == dataset),
                    key=lambda r: (r["dice"], -r["row_index"]))
    if not subset:
        raise TrainError(f"no validation tiles for dataset {dataset!r}")
    n = len(subset)
    return {
        "best": subset[-1],
        "median": subset[n // 2],
        "worst": subset[0],
        "worst2": subset[1] if n > 1 else subset[0],
        "n_tiles": n,
    }


@torch.no_grad()
def tile_prediction(model: "nn.Module", val_ds, row_index: int, device,
                    amp_enabled: bool = False) -> dict:
    """Image, ground truth and probability map for exactly ONE validation tile.

    Deliberately re-runs the forward pass rather than reusing anything cached
    by :func:`evaluate_checkpoint`: keeping a full probability map for every
    validation tile in memory is wasteful when only a handful are ever drawn,
    so the cheap per-tile summary and the expensive per-tile array are kept
    separate on purpose.
    """
    sample = val_ds[row_index]
    image = torch.from_numpy(sample["image"])[None].to(device)
    with autocast(device.type, amp_enabled):
        logits = model(image)
    prob = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
    return {
        "dataset": sample["dataset"],
        "tile_id": sample["tile_id"],
        "image": sample["image"][0],
        "truth": sample["mask"][0],
        "prob": prob,
    }


# --------------------------------------------------------------------------
# placement vs thickness -- splitting a low pixel Dice into its two causes
# --------------------------------------------------------------------------
# Verdict thresholds, every one of them placed in a gap MEASURED by
# reference_decomposition() rather than guessed. The two proximities are read
# at the project's own boundary tolerance (train.boundary_tolerance_px = 2),
# which is the distance step 2's annotation is accurate to.
#
#   case                 pred-in-true  true-in-pred  skelDice  width  curve
#   perfect                     1.00        1.00       1.00     1.00   1.00
#   placed but 1-3 px fat       1.00        0.99+      0.98+    1.97+  ~1.0
#   over-detected 1x-3x         0.34-0.60   1.00       0.44-.72 ~1.0   1.8-3.5
#   over-detected + fat         0.34        1.00       0.44     1.81   3.49
#   displaced 2 px              1.00        0.99       0.02     1.00   0.99
#   misplaced 5-8 px            0.10        0.10       0.02     1.00   ~1.0
#   noise at the same density   0.19        0.63       0.05     0.52   1.94
#
#: Below this, the true curves were not found: nothing the prediction drew is
#: near them. Noise reaches 0.63; every case that did find them reaches 0.99.
PLACEMENT_FOUND_OK = 0.70
#: Below this, most of what was drawn is not on a true boundary -- the
#: signature of OVER-DETECTION. Over-detected cases top out at 0.60; placed
#: ones sit at 1.00.
PLACEMENT_REAL_OK = 0.80
#: Above this, the predicted line is materially fatter than the true one.
#: Correctly-thin cases reach 1.00; fat ones start at 1.81.
PLACEMENT_WIDTH_HIGH = 1.40
#: Below this the two skeletons barely overlap EXACTLY, even though they are
#: within tolerance of each other -- a sub-tolerance registration offset rather
#: than a placement failure. Offset scores 0.02; everything correctly placed
#: scores 0.98+.
PLACEMENT_EXACT_OVERLAP_OK = 0.50
#: Over-detection also requires genuine structural overlap, not coincidental
#: coverage. Dense enough noise reaches "found" 0.90 purely by chance -- a
#: random pixel lands within 2 px of almost anything -- and would otherwise be
#: read as "found the curves and drew more besides". Skeleton Dice tells the
#: two apart regardless of boundary density: noise scores 0.05-0.12 across
#: every grid density tested, genuine over-detection 0.44-0.72.
PLACEMENT_STRUCTURE_OK = 0.25


def decomposition_counts(pred: np.ndarray, true: np.ndarray,
                         radii: Sequence, distances: Sequence) -> dict:
    """Every count the placement/thickness split needs, for ONE tile.

    Shared by the real evaluation and by :func:`reference_decomposition`, so
    the calibration table a notebook prints beside its results is produced by
    the same arithmetic as the results -- not by a parallel implementation that
    can drift.
    """
    import cv2
    from skimage.morphology import skeletonize

    pred = np.ascontiguousarray(pred).astype(bool)
    true = np.ascontiguousarray(true).astype(bool)
    pred_u8 = pred.astype(np.uint8)
    true_u8 = true.astype(np.uint8)

    # skeletonize() on an all-False array is a no-op, but guarding says so.
    skel_pred = skeletonize(pred) if pred.any() else np.zeros_like(pred)
    skel_true = skeletonize(true) if true.any() else np.zeros_like(true)
    skel_pred_u8 = skel_pred.astype(np.uint8)
    skel_true_u8 = skel_true.astype(np.uint8)

    disks = {r: euclidean_disk(r) for r in set(radii) | set(distances) if r > 0}

    def grow(mask_u8, base, r):
        return base if r == 0 else cv2.dilate(mask_u8, disks[r]).astype(bool)

    grown_pred = {r: grow(pred_u8, pred, r) for r in radii}
    grown_true = {r: grow(true_u8, true, r) for r in radii}
    near_pred = {d: grow(pred_u8, pred, d) for d in distances}
    near_true = {d: grow(true_u8, true, d) for d in distances}
    near_skel_pred = {d: grow(skel_pred_u8, skel_pred, d) for d in distances}
    near_skel_true = {d: grow(skel_true_u8, skel_true, d) for d in distances}

    return {
        "tiles": 1,
        "skel_tp": int((skel_pred & skel_true).sum()),
        "skel_pred_px": int(skel_pred.sum()),
        "skel_true_px": int(skel_true.sum()),
        # k = 0 is the identity, so this row IS the ordinary pixel Dice and must
        # reproduce what MetricAccumulator reported. It is kept as the anchor
        # the rest of the sweep is read against.
        "dil_tp": {r: int((grown_pred[r] & grown_true[r]).sum()) for r in radii},
        "dil_pred": {r: int(grown_pred[r].sum()) for r in radii},
        "dil_true": {r: int(grown_true[r].sum()) for r in radii},
        # (c): how much of each mask sits within d px of the other one.
        "pred_near": {d: int((pred & near_true[d]).sum()) for d in distances},
        "true_near": {d: int((true & near_pred[d]).sum()) for d in distances},
        "skel_pred_near": {d: int((skel_pred & near_skel_true[d]).sum())
                          for d in distances},
        "skel_true_near": {d: int((skel_true & near_skel_pred[d]).sum())
                          for d in distances},
    }


def _decomposition_slot(radii: Sequence, distances: Sequence) -> dict:
    return {
        "tiles": 0, "skel_tp": 0, "skel_pred_px": 0, "skel_true_px": 0,
        "dil_tp": {r: 0 for r in radii},
        "dil_pred": {r: 0 for r in radii},
        "dil_true": {r: 0 for r in radii},
        "pred_near": {d: 0 for d in distances},
        "true_near": {d: 0 for d in distances},
        "skel_pred_near": {d: 0 for d in distances},
        "skel_true_near": {d: 0 for d in distances},
    }


def _accumulate_decomposition(slot: dict, counts: dict) -> None:
    for key in ("tiles", "skel_tp", "skel_pred_px", "skel_true_px"):
        slot[key] += counts[key]
    for key in ("dil_tp", "dil_pred", "dil_true", "pred_near", "true_near",
                "skel_pred_near", "skel_true_near"):
        for index, value in counts[key].items():
            slot[key][index] += value


def summarise_decomposition(slot: dict, radii: Sequence,
                            distances: Sequence) -> dict:
    """Turn accumulated counts into the three measurements, plus a verdict.

    Counts are pooled over the dataset and the metric computed once, exactly as
    :class:`MetricAccumulator` does -- so ``pixel_dice`` here is directly
    comparable to the Dice the training loop reported, rather than being a mean
    of per-tile Dices, which is a different and larger number.
    """
    pred_px, true_px = slot["dil_pred"][0], slot["dil_true"][0]
    pixel_dice = _safe_div(2 * slot["dil_tp"][0], pred_px + true_px)
    skeleton_dice = _safe_div(2 * slot["skel_tp"],
                             slot["skel_pred_px"] + slot["skel_true_px"])
    skeleton_pred_within = {d: _safe_div(slot["skel_pred_near"][d],
                                        slot["skel_pred_px"])
                            for d in distances}
    skeleton_true_within = {d: _safe_div(slot["skel_true_near"][d],
                                        slot["skel_true_px"])
                            for d in distances}

    # BOTH directions, read at the project's own boundary tolerance. One
    # direction alone cannot tell over-detection from misplacement: in both,
    # most of what was drawn is off the truth, and only "did the true curves
    # get found" separates them.
    tolerance = min(distances, key=lambda d: abs(d - 2)) if distances else 0
    real = skeleton_pred_within.get(tolerance, 0.0)   # what we drew IS boundary
    found = skeleton_true_within.get(tolerance, 0.0)  # we FOUND the boundary

    width_ratio = _safe_div(_safe_div(pred_px, slot["skel_pred_px"]),
                            _safe_div(true_px, slot["skel_true_px"]))
    # How many times too much CURVE was drawn, independent of how fat it is.
    # Algebraically identical to (pred_frac/true_frac) / (width_pred/width_true)
    # -- the area ratio divided by the width ratio -- but taken directly from
    # the skeleton lengths, so it does not compound three roundings.
    curve_ratio = _safe_div(slot["skel_pred_px"], slot["skel_true_px"])
    too_fat = width_ratio >= PLACEMENT_WIDTH_HIGH

    if found < PLACEMENT_FOUND_OK:
        label = "MISPLACED"
        verdict = ("MISPLACED -- the true curves were not found, so thickness "
                   "is not the problem and thinning the prediction would not "
                   "help")
    elif real < PLACEMENT_REAL_OK and skeleton_dice < PLACEMENT_STRUCTURE_OK:
        # "found" is satisfied but nothing structural underlies it. At high
        # boundary density a random pixel lands within tolerance of almost
        # anything, so coverage alone is not detection -- and this branch has
        # to sit INSIDE the low-`real` case, because a sub-tolerance offset
        # also has a near-zero skeleton Dice while being perfectly placed.
        label = "MISPLACED"
        verdict = (f"MISPLACED -- {found:.0%} of the true centreline has "
                   "something within tolerance of it, but the two skeletons "
                   f"barely overlap (Dice {skeleton_dice:.3f}) and only "
                   f"{real:.0%} of what was drawn is on a boundary. That is "
                   "coincidental coverage from predicting far too much, not "
                   "detection of the real curves")
    elif real < PLACEMENT_REAL_OK:
        label = "OVER-DETECTION" + (" + THICKNESS" if too_fat else "")
        verdict = (
            f"OVER-DETECTION -- every true curve was found ({found:.0%} of the "
            f"true centreline has a prediction on it) but only {real:.0%} of "
            f"what was drawn lies on a true boundary. The model is drawing "
            f"{curve_ratio:.1f}x too many curves"
            + (f", each {width_ratio:.1f}x too fat. Two separate problems."
               if too_fat else ". Thickness is not the issue here."))
    elif too_fat:
        label = "THICKNESS"
        verdict = (f"THICKNESS -- the curves are the right ones in the right "
                   f"places ({real:.0%} of the drawn centreline is on a true "
                   f"boundary) and are {width_ratio:.1f}x too fat")
    elif skeleton_dice < PLACEMENT_EXACT_OVERLAP_OK:
        label = "OFFSET"
        verdict = (f"OFFSET -- the curves are within {tolerance} px of the "
                   "truth but barely overlap it exactly (skeleton Dice "
                   f"{skeleton_dice:.3f}). A sub-tolerance registration shift, "
                   "not a placement failure and not thickness")
    else:
        label = "GOOD"
        verdict = ("the curves are the right ones, in the right places, at the "
                   "right width; the pixel Dice is not being lost to either")
    placement_ok = label != "MISPLACED"

    return {
        "label": label,
        "width_ratio": width_ratio,
        "curve_length_ratio": curve_ratio,
        "found": found,
        "real": real,
        "tolerance_px": tolerance,
        "tiles": slot["tiles"],
        "pred_px": pred_px,
        "true_px": true_px,
        # (a) thickness removed from both sides
        "pixel_dice": pixel_dice,
        "skeleton_dice": skeleton_dice,
        "skeleton_lift": (skeleton_dice / pixel_dice) if pixel_dice > 0 else float("inf"),
        "skeleton_pred_px": slot["skel_pred_px"],
        "skeleton_true_px": slot["skel_true_px"],
        # pooled area / skeleton length -- the same quantity
        # boundary_gt.measured_line_width computes per tile.
        "width_pred": _safe_div(pred_px, slot["skel_pred_px"]),
        "width_true": _safe_div(true_px, slot["skel_true_px"]),
        # (b) tolerance added symmetrically; r = 0 is the pixel Dice itself
        "dilated_dice": {r: _safe_div(2 * slot["dil_tp"][r],
                                     slot["dil_pred"][r] + slot["dil_true"][r])
                         for r in radii},
        # (c) proximity, both directions, on the masks and on the skeletons
        "pred_within": {d: _safe_div(slot["pred_near"][d], pred_px)
                        for d in distances},
        "true_within": {d: _safe_div(slot["true_near"][d], true_px)
                        for d in distances},
        "skeleton_pred_within": skeleton_pred_within,
        "skeleton_true_within": skeleton_true_within,
        "verdict": verdict,
        "placement_ok": placement_ok,
    }


@torch.no_grad()
def decompose_error(model: "nn.Module", val_ds, thresholds: dict, device,
                    amp_enabled: bool = False, dilations: Sequence = (1, 2, 3),
                    distances: Sequence = (1, 2, 3, 5),
                    batch_size: int = 32, num_workers: int = 0) -> dict:
    """Split a low pixel Dice into PLACEMENT error and THICKNESS error.

    A pixel Dice of 0.146 has two very different explanations that no single
    number can tell apart: boundaries drawn in the right place but far too
    thick, or boundaries drawn in the wrong place. They call for opposite
    responses -- the first is fixed at the operating point or with a thinner
    target, the second means the model has not learned where boundaries are --
    so this measures the two separately:

    (a) **Dice between the two SKELETONS**, thickness removed from both sides.
        Much higher than the pixel Dice means the pixel Dice is being spent on
        width, not position.
    (b) **Dice after dilating BOTH masks** by 1, 2, 3 px: tolerance added
        symmetrically. Read this one with care -- see
        :func:`reference_decomposition`, where pure noise still reaches 0.69 at
        k=3, because dilating both sides makes almost anything overlap at this
        boundary density.
    (c) **Proximity**: the fraction of predicted boundary pixels within d px of
        a true one, and the converse. These are the two halves of the boundary
        F-score swept over its tolerance, and the skeleton version of them is
        the sharpest of the three measurements here.

    Validation only: no gradient, no optimizer, nothing retrained.
    """
    from torch.utils.data import DataLoader

    # Keep k=0 internally as the ordinary pixel-Dice anchor.  Proximity is
    # deliberately only reported at the caller's requested distances: its
    # public table is the 1, 2, 3, 5 px sweep, not an extra exact-overlap row.
    radii = tuple(sorted({0} | {int(k) for k in dilations}))
    distances = tuple(sorted({int(d) for d in distances}))
    if any(k < 0 for k in radii) or any(d < 0 for d in distances):
        raise TrainError("dilation radii and proximity distances must be non-negative")
    if not distances:
        raise TrainError("pass at least one proximity distance to decompose_error")

    present = sorted({r["dataset"] for r in val_ds.rows})
    missing = sorted(set(present) - set(thresholds))
    if missing:
        raise TrainError(
            f"decompose_error has no threshold for {missing}; pass one for "
            "every dataset in the validation split.")

    model.eval()
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    slots, row_index = {}, 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].numpy()
        with autocast(device.type, amp_enabled):
            logits = model(images)
        probs = torch.sigmoid(logits.float()).cpu().numpy()

        for i in range(probs.shape[0]):
            row = val_ds.rows[row_index]
            dataset = row["dataset"]
            pred = probs[i, 0] >= float(thresholds[dataset])
            true = masks[i, 0] > 0.5
            slot = slots.setdefault(dataset, _decomposition_slot(radii, distances))
            _accumulate_decomposition(
                slot, decomposition_counts(pred, true, radii, distances))
            row_index += 1

    if row_index != len(val_ds):
        raise TrainError(
            f"decomposed {row_index} tiles but val_ds has {len(val_ds)}.")
    return {name: summarise_decomposition(slot, radii, distances)
            for name, slot in sorted(slots.items())}


def reference_decomposition(dilations: Sequence = (1, 2, 3),
                            distances: Sequence = (1, 2, 3, 5),
                            size: int = 256, seed: int = 0) -> dict:
    """The same measurements on synthetic cases whose answer is already known.

    Printed beside the real results so "much higher" and "near" have concrete
    reference points instead of being judged by eye. Every case is measured by
    the same :func:`decomposition_counts` the real evaluation uses, so the
    calibration cannot drift away from what it is calibrating.

    The cases bracket the two failure modes: a perfectly placed prediction that
    is merely 1-3 px too fat, a prediction of the right thickness shifted 2-5
    px off, and uniform noise at the same boundary density as a floor.
    """
    import cv2

    radii = tuple(sorted({0} | {int(k) for k in dilations}))
    distances = tuple(sorted({int(d) for d in distances}))
    if any(k < 0 for k in radii) or any(d < 0 for d in distances):
        raise TrainError("dilation radii and proximity distances must be non-negative")
    if not distances:
        raise TrainError("pass at least one proximity distance to reference_decomposition")

    # A 2-px grid, the width boundary_gt.line_width_px actually produces.
    step = size // 5
    offsets = list(range(size // 8, size - 2, step))
    true = np.zeros((size, size), dtype=bool)
    for offset in offsets:
        true[offset:offset + 2, :] = True
        true[:, offset:offset + 2] = True

    def fat(k):
        return cv2.dilate(true.astype(np.uint8),
                          euclidean_disk(k)).astype(bool)

    def shifted(px):
        """Displace the grid DIAGONALLY, so nothing lands on itself.

        A purely horizontal shift would leave the full-width horizontal lines
        overlapping themselves exactly, and the case would measure as half
        placed and half misplaced regardless of the shift distance -- which is
        a real mixed case, but useless as the "misplaced" reference row.
        """
        out = np.zeros_like(true)
        out[px:, px:] = true[:-px, :-px]
        return out

    def over_detected(multiple, also_fat=False):
        """The true curves PLUS extra real curves the annotation never marked.

        This is the case the calibration set was missing, and its absence is
        why the rule used to call over-detection "misplaced": the extra curves
        roughly multiply the skeleton length, skeleton Dice falls in proportion,
        and a rule reading skeleton Dice alone cannot tell "found the truth and
        drew more besides" from "did not find the truth".

        The extras are laid between the true lines, so they are genuinely
        elsewhere in the tile rather than a fattening of what is already there.
        ``multiple`` is how many extra lines go in each gap, giving roughly
        1x, 2x and 3x the true curve length in additions.
        """
        pred = true.copy()
        for offset in offsets:
            for i in range(1, int(multiple) + 1):
                position = offset + int(step * i / (multiple + 1))
                if position + 2 < size:
                    pred[position:position + 2, :] = True
                    pred[:, position:position + 2] = True
        if also_fat:
            pred = cv2.dilate(pred.astype(np.uint8),
                              euclidean_disk(1)).astype(bool)
        return pred

    rng = np.random.default_rng(seed)
    cases = {
        "perfect": true.copy(),
        "placed, 1 px too fat": fat(1),
        "placed, 2 px too fat": fat(2),
        "placed, 3 px too fat": fat(3),
        # The true curves plus 1x, 2x and 3x their length in extra curves --
        # at the true width, and again at twice the true width, because the
        # real failure on uhcs2 is both at once and the rule has to name both.
        "over-detected 1x": over_detected(1),
        "over-detected 2x": over_detected(2),
        "over-detected 3x": over_detected(3),
        "over-detected 1x, 2x too fat": over_detected(1, also_fat=True),
        "over-detected 2x, 2x too fat": over_detected(2, also_fat=True),
        "over-detected 3x, 2x too fat": over_detected(3, also_fat=True),
        # 2 px is INSIDE train.boundary_tolerance_px, so this is a registration
        # offset rather than a placement failure -- and the verdict says so,
        # which is the honest reading of a metric measured at 2 px tolerance.
        "displaced 2 px (in tolerance)": shifted(2),
        "misplaced by 5 px": shifted(5),
        "misplaced by 8 px": shifted(8),
        "noise at the same density": rng.random(true.shape) < true.mean(),
    }

    out = {}
    for name, pred in cases.items():
        slot = _decomposition_slot(radii, distances)
        _accumulate_decomposition(
            slot, decomposition_counts(pred, true, radii, distances))
        out[name] = summarise_decomposition(slot, radii, distances)
    return out


def _diff_config(saved: dict, current: dict) -> list:
    """Human-readable list of what changed between two hashed configs."""
    lines = []
    for section in sorted(set(saved) | set(current)):
        old_section = saved.get(section) or {}
        new_section = current.get(section) or {}
        for key in sorted(set(old_section) | set(new_section)):
            old, new = old_section.get(key, "<absent>"), new_section.get(key, "<absent>")
            if old != new:
                lines.append(f"{section}.{key}: {old!r} -> {new!r}")
    return lines or ["(the hashed payload differs but no key-level diff was "
                     "found; the config schema itself changed)"]


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
METRIC_ORDER = ("iou", "dice", "precision", "recall", "boundary_f")


def report_paths(run_name: str, platform: str,
                 reports_dir: Optional[Path] = None) -> tuple:
    """``reports/train_<run>_<platform>.{md,json}``, keyed by RUN and by HOST.

    ``run_name`` is the fold for an ordinary run and ``<fold>-no-<excluded>``
    for one training on a reduced mixture, so the two arms of an exclusion
    experiment produce two reports instead of overwriting each other. A run
    with no exclusions is named for its fold exactly as before.

    The platform is part of the filename because the same fold trained on two
    hosts produces two measurements, not one measurement and one mistake. They
    differ in ways that matter and that nothing else records: different GPUs,
    different worker counts, different Drive-versus-input-dataset I/O, and on
    Kaggle a working directory that does not survive the session. Writing both
    to ``train_dev.json`` made them collide -- on disk when a second host ran,
    and in git when the two branches met -- and the collision resolved by
    whichever ran last, silently discarding the other run.

    So the filename carries the host, exactly as ``configs/dataloader.yaml``
    keys its measurements by platform for the same reason.
    """
    reports_dir = Path(reports_dir) if reports_dir else Path(REPO_ROOT) / "reports"
    if not platform:
        raise TrainError(
            "no platform to key the report by; resolve_paths() supplies it and "
            "a report written without one would collide with the other host's.")
    stem = f"train_{run_name}_{platform}"
    return reports_dir / f"{stem}.md", reports_dir / f"{stem}.json"


def load_run_reports(fold: str, reports_dir: Optional[Path] = None,
                     platform: Optional[str] = None) -> dict:
    """Every committed report for ``fold``, keyed by ``(run_name, platform)``.

    Finds the arms of an exclusion experiment: ``train_dev_colab.json`` and
    ``train_dev-no-uhcs1_colab.json`` both belong to fold ``dev`` and are
    returned together. Matching is on the parsed stem rather than a glob, so a
    fold whose name is a prefix of another cannot pull in the wrong file.
    """
    reports_dir = Path(reports_dir) if reports_dir else Path(REPO_ROOT) / "reports"
    if not reports_dir.is_dir():
        raise TrainError(f"no reports directory at {reports_dir}")

    out = {}
    for path in sorted(reports_dir.glob("train_*.json")):
        stem = path.stem[len("train_"):]
        if "_" not in stem:
            continue
        run_name, _, host = stem.rpartition("_")
        if not (run_name == fold or run_name.startswith(f"{fold}-")):
            continue
        if platform is not None and host != platform:
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:                          # noqa: BLE001
            raise TrainError(
                f"{path} is not readable JSON ({type(exc).__name__}); a "
                "half-written report is worse than a missing one because it "
                "will be compared against as if it were real.") from exc
        payload["path"] = str(path)
        out[(run_name, host)] = payload
    return out


def compare_runs(reports: dict, dataset: str, row: str = "best") -> dict:
    """One dataset's metrics across several runs, side by side.

    ``reports`` comes from :func:`load_run_reports`. ``row`` selects the fixed
    or the tuned operating point; the tuned one is the default because it is
    what ``best.pt`` was selected on.

    The point of the exclusion experiment is that ``dataset`` is scored on the
    SAME validation tiles in every arm, so these numbers are directly
    comparable. That is asserted rather than assumed: a run whose validation
    tile count for this dataset differs from the others is reported as such,
    because it would mean the arms were not scored on the same thing.
    """
    rows, tile_counts = {}, {}
    for (run_name, host), payload in sorted(reports.items()):
        history = payload.get("history") or []
        if not history:
            continue
        best_epoch = (payload.get("best") or {}).get("epoch")
        record = next((h for h in history if h.get("epoch") == best_epoch),
                      history[-1])
        entry = (record.get("metrics", {}).get("per_dataset", {})
                 .get(dataset))
        if entry is None:
            continue
        metrics = entry[row]
        label = f"{run_name} ({host})"
        rows[label] = {
            "excluded": ", ".join(payload.get("excluded_datasets") or []) or "-",
            "epoch": record["epoch"],
            "threshold": metrics["threshold"],
            **{k: metrics[k] for k in METRIC_ORDER},
            "pred_frac": metrics["pred_fraction"],
            "true_frac": metrics["true_fraction"],
            "tiles": metrics["tiles"],
            # Reports written before the hash moved to the top level still
            # carry it inside summary; read either rather than showing None
            # for a run that does have one.
            "config_hash": (payload.get("config_hash")
                            or (payload.get("summary") or {}).get("config_hash")),
        }
        tile_counts[label] = metrics["tiles"]

    comparable = len(set(tile_counts.values())) <= 1
    return {
        "dataset": dataset,
        "row": row,
        "runs": rows,
        "comparable": comparable,
        "note": ("every arm scored this dataset on the same number of "
                 f"validation tiles ({next(iter(tile_counts.values()), 0)})"
                 if comparable else
                 f"THE ARMS ARE NOT COMPARABLE: tile counts differ {tile_counts}. "
                 "Validation must be identical across arms for the difference "
                 "to mean anything."),
    }


def diff_runs(reports: dict, left: tuple, right: tuple) -> list:
    """What actually differs between two runs' hashed configs.

    Two runs of an exclusion experiment SHOULD differ in exactly one key. If
    they differ in more, the comparison is confounded and this says by what --
    which is the difference between a measurement and a story.
    """
    payloads = {}
    for key in (left, right):
        if key not in reports:
            raise TrainError(
                f"no report for {key}; available: {sorted(reports)}")
        payloads[key] = reports[key].get("hashed_config")
    if not all(payloads.values()):
        missing = [k for k, v in payloads.items() if not v]
        return [f"(hashed_config absent for {missing}; it was written by a "
                "version that predates cross-run diffing, so what differed "
                "cannot be reconstructed from the report alone)"]
    return _diff_config(payloads[left], payloads[right])


def write_report(trainer: Trainer, reports_dir: Optional[Path] = None) -> tuple:
    """reports/train_<fold>_<platform>.{md,json}. This step's committed artefact."""
    reports_dir = Path(reports_dir or trainer.resolved["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary = trainer.summary()
    history = trainer.history
    if not history:
        raise TrainError("nothing to report: no epoch completed.")
    final = history[-1]
    best_epoch = next((h for h in history if h["epoch"] == trainer.best["epoch"]),
                      final)

    payload = {
        "fold": trainer.fold,
        "run_name": trainer.run_name,
        "excluded_datasets": list(trainer.excluded),
        "config_hash": trainer.hash,
        # Carried so that comparing two runs can DIFF what actually differed
        # between them, rather than reporting that the hashes are unequal and
        # leaving the reader to guess which key moved.
        "hashed_config": trainer.hashed_config,
        "platform": trainer.platform,
        "held_out": trainer.held_out,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summary,
        "best": trainer.best,
        "freeze_transitions": trainer.freeze_transitions,
        "final_epoch": final["epoch"],
        "history": history,
    }
    md_path, json_path = report_paths(trainer.run_name, trainer.platform,
                                      reports_dir)
    json_path.write_text(json.dumps(payload, indent=1, default=str))

    exclusion_note = (
        f" (TRAINED WITHOUT {', '.join(trainer.excluded)})"
        if trainer.excluded else "")
    lines = [
        f"# Training report -- {trainer.run_name} on {trainer.platform}"
        + exclusion_note,
        "",
        f"Held-out dataset: **{trainer.held_out}**. Generated "
        f"{payload['generated_utc']} on {summary['platform']} "
        f"({summary['gpu'] or summary['device']}).",
        "",
        f"This file is keyed by run and host: "
        f"`train_{trainer.run_name}_{trainer.platform}.md`. "
        "The same fold trained on another host writes its own file beside this "
        "one rather than overwriting it -- two runs of one fold are two "
        "measurements, and they differ in GPU, worker count and I/O path.",
        "",
        (f"- **training mixture: {', '.join(trainer.excluded)} EXCLUDED.** "
         f"Trained on {sorted({r['dataset'] for r in trainer.train_ds.rows})}, "
         f"validated unchanged on {sorted(summary['val_composition'])}. "
         f"pos_weight is held at the fold's recorded "
         f"{summary['pos_weight_drift']['recorded']:.3f} in both arms "
         f"(this split alone would imply "
         f"{summary['pos_weight_drift']['implied_by_this_split']:.3f}) so that "
         "the training data is the only thing that differs."
         if trainer.excluded else
         "- training mixture: the fold's full set, nothing excluded"),
        f"- config hash `{trainer.hash}`, seed {summary['seed']['seed']}"
        f" ({summary['seed']['note']})",
        f"- {summary['epochs']} epochs, batch {summary['batch_size']}, "
        f"{summary['num_workers']} workers ({summary['sources'].get('num_workers')})",
        f"- lr {summary['lr']} (encoder {summary['encoder_lr']}), "
        f"weight decay {summary['weight_decay']}, warmup "
        f"{summary['warmup_epochs']} epochs, grad clip {summary['grad_clip']}",
        f"- pos_weight {summary['pos_weight']} "
        f"({summary['sources'].get('pos_weight')})",
        f"- best epoch {trainer.best['epoch']} by best-threshold Dice on "
        f"`{trainer.best['key']}` = {trainer.best['metric']:.4f} "
        f"at threshold {trainer.best.get('threshold')}",
        f"- validation threshold swept over "
        f"{final['metrics']['thresholds'][0]:.2f}.."
        f"{final['metrics']['thresholds'][-1]:.2f} "
        f"({len(final['metrics']['thresholds'])} points); fixed reference "
        f"{final['metrics']['fixed_threshold']:.2f}",
        "",
        "## Per-dataset validation metrics (the headline)",
        "",
        "Validation is a mixture. The pooled row is a footnote; the held-out "
        "dataset's row is the measurement this fold exists to make.",
        "",
        "Every row appears twice: at the fixed `train.threshold` and at the "
        "threshold that maximised Dice for that dataset. A single fixed "
        "threshold measures the model and the operating point together and "
        "reports the sum as if it were the model. `best.pt` is selected on the "
        "best-threshold Dice of the held-out dataset.",
        "",
        "| epoch | dataset | thr | tiles | IoU | Dice | Precision | Recall | "
        "boundary-F | pred frac | true frac |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label, record in (("best", best_epoch), ("final", final)):
        entries = list(record["metrics"]["per_dataset"].items()) + [
            ("_pooled (footnote)_", record["metrics"]["pooled"])]
        for name, entry in entries:
            marker = " (held out)" if name == trainer.held_out else ""
            for row in ("fixed", "best"):
                metrics = entry[row]
                tag = "fixed" if row == "fixed" else "**best**"
                lines.append(
                    f"| {record['epoch']} ({label}) | {name}{marker} {tag} | "
                    f"{metrics['threshold']:.2f} | {metrics['tiles']} | "
                    + " | ".join(f"{metrics[k]:.4f}" for k in METRIC_ORDER)
                    + f" | {metrics['pred_fraction']:.4f} "
                    f"| {metrics['true_fraction']:.4f} |")

    lines += [
        "",
        "## The chosen threshold, per epoch, per dataset",
        "",
        "This table is a measurement, not bookkeeping. If the held-out "
        "dataset's optimal threshold sits far from the training-side "
        "datasets', that gap IS the domain shift, expressed in the units of "
        "the decision the downstream watershed has to make -- and one global "
        "threshold will not serve both.",
        "",
        "| epoch | " + " | ".join(
            sorted(final["metrics"]["per_dataset"])) + " | spread |",
        "| --- | " + " | ".join(
            "---" for _ in final["metrics"]["per_dataset"]) + " | --- |",
    ]
    names = sorted(final["metrics"]["per_dataset"])
    for record in history:
        chosen = record["best_thresholds"]
        values = [chosen.get(name) for name in names]
        present = [v for v in values if v is not None]
        spread = (max(present) - min(present)) if len(present) > 1 else 0.0
        lines.append(
            f"| {record['epoch']} | "
            + " | ".join("-" if v is None else f"{v:.2f}" for v in values)
            + f" | {spread:.2f} |")
    if trainer.held_out in names and len(names) > 1:
        final_chosen = final["best_thresholds"]
        held = final_chosen.get(trainer.held_out)
        others = {k: v for k, v in final_chosen.items() if k != trainer.held_out}
        if held is not None and others:
            gap = max(abs(held - v) for v in others.values())
            lines += [
                "",
                f"**Final epoch.** {trainer.held_out} (held out) wants "
                f"{held:.2f}; the others want "
                + ", ".join(f"{k} {v:.2f}" for k, v in sorted(others.items()))
                + f" -- a gap of {gap:.2f}. "
                + ("That is a domain-shift finding: the held-out microscope "
                   "needs a materially different operating point, so step 7 "
                   "should set the threshold PER DATASET rather than globally."
                   if gap >= 0.15 else
                   "Close enough that one global threshold serves both, which "
                   "is the easy case for step 7."),
            ]

    lines += [
        "",
        "## Loss terms, separately",
        "",
        "They differ by orders of magnitude, so the total alone does not say "
        "which one moved.",
        "",
        "| epoch | train total | train BCE | train Dice | train clDice | "
        "val total | val BCE | val Dice | val clDice |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in history:
        lines.append(
            f"| {record['epoch']} | "
            + " | ".join(f"{record['train'][k]:.4f}"
                         for k in ("total", "bce", "dice", "cldice"))
            + " | "
            + " | ".join(f"{record['val'][k]:.4f}"
                         for k in ("total", "bce", "dice", "cldice"))
            + " |")

    probe_final = final["cldice_probe"]
    degenerate_epochs = sum(1 for h in history if h["cldice_probe"]["degenerate"])
    lines += [
        "",
        "## clDice diagnostic on real predictions",
        "",
        "Step 5 measured clDice against perturbed ground truth. This measures "
        "it against what the model actually produces, which is soft and "
        "thicker than the target. `skeleton_delta` is "
        "`mean |soft_skeleton(sigmoid(logits)) - sigmoid(logits)|`: near zero "
        "means the soft skeleton is returning its input and the term is inert.",
        "",
        "| epoch | skel(pred) | skel(true) | skel(pred) on true | t_prec | "
        "t_rec | skeleton delta | Dice | clDice | verdict |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in history:
        p = record["cldice_probe"]
        lines.append(
            f"| {record['epoch']} | {p['skel_pred_sum']:.0f} | "
            f"{p['skel_true_sum']:.0f} | {p['skel_pred_on_true']:.0f} | "
            f"{p['t_prec']:.4f} | {p['t_rec']:.4f} | "
            f"{p['skeleton_delta']:.6f} | {p['dice']:.4f} | "
            f"{p['cldice']:.4f} | "
            f"{'DEGENERATE' if p['degenerate'] else 'active'} |")
    lines += [
        "",
        f"**Finding.** {degenerate_epochs} of {len(history)} epochs came back "
        f"degenerate. Final verdict: {probe_final['verdict']}.",
        "",
        "If this says degenerate, `loss.w_cldice` has been buying nothing and "
        "the honest response is to record that here, not to retune the weight. "
        "The levers that would change it are a thicker "
        "`boundary_gt.line_width_px` (re-running steps 2 and 3) or an explicit "
        "connectivity metric at evaluation time.",
        "",
    ]
    md_path.write_text("\n".join(lines) + "\n")
    return md_path, json_path
