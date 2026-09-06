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
    "threshold": 0.5,
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
    if int(settings["warmup_epochs"]) >= int(settings["epochs"]):
        raise TrainError(
            f"train.warmup_epochs ({settings['warmup_epochs']}) must be less "
            f"than train.epochs ({settings['epochs']}); otherwise the cosine "
            "phase never runs.")
    return settings


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


def boundary_counts(pred: np.ndarray, true: np.ndarray,
                    tolerance_px: int = 2) -> tuple:
    """Counts for the boundary F-score with a tolerance (Csurka et al.).

    A predicted boundary pixel counts as correct if a TRUE boundary pixel lies
    within ``tolerance_px``, and vice versa. Two pixels of slack is the right
    amount here because the ground truth was skeletonized and re-dilated to a
    uniform 2 px in step 2 -- its exact placement is accurate to about that,
    so scoring it at exactly one pixel would be measuring the convention.

    Returns ``(hits_precision, n_pred, hits_recall, n_true)``.
    """
    import cv2

    pred = np.ascontiguousarray(pred > 0).astype(np.uint8)
    true = np.ascontiguousarray(true > 0).astype(np.uint8)
    n_pred, n_true = int(pred.sum()), int(true.sum())
    if n_pred == 0 and n_true == 0:
        return 0, 0, 0, 0
    tol = float(tolerance_px)

    # distanceTransform measures the distance to the nearest ZERO pixel, so the
    # mask is inverted: dist_to_true[y, x] is how far (y, x) is from the true
    # boundary. An empty mask gives large distances everywhere, which scores 0
    # hits -- correct, and finite.
    if n_true:
        dist_to_true = cv2.distanceTransform(1 - true, cv2.DIST_L2, 3)
        hits_p = int((pred.astype(bool) & (dist_to_true <= tol)).sum())
    else:
        hits_p = 0
    if n_pred:
        dist_to_pred = cv2.distanceTransform(1 - pred, cv2.DIST_L2, 3)
        hits_r = int((true.astype(bool) & (dist_to_pred <= tol)).sum())
    else:
        hits_r = 0
    return hits_p, n_pred, hits_r, n_true


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
    """Pixel and boundary counts, kept per dataset AND pooled.

    Counts are accumulated over pixels and the metric computed once at the end,
    rather than averaging per-tile metrics. Per-tile averaging gives a nearly
    empty tile the same weight as a dense one and makes Dice jump around on
    tiles with a handful of boundary pixels; pooling the counts weights each
    dataset by how much boundary it actually contains.
    """

    POOLED = "__pooled__"

    def __init__(self, threshold: float = 0.5, tolerance_px: int = 2):
        self.threshold = float(threshold)
        self.tolerance_px = int(tolerance_px)
        self.counts = {}

    def _slot(self, key: str) -> dict:
        return self.counts.setdefault(key, {
            "tp": 0, "fp": 0, "fn": 0, "tiles": 0, "total_px": 0,
            "true_px": 0, "pred_px": 0, "prob_sum": 0.0,
            "prob_min": float("inf"), "prob_max": float("-inf"),
            "bf_hits_p": 0, "bf_n_pred": 0, "bf_hits_r": 0, "bf_n_true": 0,
        })

    def update(self, probs, targets, dataset_names: Sequence[str]) -> None:
        """One batch. ``probs`` are probabilities, ``targets`` are 0/1."""
        probs = np.asarray(probs, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        if probs.shape != targets.shape:
            raise TrainError(
                f"probs {probs.shape} and targets {targets.shape} disagree")
        if len(dataset_names) != probs.shape[0]:
            raise TrainError(
                f"{len(dataset_names)} dataset names for {probs.shape[0]} tiles")

        for i, name in enumerate(dataset_names):
            prob = probs[i].reshape(probs.shape[-2], probs.shape[-1])
            true = targets[i].reshape(prob.shape) > 0.5
            pred = prob >= self.threshold
            tp = int((pred & true).sum())
            fp = int((pred & ~true).sum())
            fn = int((~pred & true).sum())
            hits_p, n_pred, hits_r, n_true = boundary_counts(
                pred, true, self.tolerance_px)
            for key in (name, self.POOLED):
                slot = self._slot(key)
                slot["tp"] += tp
                slot["fp"] += fp
                slot["fn"] += fn
                slot["tiles"] += 1
                slot["total_px"] += int(prob.size)
                slot["true_px"] += int(true.sum())
                slot["pred_px"] += int(pred.sum())
                slot["prob_sum"] += float(prob.sum())
                slot["prob_min"] = min(slot["prob_min"], float(prob.min()))
                slot["prob_max"] = max(slot["prob_max"], float(prob.max()))
                slot["bf_hits_p"] += hits_p
                slot["bf_n_pred"] += n_pred
                slot["bf_hits_r"] += hits_r
                slot["bf_n_true"] += n_true

    def result(self) -> dict:
        """``{"pooled": {...}, "per_dataset": {name: {...}}}``.

        The per-dataset breakdown is the headline. The pooled number is a
        footnote and is labelled as one everywhere it is printed.
        """
        if not self.counts:
            raise TrainError("no batches were accumulated; validation ran on "
                             "an empty loader.")
        per_dataset = {name: metrics_from_counts(counts)
                       for name, counts in sorted(self.counts.items())
                       if name != self.POOLED}
        return {
            "pooled": metrics_from_counts(self.counts[self.POOLED]),
            "per_dataset": per_dataset,
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
        self.run_name = run_name or self.fold

        self.hash, self.hashed_config = config_hash(
            self.model_settings, self.loss_settings, self.settings,
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
        self.best = {"metric": float("-inf"), "epoch": None, "key": None}
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

        self.train_ds = ds.TileDataset.from_manifest(
            manifest, "train", settings=self.dataset_settings, crops=crops,
            roots=roots)
        self.val_ds = ds.TileDataset.from_manifest(
            manifest, "val", settings=self.dataset_settings, crops=crops,
            roots=roots)
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

    def summary(self) -> dict:
        counts = self._model_mod.summarize(self.model) if self.model else {}
        return {
            "fold": self.fold,
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
        }

    def save(self, epoch: int, is_best: bool = False) -> dict:
        """Write last.pt, and best.pt when this epoch is the best so far."""
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
            threshold=float(self.settings["threshold"]),
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
        key = self.best_key()
        per_dataset = metrics["per_dataset"]
        if key in per_dataset:
            return float(per_dataset[key]["dice"]), key
        return float(metrics["pooled"]["dice"]), "pooled (held-out dataset absent)"

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
                self.best = {"metric": score, "epoch": epoch, "key": score_key}

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
        for key, value in record["metrics"]["pooled"].items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"val_pooled/{key}", value, epoch)
        for name, metrics in record["metrics"]["per_dataset"].items():
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"val_{name}/{key}", value, epoch)
        for key in ("skeleton_delta", "skel_pred_sum", "skel_true_sum",
                    "t_prec", "t_rec", "cldice", "dice"):
            self.writer.add_scalar(f"cldice_probe/{key}",
                                   record["cldice_probe"][key], epoch)
        self.writer.add_scalar("schedule/lr", record["lr"], epoch)
        self.writer.add_scalar("schedule/trainable_params",
                               record["trainable_params"], epoch)

    # -- figures ----------------------------------------------------------
    @torch.no_grad()
    def example_predictions(self, per_dataset: int = 1) -> list:
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
            out.append({
                "dataset": name,
                "tile_id": sample["tile_id"],
                "image": sample["image"][0],
                "truth": sample["mask"][0],
                "prob": prob,
                "pred": (prob >= float(self.settings["threshold"])),
            })
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


def write_report(trainer: Trainer, reports_dir: Optional[Path] = None) -> tuple:
    """reports/train_<fold>.{json,md}. The committed artefact of this step."""
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
        "held_out": trainer.held_out,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summary,
        "best": trainer.best,
        "freeze_transitions": trainer.freeze_transitions,
        "final_epoch": final["epoch"],
        "history": history,
    }
    json_path = reports_dir / f"train_{trainer.fold}.json"
    json_path.write_text(json.dumps(payload, indent=1, default=str))

    lines = [
        f"# Training report -- {trainer.fold}",
        "",
        f"Held-out dataset: **{trainer.held_out}**. Generated "
        f"{payload['generated_utc']} on {summary['platform']} "
        f"({summary['gpu'] or summary['device']}).",
        "",
        f"- config hash `{trainer.hash}`, seed {summary['seed']['seed']}"
        f" ({summary['seed']['note']})",
        f"- {summary['epochs']} epochs, batch {summary['batch_size']}, "
        f"{summary['num_workers']} workers ({summary['sources'].get('num_workers')})",
        f"- lr {summary['lr']} (encoder {summary['encoder_lr']}), "
        f"weight decay {summary['weight_decay']}, warmup "
        f"{summary['warmup_epochs']} epochs, grad clip {summary['grad_clip']}",
        f"- pos_weight {summary['pos_weight']} "
        f"({summary['sources'].get('pos_weight')})",
        f"- best epoch {trainer.best['epoch']} by Dice on "
        f"`{trainer.best['key']}` = {trainer.best['metric']:.4f}",
        "",
        "## Per-dataset validation metrics (the headline)",
        "",
        "Validation is a mixture. The pooled row is a footnote; the held-out "
        "dataset's row is the measurement this fold exists to make.",
        "",
        "| epoch | dataset | tiles | IoU | Dice | Precision | Recall | boundary-F |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label, record in (("best", best_epoch), ("final", final)):
        for name, metrics in record["metrics"]["per_dataset"].items():
            marker = " (held out)" if name == trainer.held_out else ""
            lines.append(
                f"| {record['epoch']} ({label}) | {name}{marker} | "
                f"{metrics['tiles']} | " + " | ".join(
                    f"{metrics[k]:.4f}" for k in METRIC_ORDER) + " |")
        pooled = record["metrics"]["pooled"]
        lines.append(
            f"| {record['epoch']} ({label}) | _pooled (footnote)_ | "
            f"{pooled['tiles']} | " + " | ".join(
                f"{pooled[k]:.4f}" for k in METRIC_ORDER) + " |")

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
    md_path = reports_dir / f"train_{trainer.fold}.md"
    md_path.write_text("\n".join(lines) + "\n")
    return md_path, json_path
