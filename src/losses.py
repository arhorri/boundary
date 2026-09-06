"""Compound boundary loss: weighted BCE + soft Dice + clDice.

    L = w_bce * BCEWithLogits(pos_weight) + w_dice * SoftDice + w_cldice * clDice

Every term consumes RAW LOGITS and does its own thing with them. BCE uses the
fused ``binary_cross_entropy_with_logits``, which is the numerically stable
form; Dice and clDice apply their own sigmoid internally. The logits are
deliberately NOT sigmoided once and handed to all three: doing that forces BCE
onto the unfused path where ``log(0)`` is reachable, and the gradient at
saturation is then computed from a probability that has already lost the
precision the fused form keeps.

Why three terms.

* **Weighted BCE** is what makes the problem learnable at all. Boundaries are
  4.8%-15.4% of pixels depending on the fold. Unweighted, the minimum of the
  first few epochs is "predict background everywhere": it is a local optimum
  the optimizer reaches immediately and leaves slowly. ``pos_weight`` is
  ``n_negative / n_positive`` on THAT FOLD's train split, read from
  configs/fold_stats.yaml -- 15.974 for fold_MetalDam against 7.111 for
  fold_uhcs2, a factor of 2.2. One constant would over-weight two folds and
  under-weight the third, so there is no default here and an unknown fold
  raises.
* **Dice** is per-image and scale-free: it measures overlap as a fraction of
  what is there, so a tile with few boundary pixels still produces a gradient
  of the same magnitude as a dense one. BCE, being a per-pixel mean, does not.
* **clDice** compares each mask against the other's soft SKELETON, and on this
  data what that buys is measured in notebook 05 rather than assumed. It is
  not extra sensitivity to gaps: what it removes is Dice's sensitivity to
  THICKNESS. Dilating every boundary by one pixel doubles the predicted pixel
  count and costs Dice ~0.34-0.39; it costs clDice ~0.007 and ~0.000 on the
  same two tiles, because a dilated line has the same centreline. That matters
  because thickness is the one property of this ground truth that is
  arbitrary -- step 2 dilated everything to a uniform
  ``boundary_gt.line_width_px = 2``, so the width is a convention, not a
  measurement, and a loss dominated by it is optimising a convention. Adding
  clDice to Dice therefore reweights the loss away from width and towards
  where the line runs.

Numerical safety. Every ratio is smoothed with ``smooth`` (default 1.0), NOT
with ``eps``. An epsilon of 1e-6 is not enough and the failure is quiet: a
confident-background prediction is ``sigmoid(-20) = 2e-9`` per pixel, which
over a 256x256 tile sums to 1.3e-4 -- a hundred times larger than a 1e-6
epsilon. On a legitimately empty val tile that turns ``(0 + eps) / (1.3e-4 +
eps)`` into a Dice loss of ~0.99 for a prediction that is correct. With
``smooth = 1.0`` the same case gives ~1e-4, and on a real tile with thousands
of boundary pixels a constant of 1 changes nothing. ``eps`` remains only as a
division guard inside the clDice harmonic mean.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class LossError(RuntimeError):
    """Raised when a loss cannot be built or applied. Never fails silently."""


try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - hosts ship torch
    raise LossError(
        "torch is not importable. Colab and Kaggle ship it preinstalled; this "
        "module is meant to run on a host, never on a local machine."
    ) from exc


# --------------------------------------------------------------------------
# defaults -- overridable from configs/default.yaml under ``loss:``
# --------------------------------------------------------------------------
DEFAULTS = {
    "w_bce": 1.0,
    "w_dice": 1.0,
    "w_cldice": 0.5,      # the topology term supports the others, not vice versa
    # Soft-skeletonize iterations. Each one is an erode + an open, so the cost
    # is linear and the reachable skeleton depth is bounded by it.
    "cldice_iters": 3,
    # O(1), not O(eps) -- see the module docstring. This is the difference
    # between "empty prediction on an empty tile scores 0" and "scores 0.99".
    "smooth": 1.0,
    "eps": 1e-6,          # division guard in the clDice harmonic mean only
}


def load_config(config_path: Optional[Path] = None) -> dict:
    """Merge ``loss:`` from configs/default.yaml over DEFAULTS."""
    from src import paths as paths_mod

    cfg = paths_mod.load_config(config_path)
    settings = dict(DEFAULTS)
    section = cfg.get("loss") or {}
    if not isinstance(section, dict):
        raise LossError(
            f"configs/default.yaml: loss must be a mapping, got "
            f"{type(section).__name__}")
    unknown = set(section) - set(DEFAULTS)
    if unknown:
        raise LossError(
            f"configs/default.yaml: unknown loss keys {sorted(unknown)}; "
            f"known keys are {sorted(DEFAULTS)}")
    for key, value in section.items():
        if value is not None:
            settings[key] = value
    for key in ("w_bce", "w_dice", "w_cldice", "smooth", "eps"):
        if float(settings[key]) < 0:
            raise LossError(f"loss.{key} must be >= 0, got {settings[key]}")
    if int(settings["cldice_iters"]) < 1:
        raise LossError(
            f"loss.cldice_iters must be >= 1, got {settings['cldice_iters']}")
    if float(settings["w_bce"]) + float(settings["w_dice"]) \
            + float(settings["w_cldice"]) <= 0:
        raise LossError("all three loss weights are 0; there is nothing to train on.")
    return settings


# --------------------------------------------------------------------------
# pos_weight -- per fold, never a constant
# --------------------------------------------------------------------------
def fold_pos_weight(fold: str, fold_stats: Optional[dict] = None) -> float:
    """``n_negative / n_positive`` on ``fold``'s train split, from step 3.

    There is deliberately no default and no fallback. The three folds differ by
    a factor of 2.2 because leave-one-dataset-out changes what is trained on;
    a wrong pos_weight does not raise, it just trains a worse model, so an
    unknown fold has to be an error.
    """
    from src import dataset as ds

    stats = fold_stats if fold_stats is not None else ds.load_fold_stats()
    folds = stats.get("folds") or {}
    if fold not in folds:
        raise LossError(
            f"fold {fold!r} is not in configs/fold_stats.yaml. Available: "
            f"{sorted(folds)}. pos_weight is a property of the fold being "
            "trained and there is no value to fall back to.")
    value = (folds[fold] or {}).get("pos_weight")
    if value is None:
        raise LossError(
            f"fold {fold!r} exists in configs/fold_stats.yaml but has no "
            "pos_weight. Re-run notebooks/03_tiling.ipynb.")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise LossError(
            f"fold {fold!r} has pos_weight {value}, which cannot be used. "
            "Re-run notebooks/03_tiling.ipynb.")
    return value


# --------------------------------------------------------------------------
# shape / value validation
# --------------------------------------------------------------------------
def _check(logits: "torch.Tensor", target: "torch.Tensor") -> None:
    if logits.shape != target.shape:
        raise LossError(
            f"logits {tuple(logits.shape)} and target {tuple(target.shape)} "
            "must have the same shape.")
    if logits.dim() != 4:
        raise LossError(
            f"expected NCHW tensors, got {logits.dim()} dimensions "
            f"{tuple(logits.shape)}.")
    if logits.shape[1] != 1:
        raise LossError(
            f"this is a binary boundary loss; expected 1 channel, got "
            f"{logits.shape[1]}.")
    if not torch.is_floating_point(logits) or not torch.is_floating_point(target):
        raise LossError(
            f"logits ({logits.dtype}) and target ({target.dtype}) must both be "
            "floating point.")
    # Cheap range check every call. The full binarity check is in the tests;
    # a target outside [0, 1] here means a mask that was interpolated, which is
    # exactly the failure src/dataset.py exists to prevent.
    tmin, tmax = float(target.min()), float(target.max())
    if tmin < 0.0 or tmax > 1.0:
        raise LossError(
            f"target values run {tmin} .. {tmax}; a boundary mask must be in "
            "[0, 1]. A mask outside it has been interpolated or normalized.")


def _flatten(x: "torch.Tensor") -> "torch.Tensor":
    """N x (C*H*W) -- every term reduces per sample, then averages over the batch."""
    return x.reshape(x.shape[0], -1)


# --------------------------------------------------------------------------
# soft morphology -- the differentiable skeleton clDice needs
# --------------------------------------------------------------------------
def soft_erode(x: "torch.Tensor") -> "torch.Tensor":
    """Min-filter, as the min of a 3x1 and a 1x3 min-pool (Shit et al., 2021).

    ``-max_pool(-x)`` IS the min-pool, and max_pool is differentiable, which is
    the whole reason erosion can appear in a loss at all.
    """
    p1 = -F.max_pool2d(-x, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
    p2 = -F.max_pool2d(-x, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
    return torch.min(p1, p2)


def soft_dilate(x: "torch.Tensor") -> "torch.Tensor":
    """Max-filter over a 3x3 neighbourhood."""
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def soft_open(x: "torch.Tensor") -> "torch.Tensor":
    return soft_dilate(soft_erode(x))


def soft_skeletonize(x: "torch.Tensor", iters: int) -> "torch.Tensor":
    """Differentiable skeleton: what survives repeated erosion, accumulated.

    What this does to a TWO-pixel line -- the width
    ``boundary_gt.line_width_px = 2`` produces -- depends on whether the line
    is isolated or part of a network, and the difference is not a detail:

    * **An isolated 2 px line comes back unchanged.** One erosion removes it
      (the 3x1 min-pool at every line pixel reaches a background row), so
      ``open(x)`` is empty and ``skel = relu(x - 0) = x``.
    * **A JUNCTION does not.** Where two 2 px lines cross, the shape is three
      or more pixels thick in both directions, so erosion survives there, the
      opening comes back as a block around the crossing, and
      ``relu(x - open(x))`` CARVES A HOLE out of the skeleton at every
      junction. Later iterations restore the eroded core but not the whole
      hole.

    So on a real boundary network the skeleton is the mask minus a patch at
    each junction -- neither "the mask itself" nor a thinned centreline.
    notebooks/05 measures this directly (input, skeleton, pixel counts, max
    absolute difference) instead of reasoning about it, and
    tests/test_losses.py pins both halves.
    """
    if int(iters) < 1:
        raise LossError(f"cldice_iters must be >= 1, got {iters}")
    opened = soft_open(x)
    skel = F.relu(x - opened)
    for _ in range(int(iters)):
        x = soft_erode(x)
        opened = soft_open(x)
        delta = F.relu(x - opened)
        # Union, kept differentiable: skel + delta - skel*delta on [0, 1].
        skel = skel + F.relu(delta - skel * delta)
    return skel


# --------------------------------------------------------------------------
# the three terms -- each takes LOGITS
# --------------------------------------------------------------------------
def bce_term(logits: "torch.Tensor", target: "torch.Tensor",
             pos_weight: "torch.Tensor") -> "torch.Tensor":
    """Class-weighted BCE on the raw logits, per sample then averaged."""
    per_pixel = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="none")
    return _flatten(per_pixel).mean(dim=1).mean()


def dice_term(logits: "torch.Tensor", target: "torch.Tensor",
              smooth: float = 1.0) -> "torch.Tensor":
    """1 - soft Dice on probabilities. Empty vs empty is 0, never NaN."""
    probs = _flatten(torch.sigmoid(logits))
    truth = _flatten(target)
    intersection = (probs * truth).sum(dim=1)
    denom = probs.sum(dim=1) + truth.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return (1.0 - dice).mean()


def cldice_term(logits: "torch.Tensor", target: "torch.Tensor",
                iters: int = 3, smooth: float = 1.0,
                eps: float = 1e-6) -> "torch.Tensor":
    """1 - clDice: the harmonic mean of topology precision and recall.

    ``t_prec`` asks how much of the PREDICTION's skeleton lies on the true
    boundary; ``t_rec`` asks how much of the TRUE skeleton is covered by the
    prediction. A break in a line costs recall, because the true centreline
    runs straight through the gap.
    """
    probs = torch.sigmoid(logits)
    skel_pred = soft_skeletonize(probs, iters)
    skel_true = soft_skeletonize(target, iters)

    sp, st = _flatten(skel_pred), _flatten(skel_true)
    p, t = _flatten(probs), _flatten(target)

    t_prec = ((sp * t).sum(dim=1) + smooth) / (sp.sum(dim=1) + smooth)
    t_rec = ((st * p).sum(dim=1) + smooth) / (st.sum(dim=1) + smooth)
    cl_dice = 2.0 * t_prec * t_rec / (t_prec + t_rec + eps)
    return (1.0 - cl_dice).mean()


def cldice_parts(logits: "torch.Tensor", target: "torch.Tensor",
                 iters: int = 3, smooth: float = 1.0,
                 eps: float = 1e-6) -> dict:
    """Every intermediate quantity :func:`cldice_term` is built from.

    Diagnostics only -- detached floats, averaged over the batch. It exists so
    that a notebook or a test can say WHY clDice returned what it returned
    without re-implementing the term next to it and drifting from it.

    The skeleton sums are the ones that matter when a clDice loss comes out at
    exactly 0: a term whose two skeletons are both non-empty and overlapping is
    genuinely indifferent to the difference between the masks, whereas one
    whose predicted skeleton is EMPTY returns ``t_prec = smooth / smooth = 1``
    and is merely degenerate. Those two look identical from the loss value
    alone and are not the same thing at all.
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        skel_pred = soft_skeletonize(probs, iters)
        skel_true = soft_skeletonize(target, iters)
        sp, st = _flatten(skel_pred), _flatten(skel_true)
        p, t = _flatten(probs), _flatten(target)
        t_prec = ((sp * t).sum(dim=1) + smooth) / (sp.sum(dim=1) + smooth)
        t_rec = ((st * p).sum(dim=1) + smooth) / (st.sum(dim=1) + smooth)
        cl = 2.0 * t_prec * t_rec / (t_prec + t_rec + eps)
        return {
            "pred_sum": float(p.sum(dim=1).mean()),
            "true_sum": float(t.sum(dim=1).mean()),
            "skel_pred_sum": float(sp.sum(dim=1).mean()),
            "skel_true_sum": float(st.sum(dim=1).mean()),
            "skel_pred_on_true": float((sp * t).sum(dim=1).mean()),
            "skel_true_on_pred": float((st * p).sum(dim=1).mean()),
            "t_prec": float(t_prec.mean()),
            "t_rec": float(t_rec.mean()),
            "cldice": float(cl.mean()),
            "loss": float((1.0 - cl).mean()),
        }


# --------------------------------------------------------------------------
# the compound loss
# --------------------------------------------------------------------------
class BoundaryLoss(nn.Module):
    """``w_bce * BCE(pos_weight) + w_dice * Dice + w_cldice * clDice``.

    ``pos_weight`` is a registered buffer, so ``loss.to(device)`` and
    ``loss.cuda()`` move it with the module and it can never end up on the
    wrong device halfway through a run.

    ``forward`` returns the scalar total, so this is a drop-in criterion.
    :meth:`components` returns every term separately, all still differentiable,
    for logging and for notebook 05's demo.
    """

    def __init__(self, pos_weight: float, settings: Optional[dict] = None,
                 fold: Optional[str] = None):
        super().__init__()
        self.settings = settings or load_config()
        pos_weight = float(pos_weight)
        if not np.isfinite(pos_weight) or pos_weight <= 0:
            raise LossError(
                f"pos_weight must be finite and positive, got {pos_weight}.")
        self.fold = fold
        self.register_buffer("pos_weight", torch.tensor(pos_weight,
                                                        dtype=torch.float32))
        self.w_bce = float(self.settings["w_bce"])
        self.w_dice = float(self.settings["w_dice"])
        self.w_cldice = float(self.settings["w_cldice"])
        self.cldice_iters = int(self.settings["cldice_iters"])
        self.smooth = float(self.settings["smooth"])
        self.eps = float(self.settings["eps"])

    @classmethod
    def for_fold(cls, fold: str, settings: Optional[dict] = None,
                 fold_stats: Optional[dict] = None) -> "BoundaryLoss":
        """Build the criterion for ONE fold, with that fold's pos_weight."""
        return cls(fold_pos_weight(fold, fold_stats=fold_stats),
                   settings=settings, fold=fold)

    def components(self, logits: "torch.Tensor",
                   target: "torch.Tensor") -> dict:
        """Every term, unweighted and weighted, plus the total. All differentiable."""
        _check(logits, target)
        target = target.to(logits.dtype)
        bce = bce_term(logits, target, self.pos_weight.to(logits.dtype))
        dice = dice_term(logits, target, smooth=self.smooth)
        cldice = cldice_term(logits, target, iters=self.cldice_iters,
                             smooth=self.smooth, eps=self.eps)
        total = self.w_bce * bce + self.w_dice * dice + self.w_cldice * cldice
        return {
            "bce": bce,
            "dice": dice,
            "cldice": cldice,
            "bce_weighted": self.w_bce * bce,
            "dice_weighted": self.w_dice * dice,
            "cldice_weighted": self.w_cldice * cldice,
            "total": total,
        }

    def forward(self, logits: "torch.Tensor",
                target: "torch.Tensor") -> "torch.Tensor":
        return self.components(logits, target)["total"]

    def describe(self) -> str:
        return (f"BoundaryLoss(fold={self.fold!r}, "
                f"pos_weight={float(self.pos_weight):.3f}, "
                f"w_bce={self.w_bce}, w_dice={self.w_dice}, "
                f"w_cldice={self.w_cldice}, cldice_iters={self.cldice_iters}, "
                f"smooth={self.smooth})")


def build_loss(fold: str, settings: Optional[dict] = None,
               fold_stats: Optional[dict] = None) -> BoundaryLoss:
    """The one entry point a training loop should use."""
    return BoundaryLoss.for_fold(fold, settings=settings, fold_stats=fold_stats)


# --------------------------------------------------------------------------
# perturbations -- used by tests/test_losses.py and notebooks/05
# --------------------------------------------------------------------------
def cut_gaps(mask: np.ndarray, spacing: int = 180, gap: int = 3) -> np.ndarray:
    """Break the boundary by erasing a ``gap`` x ``gap`` block periodically.

    ``spacing`` is counted in BOUNDARY PIXELS in raster order, not in image
    pixels, so the same call produces a comparable number of breaks on a dense
    tile and a sparse one.

    ``gap`` is 3 and not 1 on purpose, and the reason matters: the ground truth
    is dilated to ``boundary_gt.line_width_px = 2``, and erasing a single pixel
    from a 2 px line leaves the other row intact -- the line is still
    connected, and nothing has been demonstrated. A block at least as wide as
    the line is what actually severs it. The gap is still one pixel LONG along
    the line; it is one line-width ACROSS it.

    Deterministic: no randomness, so the tests and the notebook see the same
    perturbation.
    """
    mask = np.asarray(mask)
    binary = mask > 0
    if binary.ndim != 2:
        raise LossError(f"cut_gaps expects a 2-D mask, got {binary.shape}")
    if int(spacing) < 1 or int(gap) < 1:
        raise LossError(f"spacing and gap must be >= 1, got {spacing}, {gap}")
    ys, xs = np.nonzero(binary)
    if ys.size == 0:
        raise LossError(
            "cut_gaps was given an empty mask; there is no line to break. "
            "Pick a tile with boundary in it.")
    out = binary.copy()
    half = int(gap) // 2
    h, w = out.shape
    for i in range(0, ys.size, int(spacing)):
        y, x = int(ys[i]), int(xs[i])
        out[max(0, y - half):min(h, y + half + 1),
            max(0, x - half):min(w, x + half + 1)] = False
    return out.astype(mask.dtype if mask.dtype != bool else np.uint8)


def dilate_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Grow the boundary by one pixel on each side. Topology unchanged."""
    import cv2

    mask = np.asarray(mask)
    binary = (mask > 0).astype(np.uint8)
    if binary.ndim != 2:
        raise LossError(f"dilate_mask expects a 2-D mask, got {binary.shape}")
    grown = cv2.dilate(binary, np.ones((3, 3), np.uint8),
                       iterations=int(iterations))
    return grown.astype(mask.dtype if mask.dtype != bool else np.uint8)


def as_logits(mask: np.ndarray, magnitude: float = 10.0,
              device=None) -> "torch.Tensor":
    """A 1x1xHxW logit tensor that means "this mask, confidently".

    ``+m`` where the mask is set and ``-m`` where it is not. Used to score a
    constructed prediction with the same criterion the model will be scored
    with -- the demo cases have to go through the real loss, not a
    reimplementation of it.
    """
    arr = (np.asarray(mask) > 0).astype(np.float32)
    logits = torch.from_numpy((2.0 * arr - 1.0) * float(magnitude))
    logits = logits[None, None, ...]
    return logits.to(device) if device is not None else logits


def as_target(mask: np.ndarray, device=None) -> "torch.Tensor":
    """A 1x1xHxW float target of exactly 0.0 and 1.0."""
    arr = (np.asarray(mask) > 0).astype(np.float32)
    t = torch.from_numpy(arr)[None, None, ...]
    return t.to(device) if device is not None else t
