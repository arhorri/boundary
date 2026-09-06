#!/usr/bin/env python3
"""Thin CLI over src.train: train one fold, resumably.

    python scripts/train.py --fold dev
    python scripts/train.py --fold fold_MetalDam --resume
    python scripts/train.py --fold dev --epochs 2 --no-amp        # smoke test

Reads reports/manifests/<fold>.csv, configs/fold_stats.yaml (pos_weight and
sampler weights) and configs/dataloader.yaml (this host's num_workers). Writes
checkpoints and TensorBoard logs under PERSISTENT_DIR, and
reports/train_<fold>.{json,md} into the checkout.

Nothing here re-derives a decision an earlier step made: the split comes from
the manifest, pos_weight from fold_stats, num_workers from the measured entry
for THIS platform. Paths come from src/paths.py, so it runs unchanged on Colab,
Kaggle and a local checkout.

The notebook calls ``src.train.Trainer`` directly rather than shelling out to
this, so that progress bars and per-epoch figures render inline. This exists
for a headless run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import paths as paths_mod  # noqa: E402
from src import train as train_mod  # noqa: E402


def _print_epoch(record: dict, trainer) -> None:
    """One line of headline numbers per epoch, plus the per-dataset breakdown."""
    print(f"\nepoch {record['epoch']:>3}  "
          f"{record['seconds']:.1f}s  lr {record['lr']:.2e}  "
          f"trainable {record['trainable_params']:,}"
          f"{'  [encoder frozen]' if record['encoder_frozen'] else ''}"
          f"{'  <- BEST' if record['is_best'] else ''}")
    print(f"  loss  train total {record['train']['total']:.4f} "
          f"(bce {record['train']['bce']:.4f} dice {record['train']['dice']:.4f} "
          f"cldice {record['train']['cldice']:.4f})")
    print(f"        val   total {record['val']['total']:.4f} "
          f"(bce {record['val']['bce']:.4f} dice {record['val']['dice']:.4f} "
          f"cldice {record['val']['cldice']:.4f})")
    print(f"  {'dataset':<14}{'IoU':>8}{'Dice':>8}{'Prec':>8}{'Rec':>8}{'bF':>8}")
    for name, metrics in record["metrics"]["per_dataset"].items():
        marker = "  <- held out" if name == trainer.held_out else ""
        print(f"  {name:<14}" + "".join(
            f"{metrics[k]:>8.4f}" for k in train_mod.METRIC_ORDER) + marker)
    pooled = record["metrics"]["pooled"]
    print(f"  {'pooled (fn)':<14}" + "".join(
        f"{pooled[k]:>8.4f}" for k in train_mod.METRIC_ORDER))
    probe = record["cldice_probe"]
    print(f"  clDice probe: skeleton delta {probe['skeleton_delta']:.6f} "
          f"({'DEGENERATE' if probe['degenerate'] else 'active'}), "
          f"skel(pred) {probe['skel_pred_sum']:.0f} px, "
          f"skel(true) {probe['skel_true_sum']:.0f} px")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train the boundary U-Net on one fold")
    ap.add_argument("--fold", required=True,
                    help="a fold in configs/fold_stats.yaml, e.g. dev, fold_MetalDam")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override train.epochs (a short smoke run)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None,
                    help="override this host's measured value; say why in the report")
    ap.add_argument("--resume", action="store_true",
                    help="continue PERSISTENT_DIR/checkpoints/<fold>/last.pt")
    ap.add_argument("--allow-config-change", action="store_true",
                    help="resume across a config hash mismatch (deliberate only)")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-tensorboard", action="store_true")
    ap.add_argument("--reports-dir", default=None)
    ap.add_argument("--configs-dir", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    settings = train_mod.load_config()
    for key, value in (("epochs", args.epochs), ("batch_size", args.batch_size),
                       ("num_workers", args.num_workers)):
        if value is not None:
            settings[key] = value
    if args.no_amp:
        settings["amp"] = False

    resolved = paths_mod.resolve_paths()
    progress = None
    if not args.quiet:
        try:
            from tqdm.auto import tqdm

            def progress(seq, desc=""):
                return tqdm(seq, desc=desc, leave=False)
        except ImportError:
            progress = None

    trainer = train_mod.Trainer(fold=args.fold, resolved=resolved,
                                settings=settings,
                                configs_dir=args.configs_dir)
    summary = trainer.setup(progress=progress,
                            tensorboard=not args.no_tensorboard)

    print(f"fold {summary['fold']} (held out: {summary['held_out']})")
    print(f"  {summary['platform']} / {summary['gpu'] or summary['device']}, "
          f"AMP {summary['amp']}, config hash {summary['config_hash']}")
    print(f"  {summary['train_tiles']} train tiles, {summary['val_tiles']} val "
          f"tiles {summary['val_composition']}")
    print(f"  batch {summary['batch_size']}, {summary['num_workers']} workers "
          f"({summary['sources'].get('num_workers')})")
    print(f"  pos_weight {summary['pos_weight']} "
          f"({summary['sources'].get('pos_weight')})")
    for split, info in summary["cache"].items():
        print(f"  tile cache {split}: {info['source'].upper()} "
              f"{info['images']} images in {info['seconds']:.1f}s")

    if args.resume:
        status = trainer.maybe_resume(allow_config_change=args.allow_config_change)
        print(f"  resume: {status['reason']}")
        if status["resumed"]:
            print(f"  continuing from epoch {status['start_epoch']}")
    else:
        if trainer.last_path.is_file():
            print(f"! a checkpoint exists at {trainer.last_path} and --resume "
                  "was not given; it will be OVERWRITTEN after epoch 0")

    trainer.fit(epochs=args.epochs,
                on_epoch_end=None if args.quiet else _print_epoch,
                progress=progress)

    md_path, json_path = train_mod.write_report(
        trainer, reports_dir=args.reports_dir)
    print(f"\nbest epoch {trainer.best['epoch']} by Dice on "
          f"{trainer.best['key']}: {trainer.best['metric']:.4f}")
    print(f"checkpoints: {trainer.last_path}, {trainer.best_path}")
    for path in (md_path, json_path):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
