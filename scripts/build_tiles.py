#!/usr/bin/env python3
"""Thin CLI over src.tiling: index tiles, group by parent, emit fold manifests.

    python scripts/build_tiles.py
    python scripts/build_tiles.py --datasets Steel1 --quiet

Reads reports/gt_extraction.json (boundary maps and the crops step 2 applied)
and reports/audit.json (where each dataset lives). Writes manifests to
reports/<manifest_subdir>/, plus reports/parents.md, reports/tiling.{md,json}
and configs/fold_stats.yaml. No tile images are written and nothing under
data/ is touched.

Paths come from src/paths.py, so this runs unchanged on Colab, Kaggle and a
local checkout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import boundary_gt  # noqa: E402
from src import paths as paths_mod  # noqa: E402
from src import tiling  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build tile index and fold manifests")
    ap.add_argument("--gt-root", default=None,
                    help="override GT_BOUNDARIES_ROOT from src/paths.py")
    ap.add_argument("--reports-dir", default=None)
    ap.add_argument("--configs-dir", default=None)
    ap.add_argument("--manifest-dir", default=None,
                    help="override reports/<manifest_subdir>")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    resolved = paths_mod.resolve_paths()
    settings = tiling.load_config()
    reports_dir = Path(args.reports_dir) if args.reports_dir else Path(resolved["reports_dir"])
    gt_root = (Path(args.gt_root) if args.gt_root
               else Path(resolved["gt_boundaries_root"]))
    manifest_dir = (Path(args.manifest_dir) if args.manifest_dir
                    else reports_dir / settings["manifest_subdir"])

    progress = None
    if not args.quiet:
        try:
            from tqdm.auto import tqdm

            def progress(seq, desc=""):
                return tqdm(seq, desc=desc, leave=False, unit="img")
        except ImportError:
            progress = None

    audit = boundary_gt.load_audit(reports_dir)
    extraction = tiling.load_extraction(reports_dir)

    print(f"indexing boundary maps under {gt_root}")
    index = tiling.build_index(extraction, audit, gt_root, settings,
                               datasets=args.datasets, progress=progress)
    folds = tiling.build_folds(index, settings)

    # fold_steel_combined: a separate, pooled Steel1+Steel2 train/val/test
    # split (see src.tiling.build_steel_combined_fold), built ALONGSIDE the
    # LODO folds above -- never inside build_folds() -- and only when both
    # its datasets survived --datasets filtering. It is additive: nothing
    # above this line is touched by it.
    steel_cfg = settings["steel_combined"]
    steel_fold = None
    if all(d in index["datasets"] for d in steel_cfg["datasets"]):
        steel_fold = tiling.build_steel_combined_fold(index, settings)
        folds["folds"][steel_fold["name"]] = steel_fold
    else:
        missing = [d for d in steel_cfg["datasets"] if d not in index["datasets"]]
        print(f"skipping {steel_cfg['name']}: {missing} not in this index "
              "(--datasets filtered it out)")

    stats = tiling.fold_statistics(folds, settings)
    if steel_fold is not None:
        name = steel_fold["name"]
        # The same entry notebooks/06d_steel_combined.ipynb builds, from the
        # same helper: this fold's stats are assembled in exactly one place,
        # so the CLI path and the pooled-split notebook path cannot drift.
        stats[name] = tiling.steel_combined_statistics(steel_fold, settings)
        print(f"\n{name} parent/tile allocation (dataset: n_parents/n_tiles):")
        for split in ("train", "val", "test"):
            per_ds = steel_fold["parent_allocation"][split]
            rows_this_split = (steel_fold["test_rows"] if split == "test"
                               else [r for r in steel_fold["rows"] if r["split"] == split])
            counts = {ds: sum(1 for r in rows_this_split if r["dataset"] == ds)
                     for ds in steel_cfg["datasets"]}
            print(f"  {split:<5} " + "  ".join(
                f"{ds}={len(per_ds.get(ds, []))}p/{counts.get(ds, 0)}t"
                for ds in steel_cfg["datasets"]))

    manifests = tiling.write_manifests(folds, manifest_dir)
    if steel_fold is not None:
        test_manifest_path = manifest_dir / f"{steel_fold['name']}_test.csv"
        manifests[f"{steel_fold['name']}_test"] = str(
            tiling.write_manifest(steel_fold["test_rows"], test_manifest_path))
    parents_path = tiling.write_parents_md(index, reports_dir)
    md_path, json_path = tiling.write_tiling_report(
        index, folds, stats, manifests, reports_dir)
    stats_path = tiling.write_fold_stats(stats, folds, settings, args.configs_dir)

    for name, d in index["datasets"].items():
        print(f"  {name:<12} {d['n_parents']:>3} parents  {d['n_tiles']:>5} tiles  "
              f"dropped {d['n_dropped']:>4}  excluded {d['n_excluded']:>3}  "
              f"no-mask {d['n_orphans']:>3}")
        for loud in d["single_tile_drops"]:
            print(f"    ! {loud['dataset']}/{loud['image']} lost its only tile "
                  f"({loud['boundary_fraction']:.5f} boundary)")
    for name, f in stats.items():
        if name == "test":
            continue
        print(f"  {name:<16} train {f['n_train_tiles']:>5}  val {f['n_val_tiles']:>5}  "
              f"pos_weight {f['pos_weight']}")
    print(f"  test             {stats['test']['n_tiles']:>5} tiles "
          f"({', '.join(stats['test']['datasets'])})")

    for path in (parents_path, md_path, json_path, stats_path):
        print(f"wrote {path}")
    for name, path in manifests.items():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
