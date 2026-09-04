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
    stats = tiling.fold_statistics(folds, settings)

    manifests = tiling.write_manifests(folds, manifest_dir)
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
