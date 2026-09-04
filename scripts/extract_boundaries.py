#!/usr/bin/env python3
"""Thin CLI over src.boundary_gt: extract boundary ground truth for every folder.

    python scripts/extract_boundaries.py
    python scripts/extract_boundaries.py --datasets uhcs2 --limit 5
    python scripts/extract_boundaries.py --mode uhcs2=A

The mode of each folder is read from reports/audit.json. ``--mode NAME=A|B``
overrides one folder's verdict, for the case where a human disagrees with the
audit; the override is recorded in reports/gt_extraction.md.

Paths come from src/paths.py, so this runs unchanged on Colab, Kaggle and a
local checkout. Boundary PNGs go to PERSISTENT_DIR/<output_subdir>/<folder>/;
configs/hsv_ranges.yaml and reports/gt_extraction.{md,json} are written into
the repo.
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


def _parse_modes(items) -> dict:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--mode expects NAME=A or NAME=B, got {item!r}")
        name, mode = item.split("=", 1)
        out[name.strip()] = mode.strip().upper()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Extract boundary ground truth")
    ap.add_argument("--out-root", default=None,
                    help="override GT_BOUNDARIES_ROOT from src/paths.py")
    ap.add_argument("--reports-dir", default=None)
    ap.add_argument("--configs-dir", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--mode", nargs="*", default=None,
                    help="override a folder's audited mode, e.g. --mode uhcs2=A")
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N pairs per folder (smoke test)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    resolved = paths_mod.resolve_paths()
    settings = boundary_gt.load_config()
    reports_dir = Path(args.reports_dir) if args.reports_dir else Path(resolved["reports_dir"])
    out_root = (Path(args.out_root) if args.out_root
                else Path(resolved["gt_boundaries_root"]))
    if resolved["platform"] == "kaggle" and not args.out_root:
        raise SystemExit(
            f"GT_BOUNDARIES_ROOT is {out_root}, a read-only Kaggle input. Step 2 "
            "writes boundary maps and cannot run against it: run this step on "
            "Colab and upload the result as a Kaggle dataset, or pass "
            "--out-root /kaggle/working/gt_boundaries deliberately.")

    progress = None
    if not args.quiet:
        try:
            from tqdm.auto import tqdm

            def progress(seq, desc=""):
                return tqdm(seq, desc=desc, leave=False, unit="pair")
        except ImportError:
            progress = None

    audit = boundary_gt.load_audit(reports_dir)
    print(f"extracting into {out_root}")
    report = boundary_gt.extract_all(
        audit, out_root, settings,
        mode_overrides=_parse_modes(args.mode),
        datasets=args.datasets,
        limit=args.limit,
        progress=progress,
    )
    md_path, json_path = boundary_gt.write_extraction_report(report, reports_dir)
    hsv_path = boundary_gt.write_hsv_ranges(report, args.configs_dir)

    for name, d in report["datasets"].items():
        print(f"  {name:<12} MODE {d['mode']}  processed {d['n_processed']:>5}  "
              f"rejected {d['n_rejected']:>3}  excluded {d['n_excluded']:>2}  "
              f"frac {d['fraction_after']['median'] or 0:.4f}")
    for name, err in report["failures"].items():
        print(f"  {name:<12} FAILED  {err}")
    print(f"wrote {md_path}\nwrote {json_path}\nwrote {hsv_path}")
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
