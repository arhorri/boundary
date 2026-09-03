#!/usr/bin/env python3
"""Thin CLI over src.audit: audit every dataset folder and write the reports.

    python scripts/run_audit.py
    python scripts/run_audit.py --sample 40 --datasets MetalDam uhcs1

Paths come from src/paths.py, so this works unchanged on Colab, Kaggle and a
local checkout. It writes only reports/audit.json and reports/audit.md, and
never touches data/.

Notebooks call ``main()`` (or ``src.audit.audit_datasets`` directly) rather
than shelling out, so the progress bar renders inline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import audit as audit_mod  # noqa: E402
from src import paths as paths_mod  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Audit dataset folders under DATA_ROOT")
    ap.add_argument("--data-root", default=None,
                    help="override the DATA_ROOT resolved from configs/default.yaml")
    ap.add_argument("--reports-dir", default=None,
                    help="where audit.json / audit.md are written (default reports/)")
    ap.add_argument("--sample", type=int, default=audit_mod.SAMPLE_SIZE,
                    help="pairs opened per folder; pairing counts use every file")
    ap.add_argument("--seed", type=int, default=audit_mod.SAMPLE_SEED)
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="restrict to these folder names")
    ap.add_argument("--quiet", action="store_true", help="no progress bar")
    args = ap.parse_args(argv)

    resolved = paths_mod.resolve_paths()
    data_root = Path(args.data_root) if args.data_root else Path(resolved["data_root"])
    reports_dir = Path(args.reports_dir) if args.reports_dir else Path(resolved["reports_dir"])

    progress = None
    if not args.quiet:
        try:
            from tqdm.auto import tqdm

            def progress(seq, desc=""):
                return tqdm(seq, desc=desc, leave=False, unit="pair")
        except ImportError:
            progress = None

    print(f"auditing {data_root}")
    report = audit_mod.audit_datasets(
        data_root,
        datasets=args.datasets,
        sample_size=args.sample,
        seed=args.seed,
        progress=progress,
    )
    json_path, md_path = audit_mod.write_reports(report, reports_dir)

    for name, d in report["datasets"].items():
        print(f"  {name:<12} MODE {d['mode']['inferred']}  "
              f"{d['pairing']['n_pairs']:>5} pairs")
    for name, err in report["failures"].items():
        print(f"  {name:<12} FAILED  {err}")
    print(f"wrote {json_path}\nwrote {md_path}")
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
