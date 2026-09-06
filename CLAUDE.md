# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Project: Phase Boundary Segmentation (Phase 0)

## Goal
Train a U-Net to predict phase and grain boundaries directly from raw
metallurgical micrographs. Output is a boundary probability map plus a
yellow overlay. This model is a PRIOR GENERATOR for a downstream Phase 1
unsupervised segmentation stage. It is not the final segmenter.

## Execution model — READ THIS FIRST
There is NO local Python environment. You write code. You never run it.
Do not create environments, install packages, run pytest, or execute
scripts. If you believe you need to run something, write the notebook cell
that runs it instead.

Every step produces two things:
  1. library code in src/ plus a thin CLI in scripts/
  2. a notebook in notebooks/ that imports that code, runs it on Colab or
     Kaggle, verifies it, and pushes the generated reports back to this repo

Verification happens on the host. When a step is done, report
"written, not executed" and name the notebook that verifies it.

Because nothing is run here, code must be defensive: validate inputs,
fail loudly with a clear message, never fail silently.

## Pipeline architecture

Six steps so far, each one step of the chain `src/<module>.py` ->
`scripts/<cli>.py` -> `notebooks/NN_*.ipynb`. Steps communicate ONLY through
committed artefacts in `reports/` and `configs/`; a later step reads the
earlier step's JSON and never re-derives its decisions.

| step | src | script | notebook | reads | writes |
| --- | --- | --- | --- | --- | --- |
| 0 bootstrap | `paths.py` | `bootstrap_session.py` | `00_bootstrap.ipynb` | `configs/default.yaml` + `configs/<platform>.yaml` | resolved PATHS dict |
| 1 audit | `audit.py` | `run_audit.py` | `01_audit.ipynb` | DATA_ROOT | `reports/audit.{json,md}` |
| 2 boundary GT | `boundary_gt.py` | `extract_boundaries.py` | `02_boundary_gt.ipynb` | `reports/audit.json` | boundary PNGs in GT_BOUNDARIES_ROOT, `configs/hsv_ranges.yaml`, `reports/gt_extraction.{json,md}` |
| 3 tiling/folds | `tiling.py` | `build_tiles.py` | `03_tiling.ipynb` | `reports/audit.json`, `reports/gt_extraction.json` | `reports/manifests/*.csv`, `reports/parents.md`, `reports/tiling.{json,md}`, `configs/fold_stats.yaml` |
| 4 dataset/loader | `dataset.py` | — | `04_dataset.ipynb` | manifests, `configs/fold_stats.yaml` | `configs/dataloader.yaml` |
| 5 model/loss | `model.py`, `losses.py` | — | `05_model_and_loss.ipynb` | `configs/fold_stats.yaml` | `configs/default.yaml` `model:`/`loss:`/`train.batch_size` |
| 6 training | `train.py` | `train.py` | `06_train.ipynb` | manifests, `configs/fold_stats.yaml`, `configs/dataloader.yaml` | `PERSISTENT_DIR/checkpoints/<fold>/{last,best}.pt`, `reports/train_<fold>.{json,md}` |

Key consequences of that contract:

- `reports/audit.json` is the ONLY authority on a folder's MODE. `boundary_gt`
  obeys it; `--mode NAME=A|B` is the sole override and it gets recorded.
- A tile is a CSV row (dataset, parent_id, source_image, x, y,
  boundary_fraction, crop box, pad), never a written image. `src/dataset.py`
  crops from the full image at load time. Never add a tile-writing step.
- `configs/fold_stats.yaml` carries the per-fold `pos_weight` and sampler
  weights measured in step 3. Training reads them; it must not recompute them.
- Every `src/` module takes its tunables from a `DEFAULTS` dict overridable
  under its own key in `configs/default.yaml` (`boundary_gt:`, `tiling:`,
  `dataset:`), and echoes what it actually used into its markdown report.

## Path and config resolution

`src/paths.py` is the single source of truth for locations; no module may
contain a literal `/content` or `/kaggle` path. `load_config()` reads
`configs/default.yaml` and deep-merges `configs/<platform>.yaml` over it —
that overlay is the whole mechanism by which hosts differ.

`detect_platform()` returns colab | kaggle | local. The ordering is load-bearing
and was arrived at painfully: Colab ships an EMPTY `/kaggle/input`, Kaggle sets
`COLAB_RELEASE_TAG` and has an importable `google.colab` shim. Only the
`KAGGLE_*` env vars and `/kaggle/working` discriminate, so they are tested
first. The function is deliberately duplicated verbatim in
`scripts/bootstrap_session.py`, which must run before `src/` exists on the host
— keep the two copies identical.

`GT_BOUNDARIES_ROOT` is resolved INDEPENDENTLY of `PERSISTENT_DIR`: on Colab it
sits in Drive beside checkpoints, on Kaggle it is a second read-only input
dataset. Do not derive one from the other.

`/kaggle/working` does not survive the session. Treat a checkpoint left only
there as already lost; push reports and configs as you go.

## Data
`data/` holds one subfolder per dataset: MetalDam, Steel1, Steel2, uhcs1, uhcs2
(`uhcs` is an accepted alias for `uhcs1`; see `DATASET_ALIASES`). Each holds raw
images and their masks. Folder layouts and naming conventions differ between
datasets and are INFERRED by `src/audit.py`, never assumed. `data/` is
gitignored — never commit images. On the host, data is mounted from Google Drive
(Colab) or `/kaggle/input` (Kaggle); resolve it via `src/paths.py`.

Masks come in two kinds, and which kind each folder is MUST be read from
`reports/audit.json`, never assumed:
  MODE A — boundaries painted as their own distinct colour.
           Extraction: HSV colour thresholding, window derived from the pixels
           themselves and written to configs/hsv_ranges.yaml. Yields phase
           interfaces AND intra-phase grain boundaries.
  MODE B — phase-label map only, no painted boundary.
           Extraction: find_boundaries on the quantized label map.
           Yields phase interfaces ONLY. Grain boundaries are absent and
           cannot be recovered.

As currently audited, ALL FIVE folders are MODE B. The MODE A path exists and
is tested but is not exercised by this data — do not assume it has been
validated against a real MODE A folder.

Both modes share one cleanup: despeckle, CLOSE, skeletonize, dilate to a uniform
`line_width_px`. Output is a uint8 PNG whose pixels are exactly 0 or 255.

## Fold protocol (decided in step 3, do not re-litigate in later steps)
- Tiles are not independent samples. Every tile carries a `parent_id`; splits
  are made BY PARENT so tiles of one micrograph never straddle train and val.
- Steel2 is test-only — optical/colour among four grayscale SEM sets, held out
  as the domain-shift fold, never trained or validated on.
- Steel1 is split by parent with a fixed seed (15 train / 4 val of 19 parents).
- MetalDam, uhcs1, uhcs2 rotate as the held-out LODO validation fold;
  `dev` is an alias for `fold_uhcs2`.
- Steel1 and Steel2 are already exactly 256x256 and must yield exactly one tile
  each with zero padding — asserted, not assumed.

## Hard rules
- Never resize images. Handle size variation by tiling only (256 px patch,
  128 px stride; the last tile of a row/column is clamped to the image edge, not
  padded). Resizing destroys the physical micron scale.
- Masks: nearest-neighbour interpolation only. Never bilinear.
- Spatial augmentations apply to image AND mask identically, as one Compose.
  Photometric augmentations apply to the image ONLY — the mask is never passed
  to that pipeline. The separation is structural, not a flag.
- Normalize intensity per image, never with global dataset statistics.
- Split by dataset folder (leave-one-dataset-out), never by random tile.
- Never report pixel accuracy. Boundaries are a small minority class.
- No learned model may appear in the label-generation path.
- Never hardcode /content, /kaggle, a token, or a personal path. Use
  src/paths.py.
- A manifest row with non-zero pad is a stale manifest predating the clamping
  fix; refuse it loudly rather than train on fabricated pixels.

## Remote sessions

Every notebook under notebooks/ must:

- open with a markdown cell stating: what it does, what must already exist,
  what it produces, and expected runtime on a free T4
- put a markdown cell BEFORE every code cell explaining what that cell does
  and WHY. The reader is learning the pipeline, not just running it.
- have cell 1 be the standard bootstrap block, identical in every notebook:
  it reads GH_TOKEN from the host secret store, fetches
  `scripts/bootstrap_session.py` from the GitHub API, and calls `bootstrap()`,
  which returns `PATHS`
- run top to bottom on a fresh Colab AND a fresh Kaggle session with no
  manual edits. Platform differences are handled by bootstrap_session.py,
  never by the user editing a path.
- import from src/, never redefine pipeline logic inline. If a notebook needs
  a function that does not exist in src/, that function goes in src/.
- VERIFY, not just run. Every notebook ends with an explicit checks cell that
  asserts the step's output is well-formed and prints PASS or FAIL per check.
  This is where correctness is established, because nothing is run locally.
- end with a call to scripts/push_results.py so generated reports and configs
  reach the repo. Only `reports/`, `configs/` and `notebooks/` are ever staged.
- be committed with all outputs cleared
- never contain a token, key, or absolute personal path

## Commands (host only — write these into notebook cells, never run them here)

    # step 1
    python scripts/run_audit.py [--datasets MetalDam uhcs1] [--sample 40] [--quiet]
    # step 2
    python scripts/extract_boundaries.py [--datasets uhcs2] [--limit 5] [--mode uhcs2=A]
    # step 3
    python scripts/build_tiles.py [--datasets Steel1] [--quiet]
    # step 6 (the notebook calls src.train.Trainer directly; this is for headless runs)
    python scripts/train.py --fold dev [--resume] [--epochs 2] [--no-amp]
    # push generated reports/configs/notebooks back to the repo
    python scripts/push_results.py -m "step N: <description>"

Notebooks call `main()` or the `src.` function directly rather than shelling
out, so progress bars render inline.

Tests live in `tests/` and are run BY the notebook on the host, not locally:

    pytest tests/test_dataset.py -q
    pytest tests/test_losses.py -q
    pytest tests/test_train.py -q
    pytest tests/test_dataset.py -q -k binary        # one test

Tests that need real data skip themselves when it is absent; notebook 04's
checks cell FAILS if they were skipped, so a skip is not a pass.

Dependencies: `requirements-notebook.txt` is installed at notebook runtime by
the bootstrap, which skips anything already importable. Colab and Kaggle already
ship torch, numpy, scipy, opencv, scikit-image, matplotlib, pandas.

## Branches

`main` is the source of truth. All library code, notebooks, scripts and shared
config live there, and every step is developed there.

`kaggle` carries CONFIGURATION ONLY — `configs/kaggle.yaml`, naming the two
attached input datasets and the Kaggle path roots. It exists so a notebook can
be opened straight from GitHub in Kaggle by switching branch. It must never
contain a different version of a notebook or a src module. `git diff main kaggle`
must show exactly one added file.

After every step on main:

    git checkout kaggle && git merge main && git push && git checkout main

so the branch never falls behind. The merge should always be a fast-forward or
a clean merge touching nothing.

A conflict outside `configs/` means pipeline logic has leaked onto the branch.
Move it back to main: platform differences belong in `scripts/bootstrap_session.py`
and `src/paths.py`, where BOTH hosts get them. If a difference cannot be
expressed as configuration, that is a signal to change the code on main, not to
fork a file onto the branch.

## Response style
No preamble. No postamble. No summary of what you just did. Do not restate
the task. Do not explain code unless asked. Report only: files changed,
commit hash, and which notebook verifies the change.

## Git
After every step: add, commit with "step N: <short description>", push.
