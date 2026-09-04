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

## Data
`data/` holds one subfolder per dataset: MetalDam, Steel1, Steel2, uhcs, uhcs2.
Each holds raw images and their masks. Folder layouts differ between datasets.
data/ is gitignored — never commit images. On the host, data is mounted from
Google Drive (Colab) or /kaggle/input (Kaggle); resolve it via src/paths.py.

Masks come in two kinds, and which kind each folder is MUST be read from
`reports/audit.json`, never assumed:
  MODE A — boundaries painted as their own distinct colour.
           Extraction: HSV colour thresholding. Yields phase interfaces
           AND intra-phase grain boundaries.
  MODE B — phase-label map only, no painted boundary.
           Extraction: find_boundaries on the quantized label map.
           Yields phase interfaces ONLY. Grain boundaries are absent and
           cannot be recovered.

## Hard rules
- Never resize images. Handle size variation by tiling only. Resizing
  destroys the physical micron scale.
- Masks: nearest-neighbour interpolation only. Never bilinear.
- Spatial augmentations apply to image AND mask identically.
  Photometric augmentations apply to the image ONLY.
- Normalize intensity per image, never with global dataset statistics.
- Split by dataset folder (leave-one-dataset-out), never by random tile.
- Never report pixel accuracy. Boundaries are a small minority class.
- No learned model may appear in the label-generation path.
- Never hardcode /content, /kaggle, a token, or a personal path. Use
  src/paths.py.

## Remote sessions

Every notebook under notebooks/ must:

- open with a markdown cell stating: what it does, what must already exist,
  what it produces, and expected runtime on a free T4
- put a markdown cell BEFORE every code cell explaining what that cell does
  and WHY. The reader is learning the pipeline, not just running it.
- have cell 1 be the standard bootstrap block, identical in every notebook
- run top to bottom on a fresh Colab AND a fresh Kaggle session with no
  manual edits. Platform differences are handled by bootstrap_session.py,
  never by the user editing a path.
- import from src/, never redefine pipeline logic inline. If a notebook needs
  a function that does not exist in src/, that function goes in src/.
- VERIFY, not just run. Every notebook ends with an explicit checks cell that
  asserts the step's output is well-formed and prints PASS or FAIL per check.
  This is where correctness is established, because nothing is run locally.
- end with a call to scripts/push_results.py so generated reports and configs
  reach the repo
- be committed with all outputs cleared
- never contain a token, key, or absolute personal path

## Branches

`main` is the source of truth. All library code, notebooks, scripts and shared
config live there, and every step is developed there.

`kaggle` carries CONFIGURATION ONLY — `configs/kaggle.yaml`, naming the two
attached input datasets and the Kaggle path roots. It exists so a notebook can
be opened straight from GitHub in Kaggle by switching branch. It must never
contain a different version of a notebook or a src module.

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
