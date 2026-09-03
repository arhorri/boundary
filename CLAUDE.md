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

## Response style
No preamble. No postamble. No summary of what you just did. Do not restate
the task. Do not explain code unless asked. Report only: files changed,
commit hash, and which notebook verifies the change.

## Git
After every step: add, commit with "step N: <short description>", push.
