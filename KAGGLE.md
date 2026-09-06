# Running this project on Kaggle

**This file exists only on the `kaggle` branch.** `main` is unchanged and stays
the Colab path — nothing here alters how Colab behaves.

---

## What is different on Kaggle, and why

| | Colab | Kaggle |
| --- | --- | --- |
| `DATA_ROOT` | Drive folder | `<input_root>/<dataset_slug>` (read-only) |
| `GT_BOUNDARIES_ROOT` | `PERSISTENT_DIR/gt_boundaries` | `<input_root>/<gt_dataset_slug>` (read-only) |
| `input_root` | — | `/kaggle/input/datasets/<owner>` on this account, not flat `/kaggle/input` |
| `PERSISTENT_DIR` | Drive — **survives** | `/kaggle/working` — **deleted when the kernel stops** |
| secret store | `google.colab.userdata` | `kaggle_secrets.UserSecretsClient` |
| branch | `main` | `kaggle` (the bootstrap switches to it by itself) |
| tile cache on disk | on — Drive is slow | off — `/kaggle/input` is local SSD |

Two consequences drive everything below.

**Steps 1–2 cannot run on Kaggle.** `GT_BOUNDARIES_ROOT` is a read-only input
dataset, so nothing in a Kaggle session can write boundary maps. Generate them
on Colab with `02_boundary_gt.ipynb`, then upload them as a Kaggle dataset. The
bootstrap fails loudly with the directory listing if they are missing, rather
than failing later inside a data loader.

**A training run outlives a session.** `/kaggle/working` is gone when the kernel
stops. `src/kaggle_persist.py` (this branch only) closes that gap: it stages a
checkpoint out of an attached input dataset, stops the run on an epoch boundary
while there is still time to save, and prints what must be saved by hand.

---

## What was added on this branch

Everything is **additive**, so `git merge main` never conflicts:

| file | what it is |
| --- | --- |
| `configs/kaggle.yaml` | the config overlay (pre-existing), now also carrying `session.kaggle.persist` and `dataset.persist_cache: false` |
| `src/kaggle_persist.py` | **new** — session budget, checkpoint staging, survival report |
| `notebooks/06_train_kaggle.ipynb` | **new** — step 6 arranged for multi-session training |
| `KAGGLE.md` | **new** — this file |

No file that exists on `main` was modified. `git diff main kaggle --stat`
shows only additions.

---

## One-time setup

### 1. A GitHub token

Create a fine-grained PAT with **Contents: read and write** on the repo
(`github.com/settings/personal-access-tokens`).

### 2. Upload two datasets

Both are created from your existing Colab/Drive copies. In Kaggle:
**Datasets → New Dataset → Upload**.

| upload this | name it exactly | contents |
| --- | --- | --- |
| your `data/` folder | `phase11-microstructure-data` | `MetalDam/`, `Steel1/`, `Steel2/`, `uhcs1/`, `uhcs2/` |
| `PERSISTENT_DIR/gt_boundaries/` from Colab step 2 | `phase11-gt-boundaries` | one subfolder per dataset, boundary PNGs |

The slugs must match `session.kaggle.dataset_slug` and
`session.kaggle.gt_dataset_slug` in `configs/kaggle.yaml`. If you name them
differently, change that file — never a path inside a notebook.

> **Mount shape.** This account's notebooks mount attached datasets nested by
> owner — `/kaggle/input/datasets/<owner>/<slug>/` — not flat at
> `/kaggle/input/<slug>/`. `session.kaggle.input_root` is set to the owner
> directory accordingly, and both `data_root` and `gt_boundaries_root` are
> derived from it. If a session ever mounts flat instead, that one key is what
> to change; the bootstrap prints the listing of what it actually found.
>
> One wrapping folder inside each dataset is fine and needs no configuration:
> `phase11-microstructure-data/data/MetalDam/...` and
> `phase11-gt-boundaries/gt_boundaries/MetalDam/...` are both accepted, because
> `verify_data_root` and `verify_gt_root` each retry one directory down.

---

## Per-notebook setup — do this every time you create a Kaggle notebook

1. **File → Import Notebook → GitHub**, then paste
   `https://github.com/arhorri/boundary/blob/kaggle/notebooks/06_train_kaggle.ipynb`
   (or upload the `.ipynb` you exported from this branch).
2. **Add-ons → Secrets** → add `GH_TOKEN` with your PAT as the value, and tick
   the checkbox that **attaches it to this notebook**. Adding it without
   attaching is the most common failure and it looks like a missing secret.
3. **Settings → Internet: ON.** Without it the clone and the `pip install` both
   fail. This is the second most common failure.
4. **Settings → Accelerator: GPU T4 x2** (or P100). The pipeline uses one GPU;
   the second is idle.
5. **Add Input** → attach both datasets from the setup above.
6. Run all cells.

You do **not** edit any path, branch or slug in the notebook. Cell 1 says
`BRANCH = "main"` on purpose: the bootstrap clones `main`, reads
`session.kaggle.branch`, and switches the checkout to `kaggle` by itself. The
session summary it prints will say `(kaggle)`.

---

## Training across more than one session

A free Kaggle GPU session is time-limited and a 40-epoch run may not fit. The
notebook handles the split; you do one manual action between sessions.

### Session 1

Run `06_train_kaggle.ipynb` top to bottom. It will either finish the run or
stop on an epoch boundary with:

```
SESSION BUDGET REACHED -- this is a clean stop, not an error
```

Then, **before the kernel stops**:

> **File → Save Version → Quick Save**, with *Save output* enabled.

This is the only thing that makes `/kaggle/working` survive, and a notebook
cannot do it for itself. The final cell prints whether the run is incomplete and
says so in a banner.

### Session 2 (and onward)

1. Open the notebook again (or **New Version**).
2. **Add Input → Notebook Output** → attach *your own previous version's
   output*.
3. Keep the two dataset inputs attached as well.
4. Run all cells.

The staging cell prints where it found the checkpoint and which epoch it is at:

```
staged last.pt from <slug>: epoch 21, 280 MB, hash 9c8476cbb16fc03
```

and the resume cell then reports `RESUMING ... continuing at epoch 22`.

Repeat until the last cell prints `RUN COMPLETE`.

### Reading the staging output

| message | what it means |
| --- | --- |
| `no attached checkpoint for this fold` | correct for session 1; **wrong** for any later session — the output was not attached |
| `ignoring ...: belongs to fold X` | a checkpoint is attached but for a different fold; it is reported, never silently used |
| `local checkpoint already present` | `/kaggle/working` already has one — you re-ran the cell in the same session. Nothing is staged, deliberately: the local file is the run in progress |
| `STARTING CLEAN` after something was staged | raises. That inconsistency is a real problem, not a warm start |

---

## Tuning the budget

In `configs/kaggle.yaml`, under `session.kaggle.persist`:

```yaml
session_budget_hours: 8.5   # under Kaggle's 9 h limit; it kills with no grace period
reserve_minutes: 25         # kept back for checks + report + push after fit() returns
resume_input_slugs: []      # [] = scan every attached dataset
max_stage_mb: 2048          # refuse to stage something implausibly large
```

The guard stops when the time left is less than the **slowest** epoch so far,
not the mean — epoch duration on a shared host is not stationary, and a mean
lets one slow final epoch run past the deadline.

If you know the slug of the dataset holding your checkpoints, naming it in
`resume_input_slugs` makes the search deterministic and turns a missing
attachment into an immediate loud error instead of a silent clean start.

---

## Which notebooks run where

| notebook | Colab | Kaggle |
| --- | --- | --- |
| `00_bootstrap` | yes | yes (needs both datasets attached) |
| `01_audit` | yes | yes |
| `02_boundary_gt` | **yes — only here** | no: `/kaggle/input` is read-only |
| `03_tiling` | yes | yes |
| `04_dataset` | yes | **run it once here** to measure this host's `num_workers` |
| `05_model_and_loss` | yes | yes |
| `06_train` | yes | works, but loses the run at the session limit |
| `06_train_kaggle` | no — exits with a message | **yes, use this one** |

`configs/dataloader.yaml` is keyed by platform and its `kaggle` entry is still
the unmeasured default of 2 workers. Run `04_dataset.ipynb` once on Kaggle to
replace it with a measured number; the entry for `colab` is untouched by that.

---

## Where results are pushed

The bootstrap leaves the checkout on `kaggle`, so `push_results` pushes to
**`origin/kaggle`**, not `main`. That is expected. To bring the reports back:

```
git checkout main && git merge kaggle -- # or cherry-pick the reports commit
```

The "`git diff main kaggle` shows only additions" invariant applies to *code*.
Freshly pushed `reports/` from a Kaggle run will also show until you merge them
back.

---

## Troubleshooting

| symptom | cause |
| --- | --- |
| `GH_TOKEN not available on kaggle` | the secret exists but is not **attached** to this notebook |
| clone or pip hangs, then fails | Internet is off in Settings |
| `GT_BOUNDARIES_ROOT does not exist` | the `phase11-gt-boundaries` dataset is not attached, or the slug differs from `configs/kaggle.yaml` |
| `DATA_ROOT is missing expected dataset folders` | the upload is incomplete, or nested more than one level deep |
| `DATA_ROOT does not exist` and the listing shows `datasets/` | the mount is owner-nested; `session.kaggle.input_root` must point at `/kaggle/input/datasets/<owner>` |
| `platform detected as 'colab'` on Kaggle | should be impossible — `detect_platform` tests `KAGGLE_*` env vars first. Report it; do not edit a path to work around it |
| resume refuses with a config-hash diff | something in `model:`, `loss:`, `train:` or `dataset:` changed since the checkpoint. Start clean, or accept it deliberately with `allow_config_change=True` |
| checkpoint gone after a session | no version was saved. `/kaggle/working` is not storage |
