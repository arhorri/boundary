# GT mode check (FORCED): MODE B vs MODE A despite the largest-CC gate

- generated: 2026-09-30T12:01:44Z
- settings: {'line_width_px': 4.0, 'blob_thickness_px': 8.0, 'thin_width_px': 8.0, 'sample_files': 60, 'thresholds_swept': [64, 128, 192, 224, 240]}
- the largest-CC-share gate (>= 0.5) in reports/steel2_gt_mode_check.json did NOT clear (0.24) -- this report FORCES the MODE B vs MODE A comparison anyway; no gate in Cell 27 was changed.

File-size check: 60 files read, all exactly [256, 256] -- every sampled file is a single 256x256 image, and Steel1/Steel2 tiles are already exactly 256x256 with one tile per image and zero padding (CLAUDE.md's own hard rule) -- so 'file' and 'tile' are the SAME unit here; there is no separate crop step to account for.

## 1. Threshold sensitivity (same 60-file Steel2 sample)

`dark = raw grayscale value < threshold`. `current (K=2 snap)` is `snap_to_palette` against the real black/white palette, included for comparison, not swept.

| cut | dark share | share thin | largest CC share | components | skeleton px/file (mean/median) |
| --- | --- | --- | --- | --- | --- |
| 64 | 0.196 | 90.22% | 23.58% | 11843 | 3416 / 3378 |
| 128 | 0.196 | 90.22% | 23.58% | 11843 | 3416 / 3378 |
| 192 | 0.196 | 90.22% | 23.58% | 11843 | 3416 / 3378 |
| 224 | 0.196 | 90.22% | 23.58% | 11843 | 3416 / 3378 |
| 240 | 0.198 | 90.57% | 23.40% | 17268 | 3557 / 3529 |
| current (K=2 snap) | 0.196 | 90.22% | 23.58% | 11843 | 3416 / - |

Best cut by largest-CC-share among the swept thresholds: **64** (fragmentation did NOT clearly drop moving toward white).

## 2/3. MODE B vs proposed MODE A -- forced, both cuts

### Variant: k2 (current K=2 snap -- what extraction actually used)

4 gallery tiles (Cell 26's ranking):

| tile | regions (on-disk GT) | regions (B) | regions (A) | area p50 (B) | area p50 (A) |
| --- | --- | --- | --- | --- | --- |
| `87450661_tile_2560_512.png` | 207 | 207 | 66 | 82 | 236 |
| `87450661_tile_2048_2048.png` | 205 | 205 | 61 | 79 | 266 |
| `87450741_tile_1024_1536.png` | 201 | 201 | 54 | 64 | 96 |
| `87450661_tile_1280_1792.png` | 200 | 200 | 64 | 72 | 136 |

Full Steel2 profile (504 files):

| | regions/tile | area p25 | area p50 | area p75 | share < 50 px | share < 100 px |
| --- | --- | --- | --- | --- | --- | --- |
| MODE B | 102.8 | 38 | 78 | 192 | 33.8% | 57.9% |
| MODE A | 28.7 | 64 | 146 | 395 | 19.4% | 37.8% |

### Variant: best_cut (threshold 64)

4 gallery tiles (Cell 26's ranking):

| tile | regions (on-disk GT) | regions (B) | regions (A) | area p50 (B) | area p50 (A) |
| --- | --- | --- | --- | --- | --- |
| `87450661_tile_2560_512.png` | 207 | 207 | 66 | 82 | 236 |
| `87450661_tile_2048_2048.png` | 205 | 205 | 61 | 79 | 266 |
| `87450741_tile_1024_1536.png` | 201 | 201 | 54 | 64 | 96 |
| `87450661_tile_1280_1792.png` | 200 | 200 | 64 | 72 | 136 |

Full Steel2 profile (504 files):

| | regions/tile | area p25 | area p50 | area p75 | share < 50 px | share < 100 px |
| --- | --- | --- | --- | --- | --- | --- |
| MODE B | 102.8 | 38 | 78 | 192 | 33.8% | 57.9% |
| MODE A | 28.7 | 64 | 146 | 395 | 19.4% | 37.8% |

## 4. Gallery figures

See the notebook cell's output: one figure per variant, 4 rows (the same gallery tiles) x 4 columns (raw image, the current boundary GT as committed on disk, MODE B boundary under that variant's binarization, MODE A boundary under the same binarization), each panel captioned with its region count. Under the K=2 variant, MODE B reproduces the on-disk GT; under the best cut it shows what MODE B alone would give if only the binarization changed. Not persisted as image files here -- the notebook is the artefact.

## What this report does NOT do

- does not change MC_SHARE_THIN_MIN / MC_LARGEST_CC_MIN in Cell 27
- does not wire MODE A, or any alternate threshold, into extract_folder
- does not regenerate any boundary PNG under `GT_BOUNDARIES_ROOT`
- does not rebuild `fold_steel_combined`'s manifests or `configs/fold_stats.yaml`
- does not retrain or touch any checkpoint
