# fold_steel_combined

- generated: 2026-09-26T09:10:39Z
- pooled from ['Steel1', 'Steel2'], split BY PARENT with seed 0
- held out: `mixed(Steel1+Steel2)` — this fold has no single held-out DATASET in the LODO sense. The label is descriptive and matches no dataset in any per-dataset metrics breakdown by construction, which is what makes the held-out-specific paths in `src/train.py` fall back to their pooled behaviour instead of mislabelling a real dataset as this fold's held-out one.
- pos_weight **6.516**, measured on THIS fold's own train split, never inherited from another fold

## Parents and tiles per split

Printed per dataset so an imbalance is visible rather than absorbed into a total. Steel2 has only 4 parents, so `min_parents_per_split: 1` is what guarantees it appears in every split at all.

| split | Steel1 p/t | Steel2 p/t | tiles | share |
| --- | --- | --- | --- | --- |
| train | 15p / 713t | 2p / 252t | 965 | 0.685 |
| val | 2p / 95t | 1p / 126t | 221 | 0.157 |
| test | 2p / 96t | 1p / 126t | 222 | 0.158 |

Total 1408 tiles. Target 0.7/0.15/0.15, achieved 0.685/0.157/0.158 by TILE count (the ratio is measured in tiles, not parents: Steel1 and Steel2 have very different tiles-per-parent).

## Boundary fraction

| split | min | mean | median | max | n |
| --- | --- | --- | --- | --- | --- |
| train | 0.011673 | 0.133041 | 0.103241 | 0.420181 | 965 |
| val | 0.007782 | 0.19511 | 0.193954 | 0.370697 | 221 |
| test | 0.015274 | 0.218955 | 0.239509 | 0.421494 | 222 |

## Parent allocation

Every parent appears in exactly one split; a parent's tiles never straddle two splits. Listed in full so the split can be checked by eye and reproduced.

- **train**
  - Steel1 (15): `im_val_cut10-400-01-H-12mm-4000x14`, `im_val_cut10-400-01-H-9mm-4000x10`, `im_val_cut10-400-01-H-9mm-4000x12`, `im_val_cut10-400-01-V-12mm-4000x16`, `im_val_cut10-400-01-V-12mm-4000x17`, `im_val_cut10-400-01-V-6mm-4000x07`, `im_val_cut10-400-01-V-6mm-4000x08`, `im_val_cut10-400-01-V-9mm-4000x12`, `im_val_cut10_400-H-12mm-4000x14`, `im_val_cut10_400-H-6mm-4000x06`, `im_val_cut10_400-H-9mm-4000x10`, `im_val_cut10_400-V-12mm-4000x14`, `im_val_cut10_400-V-6mm-4000x06`, `im_val_cut10_400-V-6mm-4000x07`, `im_val_cut10_400-V-9mm-4000x10`
  - Steel2 (2): `87440491`, `87450661`
- **val**
  - Steel1 (2): `im_val_cut10_400-H-9mm-4000x11`, `im_val_cut10_400-V-9mm-4000x11`
  - Steel2 (1): `87431041`
- **test**
  - Steel1 (2): `im_val_cut10-400-01-H-6mm-4000x06`, `im_val_cut10_400-H-6mm-4000x07`
  - Steel2 (1): `87450741`

## Manifests

- `fold_steel_combined`: `/content/boundary/reports/manifests/fold_steel_combined.csv`
- `fold_steel_combined_test`: `/content/boundary/reports/manifests/fold_steel_combined_test.csv`
