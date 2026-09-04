# Tiling and folds

- generated: 2026-09-04T10:14:11Z
- patch 256 px, stride 128 px (50% overlap), reflect padding on the right and bottom edges only
- tiles below 0.005 boundary fraction are dropped
- nothing is resized and no tile images are written: a tile is a row in a manifest and the loader crops it on the fly

## Tiles per dataset

| dataset | parents | images | tiles | dropped (low boundary) | excluded | no mask | no boundary map |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MetalDam | 42 | 42 | 1584 | 0 | 0 | 1 | 0 |
| Steel1 | 19 | 902 | 902 | 5 | 0 | 48 | 0 |
| Steel2 | 4 | 504 | 504 | 0 | 0 | 1 | 0 |
| uhcs1 | 24 | 24 | 360 | 0 | 0 | 0 | 0 |
| uhcs2 | 23 | 23 | 325 | 20 | 0 | 0 | 1 |

## Images lost entirely to the boundary filter

Each of these is a pre-tiled image whose ONLY tile fell below the threshold, so the image is absent from every split.

- **Steel1/im_val_cut10-400-01-H-12mm-4000x14_7_1.png** — boundary fraction 0.00021
- **Steel1/im_val_cut10_400-H-9mm-4000x11_2_3.png** — boundary fraction 0.00476
- **Steel1/im_val_cut10_400-H-9mm-4000x11_6_1.png** — boundary fraction 0.00369
- **Steel1/im_val_cut10_400-V-12mm-4000x14_3_6.png** — boundary fraction 0.00107
- **Steel1/im_val_cut10_400-V-6mm-4000x07_2_3.png** — boundary fraction 0.00180

## Folds

Steel2 is test-only: the domain-shift fold, evaluated but never learned from. Steel1 is split by parent with a fixed seed, so its tiles never straddle train and val. The remaining three datasets rotate as the held-out validation set.

| fold | held out | train tiles | val tiles | train parents | val parents | pos_weight | train frac (mean) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fold_MetalDam | MetalDam | 1398 | 1773 | 62 | 46 | 15.527 | 0.060509 |
| fold_uhcs1 | uhcs1 | 2622 | 549 | 80 | 28 | 7.144 | 0.122783 |
| fold_uhcs2 | uhcs2 | 2657 | 514 | 81 | 27 | 6.825 | 0.127793 |
| dev (= fold_uhcs2) | uhcs2 | 2657 | 514 | 81 | 27 | 6.825 | 0.127793 |

- test manifest: ['Steel2'], 504 tiles from 4 parents, mean boundary fraction 0.131535

## Sampling weights

Raw tile counts do not measure independent information: Steel1's tiles come from a handful of micrographs. The sampler weights each dataset so its share of an epoch matches its share of the independent scenes.

### fold_MetalDam

- mode: `parents`

| dataset | parents | raw tiles | weight | effective tiles/epoch |
| --- | --- | --- | --- | --- |
| Steel1 | 15 | 713 | 0.47437 | 338 |
| uhcs1 | 24 | 360 | 1.503226 | 541 |
| uhcs2 | 23 | 325 | 1.595732 | 519 |

### fold_uhcs1

- mode: `parents`

| dataset | parents | raw tiles | weight | effective tiles/epoch |
| --- | --- | --- | --- | --- |
| MetalDam | 42 | 1584 | 0.869034 | 1377 |
| Steel1 | 15 | 713 | 0.689516 | 492 |
| uhcs2 | 23 | 325 | 2.319462 | 754 |

### fold_uhcs2

- mode: `parents`

| dataset | parents | raw tiles | weight | effective tiles/epoch |
| --- | --- | --- | --- | --- |
| MetalDam | 42 | 1584 | 0.869762 | 1378 |
| Steel1 | 15 | 713 | 0.690094 | 492 |
| uhcs1 | 24 | 360 | 2.186831 | 787 |

### dev

- mode: `parents`

| dataset | parents | raw tiles | weight | effective tiles/epoch |
| --- | --- | --- | --- | --- |
| MetalDam | 42 | 1584 | 0.869762 | 1378 |
| Steel1 | 15 | 713 | 0.690094 | 492 |
| uhcs1 | 24 | 360 | 2.186831 | 787 |

## Manifests

- `fold_MetalDam`: `/content/boundary/reports/manifests/fold_MetalDam.csv`
- `fold_uhcs1`: `/content/boundary/reports/manifests/fold_uhcs1.csv`
- `fold_uhcs2`: `/content/boundary/reports/manifests/fold_uhcs2.csv`
- `dev`: `/content/boundary/reports/manifests/dev.csv`
- `test`: `/content/boundary/reports/manifests/test.csv`
