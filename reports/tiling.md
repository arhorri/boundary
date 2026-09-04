# Tiling and folds

- generated: 2026-09-04T11:16:10Z
- patch 256 px, stride 128 px (50% overlap); the last tile of each row and column is clamped to the image edge, so no pixel is fabricated. Padding applies only to an image smaller than the patch in an axis (reflect mode), and every such pixel is counted below.
- tiles below 0.005 boundary fraction are dropped
- nothing is resized and no tile images are written: a tile is a row in a manifest and the loader crops it on the fly

## Tiles per dataset

| dataset | parents | images | tiles | padded px | dropped (low boundary) | excluded | no mask | no boundary map |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MetalDam | 42 | 42 | 1332 | 0 | 0 | 0 | 1 | 0 |
| Steel1 | 19 | 902 | 902 | 0 | 5 | 0 | 48 | 0 |
| Steel2 | 4 | 504 | 504 | 0 | 0 | 0 | 1 | 0 |
| uhcs1 | 24 | 24 | 288 | 0 | 0 | 0 | 0 | 0 |
| uhcs2 | 23 | 23 | 265 | 0 | 11 | 0 | 0 | 1 |

**Fabricated pixels: 0.** Every tile is real image data.

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
| fold_MetalDam | MetalDam | 1266 | 1521 | 62 | 46 | 15.974 | 0.058913 |
| fold_uhcs1 | uhcs1 | 2310 | 477 | 80 | 28 | 7.45 | 0.118337 |
| fold_uhcs2 | uhcs2 | 2333 | 454 | 81 | 27 | 7.111 | 0.123292 |
| dev (= fold_uhcs2) | uhcs2 | 2333 | 454 | 81 | 27 | 7.111 | 0.123292 |

- test manifest: ['Steel2'], 504 tiles from 4 parents, mean boundary fraction 0.131535

## Sampling weights

Raw tile counts do not measure independent information: Steel1's tiles come from a handful of micrographs. The sampler weights each dataset so its share of an epoch matches its share of the independent scenes.

### fold_MetalDam

- mode: `parents`

| dataset | parents | raw tiles | weight | effective tiles/epoch |
| --- | --- | --- | --- | --- |
| Steel1 | 15 | 713 | 0.42958 | 306 |
| uhcs1 | 24 | 288 | 1.701613 | 490 |
| uhcs2 | 23 | 265 | 1.772246 | 470 |

### fold_uhcs1

- mode: `parents`

| dataset | parents | raw tiles | weight | effective tiles/epoch |
| --- | --- | --- | --- | --- |
| MetalDam | 42 | 1332 | 0.910473 | 1213 |
| Steel1 | 15 | 713 | 0.607468 | 433 |
| uhcs2 | 23 | 265 | 2.506132 | 664 |

### fold_uhcs2

- mode: `parents`

| dataset | parents | raw tiles | weight | effective tiles/epoch |
| --- | --- | --- | --- | --- |
| MetalDam | 42 | 1332 | 0.908186 | 1210 |
| Steel1 | 15 | 713 | 0.605943 | 432 |
| uhcs1 | 24 | 288 | 2.400206 | 691 |

### dev

- mode: `parents`

| dataset | parents | raw tiles | weight | effective tiles/epoch |
| --- | --- | --- | --- | --- |
| MetalDam | 42 | 1332 | 0.908186 | 1210 |
| Steel1 | 15 | 713 | 0.605943 | 432 |
| uhcs1 | 24 | 288 | 2.400206 | 691 |

## Manifests

- `fold_MetalDam`: `/content/boundary/reports/manifests/fold_MetalDam.csv`
- `fold_uhcs1`: `/content/boundary/reports/manifests/fold_uhcs1.csv`
- `fold_uhcs2`: `/content/boundary/reports/manifests/fold_uhcs2.csv`
- `dev`: `/content/boundary/reports/manifests/dev.csv`
- `test`: `/content/boundary/reports/manifests/test.csv`
