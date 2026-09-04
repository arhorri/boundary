# Parent grouping

A *parent* is the source micrograph a tile was cut from. Splits are made by parent, never by tile: two tiles 128 px apart show the same grains under the same etch, so a validation tile from a training parent is leakage and its score means nothing.

| dataset | parents | source images | tiles | tiles per parent (min/median/max) |
| --- | --- | --- | --- | --- |
| MetalDam | 42 | 42 | 1332 | 28/28/54 |
| Steel1 | 19 | 902 | 902 | 43/48/48 |
| Steel2 | 4 | 504 | 504 | 126/126/126 |
| uhcs1 | 24 | 24 | 288 | 12/12/12 |
| uhcs2 | 23 | 23 | 265 | 10/12/12 |

## MetalDam

- rule: `filename is the parent`
- 42 parents, 1332 tiles kept, 0 dropped below min_boundary_frac, 0 excluded, 1 images with no mask

| parent | source images | tiles |
| --- | --- | --- |
| `micrograph0` | 1 | 28 |
| `micrograph1` | 1 | 28 |
| `micrograph10` | 1 | 28 |
| `micrograph11` | 1 | 28 |
| `micrograph12` | 1 | 28 |
| `micrograph13` | 1 | 28 |
| `micrograph14` | 1 | 28 |
| `micrograph15` | 1 | 28 |
| `micrograph16` | 1 | 28 |
| `micrograph17` | 1 | 28 |
| `micrograph18` | 1 | 28 |
| `micrograph19` | 1 | 28 |
| `micrograph2` | 1 | 28 |
| `micrograph20` | 1 | 28 |
| `micrograph21` | 1 | 54 |
| `micrograph22` | 1 | 54 |
| `micrograph23` | 1 | 28 |
| `micrograph24` | 1 | 28 |
| `micrograph25` | 1 | 28 |
| `micrograph26` | 1 | 28 |
| `micrograph27` | 1 | 28 |
| `micrograph28` | 1 | 28 |
| `micrograph29` | 1 | 28 |
| `micrograph3` | 1 | 28 |
| `micrograph30` | 1 | 28 |
| `micrograph31` | 1 | 28 |
| `micrograph32` | 1 | 54 |
| `micrograph33` | 1 | 28 |
| `micrograph34` | 1 | 28 |
| `micrograph35` | 1 | 28 |
| `micrograph36` | 1 | 28 |
| `micrograph37` | 1 | 28 |
| `micrograph38` | 1 | 28 |
| `micrograph39` | 1 | 28 |
| `micrograph4` | 1 | 54 |
| `micrograph40` | 1 | 28 |
| `micrograph41` | 1 | 54 |
| `micrograph5` | 1 | 28 |
| `micrograph6` | 1 | 28 |
| `micrograph7` | 1 | 28 |
| `micrograph8` | 1 | 54 |
| `micrograph9` | 1 | 28 |

### Images with no mask (1)

These never had ground truth, so they cannot be tiled. 0 of them are named explicitly in `tiling.exclusions`; the rest are listed here so that none of them vanishes silently.
- `micrograph100.jpg`

## Steel1

- rule: `_\d+_\d+$`
- 19 parents, 902 tiles kept, 5 dropped below min_boundary_frac, 0 excluded, 48 images with no mask

| parent | source images | tiles |
| --- | --- | --- |
| `im_val_cut10-400-01-H-12mm-4000x14` | 47 | 47 |
| `im_val_cut10-400-01-H-6mm-4000x06` | 48 | 48 |
| `im_val_cut10-400-01-H-9mm-4000x10` | 48 | 48 |
| `im_val_cut10-400-01-H-9mm-4000x12` | 48 | 48 |
| `im_val_cut10-400-01-V-12mm-4000x16` | 48 | 48 |
| `im_val_cut10-400-01-V-12mm-4000x17` | 48 | 48 |
| `im_val_cut10-400-01-V-6mm-4000x07` | 48 | 48 |
| `im_val_cut10-400-01-V-6mm-4000x08` | 48 | 48 |
| `im_val_cut10-400-01-V-9mm-4000x12` | 48 | 48 |
| `im_val_cut10_400-H-12mm-4000x14` | 48 | 48 |
| `im_val_cut10_400-H-6mm-4000x06` | 48 | 48 |
| `im_val_cut10_400-H-6mm-4000x07` | 48 | 48 |
| `im_val_cut10_400-H-9mm-4000x10` | 48 | 48 |
| `im_val_cut10_400-H-9mm-4000x11` | 45 | 45 |
| `im_val_cut10_400-V-12mm-4000x14` | 47 | 47 |
| `im_val_cut10_400-V-6mm-4000x06` | 48 | 48 |
| `im_val_cut10_400-V-6mm-4000x07` | 43 | 43 |
| `im_val_cut10_400-V-9mm-4000x10` | 48 | 48 |
| `im_val_cut10_400-V-9mm-4000x11` | 48 | 48 |

### Images with no mask (48)

These never had ground truth, so they cannot be tiled. 48 of them are named explicitly in `tiling.exclusions`; the rest are listed here so that none of them vanishes silently.
- `im_val_cut10-400-01-V-9mm-4000x13_1_1.png`
- `im_val_cut10-400-01-V-9mm-4000x13_1_2.png`
- `im_val_cut10-400-01-V-9mm-4000x13_1_3.png`
- `im_val_cut10-400-01-V-9mm-4000x13_1_4.png`
- `im_val_cut10-400-01-V-9mm-4000x13_1_5.png`
- `im_val_cut10-400-01-V-9mm-4000x13_1_6.png`
- `im_val_cut10-400-01-V-9mm-4000x13_2_1.png`
- `im_val_cut10-400-01-V-9mm-4000x13_2_2.png`
- `im_val_cut10-400-01-V-9mm-4000x13_2_3.png`
- `im_val_cut10-400-01-V-9mm-4000x13_2_4.png`
- `im_val_cut10-400-01-V-9mm-4000x13_2_5.png`
- `im_val_cut10-400-01-V-9mm-4000x13_2_6.png`
- `im_val_cut10-400-01-V-9mm-4000x13_3_1.png`
- `im_val_cut10-400-01-V-9mm-4000x13_3_2.png`
- `im_val_cut10-400-01-V-9mm-4000x13_3_3.png`
- `im_val_cut10-400-01-V-9mm-4000x13_3_4.png`
- `im_val_cut10-400-01-V-9mm-4000x13_3_5.png`
- `im_val_cut10-400-01-V-9mm-4000x13_3_6.png`
- `im_val_cut10-400-01-V-9mm-4000x13_4_1.png`
- `im_val_cut10-400-01-V-9mm-4000x13_4_2.png`
- `im_val_cut10-400-01-V-9mm-4000x13_4_3.png`
- `im_val_cut10-400-01-V-9mm-4000x13_4_4.png`
- `im_val_cut10-400-01-V-9mm-4000x13_4_5.png`
- `im_val_cut10-400-01-V-9mm-4000x13_4_6.png`
- `im_val_cut10-400-01-V-9mm-4000x13_5_1.png`
- `im_val_cut10-400-01-V-9mm-4000x13_5_2.png`
- `im_val_cut10-400-01-V-9mm-4000x13_5_3.png`
- `im_val_cut10-400-01-V-9mm-4000x13_5_4.png`
- `im_val_cut10-400-01-V-9mm-4000x13_5_5.png`
- `im_val_cut10-400-01-V-9mm-4000x13_5_6.png`
- `im_val_cut10-400-01-V-9mm-4000x13_6_1.png`
- `im_val_cut10-400-01-V-9mm-4000x13_6_2.png`
- `im_val_cut10-400-01-V-9mm-4000x13_6_3.png`
- `im_val_cut10-400-01-V-9mm-4000x13_6_4.png`
- `im_val_cut10-400-01-V-9mm-4000x13_6_5.png`
- `im_val_cut10-400-01-V-9mm-4000x13_6_6.png`
- `im_val_cut10-400-01-V-9mm-4000x13_7_1.png`
- `im_val_cut10-400-01-V-9mm-4000x13_7_2.png`
- `im_val_cut10-400-01-V-9mm-4000x13_7_3.png`
- `im_val_cut10-400-01-V-9mm-4000x13_7_4.png`
- `im_val_cut10-400-01-V-9mm-4000x13_7_5.png`
- `im_val_cut10-400-01-V-9mm-4000x13_7_6.png`
- `im_val_cut10-400-01-V-9mm-4000x13_8_1.png`
- `im_val_cut10-400-01-V-9mm-4000x13_8_2.png`
- `im_val_cut10-400-01-V-9mm-4000x13_8_3.png`
- `im_val_cut10-400-01-V-9mm-4000x13_8_4.png`
- `im_val_cut10-400-01-V-9mm-4000x13_8_5.png`
- `im_val_cut10-400-01-V-9mm-4000x13_8_6.png`

## Steel2

- rule: `_tile_\d+_\d+$`
- 4 parents, 504 tiles kept, 0 dropped below min_boundary_frac, 0 excluded, 1 images with no mask

| parent | source images | tiles |
| --- | --- | --- |
| `87431041` | 126 | 126 |
| `87440491` | 126 | 126 |
| `87450661` | 126 | 126 |
| `87450741` | 126 | 126 |

### Images with no mask (1)

These never had ground truth, so they cannot be tiled. 0 of them are named explicitly in `tiling.exclusions`; the rest are listed here so that none of them vanishes silently.
- `87431041_tile_768_512 (1).png`

## uhcs1

- rule: `filename is the parent`
- 24 parents, 288 tiles kept, 0 dropped below min_boundary_frac, 0 excluded, 0 images with no mask

| parent | source images | tiles |
| --- | --- | --- |
| `800C-24H-Q-1` | 1 | 12 |
| `800C-24H-Q-2` | 1 | 12 |
| `800C-24H-Q-3` | 1 | 12 |
| `800C-24H-Q-4` | 1 | 12 |
| `800C-24H-Q-5` | 1 | 12 |
| `800C-24H-Q-6` | 1 | 12 |
| `800C-24H-Q-7` | 1 | 12 |
| `800C-3H-Q-1` | 1 | 12 |
| `800C-3H-Q-2` | 1 | 12 |
| `800C-3H-Q-3` | 1 | 12 |
| `800C-3H-Q-4` | 1 | 12 |
| `800C-85H-Q-1` | 1 | 12 |
| `800C-85H-Q-2` | 1 | 12 |
| `800C-85H-Q-3` | 1 | 12 |
| `800C-85H-Q-5` | 1 | 12 |
| `800C-85H-Q-6` | 1 | 12 |
| `800C-8H-Q-1` | 1 | 12 |
| `800C-8H-Q-3` | 1 | 12 |
| `800C-8H-Q-4` | 1 | 12 |
| `970C-5M-Q-1` | 1 | 12 |
| `970C-5M-Q-2` | 1 | 12 |
| `970C-5M-Q-3` | 1 | 12 |
| `970C-5M-Q-4` | 1 | 12 |
| `970C-5M-Q-5` | 1 | 12 |

## uhcs2

- rule: `filename is the parent`
- 23 parents, 265 tiles kept, 11 dropped below min_boundary_frac, 0 excluded, 0 images with no mask

| parent | source images | tiles |
| --- | --- | --- |
| `uhcs0006` | 1 | 12 |
| `uhcs0007` | 1 | 12 |
| `uhcs0075` | 1 | 12 |
| `uhcs0124` | 1 | 10 |
| `uhcs0220` | 1 | 12 |
| `uhcs0235` | 1 | 11 |
| `uhcs0295` | 1 | 10 |
| `uhcs0312` | 1 | 12 |
| `uhcs0333` | 1 | 12 |
| `uhcs0357` | 1 | 12 |
| `uhcs0360` | 1 | 11 |
| `uhcs0477` | 1 | 12 |
| `uhcs0495` | 1 | 12 |
| `uhcs0579` | 1 | 12 |
| `uhcs0599` | 1 | 12 |
| `uhcs1061` | 1 | 11 |
| `uhcs1150` | 1 | 12 |
| `uhcs1176` | 1 | 12 |
| `uhcs1219` | 1 | 12 |
| `uhcs1289` | 1 | 11 |
| `uhcs1528` | 1 | 12 |
| `uhcs1579` | 1 | 10 |
| `uhcs1648` | 1 | 11 |
