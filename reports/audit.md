# Dataset audit

- generated: 2026-09-03T23:42:26Z
- data root: `/content/drive/MyDrive/phase11-data/data`
- pairing counted over ALL files; pixel statistics over 24 sampled pairs per folder (seed 0)

MODE A masks paint boundaries as their own colour, so phase interfaces AND intra-phase grain boundaries are present. MODE B masks are phase-label maps: `find_boundaries` recovers phase interfaces only, and grain boundaries are absent from the ground truth.

## Verdicts

| folder | mode | pairs | unmatched | mask colours (median) | boundary frac A | boundary frac B |
| --- | --- | --- | --- | --- | --- | --- |
| MetalDam | **B** | 42 | 1 img / 0 mask | 4 | 0.0000 | 0.1736 |
| Steel1 | **B** | 907 | 48 img / 0 mask | 2 | 0.0000 | 0.0329 |
| Steel2 | **B** | 504 | 1 img / 0 mask | 48 | 0.0000 | 0.6487 |
| uhcs1 | **B** | 24 | 0 img / 0 mask | 2 | 0.0000 | 0.1151 |
| uhcs2 | **B** | 24 | 0 img / 0 mask | 4 | 0.0000 | 0.0485 |

## MetalDam

**MODE B** — 4 non-majority colours tested; the best, [75, 176, 40], is line-like in only 1/24 sampled masks, below the 60% quorum

### Pairing

- convention inferred: `split-dirs: mask stem == image stem`
- images dir: `/content/drive/MyDrive/phase11-data/data/MetalDam/images`
- masks dir: `/content/drive/MyDrive/phase11-data/data/MetalDam/masks`
- 43 images, 42 masks, **42 matched pairs**
- unmatched images (1): micrograph100.jpg
- unmatched masks (0): none

### Files

- image: format {'JPEG': 24}, bit depth {8: 24}, colour {'rgb-but-gray': 24}
- image size w 1024/1024/1280, h 703/703/895 (min/median/max)
- mask: format {'PNG': 24}, bit depth {8: 24}, colour {'rgb': 24}
- mask size matches image: True

### Mask palette

- unique colours per mask (min/median/max): 2/4/5
- quantized: True

| colour | pixels | fraction |
| --- | --- | --- |
| `[255, 255, 0]` | 10359872 | 0.558360 |
| `[21, 100, 255]` | 6684071 | 0.360247 |
| `[254, 40, 40]` | 1354323 | 0.072993 |
| `[0, 0, 0]` | 95115 | 0.005126 |
| `[75, 176, 40]` | 60731 | 0.003273 |

### MODE A evidence — every non-majority colour tested

4 colours tested over 24 sampled masks; a colour clears MODE A at 14.4 qualifying masks. Pixel fraction is reported but does not gate the verdict: a colour qualifies on thickness, frame span and network length alone.

| colour | HSV | votes | present in | frac (med) | thick px (med) | span (med) | skel spans (med) | components (med) | failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `[75, 176, 40]` | [104.6, 0.773, 0.69] | 1/24 | 9 | 0.0048 | 3.46 | 0.665 | 1.2 | 142 | thin, span, extent |
| `[21, 100, 255]` | [219.7, 0.918, 1.0] | 0/24 | 24 | 0.3819 | 7.67 | 0.986 | 34.1 | 296 | thin |
| `[254, 40, 40]` | [0.0, 0.843, 0.996] | 0/24 | 18 | 0.0829 | 12.26 | 0.873 | 4.4 | 14 | thin, span, extent |
| `[0, 0, 0]` | [0.0, 0.0, 0.0] | 0/24 | 9 | 0.0130 | 19.72 | 0.093 | 0.2 | 2 | thin, span, extent |

### MODE B structure

- labels per mask (min/median/max): 2/4/5
- majority label is **tessellation** ({'tessellation': 24}), median 307 components
- components per label (min/median/max): 1/225/3603
- component area, median per label (min/median/max): 1/24/28685 px
- smallest component area seen: 1 px; largest: 692233 px

### Boundary pixel fraction each mode would produce

- MODE A (colours that cleared the quorum): 0.0000
- MODE A (any per-file line-like colour, diagnostic only): 0.0000
- MODE B (find_boundaries on the label map): 0.1736

## Steel1

**MODE B** — 1 non-majority colours tested; the best, [255], is line-like in only 0/24 sampled masks, below the 60% quorum

### Pairing

- convention inferred: `split-dirs: mask stem == image stem + '_mask'`
- images dir: `/content/drive/MyDrive/phase11-data/data/Steel1/images`
- masks dir: `/content/drive/MyDrive/phase11-data/data/Steel1/masks`
- 955 images, 907 masks, **907 matched pairs**
- unmatched images (48): im_val_cut10-400-01-V-9mm-4000x13_1_1.png, im_val_cut10-400-01-V-9mm-4000x13_1_2.png, im_val_cut10-400-01-V-9mm-4000x13_1_3.png, im_val_cut10-400-01-V-9mm-4000x13_1_4.png, im_val_cut10-400-01-V-9mm-4000x13_1_5.png, im_val_cut10-400-01-V-9mm-4000x13_1_6.png, im_val_cut10-400-01-V-9mm-4000x13_2_1.png, im_val_cut10-400-01-V-9mm-4000x13_2_2.png, im_val_cut10-400-01-V-9mm-4000x13_2_3.png, im_val_cut10-400-01-V-9mm-4000x13_2_4.png, im_val_cut10-400-01-V-9mm-4000x13_2_5.png, im_val_cut10-400-01-V-9mm-4000x13_2_6.png, im_val_cut10-400-01-V-9mm-4000x13_3_1.png, im_val_cut10-400-01-V-9mm-4000x13_3_2.png, im_val_cut10-400-01-V-9mm-4000x13_3_3.png, im_val_cut10-400-01-V-9mm-4000x13_3_4.png, im_val_cut10-400-01-V-9mm-4000x13_3_5.png, im_val_cut10-400-01-V-9mm-4000x13_3_6.png, im_val_cut10-400-01-V-9mm-4000x13_4_1.png, im_val_cut10-400-01-V-9mm-4000x13_4_2.png ...
- unmatched masks (0): none

### Files

- image: format {'PNG': 24}, bit depth {8: 24}, colour {'rgb-but-gray': 24}
- image size w 256/256/256, h 256/256/256 (min/median/max)
- mask: format {'PNG': 24}, bit depth {8: 24}, colour {'grayscale': 24}
- mask size matches image: True

### Mask palette

- unique colours per mask (min/median/max): 2/2/2
- quantized: True

| colour | pixels | fraction |
| --- | --- | --- |
| `[0]` | 1388653 | 0.882882 |
| `[255]` | 184211 | 0.117118 |

### MODE A evidence — every non-majority colour tested

1 colours tested over 24 sampled masks; a colour clears MODE A at 14.4 qualifying masks. Pixel fraction is reported but does not gate the verdict: a colour qualifies on thickness, frame span and network length alone.

| colour | HSV | votes | present in | frac (med) | thick px (med) | span (med) | skel spans (med) | components (med) | failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `[255]` | [0.0, 0.0, 1.0] | 0/24 | 24 | 0.1143 | 13.20 | 0.988 | 2.3 | 8 | thin, span, extent |

### MODE B structure

- labels per mask (min/median/max): 2/2/2
- majority label is **tessellation** ({'tessellation': 12, 'islands': 12}), median 2 components
- components per label (min/median/max): 1/4/18
- component area, median per label (min/median/max): 55/1332/62972 px
- smallest component area seen: 1 px; largest: 62972 px

### Boundary pixel fraction each mode would produce

- MODE A (colours that cleared the quorum): 0.0000
- MODE A (any per-file line-like colour, diagnostic only): 0.0000
- MODE B (find_boundaries on the label map): 0.0329

## Steel2

**MODE B** — 62 non-majority colours tested; the best, [249, 249, 249], is line-like in only 13/24 sampled masks, below the 60% quorum

### Pairing

- convention inferred: `split-dirs: mask stem == image stem`
- images dir: `/content/drive/MyDrive/phase11-data/data/Steel2/images`
- masks dir: `/content/drive/MyDrive/phase11-data/data/Steel2/masks`
- 505 images, 504 masks, **504 matched pairs**
- unmatched images (1): 87431041_tile_768_512 (1).png
- unmatched masks (0): none

### Files

- image: format {'PNG': 24}, bit depth {8: 24}, colour {'rgb': 24}
- image size w 256/256/256, h 256/256/256 (min/median/max)
- mask: format {'PNG': 24}, bit depth {8: 24}, colour {'grayscale': 11, 'rgb-but-gray': 13}
- mask size matches image: True

### Mask palette

- unique colours per mask (min/median/max): 45/48/52
- quantized: True

| colour | pixels | fraction |
| --- | --- | --- |
| `[255, 255, 255]` | 475973 | 0.311679 |
| `[255]` | 455805 | 0.298472 |
| `[0, 0, 0]` | 122592 | 0.080276 |
| `[0]` | 84069 | 0.055050 |
| `[254, 254, 254]` | 23853 | 0.015620 |
| `[253, 253, 253]` | 23347 | 0.015288 |
| `[252, 252, 252]` | 20723 | 0.013570 |
| `[251, 251, 251]` | 18046 | 0.011817 |
| `[254]` | 17742 | 0.011618 |
| `[253]` | 17324 | 0.011344 |
| `[250, 250, 250]` | 15735 | 0.010304 |
| `[252]` | 15509 | 0.010156 |
| `[1, 1, 1]` | 14879 | 0.009743 |
| `[2, 2, 2]` | 13967 | 0.009146 |
| `[251]` | 13217 | 0.008655 |
| `[249, 249, 249]` | 12720 | 0.008329 |
| `[3, 3, 3]` | 12645 | 0.008280 |
| `[250]` | 11759 | 0.007700 |
| `[4, 4, 4]` | 11046 | 0.007233 |
| `[248, 248, 248]` | 10617 | 0.006952 |

### MODE A evidence — every non-majority colour tested

62 colours tested over 24 sampled masks; a colour clears MODE A at 14.4 qualifying masks. Pixel fraction is reported but does not gate the verdict: a colour qualifies on thickness, frame span and network length alone.

| colour | HSV | votes | present in | frac (med) | thick px (med) | span (med) | skel spans (med) | components (med) | failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `[249, 249, 249]` | [0.0, 0.0, 0.976] | 13/24 | 13 | 0.0151 | 1.00 | 1.000 | 3.9 | 896 | - |
| `[248, 248, 248]` | [0.0, 0.0, 0.973] | 13/24 | 13 | 0.0125 | 1.00 | 1.000 | 3.2 | 739 | - |
| `[250, 250, 250]` | [0.0, 0.0, 0.98] | 13/24 | 13 | 0.0188 | 1.00 | 1.000 | 4.8 | 1081 | - |
| `[251, 251, 251]` | [0.0, 0.0, 0.984] | 13/24 | 13 | 0.0215 | 1.01 | 1.000 | 5.5 | 1207 | - |
| `[252, 252, 252]` | [0.0, 0.0, 0.988] | 13/24 | 13 | 0.0251 | 1.02 | 1.000 | 6.3 | 1352 | - |
| `[2, 2, 2]` | [0.0, 0.0, 0.008] | 13/24 | 13 | 0.0162 | 1.02 | 1.000 | 4.1 | 836 | - |
| `[254, 254, 254]` | [0.0, 0.0, 0.996] | 13/24 | 13 | 0.0281 | 1.02 | 1.000 | 7.1 | 1455 | - |
| `[253, 253, 253]` | [0.0, 0.0, 0.992] | 13/24 | 13 | 0.0276 | 1.02 | 1.000 | 6.9 | 1438 | - |
| `[1, 1, 1]` | [0.0, 0.0, 0.004] | 13/24 | 13 | 0.0164 | 1.02 | 1.000 | 4.1 | 837 | - |
| `[0, 0, 0]` | [0.0, 0.0, 0.0] | 13/24 | 13 | 0.1422 | 1.23 | 1.000 | 28.8 | 390 | - |
| `[247, 247, 247]` | [0.0, 0.0, 0.969] | 12/24 | 13 | 0.0092 | 1.00 | 0.996 | 2.3 | 562 | extent |
| `[4, 4, 4]` | [0.0, 0.0, 0.016] | 12/24 | 13 | 0.0128 | 1.01 | 1.000 | 3.2 | 729 | extent |
| `[3, 3, 3]` | [0.0, 0.0, 0.012] | 12/24 | 13 | 0.0141 | 1.01 | 1.000 | 3.6 | 764 | extent |
| `[249]` | [0.0, 0.0, 0.976] | 11/24 | 11 | 0.0132 | 1.00 | 1.000 | 3.4 | 784 | - |
| `[6, 6, 6]` | [0.0, 0.0, 0.024] | 11/24 | 13 | 0.0091 | 1.00 | 0.996 | 2.3 | 530 | extent |
| `[250]` | [0.0, 0.0, 0.98] | 11/24 | 11 | 0.0166 | 1.00 | 1.000 | 4.3 | 957 | - |
| `[5, 5, 5]` | [0.0, 0.0, 0.02] | 11/24 | 13 | 0.0103 | 1.00 | 1.000 | 2.6 | 604 | extent |
| `[251]` | [0.0, 0.0, 0.984] | 11/24 | 11 | 0.0187 | 1.01 | 1.000 | 4.7 | 1073 | - |
| `[252]` | [0.0, 0.0, 0.988] | 11/24 | 11 | 0.0221 | 1.02 | 1.000 | 5.6 | 1210 | - |
| `[254]` | [0.0, 0.0, 0.996] | 11/24 | 11 | 0.0247 | 1.02 | 1.000 | 6.2 | 1259 | - |
| `[253]` | [0.0, 0.0, 0.992] | 11/24 | 11 | 0.0245 | 1.02 | 1.000 | 6.1 | 1270 | - |
| `[0]` | [0.0, 0.0, 0.0] | 11/24 | 11 | 0.1333 | 1.24 | 1.000 | 27.6 | 382 | - |
| `[248]` | [0.0, 0.0, 0.973] | 10/24 | 11 | 0.0105 | 1.00 | 1.000 | 2.7 | 628 | extent |
| `[1]` | [0.0, 0.0, 0.004] | 9/24 | 11 | 0.0165 | 1.02 | 1.000 | 4.1 | 823 | extent |
| `[4]` | [0.0, 0.0, 0.016] | 8/24 | 11 | 0.0117 | 1.01 | 1.000 | 3.0 | 657 | extent |
| `[3]` | [0.0, 0.0, 0.012] | 8/24 | 11 | 0.0130 | 1.01 | 1.000 | 3.3 | 718 | extent |
| `[2]` | [0.0, 0.0, 0.008] | 8/24 | 11 | 0.0147 | 1.02 | 1.000 | 3.7 | 774 | extent |
| `[6]` | [0.0, 0.0, 0.024] | 7/24 | 11 | 0.0082 | 1.00 | 0.996 | 2.1 | 490 | extent |
| `[246, 246, 246]` | [0.0, 0.0, 0.965] | 7/24 | 13 | 0.0080 | 1.00 | 0.996 | 2.0 | 496 | extent |
| `[5]` | [0.0, 0.0, 0.02] | 7/24 | 11 | 0.0106 | 1.00 | 0.996 | 2.7 | 606 | extent |
| `[7, 7, 7]` | [0.0, 0.0, 0.027] | 5/24 | 13 | 0.0072 | 1.00 | 0.996 | 1.8 | 437 | extent |
| `[247]` | [0.0, 0.0, 0.969] | 4/24 | 11 | 0.0077 | 1.00 | 0.996 | 2.0 | 471 | extent |
| `[246]` | [0.0, 0.0, 0.965] | 3/24 | 11 | 0.0068 | 1.00 | 0.992 | 1.8 | 422 | extent |
| `[7]` | [0.0, 0.0, 0.027] | 3/24 | 11 | 0.0067 | 1.00 | 0.992 | 1.7 | 405 | extent |
| `[8, 8, 8]` | [0.0, 0.0, 0.031] | 2/24 | 13 | 0.0059 | 1.00 | 0.996 | 1.5 | 365 | extent |
| `[245]` | [0.0, 0.0, 0.961] | 0/24 | 11 | 0.0060 | 1.00 | 0.988 | 1.5 | 365 | extent |
| `[244]` | [0.0, 0.0, 0.957] | 0/24 | 11 | 0.0045 | 1.00 | 0.984 | 1.2 | 292 | extent |
| `[8]` | [0.0, 0.0, 0.031] | 0/24 | 11 | 0.0057 | 1.00 | 0.988 | 1.4 | 346 | extent |
| `[9]` | [0.0, 0.0, 0.035] | 0/24 | 11 | 0.0043 | 1.00 | 0.984 | 1.1 | 264 | extent |
| `[243]` | [0.0, 0.0, 0.953] | 0/24 | 11 | 0.0028 | 1.00 | 0.984 | 0.7 | 184 | extent |
| `[10]` | [0.0, 0.0, 0.039] | 0/24 | 10 | 0.0038 | 1.00 | 0.979 | 1.0 | 234 | extent |
| `[242]` | [0.0, 0.0, 0.949] | 0/24 | 11 | 0.0018 | 1.00 | 0.969 | 0.5 | 118 | extent |
| `[11]` | [0.0, 0.0, 0.043] | 0/24 | 9 | 0.0027 | 1.00 | 0.977 | 0.7 | 170 | extent |
| `[12]` | [0.0, 0.0, 0.047] | 0/24 | 8 | 0.0019 | 1.00 | 0.961 | 0.5 | 122 | extent |
| `[241]` | [0.0, 0.0, 0.945] | 0/24 | 8 | 0.0012 | 1.00 | 0.965 | 0.3 | 78 | extent |
| `[13]` | [0.0, 0.0, 0.051] | 0/24 | 7 | 0.0014 | 1.00 | 0.981 | 0.4 | 92 | extent |
| `[14]` | [0.0, 0.0, 0.055] | 0/24 | 2 | 0.0010 | 1.00 | 0.927 | 0.3 | 68 | extent |
| `[239]` | [0.0, 0.0, 0.937] | 0/24 | 5 | 0.0011 | 1.00 | 0.961 | 0.3 | 71 | extent |
| `[240]` | [0.0, 0.0, 0.941] | 0/24 | 1 | 0.0010 | 1.00 | 0.851 | 0.2 | 64 | extent |
| `[245, 245, 245]` | [0.0, 0.0, 0.961] | 0/24 | 13 | 0.0070 | 1.00 | 0.992 | 1.8 | 431 | extent |
| `[244, 244, 244]` | [0.0, 0.0, 0.957] | 0/24 | 13 | 0.0050 | 1.00 | 0.988 | 1.3 | 323 | extent |
| `[9, 9, 9]` | [0.0, 0.0, 0.035] | 0/24 | 13 | 0.0043 | 1.00 | 0.988 | 1.1 | 270 | extent |
| `[10, 10, 10]` | [0.0, 0.0, 0.039] | 0/24 | 13 | 0.0036 | 1.00 | 0.988 | 0.9 | 230 | extent |
| `[243, 243, 243]` | [0.0, 0.0, 0.953] | 0/24 | 13 | 0.0031 | 1.00 | 0.977 | 0.8 | 201 | extent |
| `[11, 11, 11]` | [0.0, 0.0, 0.043] | 0/24 | 13 | 0.0026 | 1.00 | 0.977 | 0.7 | 169 | extent |
| `[242, 242, 242]` | [0.0, 0.0, 0.949] | 0/24 | 13 | 0.0023 | 1.00 | 0.973 | 0.6 | 148 | extent |
| `[12, 12, 12]` | [0.0, 0.0, 0.047] | 0/24 | 13 | 0.0018 | 1.00 | 0.965 | 0.5 | 118 | extent |
| `[239, 239, 239]` | [0.0, 0.0, 0.937] | 0/24 | 11 | 0.0011 | 1.00 | 0.946 | 0.3 | 70 | extent |
| `[241, 241, 241]` | [0.0, 0.0, 0.945] | 0/24 | 13 | 0.0014 | 1.00 | 0.942 | 0.4 | 93 | extent |
| `[13, 13, 13]` | [0.0, 0.0, 0.051] | 0/24 | 11 | 0.0015 | 1.00 | 0.969 | 0.4 | 95 | extent |
| `[240, 240, 240]` | [0.0, 0.0, 0.941] | 0/24 | 9 | 0.0011 | 1.00 | 0.950 | 0.3 | 70 | extent |
| `[14, 14, 14]` | [0.0, 0.0, 0.055] | 0/24 | 4 | 0.0011 | 1.00 | 0.952 | 0.3 | 73 | extent |

### MODE B structure

- labels per mask (min/median/max): 32/32/32
- majority label is **tessellation** ({'tessellation': 24}), median 3068 components
- components per label (min/median/max): 19/476/3778
- component area, median per label (min/median/max): 1/1/1 px
- smallest component area seen: 1 px; largest: 48317 px

### Boundary pixel fraction each mode would produce

- MODE A (colours that cleared the quorum): 0.0000
- MODE A (any per-file line-like colour, diagnostic only): 0.3729
- MODE B (find_boundaries on the label map): 0.6487

## uhcs1

**MODE B** — 1 non-majority colours tested; the best, [255], is line-like in only 1/24 sampled masks, below the 60% quorum

### Pairing

- convention inferred: `split-dirs: mask stem == image stem`
- images dir: `/content/drive/MyDrive/phase11-data/data/uhcs1/images`
- masks dir: `/content/drive/MyDrive/phase11-data/data/uhcs1/masks`
- 24 images, 24 masks, **24 matched pairs**
- unmatched images (0): none
- unmatched masks (0): none

### Files

- image: format {'PNG': 24}, bit depth {8: 24}, colour {'grayscale': 24}
- image size w 645/645/645, h 475/475/475 (min/median/max)
- mask: format {'PNG': 24}, bit depth {8: 24}, colour {'grayscale': 24}
- mask size matches image: True

### Mask palette

- unique colours per mask (min/median/max): 2/2/2
- quantized: True

| colour | pixels | fraction |
| --- | --- | --- |
| `[0]` | 6234878 | 0.847937 |
| `[255]` | 1118122 | 0.152063 |

### MODE A evidence — every non-majority colour tested

1 colours tested over 24 sampled masks; a colour clears MODE A at 14.4 qualifying masks. Pixel fraction is reported but does not gate the verdict: a colour qualifies on thickness, frame span and network length alone.

| colour | HSV | votes | present in | frac (med) | thick px (med) | span (med) | skel spans (med) | components (med) | failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `[255]` | [0.0, 0.0, 1.0] | 1/24 | 24 | 0.1559 | 11.44 | 0.984 | 6.7 | 751 | thin |

### MODE B structure

- labels per mask (min/median/max): 2/2/2
- majority label is **islands** ({'islands': 15, 'tessellation': 9}), median 1 components
- components per label (min/median/max): 1/60/5075
- component area, median per label (min/median/max): 1/36/267646 px
- smallest component area seen: 1 px; largest: 276546 px

### Boundary pixel fraction each mode would produce

- MODE A (colours that cleared the quorum): 0.0000
- MODE A (any per-file line-like colour, diagnostic only): 0.0000
- MODE B (find_boundaries on the label map): 0.1151

## uhcs2

**MODE B** — 3 non-majority colours tested; the best, [75, 176, 40], is line-like in only 2/24 sampled masks, below the 60% quorum

### Pairing

- convention inferred: `split-dirs: mask stem == image stem`
- images dir: `/content/drive/MyDrive/phase11-data/data/uhcs2/images`
- masks dir: `/content/drive/MyDrive/phase11-data/data/uhcs2/masks`
- 24 images, 24 masks, **24 matched pairs**
- unmatched images (0): none
- unmatched masks (0): none

### Files

- image: format {'JPEG': 24}, bit depth {8: 24}, colour {'grayscale': 24}
- image size w 645/645/645, h 484/484/484 (min/median/max)
- mask: format {'PNG': 24}, bit depth {8: 24}, colour {'rgb': 24}
- mask size matches image: True

### Mask palette

- unique colours per mask (min/median/max): 1/4/4
- quantized: True

| colour | pixels | fraction |
| --- | --- | --- |
| `[254, 40, 40]` | 4799762 | 0.640624 |
| `[21, 100, 255]` | 1207721 | 0.161195 |
| `[255, 255, 0]` | 972313 | 0.129775 |
| `[75, 176, 40]` | 512524 | 0.068407 |

### MODE A evidence — every non-majority colour tested

3 colours tested over 24 sampled masks; a colour clears MODE A at 14.4 qualifying masks. Pixel fraction is reported but does not gate the verdict: a colour qualifies on thickness, frame span and network length alone.

| colour | HSV | votes | present in | frac (med) | thick px (med) | span (med) | skel spans (med) | components (med) | failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `[75, 176, 40]` | [104.6, 0.773, 0.69] | 2/24 | 23 | 0.0169 | 5.10 | 0.702 | 1.9 | 14 | thin, span, extent |
| `[21, 100, 255]` | [219.7, 0.918, 1.0] | 0/24 | 23 | 0.1591 | 13.87 | 1.000 | 4.3 | 10 | thin, span, extent |
| `[255, 255, 0]` | [60.0, 1.0, 1.0] | 0/24 | 23 | 0.1415 | 28.03 | 0.758 | 1.6 | 3 | thin, span, extent |

### MODE B structure

- labels per mask (min/median/max): 1/4/4
- majority label is **tessellation** ({'tessellation': 23, 'islands': 1}), median 8 components
- components per label (min/median/max): 1/9/137
- component area, median per label (min/median/max): 1/343/312180 px
- smallest component area seen: 1 px; largest: 312180 px

### Boundary pixel fraction each mode would produce

- MODE A (colours that cleared the quorum): 0.0000
- MODE A (any per-file line-like colour, diagnostic only): 0.0000
- MODE B (find_boundaries on the label map): 0.0485
