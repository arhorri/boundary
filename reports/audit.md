# Dataset audit

- generated: 2026-09-03T23:01:05Z
- data root: `/content/drive/MyDrive/phase11-data/data`
- pairing counted over ALL files; pixel statistics over 24 sampled pairs per folder (seed 0)

MODE A masks paint boundaries as their own colour, so phase interfaces AND intra-phase grain boundaries are present. MODE B masks are phase-label maps: `find_boundaries` recovers phase interfaces only, and grain boundaries are absent from the ground truth.

## Verdicts

| folder | mode | pairs | unmatched | mask colours (median) | boundary frac A | boundary frac B |
| --- | --- | --- | --- | --- | --- | --- |
| MetalDam | **B** | 42 | 1 img / 0 mask | 4 | 0.0000 | 0.1736 |
| Steel1 | **B** | 907 | 48 img / 0 mask | 2 | 0.0000 | 0.0329 |
| Steel2 | **B** | 504 | 1 img / 0 mask | 48 | 0.0146 | 0.6487 |
| uhcs1 | **B** | 24 | 0 img / 0 mask | 2 | 0.0000 | 0.1151 |
| uhcs2 | **B** | 24 | 0 img / 0 mask | 4 | 0.0000 | 0.0485 |

## MetalDam

**MODE B** — colour [75, 176, 40] looks line-like in only 4/24 sampled masks, below the 60% quorum

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

### MODE A evidence

- candidate colour: `[75, 176, 40]` (HSV [104.6, 0.773, 0.69])
- qualifying in 4/24 sampled masks (quorum 14.4)
- pixel fraction (min/median/max): 0.0024/0.0048/0.0343
- skeleton-to-area: 0.2894 (mean thickness 3.46 px)
- components: 142, frame span 0.665

### MODE B structure

- labels per mask (min/median/max): 2/4/5
- majority label is **tessellation** ({'tessellation': 24}), median 307 components
- components per label (min/median/max): 1/225/3603
- component area, median per label (min/median/max): 1/24/28685 px
- smallest component area seen: 1 px; largest: 692233 px

### Boundary pixel fraction each mode would produce

- MODE A (painted colour): 0.0000
- MODE B (find_boundaries on the label map): 0.1736

## Steel1

**MODE B** — no colour is simultaneously sparse, thin and frame-spanning: the mask is a phase-label map, so grain boundaries are absent

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

### MODE A evidence

- candidate colour: `None` (HSV None)
- qualifying in 0/24 sampled masks (quorum 14.4)
- pixel fraction (min/median/max): -/-/-
- skeleton-to-area: - (mean thickness - px)
- components: -, frame span -

### MODE B structure

- labels per mask (min/median/max): 2/2/2
- majority label is **tessellation** ({'tessellation': 12, 'islands': 12}), median 2 components
- components per label (min/median/max): 1/4/18
- component area, median per label (min/median/max): 55/1332/62972 px
- smallest component area seen: 1 px; largest: 62972 px

### Boundary pixel fraction each mode would produce

- MODE A (painted colour): 0.0000
- MODE B (find_boundaries on the label map): 0.0329

## Steel2

**MODE B** — colour [249] looks line-like in only 8/24 sampled masks, below the 60% quorum

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

### MODE A evidence

- candidate colour: `[249]` (HSV [0.0, 0.0, 0.976])
- qualifying in 8/24 sampled masks (quorum 14.4)
- pixel fraction (min/median/max): 0.0089/0.0132/0.0153
- skeleton-to-area: 1.0000 (mean thickness 1.00 px)
- components: 784, frame span 1.000

### MODE B structure

- labels per mask (min/median/max): 32/32/32
- majority label is **tessellation** ({'tessellation': 24}), median 3068 components
- components per label (min/median/max): 19/476/3778
- component area, median per label (min/median/max): 1/1/1 px
- smallest component area seen: 1 px; largest: 48317 px

### Boundary pixel fraction each mode would produce

- MODE A (painted colour): 0.0146
- MODE B (find_boundaries on the label map): 0.6487

## uhcs1

**MODE B** — colour [255] looks line-like in only 1/24 sampled masks, below the 60% quorum

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

### MODE A evidence

- candidate colour: `[255]` (HSV [0.0, 0.0, 1.0])
- qualifying in 1/24 sampled masks (quorum 14.4)
- pixel fraction (min/median/max): 0.0973/0.1559/0.2395
- skeleton-to-area: 0.0879 (mean thickness 11.44 px)
- components: 751, frame span 0.984

### MODE B structure

- labels per mask (min/median/max): 2/2/2
- majority label is **islands** ({'islands': 15, 'tessellation': 9}), median 1 components
- components per label (min/median/max): 1/60/5075
- component area, median per label (min/median/max): 1/36/267646 px
- smallest component area seen: 1 px; largest: 276546 px

### Boundary pixel fraction each mode would produce

- MODE A (painted colour): 0.0000
- MODE B (find_boundaries on the label map): 0.1151

## uhcs2

**MODE B** — colour [75, 176, 40] looks line-like in only 4/24 sampled masks, below the 60% quorum

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

### MODE A evidence

- candidate colour: `[75, 176, 40]` (HSV [104.6, 0.773, 0.69])
- qualifying in 4/24 sampled masks (quorum 14.4)
- pixel fraction (min/median/max): 0.0029/0.0169/0.1633
- skeleton-to-area: 0.1962 (mean thickness 5.10 px)
- components: 14, frame span 0.702

### MODE B structure

- labels per mask (min/median/max): 1/4/4
- majority label is **tessellation** ({'tessellation': 23, 'islands': 1}), median 8 components
- components per label (min/median/max): 1/9/137
- component area, median per label (min/median/max): 1/343/312180 px
- smallest component area seen: 1 px; largest: 312180 px

### Boundary pixel fraction each mode would produce

- MODE A (painted colour): 0.0000
- MODE B (find_boundaries on the label map): 0.0485
