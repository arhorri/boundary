# Boundary ground truth extraction

- generated: 2026-09-04T09:04:16Z
- from audit: 2026-09-03T23:42:26Z
- output root: `/content/drive/MyDrive/phase11-persistent/gt_boundaries`
- line width: 2 px, speckle removal 'speckle' at 3x3, CLOSE 3x3
- speckle removal drops connected components smaller than 9 px. A literal 3x3 morphological OPEN would erase the whole map: its erosion needs a full 3x3 block of foreground, and a boundary line is 1-3 px wide. Set `boundary_gt.open_mode: morph` to force the literal version.
- size tolerance: 2 px — a mask within this many pixels of its image is centre-cropped to the common size, not rejected. Cropped, never resized.
- mode overrides applied: none
- artifact colour folding: off (watched: {'MetalDam': [[75, 176, 40]], 'uhcs2': [[75, 176, 40]]})

MODE A extracts a painted boundary colour by HSV thresholding and yields phase interfaces AND grain boundaries. MODE B derives boundaries from a phase-label map with `find_boundaries` and yields phase interfaces ONLY -- grain boundaries are absent from the source masks and are not invented here.

| folder | mode | K | processed | reconciled | rejected | excluded | frac before | frac after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MetalDam | **B** | 5 | 42 | 2 | 0 | 0 | 0.1651 | 0.1702 |
| Steel1 | **B** | 2 | 907 | 0 | 0 | 0 | 0.0381 | 0.0432 |
| Steel2 | **B** | 2 | 504 | 0 | 0 | 0 | 0.1787 | 0.1408 |
| uhcs1 | **B** | 2 | 24 | 0 | 0 | 0 | 0.1151 | 0.1027 |
| uhcs2 | **B** | 4 | 23 | 0 | 0 | 1 | 0.0505 | 0.0487 |

## MetalDam

- mode: **B**
- output: `/content/drive/MyDrive/phase11-persistent/gt_boundaries/MetalDam`
- pairs: 42 -> processed 42 (2 size-reconciled), rejected 0, excluded 0
- boundary pixel fraction before cleanup (min/median/max): 0.1059/0.1651/0.2404
- boundary pixel fraction after cleanup (min/median/max): 0.1175/0.1702/0.2534
- intermediate medians: after speckle removal 0.1650, after CLOSE 0.1808
- K = 5 palette classes: `[255, 255, 0]`, `[21, 100, 255]`, `[254, 40, 40]`, `[0, 0, 0]`, `[75, 176, 40]`
- K is the number of colour *peaks*, not the raw unique-colour count: colours within 48 RGB of a heavier colour are anti-aliasing or JPEG ramp values and snap to it.

### Watched artifact colours

A colour kept as its own class gets a closed boundary loop drawn around every region of it. `share_of_boundary` is how much of this folder's extracted boundary lies on that colour's outline -- the cost of treating it as a phase.

| colour | folded | present in | pixel frac (med) | share of boundary (med) |
| --- | --- | --- | --- | --- |
| `[75, 176, 40]` | no | 17/42 | 0.0031 | 0.0173 |

### Size-reconciled pairs (2)

Mask and image dimensions disagreed by no more than the 2 px tolerance, so both were centre-cropped to their common size and the pair was kept. The boundary PNG is written at the final size; `image crop` is the identical crop the raw image needs when it is loaded.

| pair | image | mask | final | image crop (l,t,w,h) |
| --- | --- | --- | --- | --- |
| `micrograph15.png` | 1024x703 | 1024x702 | 1024x702 | 0,0,1024,702 |
| `micrograph19.png` | 1024x703 | 1024x702 | 1024x702 | 0,0,1024,702 |

## Steel1

- mode: **B**
- output: `/content/drive/MyDrive/phase11-persistent/gt_boundaries/Steel1`
- pairs: 907 -> processed 907 (0 size-reconciled), rejected 0, excluded 0
- boundary pixel fraction before cleanup (min/median/max): 0.0002/0.0381/0.1116
- boundary pixel fraction after cleanup (min/median/max): 0.0002/0.0432/0.1229
- intermediate medians: after speckle removal 0.0381, after CLOSE 0.0396
- K = 2 palette classes: `[0, 0, 0]`, `[255, 255, 255]`
- K is the number of colour *peaks*, not the raw unique-colour count: colours within 48 RGB of a heavier colour are anti-aliasing or JPEG ramp values and snap to it.

## Steel2

- mode: **B**
- output: `/content/drive/MyDrive/phase11-persistent/gt_boundaries/Steel2`
- pairs: 504 -> processed 504 (0 size-reconciled), rejected 0, excluded 0
- boundary pixel fraction before cleanup (min/median/max): 0.0338/0.1787/0.2801
- boundary pixel fraction after cleanup (min/median/max): 0.0171/0.1408/0.2068
- intermediate medians: after speckle removal 0.1755, after CLOSE 0.2184
- K = 2 palette classes: `[255, 255, 255]`, `[0, 0, 0]`
- K is the number of colour *peaks*, not the raw unique-colour count: colours within 48 RGB of a heavier colour are anti-aliasing or JPEG ramp values and snap to it.

## uhcs1

- mode: **B**
- output: `/content/drive/MyDrive/phase11-persistent/gt_boundaries/uhcs1`
- pairs: 24 -> processed 24 (0 size-reconciled), rejected 0, excluded 0
- boundary pixel fraction before cleanup (min/median/max): 0.0308/0.1151/0.2255
- boundary pixel fraction after cleanup (min/median/max): 0.0346/0.1027/0.1494
- intermediate medians: after speckle removal 0.1141, after CLOSE 0.1278
- K = 2 palette classes: `[0, 0, 0]`, `[255, 255, 255]`
- K is the number of colour *peaks*, not the raw unique-colour count: colours within 48 RGB of a heavier colour are anti-aliasing or JPEG ramp values and snap to it.

## uhcs2

- mode: **B**
- output: `/content/drive/MyDrive/phase11-persistent/gt_boundaries/uhcs2`
- pairs: 24 -> processed 23 (0 size-reconciled), rejected 0, excluded 1
- boundary pixel fraction before cleanup (min/median/max): 0.0232/0.0505/0.1098
- boundary pixel fraction after cleanup (min/median/max): 0.0226/0.0487/0.1042
- intermediate medians: after speckle removal 0.0505, after CLOSE 0.0550
- K = 4 palette classes: `[254, 40, 40]`, `[21, 100, 255]`, `[255, 255, 0]`, `[75, 176, 40]`
- K is the number of colour *peaks*, not the raw unique-colour count: colours within 48 RGB of a heavier colour are anti-aliasing or JPEG ramp values and snap to it.

### Watched artifact colours

A colour kept as its own class gets a closed boundary loop drawn around every region of it. `share_of_boundary` is how much of this folder's extracted boundary lies on that colour's outline -- the cost of treating it as a phase.

| colour | folded | present in | pixel frac (med) | share of boundary (med) |
| --- | --- | --- | --- | --- |
| `[75, 176, 40]` | no | 23/23 | 0.0169 | 0.2446 |

### Excluded before processing

- `uhcs0596.png` — listed in boundary_gt.exclusions
