# Gap closing -- fold_steel_combined (colab)

Checkpoint: epoch 21, config hash `067fecc2e06a7d0a`. Two shapes swept together with threshold, on VAL, selected by PQ per dataset: `morph_close` (binary closing at a radius) and `skeleton_bridge` (bridge skeleton endpoints within a distance, then redilate). Scored on the `binary_boundary` partition -- the same conversion the ground truth goes through -- so this is comparable to Cell 21's post-processing sweep, not to Cell 20's watershed numbers directly.

## VAL sweep (best 5 rows per dataset by PQ)

### Steel1 (control -- MISPLACED, a model problem)

| config | mode | threshold | closing_radius | bridge_px | pq | sq | rq | over-seg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| morph_close_r1@0.70 **<- selected** | morph_close | 0.70 | 1.0 | -- | 0.2248 | 0.7282 | 0.3069 | 0.821 |
| morph_close_r1@tuned | morph_close | 0.70 | 1.0 | -- | 0.2248 | 0.7282 | 0.3069 | 0.821 |
| skeleton_bridge_6px@0.70 | skeleton_bridge | 0.70 | -- | 6.0 | 0.2150 | 0.7121 | 0.2977 | 0.758 |
| skeleton_bridge_6px@tuned | skeleton_bridge | 0.70 | -- | 6.0 | 0.2150 | 0.7121 | 0.2977 | 0.758 |
| skeleton_bridge_2px@0.70 | skeleton_bridge | 0.70 | -- | 2.0 | 0.2147 | 0.7121 | 0.2973 | 0.761 |
### Steel2 (hypothesis: gapped boundaries merge regions)

| config | mode | threshold | closing_radius | bridge_px | pq | sq | rq | over-seg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skeleton_bridge_6px@0.85 **<- selected** | skeleton_bridge | 0.85 | -- | 6.0 | 0.2102 | 0.6836 | 0.3055 | 0.576 |
| skeleton_bridge_6px@tuned | skeleton_bridge | 0.85 | -- | 6.0 | 0.2102 | 0.6836 | 0.3055 | 0.576 |
| skeleton_bridge_4px@0.85 | skeleton_bridge | 0.85 | -- | 4.0 | 0.2100 | 0.6835 | 0.3053 | 0.575 |
| skeleton_bridge_4px@tuned | skeleton_bridge | 0.85 | -- | 4.0 | 0.2100 | 0.6835 | 0.3053 | 0.575 |
| skeleton_bridge_2px@0.85 | skeleton_bridge | 0.85 | -- | 2.0 | 0.2100 | 0.6835 | 0.3053 | 0.575 |

## TEST -- selected gap-closing config vs none@tuned vs watershed@val-best marker

| dataset | config | pq | sq | rq | over-seg | label |
| --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | selected gap-closing (val-best) | 0.2777 | 0.7542 | 0.3663 | 0.765 | MISPLACED |
| Steel2 (hypothesis: gapped boundaries merge regions) | selected gap-closing (val-best) | 0.1725 | 0.6657 | 0.2544 | 0.456 | OFFSET |
| Steel1 (control -- MISPLACED, a model problem) | none@tuned | 0.2721 | 0.7471 | 0.3634 | 0.813 | MISPLACED |
| Steel2 (hypothesis: gapped boundaries merge regions) | none@tuned | 0.1691 | 0.6623 | 0.2504 | 0.470 | THICKNESS |
| Steel1 (control -- MISPLACED, a model problem) | watershed@val-best marker (extended) | 0.3434 | 0.7442 | 0.4581 | 0.992 | MISPLACED |
| Steel2 (hypothesis: gapped boundaries merge regions) | watershed@val-best marker (extended) | 0.1770 | 0.6963 | 0.2528 | 0.493 | THICKNESS |

## Watershed marker (default 0.3; extended sweep [0.4, 0.45, 0.5, 0.55, 0.6, 0.7] on VAL)

| split | config | dataset | marker | pq | sq | rq | over-seg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | watershed_prob@marker0.40 | Steel1 (control -- MISPLACED, a model problem) | 0.40 | 0.2334 | 0.7146 | 0.3298 | 1.213 |
| val | watershed_prob@marker0.40 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.40 | 0.1149 | 0.7036 | 0.1641 | 0.474 |
| val | watershed_prob@marker0.45 | Steel1 (control -- MISPLACED, a model problem) | 0.45 | 0.2466 | 0.7202 | 0.3453 | 1.154 |
| val | watershed_prob@marker0.45 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.45 | 0.1264 | 0.7006 | 0.1804 | 0.502 |
| val | watershed_prob@marker0.50 | Steel1 (control -- MISPLACED, a model problem) | 0.50 | 0.2435 | 0.7213 | 0.3408 | 1.186 |
| val | watershed_prob@marker0.50 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.50 | 0.1387 | 0.7000 | 0.1979 | 0.517 |
| val | watershed_prob@marker0.55 **<- selected** | Steel1 (control -- MISPLACED, a model problem) | 0.55 | 0.2517 | 0.7254 | 0.3511 | 1.139 |
| val | watershed_prob@marker0.55 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.55 | 0.1489 | 0.7026 | 0.2117 | 0.527 |
| val | watershed_prob@marker0.60 | Steel1 (control -- MISPLACED, a model problem) | 0.60 | 0.2496 | 0.7246 | 0.3474 | 1.028 |
| val | watershed_prob@marker0.60 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.60 | 0.1585 | 0.7004 | 0.2260 | 0.548 |
| val | watershed_prob@marker0.70 | Steel1 (control -- MISPLACED, a model problem) | 0.70 | 0.2224 | 0.7267 | 0.3045 | 0.808 |
| val | watershed_prob@marker0.70 **<- selected** | Steel2 (hypothesis: gapped boundaries merge regions) | 0.70 | 0.1809 | 0.6949 | 0.2592 | 0.565 |
| test | watershed@val-best marker (extended) | Steel1 (control -- MISPLACED, a model problem) | 0.55 | 0.3434 | 0.7442 | 0.4581 | 0.992 |
| test | watershed@val-best marker (extended) | Steel2 (hypothesis: gapped boundaries merge regions) | 0.70 | 0.1770 | 0.6963 | 0.2528 | 0.493 |

## FN class breakdown and MERGED count, before vs after (TEST, none@tuned vs the VAL-selected gap-closing config)

| dataset | config | FN/tile | TINY | MERGED | SPLIT | OTHER |
| --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | before (none@tuned) | 12.1 | 582 (50%) | 528 (45%) | 33 (3%) | 18 (2%) |
| Steel2 (hypothesis: gapped boundaries merge regions) | before (none@tuned) | 103.9 | 5730 (44%) | 5847 (45%) | 1155 (9%) | 356 (3%) |
| Steel1 (control -- MISPLACED, a model problem) | after (selected) | 12.3 | 584 (49%) | 527 (44%) | 49 (4%) | 25 (2%) |
| Steel2 (hypothesis: gapped boundaries merge regions) | after (selected) | 103.6 | 5715 (44%) | 5850 (45%) | 1128 (9%) | 366 (3%) |

## Secondary diagnostic -- PQ excluding GT regions under the dataset's TINY floor (never a selection metric)

| dataset | floor px | pq before | pq after | true regions excluded |
| --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 50 | 0.3236 | 0.3308 | 6 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 50 | 0.2328 | 0.2369 | 45 |

## Checks

- PASS marker selected on VAL rows only
- PASS gap-closing config selected on VAL rows only
- PASS TEST scored exactly the three pre-declared configurations -- selected gap-closing (val-best), none@tuned, watershed@val-best marker (extended)
- PASS Steel1: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00
- PASS Steel2: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00
