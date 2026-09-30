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
| skeleton_bridge_6px@0.85 **<- selected** | skeleton_bridge | 0.85 | -- | 6.0 | 0.2100 | 0.6834 | 0.3054 | 0.576 |
| skeleton_bridge_6px@tuned | skeleton_bridge | 0.85 | -- | 6.0 | 0.2100 | 0.6834 | 0.3054 | 0.576 |
| skeleton_bridge_4px@0.85 | skeleton_bridge | 0.85 | -- | 4.0 | 0.2098 | 0.6833 | 0.3052 | 0.576 |
| skeleton_bridge_4px@tuned | skeleton_bridge | 0.85 | -- | 4.0 | 0.2098 | 0.6833 | 0.3052 | 0.576 |
| skeleton_bridge_2px@0.85 | skeleton_bridge | 0.85 | -- | 2.0 | 0.2098 | 0.6833 | 0.3052 | 0.576 |

## TEST -- selected gap-closing config vs none@tuned vs watershed@val-best marker

| dataset | config | pq | sq | rq | over-seg | label |
| --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | selected gap-closing (val-best) | 0.2777 | 0.7542 | 0.3663 | 0.765 | MISPLACED |
| Steel2 (hypothesis: gapped boundaries merge regions) | selected gap-closing (val-best) | 0.1723 | 0.6661 | 0.2540 | 0.456 | OFFSET |
| Steel1 (control -- MISPLACED, a model problem) | none@tuned | 0.2724 | 0.7472 | 0.3637 | 0.811 | MISPLACED |
| Steel2 (hypothesis: gapped boundaries merge regions) | none@tuned | 0.1693 | 0.6624 | 0.2507 | 0.470 | THICKNESS |
| Steel1 (control -- MISPLACED, a model problem) | watershed@val-best marker (extended) | 0.3434 | 0.7442 | 0.4581 | 0.992 | MISPLACED |
| Steel2 (hypothesis: gapped boundaries merge regions) | watershed@val-best marker (extended) | 0.1769 | 0.6964 | 0.2527 | 0.493 | THICKNESS |

## Watershed marker (default 0.3; extended sweep [0.4, 0.45, 0.5, 0.55, 0.6, 0.7] on VAL)

| split | config | dataset | marker | pq | sq | rq | over-seg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | watershed_prob@marker0.40 | Steel1 (control -- MISPLACED, a model problem) | 0.40 | 0.2335 | 0.7146 | 0.3299 | 1.212 |
| val | watershed_prob@marker0.40 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.40 | 0.1150 | 0.7035 | 0.1643 | 0.474 |
| val | watershed_prob@marker0.45 | Steel1 (control -- MISPLACED, a model problem) | 0.45 | 0.2466 | 0.7202 | 0.3453 | 1.154 |
| val | watershed_prob@marker0.45 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.45 | 0.1263 | 0.7008 | 0.1803 | 0.502 |
| val | watershed_prob@marker0.50 | Steel1 (control -- MISPLACED, a model problem) | 0.50 | 0.2435 | 0.7213 | 0.3407 | 1.186 |
| val | watershed_prob@marker0.50 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.50 | 0.1386 | 0.7001 | 0.1977 | 0.517 |
| val | watershed_prob@marker0.55 **<- selected** | Steel1 (control -- MISPLACED, a model problem) | 0.55 | 0.2517 | 0.7254 | 0.3511 | 1.139 |
| val | watershed_prob@marker0.55 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.55 | 0.1487 | 0.7030 | 0.2114 | 0.527 |
| val | watershed_prob@marker0.60 | Steel1 (control -- MISPLACED, a model problem) | 0.60 | 0.2496 | 0.7246 | 0.3475 | 1.028 |
| val | watershed_prob@marker0.60 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.60 | 0.1588 | 0.7001 | 0.2264 | 0.548 |
| val | watershed_prob@marker0.70 | Steel1 (control -- MISPLACED, a model problem) | 0.70 | 0.2223 | 0.7267 | 0.3045 | 0.809 |
| val | watershed_prob@marker0.70 **<- selected** | Steel2 (hypothesis: gapped boundaries merge regions) | 0.70 | 0.1810 | 0.6947 | 0.2594 | 0.566 |
| test | watershed@val-best marker (extended) | Steel1 (control -- MISPLACED, a model problem) | 0.55 | 0.3434 | 0.7442 | 0.4581 | 0.992 |
| test | watershed@val-best marker (extended) | Steel2 (hypothesis: gapped boundaries merge regions) | 0.70 | 0.1769 | 0.6964 | 0.2527 | 0.493 |

## FN class breakdown and MERGED count, before vs after (TEST, none@tuned vs the VAL-selected gap-closing config)

| dataset | config | FN/tile | TINY | MERGED | SPLIT | OTHER |
| --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | before (none@tuned) | 12.1 | 582 (50%) | 528 (45%) | 33 (3%) | 18 (2%) |
| Steel2 (hypothesis: gapped boundaries merge regions) | before (none@tuned) | 103.9 | 5730 (44%) | 5849 (45%) | 1150 (9%) | 357 (3%) |
| Steel1 (control -- MISPLACED, a model problem) | after (selected) | 12.3 | 584 (49%) | 527 (44%) | 49 (4%) | 25 (2%) |
| Steel2 (hypothesis: gapped boundaries merge regions) | after (selected) | 103.7 | 5716 (44%) | 5853 (45%) | 1128 (9%) | 368 (3%) |

## Secondary diagnostic -- PQ excluding GT regions under the dataset's TINY floor (never a selection metric)

| dataset | floor px | pq before | pq after | true regions excluded |
| --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 50 | 0.3238 | 0.3308 | 6 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 50 | 0.2331 | 0.2367 | 45 |

## Checks

- PASS marker selected on VAL rows only
- PASS gap-closing config selected on VAL rows only
- PASS TEST scored exactly the three pre-declared configurations -- selected gap-closing (val-best), none@tuned, watershed@val-best marker (extended)
- PASS Steel1: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00
- PASS Steel2: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00
