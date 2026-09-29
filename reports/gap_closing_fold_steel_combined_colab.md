# Gap closing -- fold_steel_combined (colab)

Checkpoint: epoch 21, config hash `067fecc2e06a7d0a`. Two shapes swept together with threshold, on VAL, selected by PQ per dataset: `morph_close` (binary closing at a radius) and `skeleton_bridge` (bridge skeleton endpoints within a distance, then redilate). Scored on the `binary_boundary` partition -- the same conversion the ground truth goes through -- so this is comparable to Cell 21's post-processing sweep, not to Cell 20's watershed numbers directly.

## VAL sweep (best 5 rows per dataset by PQ)

### Steel1 (control -- MISPLACED, a model problem)

| config | mode | threshold | closing_radius | bridge_px | pq | sq | rq | over-seg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| morph_close_r1@0.70 **<- selected** | morph_close | 0.70 | 1.0 | -- | 0.2254 | 0.7276 | 0.3080 | 0.821 |
| morph_close_r1@tuned | morph_close | 0.70 | 1.0 | -- | 0.2254 | 0.7276 | 0.3080 | 0.821 |
| skeleton_bridge_6px@0.70 | skeleton_bridge | 0.70 | -- | 6.0 | 0.2149 | 0.7119 | 0.2976 | 0.759 |
| skeleton_bridge_6px@tuned | skeleton_bridge | 0.70 | -- | 6.0 | 0.2149 | 0.7119 | 0.2976 | 0.759 |
| skeleton_bridge_2px@0.70 | skeleton_bridge | 0.70 | -- | 2.0 | 0.2146 | 0.7119 | 0.2972 | 0.761 |
### Steel2 (hypothesis: gapped boundaries merge regions)

| config | mode | threshold | closing_radius | bridge_px | pq | sq | rq | over-seg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skeleton_bridge_6px@0.85 **<- selected** | skeleton_bridge | 0.85 | -- | 6.0 | 0.2098 | 0.6838 | 0.3049 | 0.576 |
| skeleton_bridge_6px@tuned | skeleton_bridge | 0.85 | -- | 6.0 | 0.2098 | 0.6838 | 0.3049 | 0.576 |
| skeleton_bridge_4px@0.85 | skeleton_bridge | 0.85 | -- | 4.0 | 0.2097 | 0.6837 | 0.3048 | 0.575 |
| skeleton_bridge_4px@tuned | skeleton_bridge | 0.85 | -- | 4.0 | 0.2097 | 0.6837 | 0.3048 | 0.575 |
| skeleton_bridge_2px@0.85 | skeleton_bridge | 0.85 | -- | 2.0 | 0.2097 | 0.6837 | 0.3048 | 0.575 |

## TEST -- selected gap-closing config vs none@tuned vs watershed@val-best marker

| dataset | config | pq | sq | rq | over-seg | label |
| --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | selected gap-closing (val-best) | 0.2770 | 0.7541 | 0.3653 | 0.771 | MISPLACED |
| Steel2 (hypothesis: gapped boundaries merge regions) | selected gap-closing (val-best) | 0.1720 | 0.6660 | 0.2536 | 0.456 | OFFSET |
| Steel1 (control -- MISPLACED, a model problem) | none@tuned | 0.2721 | 0.7471 | 0.3634 | 0.813 | MISPLACED |
| Steel2 (hypothesis: gapped boundaries merge regions) | none@tuned | 0.1691 | 0.6622 | 0.2505 | 0.469 | THICKNESS |
| Steel1 (control -- MISPLACED, a model problem) | watershed@val-best marker (extended) | 0.3435 | 0.7442 | 0.4582 | 0.990 | MISPLACED |
| Steel2 (hypothesis: gapped boundaries merge regions) | watershed@val-best marker (extended) | 0.1770 | 0.6961 | 0.2530 | 0.493 | THICKNESS |

## Watershed marker (default 0.3; extended sweep [0.4, 0.45, 0.5, 0.55, 0.6, 0.7] on VAL)

| split | config | dataset | marker | pq | sq | rq | over-seg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | watershed_prob@marker0.40 | Steel1 (control -- MISPLACED, a model problem) | 0.40 | 0.2325 | 0.7149 | 0.3284 | 1.210 |
| val | watershed_prob@marker0.40 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.40 | 0.1147 | 0.7039 | 0.1639 | 0.474 |
| val | watershed_prob@marker0.45 | Steel1 (control -- MISPLACED, a model problem) | 0.45 | 0.2468 | 0.7203 | 0.3456 | 1.153 |
| val | watershed_prob@marker0.45 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.45 | 0.1262 | 0.7012 | 0.1800 | 0.502 |
| val | watershed_prob@marker0.50 | Steel1 (control -- MISPLACED, a model problem) | 0.50 | 0.2436 | 0.7213 | 0.3410 | 1.184 |
| val | watershed_prob@marker0.50 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.50 | 0.1386 | 0.7001 | 0.1978 | 0.517 |
| val | watershed_prob@marker0.55 **<- selected** | Steel1 (control -- MISPLACED, a model problem) | 0.55 | 0.2516 | 0.7254 | 0.3509 | 1.139 |
| val | watershed_prob@marker0.55 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.55 | 0.1488 | 0.7027 | 0.2115 | 0.527 |
| val | watershed_prob@marker0.60 | Steel1 (control -- MISPLACED, a model problem) | 0.60 | 0.2495 | 0.7246 | 0.3473 | 1.029 |
| val | watershed_prob@marker0.60 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.60 | 0.1586 | 0.7001 | 0.2263 | 0.548 |
| val | watershed_prob@marker0.70 | Steel1 (control -- MISPLACED, a model problem) | 0.70 | 0.2226 | 0.7267 | 0.3048 | 0.807 |
| val | watershed_prob@marker0.70 **<- selected** | Steel2 (hypothesis: gapped boundaries merge regions) | 0.70 | 0.1812 | 0.6942 | 0.2599 | 0.565 |
| test | watershed@val-best marker (extended) | Steel1 (control -- MISPLACED, a model problem) | 0.55 | 0.3435 | 0.7442 | 0.4582 | 0.990 |
| test | watershed@val-best marker (extended) | Steel2 (hypothesis: gapped boundaries merge regions) | 0.70 | 0.1770 | 0.6961 | 0.2530 | 0.493 |

## FN class breakdown and MERGED count, before vs after (TEST, none@tuned vs the VAL-selected gap-closing config)

| dataset | config | FN/tile | TINY | MERGED | SPLIT | OTHER |
| --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | before (none@tuned) | 12.1 | 582 (50%) | 527 (45%) | 33 (3%) | 19 (2%) |
| Steel2 (hypothesis: gapped boundaries merge regions) | before (none@tuned) | 103.9 | 5731 (44%) | 5847 (45%) | 1152 (9%) | 357 (3%) |
| Steel1 (control -- MISPLACED, a model problem) | after (selected) | 12.3 | 584 (49%) | 527 (44%) | 49 (4%) | 25 (2%) |
| Steel2 (hypothesis: gapped boundaries merge regions) | after (selected) | 103.7 | 5717 (44%) | 5852 (45%) | 1132 (9%) | 367 (3%) |

## Secondary diagnostic -- PQ excluding GT regions under the dataset's TINY floor (never a selection metric)

| dataset | floor px | pq before | pq after | true regions excluded |
| --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 50 | 0.3235 | 0.3299 | 6 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 50 | 0.2329 | 0.2365 | 45 |

## Checks

- PASS marker selected on VAL rows only
- PASS gap-closing config selected on VAL rows only
- PASS TEST scored exactly the three pre-declared configurations -- selected gap-closing (val-best), none@tuned, watershed@val-best marker (extended)
- PASS Steel1: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00
- PASS Steel2: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00
