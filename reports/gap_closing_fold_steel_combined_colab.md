# Gap closing -- fold_steel_combined (colab)

Checkpoint: epoch 21, config hash `067fecc2e06a7d0a`. Two shapes swept together with threshold, on VAL, selected by PQ per dataset: `morph_close` (binary closing at a radius) and `skeleton_bridge` (bridge skeleton endpoints within a distance, then redilate). Scored on the `binary_boundary` partition -- the same conversion the ground truth goes through -- so this is comparable to Cell 21's post-processing sweep, not to Cell 20's watershed numbers directly.

## VAL sweep (best 5 rows per dataset by PQ)

### Steel1 (control -- MISPLACED, a model problem)

| config | mode | threshold | closing_radius | bridge_px | pq | sq | rq | over-seg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| morph_close_r1@0.70 **<- selected** | morph_close | 0.70 | 1.0 | -- | 0.2247 | 0.7275 | 0.3071 | 0.825 |
| morph_close_r1@tuned | morph_close | 0.70 | 1.0 | -- | 0.2247 | 0.7275 | 0.3071 | 0.825 |
| skeleton_bridge_6px@0.70 | skeleton_bridge | 0.70 | -- | 6.0 | 0.2144 | 0.7122 | 0.2970 | 0.765 |
| skeleton_bridge_6px@tuned | skeleton_bridge | 0.70 | -- | 6.0 | 0.2144 | 0.7122 | 0.2970 | 0.765 |
| skeleton_bridge_2px@0.70 | skeleton_bridge | 0.70 | -- | 2.0 | 0.2141 | 0.7122 | 0.2965 | 0.767 |
### Steel2 (hypothesis: gapped boundaries merge regions)

| config | mode | threshold | closing_radius | bridge_px | pq | sq | rq | over-seg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skeleton_bridge_6px@0.85 **<- selected** | skeleton_bridge | 0.85 | -- | 6.0 | 0.2099 | 0.6836 | 0.3052 | 0.576 |
| skeleton_bridge_6px@tuned | skeleton_bridge | 0.85 | -- | 6.0 | 0.2099 | 0.6836 | 0.3052 | 0.576 |
| skeleton_bridge_4px@0.85 | skeleton_bridge | 0.85 | -- | 4.0 | 0.2098 | 0.6835 | 0.3051 | 0.575 |
| skeleton_bridge_4px@tuned | skeleton_bridge | 0.85 | -- | 4.0 | 0.2098 | 0.6835 | 0.3051 | 0.575 |
| skeleton_bridge_2px@0.85 | skeleton_bridge | 0.85 | -- | 2.0 | 0.2098 | 0.6835 | 0.3051 | 0.575 |

## TEST -- selected gap-closing config vs none@tuned vs watershed@val-best marker

| dataset | config | pq | sq | rq | over-seg | label |
| --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | selected gap-closing (val-best) | 0.2773 | 0.7543 | 0.3657 | 0.768 | MISPLACED |
| Steel2 (hypothesis: gapped boundaries merge regions) | selected gap-closing (val-best) | 0.1722 | 0.6660 | 0.2538 | 0.456 | OFFSET |
| Steel1 (control -- MISPLACED, a model problem) | none@tuned | 0.2720 | 0.7471 | 0.3633 | 0.813 | MISPLACED |
| Steel2 (hypothesis: gapped boundaries merge regions) | none@tuned | 0.1693 | 0.6621 | 0.2508 | 0.469 | THICKNESS |
| Steel1 (control -- MISPLACED, a model problem) | watershed@val-best marker (extended) | 0.3436 | 0.7441 | 0.4584 | 0.991 | MISPLACED |
| Steel2 (hypothesis: gapped boundaries merge regions) | watershed@val-best marker (extended) | 0.1770 | 0.6962 | 0.2529 | 0.493 | THICKNESS |

## Watershed marker (default 0.3; extended sweep [0.4, 0.45, 0.5, 0.55, 0.6, 0.7] on VAL)

| split | config | dataset | marker | pq | sq | rq | over-seg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | watershed_prob@marker0.40 | Steel1 (control -- MISPLACED, a model problem) | 0.40 | 0.2325 | 0.7141 | 0.3286 | 1.217 |
| val | watershed_prob@marker0.40 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.40 | 0.1151 | 0.7036 | 0.1646 | 0.474 |
| val | watershed_prob@marker0.45 | Steel1 (control -- MISPLACED, a model problem) | 0.45 | 0.2467 | 0.7201 | 0.3455 | 1.153 |
| val | watershed_prob@marker0.45 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.45 | 0.1260 | 0.7010 | 0.1799 | 0.502 |
| val | watershed_prob@marker0.50 | Steel1 (control -- MISPLACED, a model problem) | 0.50 | 0.2436 | 0.7212 | 0.3410 | 1.184 |
| val | watershed_prob@marker0.50 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.50 | 0.1386 | 0.7002 | 0.1978 | 0.517 |
| val | watershed_prob@marker0.55 **<- selected** | Steel1 (control -- MISPLACED, a model problem) | 0.55 | 0.2514 | 0.7250 | 0.3510 | 1.140 |
| val | watershed_prob@marker0.55 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.55 | 0.1491 | 0.7024 | 0.2121 | 0.527 |
| val | watershed_prob@marker0.60 | Steel1 (control -- MISPLACED, a model problem) | 0.60 | 0.2496 | 0.7246 | 0.3474 | 1.028 |
| val | watershed_prob@marker0.60 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.60 | 0.1590 | 0.7004 | 0.2268 | 0.548 |
| val | watershed_prob@marker0.70 | Steel1 (control -- MISPLACED, a model problem) | 0.70 | 0.2219 | 0.7267 | 0.3038 | 0.808 |
| val | watershed_prob@marker0.70 **<- selected** | Steel2 (hypothesis: gapped boundaries merge regions) | 0.70 | 0.1807 | 0.6950 | 0.2588 | 0.566 |
| test | watershed@val-best marker (extended) | Steel1 (control -- MISPLACED, a model problem) | 0.55 | 0.3436 | 0.7441 | 0.4584 | 0.991 |
| test | watershed@val-best marker (extended) | Steel2 (hypothesis: gapped boundaries merge regions) | 0.70 | 0.1770 | 0.6962 | 0.2529 | 0.493 |

## FN class breakdown and MERGED count, before vs after (TEST, none@tuned vs the VAL-selected gap-closing config)

| dataset | config | FN/tile | TINY | MERGED | SPLIT | OTHER |
| --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | before (none@tuned) | 12.1 | 582 (50%) | 528 (45%) | 33 (3%) | 18 (2%) |
| Steel2 (hypothesis: gapped boundaries merge regions) | before (none@tuned) | 103.8 | 5730 (44%) | 5847 (45%) | 1149 (9%) | 358 (3%) |
| Steel1 (control -- MISPLACED, a model problem) | after (selected) | 12.3 | 584 (49%) | 527 (44%) | 49 (4%) | 25 (2%) |
| Steel2 (hypothesis: gapped boundaries merge regions) | after (selected) | 103.7 | 5717 (44%) | 5849 (45%) | 1134 (9%) | 367 (3%) |

## Secondary diagnostic -- PQ excluding GT regions under the dataset's TINY floor (never a selection metric)

| dataset | floor px | pq before | pq after | true regions excluded |
| --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 50 | 0.3234 | 0.3303 | 6 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 50 | 0.2332 | 0.2367 | 45 |

## Checks

- PASS marker selected on VAL rows only
- PASS gap-closing config selected on VAL rows only
- PASS TEST scored exactly the three pre-declared configurations -- selected gap-closing (val-best), none@tuned, watershed@val-best marker (extended)
- PASS Steel1: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00
- PASS Steel2: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00
