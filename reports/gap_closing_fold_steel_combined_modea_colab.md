# Gap closing -- fold_steel_combined_modea (colab)

Checkpoint: epoch 32, config hash `067fecc2e06a7d0a`. Two shapes swept together with threshold, on VAL, selected by PQ per dataset: `morph_close` (binary closing at a radius) and `skeleton_bridge` (bridge skeleton endpoints within a distance, then redilate). Scored on the `binary_boundary` partition -- the same conversion the ground truth goes through -- so this is comparable to Cell 21's post-processing sweep, not to Cell 20's watershed numbers directly.

## VAL sweep (best 5 rows per dataset by PQ)

### Steel1 (control -- MISPLACED, a model problem)

| config | mode | threshold | closing_radius | bridge_px | pq | sq | rq | over-seg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skeleton_bridge_4px@0.70 **<- selected** | skeleton_bridge | 0.70 | -- | 4.0 | 0.2366 | 0.7329 | 0.3250 | 0.805 |
| skeleton_bridge_2px@0.70 | skeleton_bridge | 0.70 | -- | 2.0 | 0.2366 | 0.7329 | 0.3250 | 0.805 |
| skeleton_bridge_6px@0.70 | skeleton_bridge | 0.70 | -- | 6.0 | 0.2366 | 0.7330 | 0.3249 | 0.805 |
| skeleton_bridge_2px@0.75 | skeleton_bridge | 0.75 | -- | 2.0 | 0.2329 | 0.7345 | 0.3182 | 0.737 |
| skeleton_bridge_2px@tuned | skeleton_bridge | 0.75 | -- | 2.0 | 0.2329 | 0.7345 | 0.3182 | 0.737 |
### Steel2 (hypothesis: gapped boundaries merge regions)

| config | mode | threshold | closing_radius | bridge_px | pq | sq | rq | over-seg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skeleton_bridge_4px@0.70 **<- selected** | skeleton_bridge | 0.70 | -- | 4.0 | 0.3887 | 0.8061 | 0.4807 | 0.941 |
| skeleton_bridge_2px@0.70 | skeleton_bridge | 0.70 | -- | 2.0 | 0.3887 | 0.8061 | 0.4807 | 0.941 |
| skeleton_bridge_6px@0.70 | skeleton_bridge | 0.70 | -- | 6.0 | 0.3884 | 0.8061 | 0.4804 | 0.945 |
| skeleton_bridge_6px@0.75 | skeleton_bridge | 0.75 | -- | 6.0 | 0.3880 | 0.8090 | 0.4780 | 0.858 |
| skeleton_bridge_2px@0.75 | skeleton_bridge | 0.75 | -- | 2.0 | 0.3878 | 0.8095 | 0.4775 | 0.851 |

## TEST -- selected gap-closing config vs none@tuned vs watershed@val-best marker

| dataset | config | pq | sq | rq | over-seg | label |
| --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | selected gap-closing (val-best) | 0.2979 | 0.7400 | 0.4007 | 0.710 | MISPLACED |
| Steel2 (hypothesis: gapped boundaries merge regions) | selected gap-closing (val-best) | 0.3649 | 0.8031 | 0.4537 | 0.792 | OFFSET |
| Steel1 (control -- MISPLACED, a model problem) | none@tuned | 0.2562 | 0.7479 | 0.3421 | 0.730 | MISPLACED |
| Steel2 (hypothesis: gapped boundaries merge regions) | none@tuned | 0.2901 | 0.7891 | 0.3671 | 0.554 | OFFSET |
| Steel1 (control -- MISPLACED, a model problem) | watershed@val-best marker (extended) | 0.3408 | 0.7413 | 0.4571 | 0.937 | MISPLACED |
| Steel2 (hypothesis: gapped boundaries merge regions) | watershed@val-best marker (extended) | 0.3605 | 0.7652 | 0.4699 | 1.071 | OFFSET |

## Watershed marker (default 0.3; extended sweep [0.4, 0.45, 0.5, 0.55, 0.6, 0.7] on VAL)

| split | config | dataset | marker | pq | sq | rq | over-seg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | watershed_prob@marker0.40 | Steel1 (control -- MISPLACED, a model problem) | 0.40 | 0.2466 | 0.7178 | 0.3461 | 1.119 |
| val | watershed_prob@marker0.40 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.40 | 0.3186 | 0.7730 | 0.4133 | 1.465 |
| val | watershed_prob@marker0.45 | Steel1 (control -- MISPLACED, a model problem) | 0.45 | 0.2518 | 0.7215 | 0.3516 | 1.112 |
| val | watershed_prob@marker0.45 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.45 | 0.3392 | 0.7729 | 0.4399 | 1.408 |
| val | watershed_prob@marker0.50 | Steel1 (control -- MISPLACED, a model problem) | 0.50 | 0.2523 | 0.7192 | 0.3526 | 1.121 |
| val | watershed_prob@marker0.50 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.50 | 0.3518 | 0.7721 | 0.4564 | 1.337 |
| val | watershed_prob@marker0.55 | Steel1 (control -- MISPLACED, a model problem) | 0.55 | 0.2579 | 0.7227 | 0.3604 | 1.083 |
| val | watershed_prob@marker0.55 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.55 | 0.3639 | 0.7698 | 0.4736 | 1.272 |
| val | watershed_prob@marker0.60 **<- selected** | Steel1 (control -- MISPLACED, a model problem) | 0.60 | 0.2587 | 0.7257 | 0.3600 | 1.072 |
| val | watershed_prob@marker0.60 **<- selected** | Steel2 (hypothesis: gapped boundaries merge regions) | 0.60 | 0.3648 | 0.7698 | 0.4733 | 1.198 |
| val | watershed_prob@marker0.70 | Steel1 (control -- MISPLACED, a model problem) | 0.70 | 0.2429 | 0.7310 | 0.3347 | 0.887 |
| val | watershed_prob@marker0.70 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.70 | 0.3577 | 0.7696 | 0.4639 | 0.908 |
| test | watershed@val-best marker (extended) | Steel1 (control -- MISPLACED, a model problem) | 0.60 | 0.3408 | 0.7413 | 0.4571 | 0.937 |
| test | watershed@val-best marker (extended) | Steel2 (hypothesis: gapped boundaries merge regions) | 0.60 | 0.3605 | 0.7652 | 0.4699 | 1.071 |

## FN class breakdown and MERGED count, before vs after (TEST, none@tuned vs the VAL-selected gap-closing config)

| dataset | config | FN/tile | TINY | MERGED | SPLIT | OTHER |
| --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | before (none@tuned) | 12.5 | 581 (49%) | 576 (48%) | 24 (2%) | 16 (1%) |
| Steel2 (hypothesis: gapped boundaries merge regions) | before (none@tuned) | 23.9 | 855 (28%) | 2098 (70%) | 32 (1%) | 29 (1%) |
| Steel1 (control -- MISPLACED, a model problem) | after (selected) | 11.9 | 584 (51%) | 517 (45%) | 23 (2%) | 19 (2%) |
| Steel2 (hypothesis: gapped boundaries merge regions) | after (selected) | 20.3 | 818 (32%) | 1585 (62%) | 92 (4%) | 60 (2%) |

## Secondary diagnostic -- PQ excluding GT regions under the dataset's TINY floor (never a selection metric)

| dataset | floor px | pq before | pq after | true regions excluded |
| --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 50 | 0.3104 | 0.3624 | 6 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 50 | 0.3356 | 0.4058 | 6 |

## Checks

- PASS marker selected on VAL rows only
- PASS gap-closing config selected on VAL rows only
- PASS TEST scored exactly the three pre-declared configurations -- selected gap-closing (val-best), none@tuned, watershed@val-best marker (extended)
- PASS Steel1: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00
- PASS Steel2: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00
