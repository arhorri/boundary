# Gap closing -- fold_steel_combined_modea (colab)

Checkpoint: epoch 32, config hash `067fecc2e06a7d0a`. Two shapes swept together with threshold, on VAL, selected by PQ per dataset: `morph_close` (binary closing at a radius) and `skeleton_bridge` (bridge skeleton endpoints within a distance, then redilate). Scored on the `binary_boundary` partition -- the same conversion the ground truth goes through -- so this is comparable to Cell 21's post-processing sweep, not to Cell 20's watershed numbers directly.

## VAL sweep (best 5 rows per dataset by PQ)

### Steel1 (control -- MISPLACED, a model problem)

| config | mode | threshold | closing_radius | bridge_px | pq | sq | rq | over-seg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skeleton_bridge_4px@0.70 **<- selected** | skeleton_bridge | 0.70 | -- | 4.0 | 0.2369 | 0.7325 | 0.3256 | 0.805 |
| skeleton_bridge_2px@0.70 | skeleton_bridge | 0.70 | -- | 2.0 | 0.2369 | 0.7325 | 0.3256 | 0.805 |
| skeleton_bridge_6px@0.70 | skeleton_bridge | 0.70 | -- | 6.0 | 0.2368 | 0.7325 | 0.3255 | 0.805 |
| skeleton_bridge_2px@0.75 | skeleton_bridge | 0.75 | -- | 2.0 | 0.2317 | 0.7251 | 0.3170 | 0.736 |
| skeleton_bridge_2px@tuned | skeleton_bridge | 0.75 | -- | 2.0 | 0.2317 | 0.7251 | 0.3170 | 0.736 |
### Steel2 (hypothesis: gapped boundaries merge regions)

| config | mode | threshold | closing_radius | bridge_px | pq | sq | rq | over-seg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skeleton_bridge_4px@0.70 **<- selected** | skeleton_bridge | 0.70 | -- | 4.0 | 0.3887 | 0.8060 | 0.4807 | 0.942 |
| skeleton_bridge_2px@0.70 | skeleton_bridge | 0.70 | -- | 2.0 | 0.3887 | 0.8060 | 0.4807 | 0.942 |
| skeleton_bridge_6px@0.70 | skeleton_bridge | 0.70 | -- | 6.0 | 0.3884 | 0.8060 | 0.4804 | 0.945 |
| skeleton_bridge_6px@0.75 | skeleton_bridge | 0.75 | -- | 6.0 | 0.3879 | 0.8096 | 0.4776 | 0.859 |
| skeleton_bridge_2px@0.75 | skeleton_bridge | 0.75 | -- | 2.0 | 0.3878 | 0.8101 | 0.4773 | 0.852 |

## TEST -- selected gap-closing config vs none@tuned vs watershed@val-best marker

| dataset | config | pq | sq | rq | over-seg | label |
| --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | selected gap-closing (val-best) | 0.2984 | 0.7398 | 0.4013 | 0.704 | MISPLACED |
| Steel2 (hypothesis: gapped boundaries merge regions) | selected gap-closing (val-best) | 0.3636 | 0.8036 | 0.4521 | 0.794 | OFFSET |
| Steel1 (control -- MISPLACED, a model problem) | none@tuned | 0.2562 | 0.7481 | 0.3421 | 0.730 | MISPLACED |
| Steel2 (hypothesis: gapped boundaries merge regions) | none@tuned | 0.2903 | 0.7890 | 0.3675 | 0.554 | OFFSET |
| Steel1 (control -- MISPLACED, a model problem) | watershed@val-best marker (extended) | 0.3405 | 0.7414 | 0.4567 | 0.936 | MISPLACED |
| Steel2 (hypothesis: gapped boundaries merge regions) | watershed@val-best marker (extended) | 0.3601 | 0.7659 | 0.4691 | 1.072 | OFFSET |

## Watershed marker (default 0.3; extended sweep [0.4, 0.45, 0.5, 0.55, 0.6, 0.7] on VAL)

| split | config | dataset | marker | pq | sq | rq | over-seg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | watershed_prob@marker0.40 | Steel1 (control -- MISPLACED, a model problem) | 0.40 | 0.2468 | 0.7179 | 0.3464 | 1.117 |
| val | watershed_prob@marker0.40 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.40 | 0.3186 | 0.7729 | 0.4135 | 1.465 |
| val | watershed_prob@marker0.45 | Steel1 (control -- MISPLACED, a model problem) | 0.45 | 0.2513 | 0.7214 | 0.3510 | 1.111 |
| val | watershed_prob@marker0.45 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.45 | 0.3391 | 0.7726 | 0.4399 | 1.407 |
| val | watershed_prob@marker0.50 | Steel1 (control -- MISPLACED, a model problem) | 0.50 | 0.2523 | 0.7192 | 0.3525 | 1.121 |
| val | watershed_prob@marker0.50 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.50 | 0.3517 | 0.7722 | 0.4562 | 1.337 |
| val | watershed_prob@marker0.55 | Steel1 (control -- MISPLACED, a model problem) | 0.55 | 0.2583 | 0.7227 | 0.3609 | 1.078 |
| val | watershed_prob@marker0.55 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.55 | 0.3639 | 0.7700 | 0.4734 | 1.272 |
| val | watershed_prob@marker0.60 **<- selected** | Steel1 (control -- MISPLACED, a model problem) | 0.60 | 0.2587 | 0.7257 | 0.3600 | 1.072 |
| val | watershed_prob@marker0.60 **<- selected** | Steel2 (hypothesis: gapped boundaries merge regions) | 0.60 | 0.3646 | 0.7694 | 0.4732 | 1.196 |
| val | watershed_prob@marker0.70 | Steel1 (control -- MISPLACED, a model problem) | 0.70 | 0.2423 | 0.7310 | 0.3339 | 0.887 |
| val | watershed_prob@marker0.70 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.70 | 0.3574 | 0.7696 | 0.4635 | 0.907 |
| test | watershed@val-best marker (extended) | Steel1 (control -- MISPLACED, a model problem) | 0.60 | 0.3405 | 0.7414 | 0.4567 | 0.936 |
| test | watershed@val-best marker (extended) | Steel2 (hypothesis: gapped boundaries merge regions) | 0.60 | 0.3601 | 0.7659 | 0.4691 | 1.072 |

## FN class breakdown and MERGED count, before vs after (TEST, none@tuned vs the VAL-selected gap-closing config)

| dataset | config | FN/tile | TINY | MERGED | SPLIT | OTHER |
| --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | before (none@tuned) | 12.5 | 581 (49%) | 576 (48%) | 24 (2%) | 16 (1%) |
| Steel2 (hypothesis: gapped boundaries merge regions) | before (none@tuned) | 23.9 | 855 (28%) | 2095 (70%) | 31 (1%) | 30 (1%) |
| Steel1 (control -- MISPLACED, a model problem) | after (selected) | 11.9 | 584 (51%) | 517 (45%) | 23 (2%) | 19 (2%) |
| Steel2 (hypothesis: gapped boundaries merge regions) | after (selected) | 20.3 | 818 (32%) | 1589 (62%) | 91 (4%) | 60 (2%) |

## Secondary diagnostic -- PQ excluding GT regions under the dataset's TINY floor (never a selection metric)

| dataset | floor px | pq before | pq after | true regions excluded |
| --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 50 | 0.3103 | 0.3629 | 6 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 50 | 0.3358 | 0.4043 | 6 |

## Checks

- PASS marker selected on VAL rows only
- PASS gap-closing config selected on VAL rows only
- PASS TEST scored exactly the three pre-declared configurations -- selected gap-closing (val-best), none@tuned, watershed@val-best marker (extended)
- PASS Steel1: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00
- PASS Steel2: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00
