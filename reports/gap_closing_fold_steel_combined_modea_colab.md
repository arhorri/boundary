# Gap closing -- fold_steel_combined_modea (colab)

Checkpoint: epoch 32, config hash `067fecc2e06a7d0a`. Two shapes swept together with threshold, on VAL, selected by PQ per dataset: `morph_close` (binary closing at a radius) and `skeleton_bridge` (bridge skeleton endpoints within a distance, then redilate). Scored on the `binary_boundary` partition -- the same conversion the ground truth goes through -- so this is comparable to Cell 21's post-processing sweep, not to Cell 20's watershed numbers directly.

## VAL sweep (best 5 rows per dataset by PQ)

### Steel1 (control -- MISPLACED, a model problem)

| config | mode | threshold | closing_radius | bridge_px | pq | sq | rq | over-seg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skeleton_bridge_4px@0.70 **<- selected** | skeleton_bridge | 0.70 | -- | 4.0 | 0.2367 | 0.7327 | 0.3252 | 0.805 |
| skeleton_bridge_2px@0.70 | skeleton_bridge | 0.70 | -- | 2.0 | 0.2367 | 0.7327 | 0.3252 | 0.805 |
| skeleton_bridge_6px@0.70 | skeleton_bridge | 0.70 | -- | 6.0 | 0.2366 | 0.7327 | 0.3252 | 0.806 |
| skeleton_bridge_6px@0.75 | skeleton_bridge | 0.75 | -- | 6.0 | 0.2330 | 0.7342 | 0.3184 | 0.741 |
| skeleton_bridge_6px@tuned | skeleton_bridge | 0.75 | -- | 6.0 | 0.2330 | 0.7342 | 0.3184 | 0.741 |
### Steel2 (hypothesis: gapped boundaries merge regions)

| config | mode | threshold | closing_radius | bridge_px | pq | sq | rq | over-seg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skeleton_bridge_4px@0.70 **<- selected** | skeleton_bridge | 0.70 | -- | 4.0 | 0.3886 | 0.8061 | 0.4805 | 0.942 |
| skeleton_bridge_2px@0.70 | skeleton_bridge | 0.70 | -- | 2.0 | 0.3886 | 0.8061 | 0.4805 | 0.942 |
| skeleton_bridge_6px@0.75 | skeleton_bridge | 0.75 | -- | 6.0 | 0.3884 | 0.8098 | 0.4781 | 0.859 |
| skeleton_bridge_6px@0.70 | skeleton_bridge | 0.70 | -- | 6.0 | 0.3884 | 0.8061 | 0.4802 | 0.946 |
| skeleton_bridge_2px@0.75 | skeleton_bridge | 0.75 | -- | 2.0 | 0.3883 | 0.8101 | 0.4777 | 0.852 |

## TEST -- selected gap-closing config vs none@tuned vs watershed@val-best marker

| dataset | config | pq | sq | rq | over-seg | label |
| --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | selected gap-closing (val-best) | 0.2985 | 0.7397 | 0.4016 | 0.709 | MISPLACED |
| Steel2 (hypothesis: gapped boundaries merge regions) | selected gap-closing (val-best) | 0.3637 | 0.8034 | 0.4522 | 0.795 | OFFSET |
| Steel1 (control -- MISPLACED, a model problem) | none@tuned | 0.2563 | 0.7481 | 0.3422 | 0.729 | MISPLACED |
| Steel2 (hypothesis: gapped boundaries merge regions) | none@tuned | 0.2902 | 0.7889 | 0.3674 | 0.555 | OFFSET |
| Steel1 (control -- MISPLACED, a model problem) | watershed@val-best marker (extended) | 0.3408 | 0.7414 | 0.4571 | 0.937 | MISPLACED |
| Steel2 (hypothesis: gapped boundaries merge regions) | watershed@val-best marker (extended) | 0.3592 | 0.7657 | 0.4681 | 1.071 | OFFSET |

## Watershed marker (default 0.3; extended sweep [0.4, 0.45, 0.5, 0.55, 0.6, 0.7] on VAL)

| split | config | dataset | marker | pq | sq | rq | over-seg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | watershed_prob@marker0.40 | Steel1 (control -- MISPLACED, a model problem) | 0.40 | 0.2472 | 0.7174 | 0.3471 | 1.119 |
| val | watershed_prob@marker0.40 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.40 | 0.3186 | 0.7729 | 0.4135 | 1.463 |
| val | watershed_prob@marker0.45 | Steel1 (control -- MISPLACED, a model problem) | 0.45 | 0.2512 | 0.7211 | 0.3510 | 1.113 |
| val | watershed_prob@marker0.45 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.45 | 0.3391 | 0.7726 | 0.4399 | 1.407 |
| val | watershed_prob@marker0.50 | Steel1 (control -- MISPLACED, a model problem) | 0.50 | 0.2523 | 0.7192 | 0.3526 | 1.122 |
| val | watershed_prob@marker0.50 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.50 | 0.3515 | 0.7723 | 0.4560 | 1.337 |
| val | watershed_prob@marker0.55 | Steel1 (control -- MISPLACED, a model problem) | 0.55 | 0.2583 | 0.7229 | 0.3608 | 1.081 |
| val | watershed_prob@marker0.55 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.55 | 0.3639 | 0.7699 | 0.4735 | 1.271 |
| val | watershed_prob@marker0.60 **<- selected** | Steel1 (control -- MISPLACED, a model problem) | 0.60 | 0.2586 | 0.7258 | 0.3598 | 1.074 |
| val | watershed_prob@marker0.60 **<- selected** | Steel2 (hypothesis: gapped boundaries merge regions) | 0.60 | 0.3649 | 0.7695 | 0.4735 | 1.195 |
| val | watershed_prob@marker0.70 | Steel1 (control -- MISPLACED, a model problem) | 0.70 | 0.2428 | 0.7311 | 0.3345 | 0.888 |
| val | watershed_prob@marker0.70 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.70 | 0.3583 | 0.7700 | 0.4646 | 0.909 |
| test | watershed@val-best marker (extended) | Steel1 (control -- MISPLACED, a model problem) | 0.60 | 0.3408 | 0.7414 | 0.4571 | 0.937 |
| test | watershed@val-best marker (extended) | Steel2 (hypothesis: gapped boundaries merge regions) | 0.60 | 0.3592 | 0.7657 | 0.4681 | 1.071 |

## FN class breakdown and MERGED count, before vs after (TEST, none@tuned vs the VAL-selected gap-closing config)

| dataset | config | FN/tile | TINY | MERGED | SPLIT | OTHER |
| --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | before (none@tuned) | 12.5 | 581 (49%) | 576 (48%) | 24 (2%) | 16 (1%) |
| Steel2 (hypothesis: gapped boundaries merge regions) | before (none@tuned) | 23.9 | 855 (28%) | 2094 (70%) | 32 (1%) | 30 (1%) |
| Steel1 (control -- MISPLACED, a model problem) | after (selected) | 11.9 | 584 (51%) | 517 (45%) | 23 (2%) | 18 (2%) |
| Steel2 (hypothesis: gapped boundaries merge regions) | after (selected) | 20.3 | 818 (32%) | 1585 (62%) | 93 (4%) | 63 (2%) |

## Secondary diagnostic -- PQ excluding GT regions under the dataset's TINY floor (never a selection metric)

| dataset | floor px | pq before | pq after | true regions excluded |
| --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 50 | 0.3102 | 0.3630 | 6 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 50 | 0.3358 | 0.4045 | 6 |

## Checks

- PASS marker selected on VAL rows only
- PASS gap-closing config selected on VAL rows only
- PASS TEST scored exactly the three pre-declared configurations -- selected gap-closing (val-best), none@tuned, watershed@val-best marker (extended)
- PASS Steel1: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00
- PASS Steel2: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00
