# False-negative attribution -- fold_steel_combined_modea (colab)

Checkpoint: epoch 32, config hash `067fecc2e06a7d0a`. An unmatched true region (best IoU <= 0.5, exactly PQ's pq_fn) is TINY (area below the dataset's threshold, tested first), MERGED (>50% inside one predicted region that also holds >50% of another true region), SPLIT (no predicted region holds >50%) or OTHER. MERGED gap = true-boundary pixels between the merged regions not within the gap tolerance of the predicted boundary.

## VAL -- none@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 95 | 13.4 | 552 (43%) | 676 (53%) | 24 (2%) | 17 (1%) | 107 | 55 / 184 | 0.34 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 18.0 | 655 (29%) | 1561 (69%) | 21 (1%) | 26 (1%) | 102 | 8 / 102 | 0.12 |

## VAL -- reference: watershed_prob@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 95 | 12.1 | 554 (48%) | 437 (38%) | 90 (8%) | 72 (6%) | 69 | 16 / 108 | 0.43 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 14.8 | 657 (35%) | 825 (44%) | 119 (6%) | 264 (14%) | 71 | 0 / 40 | 0.19 |

## TEST -- none@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 96 | 12.5 | 581 (49%) | 576 (48%) | 24 (2%) | 16 (1%) | 60 | 28 / 182 | 0.20 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 23.9 | 855 (28%) | 2095 (70%) | 31 (1%) | 30 (1%) | 100 | 7 / 98 | 0.11 |

## TEST -- reference: watershed_prob@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 96 | 11.1 | 583 (55%) | 335 (32%) | 81 (8%) | 64 (6%) | 29 | 10 / 95 | 0.24 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 18.4 | 861 (37%) | 1040 (45%) | 149 (6%) | 266 (11%) | 66 | 0 / 36 | 0.21 |

## Watershed marker threshold (default 0.3; swept [0.2, 0.25, 0.3, 0.35, 0.4] on VAL)

| split | config | dataset | marker | PQ | SQ | RQ | over-seg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | watershed_prob@marker0.20 | Steel1 (control -- MISPLACED, a model problem) | 0.20 | 0.2178 | 0.7159 | 0.3055 | 1.065 |
| val | watershed_prob@marker0.20 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.20 | 0.2276 | 0.7662 | 0.2981 | 1.688 |
| val | watershed_prob@marker0.25 | Steel1 (control -- MISPLACED, a model problem) | 0.25 | 0.2304 | 0.7163 | 0.3223 | 1.064 |
| val | watershed_prob@marker0.25 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.25 | 0.2544 | 0.7684 | 0.3316 | 1.649 |
| val | watershed_prob@marker0.30 | Steel1 (control -- MISPLACED, a model problem) | 0.30 | 0.2433 | 0.7133 | 0.3410 | 1.062 |
| val | watershed_prob@marker0.30 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.30 | 0.2790 | 0.7693 | 0.3633 | 1.593 |
| val | watershed_prob@marker0.35 | Steel1 (control -- MISPLACED, a model problem) | 0.35 | 0.2402 | 0.7154 | 0.3375 | 1.119 |
| val | watershed_prob@marker0.35 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.35 | 0.2986 | 0.7714 | 0.3878 | 1.540 |
| val | watershed_prob@marker0.40 **<- selected** | Steel1 (control -- MISPLACED, a model problem) | 0.40 | 0.2468 | 0.7179 | 0.3464 | 1.117 |
| val | watershed_prob@marker0.40 **<- selected** | Steel2 (hypothesis: gapped boundaries merge regions) | 0.40 | 0.3186 | 0.7729 | 0.4135 | 1.465 |
| test | watershed_prob@val-best marker | Steel1 (control -- MISPLACED, a model problem) | 0.40 | 0.3496 | 0.7458 | 0.4651 | 0.940 |
| test | watershed_prob@val-best marker | Steel2 (hypothesis: gapped boundaries merge regions) | 0.40 | 0.3474 | 0.7730 | 0.4496 | 1.294 |
| test | none@tuned | Steel1 (control -- MISPLACED, a model problem) | 0.30 | 0.2562 | 0.7481 | 0.3421 | 0.730 |
| test | none@tuned | Steel2 (hypothesis: gapped boundaries merge regions) | 0.30 | 0.2903 | 0.7890 | 0.3675 | 0.554 |
| test | reference: watershed_prob@tuned | Steel1 (control -- MISPLACED, a model problem) | 0.30 | 0.3235 | 0.7512 | 0.4294 | 0.942 |
| test | reference: watershed_prob@tuned | Steel2 (hypothesis: gapped boundaries merge regions) | 0.30 | 0.3109 | 0.7690 | 0.4041 | 1.409 |

## Checks

- PASS marker selected on VAL rows only
- PASS TEST scored exactly the three pre-declared configurations -- watershed_prob@val-best marker, none@tuned, reference: watershed_prob@tuned
- PASS val none@tuned Steel1: attributed FN/tile equals pq_fn -- 13.3579 vs 13.3579
- PASS val none@tuned Steel2: attributed FN/tile equals pq_fn -- 17.9603 vs 17.9603
- PASS val reference: watershed_prob@tuned Steel1: attributed FN/tile equals pq_fn -- 12.1368 vs 12.1368
- PASS val reference: watershed_prob@tuned Steel2: attributed FN/tile equals pq_fn -- 14.8016 vs 14.8016
- PASS test none@tuned Steel1: attributed FN/tile equals pq_fn -- 12.4688 vs 12.4688
- PASS test none@tuned Steel2: attributed FN/tile equals pq_fn -- 23.8968 vs 23.8968
- PASS test reference: watershed_prob@tuned Steel1: attributed FN/tile equals pq_fn -- 11.0729 vs 11.0729
- PASS test reference: watershed_prob@tuned Steel2: attributed FN/tile equals pq_fn -- 18.3810 vs 18.3810
- PASS Steel1: TEST watershed reference reproduces Cell 20 -- max |diff| 0.00e+00
- PASS Steel2: TEST watershed reference reproduces Cell 20 -- max |diff| 0.00e+00
