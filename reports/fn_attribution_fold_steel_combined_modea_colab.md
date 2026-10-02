# False-negative attribution -- fold_steel_combined_modea (colab)

Checkpoint: epoch 32, config hash `067fecc2e06a7d0a`. An unmatched true region (best IoU <= 0.5, exactly PQ's pq_fn) is TINY (area below the dataset's threshold, tested first), MERGED (>50% inside one predicted region that also holds >50% of another true region), SPLIT (no predicted region holds >50%) or OTHER. MERGED gap = true-boundary pixels between the merged regions not within the gap tolerance of the predicted boundary.

## VAL -- none@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 95 | 13.3 | 552 (44%) | 675 (53%) | 24 (2%) | 17 (1%) | 107 | 55 / 183 | 0.34 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 18.0 | 655 (29%) | 1562 (69%) | 21 (1%) | 26 (1%) | 102 | 8 / 102 | 0.12 |

## VAL -- reference: watershed_prob@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 95 | 12.1 | 554 (48%) | 438 (38%) | 90 (8%) | 72 (6%) | 70 | 16 / 108 | 0.43 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 14.8 | 656 (35%) | 828 (44%) | 118 (6%) | 264 (14%) | 71 | 0 / 40 | 0.19 |

## TEST -- none@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 96 | 12.5 | 581 (49%) | 576 (48%) | 24 (2%) | 16 (1%) | 60 | 28 / 182 | 0.20 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 23.9 | 855 (28%) | 2096 (70%) | 31 (1%) | 30 (1%) | 100 | 7 / 98 | 0.11 |

## TEST -- reference: watershed_prob@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 96 | 11.1 | 583 (55%) | 334 (31%) | 80 (8%) | 65 (6%) | 29 | 10 / 96 | 0.24 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 18.4 | 861 (37%) | 1041 (45%) | 151 (7%) | 266 (11%) | 66 | 0 / 36 | 0.21 |

## Watershed marker threshold (default 0.3; swept [0.2, 0.25, 0.3, 0.35, 0.4] on VAL)

| split | config | dataset | marker | PQ | SQ | RQ | over-seg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | watershed_prob@marker0.20 | Steel1 (control -- MISPLACED, a model problem) | 0.20 | 0.2178 | 0.7159 | 0.3056 | 1.064 |
| val | watershed_prob@marker0.20 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.20 | 0.2274 | 0.7661 | 0.2978 | 1.688 |
| val | watershed_prob@marker0.25 | Steel1 (control -- MISPLACED, a model problem) | 0.25 | 0.2299 | 0.7167 | 0.3216 | 1.066 |
| val | watershed_prob@marker0.25 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.25 | 0.2542 | 0.7679 | 0.3314 | 1.649 |
| val | watershed_prob@marker0.30 | Steel1 (control -- MISPLACED, a model problem) | 0.30 | 0.2429 | 0.7138 | 0.3404 | 1.063 |
| val | watershed_prob@marker0.30 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.30 | 0.2786 | 0.7687 | 0.3629 | 1.592 |
| val | watershed_prob@marker0.35 | Steel1 (control -- MISPLACED, a model problem) | 0.35 | 0.2408 | 0.7152 | 0.3385 | 1.118 |
| val | watershed_prob@marker0.35 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.35 | 0.2987 | 0.7713 | 0.3879 | 1.538 |
| val | watershed_prob@marker0.40 **<- selected** | Steel1 (control -- MISPLACED, a model problem) | 0.40 | 0.2467 | 0.7179 | 0.3462 | 1.119 |
| val | watershed_prob@marker0.40 **<- selected** | Steel2 (hypothesis: gapped boundaries merge regions) | 0.40 | 0.3189 | 0.7729 | 0.4138 | 1.465 |
| test | watershed_prob@val-best marker | Steel1 (control -- MISPLACED, a model problem) | 0.40 | 0.3503 | 0.7461 | 0.4658 | 0.941 |
| test | watershed_prob@val-best marker | Steel2 (hypothesis: gapped boundaries merge regions) | 0.40 | 0.3472 | 0.7730 | 0.4495 | 1.293 |
| test | none@tuned | Steel1 (control -- MISPLACED, a model problem) | 0.30 | 0.2558 | 0.7480 | 0.3416 | 0.734 |
| test | none@tuned | Steel2 (hypothesis: gapped boundaries merge regions) | 0.30 | 0.2902 | 0.7891 | 0.3673 | 0.555 |
| test | reference: watershed_prob@tuned | Steel1 (control -- MISPLACED, a model problem) | 0.30 | 0.3237 | 0.7513 | 0.4296 | 0.944 |
| test | reference: watershed_prob@tuned | Steel2 (hypothesis: gapped boundaries merge regions) | 0.30 | 0.3107 | 0.7696 | 0.4037 | 1.408 |

## Checks

- PASS marker selected on VAL rows only
- PASS TEST scored exactly the three pre-declared configurations -- watershed_prob@val-best marker, none@tuned, reference: watershed_prob@tuned
- PASS val none@tuned Steel1: attributed FN/tile equals pq_fn -- 13.3474 vs 13.3474
- PASS val none@tuned Steel2: attributed FN/tile equals pq_fn -- 17.9683 vs 17.9683
- PASS val reference: watershed_prob@tuned Steel1: attributed FN/tile equals pq_fn -- 12.1474 vs 12.1474
- PASS val reference: watershed_prob@tuned Steel2: attributed FN/tile equals pq_fn -- 14.8095 vs 14.8095
- PASS test none@tuned Steel1: attributed FN/tile equals pq_fn -- 12.4688 vs 12.4688
- PASS test none@tuned Steel2: attributed FN/tile equals pq_fn -- 23.9048 vs 23.9048
- PASS test reference: watershed_prob@tuned Steel1: attributed FN/tile equals pq_fn -- 11.0625 vs 11.0625
- PASS test reference: watershed_prob@tuned Steel2: attributed FN/tile equals pq_fn -- 18.4048 vs 18.4048
- PASS Steel1: TEST watershed reference reproduces Cell 20 -- max |diff| 0.00e+00
- PASS Steel2: TEST watershed reference reproduces Cell 20 -- max |diff| 0.00e+00
