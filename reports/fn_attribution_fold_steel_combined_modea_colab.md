# False-negative attribution -- fold_steel_combined_modea (colab)

Checkpoint: epoch 32, config hash `067fecc2e06a7d0a`. An unmatched true region (best IoU <= 0.5, exactly PQ's pq_fn) is TINY (area below the dataset's threshold, tested first), MERGED (>50% inside one predicted region that also holds >50% of another true region), SPLIT (no predicted region holds >50%) or OTHER. MERGED gap = true-boundary pixels between the merged regions not within the gap tolerance of the predicted boundary.

## VAL -- none@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 95 | 13.4 | 552 (43%) | 676 (53%) | 24 (2%) | 17 (1%) | 107 | 55 / 184 | 0.34 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 18.0 | 655 (29%) | 1561 (69%) | 21 (1%) | 26 (1%) | 102 | 8 / 103 | 0.12 |

## VAL -- reference: watershed_prob@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 95 | 12.1 | 554 (48%) | 438 (38%) | 90 (8%) | 72 (6%) | 70 | 16 / 108 | 0.43 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 14.8 | 657 (35%) | 825 (44%) | 120 (6%) | 265 (14%) | 71 | 0 / 40 | 0.19 |

## TEST -- none@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 96 | 12.5 | 581 (49%) | 576 (48%) | 24 (2%) | 16 (1%) | 60 | 28 / 182 | 0.20 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 23.9 | 855 (28%) | 2094 (70%) | 32 (1%) | 30 (1%) | 100 | 7 / 98 | 0.11 |

## TEST -- reference: watershed_prob@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 96 | 11.1 | 583 (55%) | 336 (32%) | 81 (8%) | 62 (6%) | 29 | 10 / 96 | 0.24 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 18.4 | 861 (37%) | 1041 (45%) | 150 (6%) | 267 (12%) | 66 | 0 / 36 | 0.21 |

## Watershed marker threshold (default 0.3; swept [0.2, 0.25, 0.3, 0.35, 0.4] on VAL)

| split | config | dataset | marker | PQ | SQ | RQ | over-seg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | watershed_prob@marker0.20 | Steel1 (control -- MISPLACED, a model problem) | 0.20 | 0.2179 | 0.7159 | 0.3056 | 1.065 |
| val | watershed_prob@marker0.20 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.20 | 0.2274 | 0.7662 | 0.2978 | 1.689 |
| val | watershed_prob@marker0.25 | Steel1 (control -- MISPLACED, a model problem) | 0.25 | 0.2299 | 0.7166 | 0.3216 | 1.065 |
| val | watershed_prob@marker0.25 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.25 | 0.2542 | 0.7682 | 0.3314 | 1.649 |
| val | watershed_prob@marker0.30 | Steel1 (control -- MISPLACED, a model problem) | 0.30 | 0.2431 | 0.7137 | 0.3408 | 1.060 |
| val | watershed_prob@marker0.30 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.30 | 0.2785 | 0.7693 | 0.3627 | 1.594 |
| val | watershed_prob@marker0.35 | Steel1 (control -- MISPLACED, a model problem) | 0.35 | 0.2408 | 0.7152 | 0.3386 | 1.117 |
| val | watershed_prob@marker0.35 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.35 | 0.2987 | 0.7713 | 0.3879 | 1.539 |
| val | watershed_prob@marker0.40 **<- selected** | Steel1 (control -- MISPLACED, a model problem) | 0.40 | 0.2472 | 0.7174 | 0.3471 | 1.119 |
| val | watershed_prob@marker0.40 **<- selected** | Steel2 (hypothesis: gapped boundaries merge regions) | 0.40 | 0.3186 | 0.7729 | 0.4135 | 1.463 |
| test | watershed_prob@val-best marker | Steel1 (control -- MISPLACED, a model problem) | 0.40 | 0.3498 | 0.7456 | 0.4656 | 0.940 |
| test | watershed_prob@val-best marker | Steel2 (hypothesis: gapped boundaries merge regions) | 0.40 | 0.3471 | 0.7734 | 0.4492 | 1.295 |
| test | none@tuned | Steel1 (control -- MISPLACED, a model problem) | 0.30 | 0.2563 | 0.7481 | 0.3422 | 0.729 |
| test | none@tuned | Steel2 (hypothesis: gapped boundaries merge regions) | 0.30 | 0.2902 | 0.7889 | 0.3674 | 0.555 |
| test | reference: watershed_prob@tuned | Steel1 (control -- MISPLACED, a model problem) | 0.30 | 0.3238 | 0.7511 | 0.4299 | 0.943 |
| test | reference: watershed_prob@tuned | Steel2 (hypothesis: gapped boundaries merge regions) | 0.30 | 0.3106 | 0.7695 | 0.4036 | 1.410 |

## Checks

- PASS marker selected on VAL rows only
- PASS TEST scored exactly the three pre-declared configurations -- watershed_prob@val-best marker, none@tuned, reference: watershed_prob@tuned
- PASS val none@tuned Steel1: attributed FN/tile equals pq_fn -- 13.3579 vs 13.3579
- PASS val none@tuned Steel2: attributed FN/tile equals pq_fn -- 17.9603 vs 17.9603
- PASS val reference: watershed_prob@tuned Steel1: attributed FN/tile equals pq_fn -- 12.1474 vs 12.1474
- PASS val reference: watershed_prob@tuned Steel2: attributed FN/tile equals pq_fn -- 14.8175 vs 14.8175
- PASS test none@tuned Steel1: attributed FN/tile equals pq_fn -- 12.4688 vs 12.4688
- PASS test none@tuned Steel2: attributed FN/tile equals pq_fn -- 23.8968 vs 23.8968
- PASS test reference: watershed_prob@tuned Steel1: attributed FN/tile equals pq_fn -- 11.0625 vs 11.0625
- PASS test reference: watershed_prob@tuned Steel2: attributed FN/tile equals pq_fn -- 18.4048 vs 18.4048
- PASS Steel1: TEST watershed reference reproduces Cell 20 -- max |diff| 0.00e+00
- PASS Steel2: TEST watershed reference reproduces Cell 20 -- max |diff| 0.00e+00
