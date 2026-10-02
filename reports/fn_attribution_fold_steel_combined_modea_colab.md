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
| Steel1 (control -- MISPLACED, a model problem) | 95 | 12.1 | 554 (48%) | 437 (38%) | 90 (8%) | 72 (6%) | 69 | 16 / 108 | 0.43 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 14.8 | 656 (35%) | 826 (44%) | 120 (6%) | 265 (14%) | 71 | 0 / 40 | 0.19 |

## TEST -- none@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 96 | 12.5 | 581 (49%) | 576 (48%) | 24 (2%) | 16 (1%) | 60 | 28 / 182 | 0.20 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 23.9 | 855 (28%) | 2098 (70%) | 32 (1%) | 29 (1%) | 100 | 7 / 97 | 0.11 |

## TEST -- reference: watershed_prob@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 96 | 11.1 | 583 (55%) | 335 (32%) | 81 (8%) | 63 (6%) | 29 | 10 / 95 | 0.24 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 18.4 | 861 (37%) | 1041 (45%) | 150 (6%) | 267 (12%) | 66 | 0 / 36 | 0.21 |

## Watershed marker threshold (default 0.3; swept [0.2, 0.25, 0.3, 0.35, 0.4] on VAL)

| split | config | dataset | marker | PQ | SQ | RQ | over-seg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | watershed_prob@marker0.20 | Steel1 (control -- MISPLACED, a model problem) | 0.20 | 0.2177 | 0.7159 | 0.3054 | 1.065 |
| val | watershed_prob@marker0.20 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.20 | 0.2280 | 0.7666 | 0.2985 | 1.690 |
| val | watershed_prob@marker0.25 | Steel1 (control -- MISPLACED, a model problem) | 0.25 | 0.2305 | 0.7163 | 0.3224 | 1.063 |
| val | watershed_prob@marker0.25 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.25 | 0.2544 | 0.7681 | 0.3316 | 1.649 |
| val | watershed_prob@marker0.30 | Steel1 (control -- MISPLACED, a model problem) | 0.30 | 0.2432 | 0.7134 | 0.3410 | 1.063 |
| val | watershed_prob@marker0.30 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.30 | 0.2784 | 0.7688 | 0.3626 | 1.592 |
| val | watershed_prob@marker0.35 | Steel1 (control -- MISPLACED, a model problem) | 0.35 | 0.2401 | 0.7154 | 0.3373 | 1.120 |
| val | watershed_prob@marker0.35 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.35 | 0.2989 | 0.7711 | 0.3882 | 1.538 |
| val | watershed_prob@marker0.40 **<- selected** | Steel1 (control -- MISPLACED, a model problem) | 0.40 | 0.2466 | 0.7178 | 0.3461 | 1.119 |
| val | watershed_prob@marker0.40 **<- selected** | Steel2 (hypothesis: gapped boundaries merge regions) | 0.40 | 0.3186 | 0.7730 | 0.4133 | 1.465 |
| test | watershed_prob@val-best marker | Steel1 (control -- MISPLACED, a model problem) | 0.40 | 0.3499 | 0.7456 | 0.4657 | 0.940 |
| test | watershed_prob@val-best marker | Steel2 (hypothesis: gapped boundaries merge regions) | 0.40 | 0.3475 | 0.7731 | 0.4497 | 1.293 |
| test | none@tuned | Steel1 (control -- MISPLACED, a model problem) | 0.30 | 0.2562 | 0.7479 | 0.3421 | 0.730 |
| test | none@tuned | Steel2 (hypothesis: gapped boundaries merge regions) | 0.30 | 0.2901 | 0.7891 | 0.3671 | 0.554 |
| test | reference: watershed_prob@tuned | Steel1 (control -- MISPLACED, a model problem) | 0.30 | 0.3239 | 0.7513 | 0.4300 | 0.942 |
| test | reference: watershed_prob@tuned | Steel2 (hypothesis: gapped boundaries merge regions) | 0.30 | 0.3109 | 0.7692 | 0.4040 | 1.408 |

## Checks

- PASS marker selected on VAL rows only
- PASS TEST scored exactly the three pre-declared configurations -- watershed_prob@val-best marker, none@tuned, reference: watershed_prob@tuned
- PASS val none@tuned Steel1: attributed FN/tile equals pq_fn -- 13.3474 vs 13.3474
- PASS val none@tuned Steel2: attributed FN/tile equals pq_fn -- 17.9683 vs 17.9683
- PASS val reference: watershed_prob@tuned Steel1: attributed FN/tile equals pq_fn -- 12.1368 vs 12.1368
- PASS val reference: watershed_prob@tuned Steel2: attributed FN/tile equals pq_fn -- 14.8175 vs 14.8175
- PASS test none@tuned Steel1: attributed FN/tile equals pq_fn -- 12.4688 vs 12.4688
- PASS test none@tuned Steel2: attributed FN/tile equals pq_fn -- 23.9206 vs 23.9206
- PASS test reference: watershed_prob@tuned Steel1: attributed FN/tile equals pq_fn -- 11.0625 vs 11.0625
- PASS test reference: watershed_prob@tuned Steel2: attributed FN/tile equals pq_fn -- 18.4048 vs 18.4048
- PASS Steel1: TEST watershed reference reproduces Cell 20 -- max |diff| 0.00e+00
- PASS Steel2: TEST watershed reference reproduces Cell 20 -- max |diff| 0.00e+00
