# False-negative attribution -- fold_steel_combined_modea (colab)

Checkpoint: epoch 32, config hash `067fecc2e06a7d0a`. An unmatched true region (best IoU <= 0.5, exactly PQ's pq_fn) is TINY (area below the dataset's threshold, tested first), MERGED (>50% inside one predicted region that also holds >50% of another true region), SPLIT (no predicted region holds >50%) or OTHER. MERGED gap = true-boundary pixels between the merged regions not within the gap tolerance of the predicted boundary.

## VAL -- none@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 95 | 13.4 | 552 (43%) | 676 (53%) | 24 (2%) | 17 (1%) | 107 | 55 / 184 | 0.34 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 18.0 | 655 (29%) | 1562 (69%) | 21 (1%) | 26 (1%) | 102 | 8 / 102 | 0.12 |

## VAL -- reference: watershed_prob@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 95 | 12.1 | 554 (48%) | 438 (38%) | 90 (8%) | 72 (6%) | 70 | 16 / 108 | 0.43 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 14.8 | 656 (35%) | 824 (44%) | 120 (6%) | 265 (14%) | 71 | 0 / 40 | 0.19 |

## TEST -- none@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 96 | 12.5 | 581 (49%) | 576 (48%) | 24 (2%) | 16 (1%) | 60 | 28 / 182 | 0.20 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 23.9 | 855 (28%) | 2096 (70%) | 32 (1%) | 30 (1%) | 100 | 7 / 98 | 0.11 |

## TEST -- reference: watershed_prob@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 96 | 11.1 | 583 (55%) | 335 (32%) | 81 (8%) | 63 (6%) | 29 | 10 / 95 | 0.24 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 18.4 | 861 (37%) | 1042 (45%) | 149 (6%) | 266 (11%) | 66 | 0 / 36 | 0.21 |

## Watershed marker threshold (default 0.3; swept [0.2, 0.25, 0.3, 0.35, 0.4] on VAL)

| split | config | dataset | marker | PQ | SQ | RQ | over-seg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | watershed_prob@marker0.20 | Steel1 (control -- MISPLACED, a model problem) | 0.20 | 0.2181 | 0.7159 | 0.3058 | 1.064 |
| val | watershed_prob@marker0.20 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.20 | 0.2275 | 0.7658 | 0.2980 | 1.689 |
| val | watershed_prob@marker0.25 | Steel1 (control -- MISPLACED, a model problem) | 0.25 | 0.2311 | 0.7168 | 0.3232 | 1.061 |
| val | watershed_prob@marker0.25 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.25 | 0.2544 | 0.7682 | 0.3316 | 1.649 |
| val | watershed_prob@marker0.30 | Steel1 (control -- MISPLACED, a model problem) | 0.30 | 0.2432 | 0.7137 | 0.3408 | 1.060 |
| val | watershed_prob@marker0.30 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.30 | 0.2786 | 0.7688 | 0.3630 | 1.593 |
| val | watershed_prob@marker0.35 | Steel1 (control -- MISPLACED, a model problem) | 0.35 | 0.2409 | 0.7152 | 0.3386 | 1.117 |
| val | watershed_prob@marker0.35 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.35 | 0.2987 | 0.7711 | 0.3880 | 1.539 |
| val | watershed_prob@marker0.40 **<- selected** | Steel1 (control -- MISPLACED, a model problem) | 0.40 | 0.2469 | 0.7174 | 0.3467 | 1.118 |
| val | watershed_prob@marker0.40 **<- selected** | Steel2 (hypothesis: gapped boundaries merge regions) | 0.40 | 0.3185 | 0.7728 | 0.4134 | 1.465 |
| test | watershed_prob@val-best marker | Steel1 (control -- MISPLACED, a model problem) | 0.40 | 0.3498 | 0.7456 | 0.4655 | 0.941 |
| test | watershed_prob@val-best marker | Steel2 (hypothesis: gapped boundaries merge regions) | 0.40 | 0.3472 | 0.7735 | 0.4491 | 1.294 |
| test | none@tuned | Steel1 (control -- MISPLACED, a model problem) | 0.30 | 0.2563 | 0.7481 | 0.3421 | 0.730 |
| test | none@tuned | Steel2 (hypothesis: gapped boundaries merge regions) | 0.30 | 0.2899 | 0.7890 | 0.3669 | 0.555 |
| test | reference: watershed_prob@tuned | Steel1 (control -- MISPLACED, a model problem) | 0.30 | 0.3238 | 0.7513 | 0.4298 | 0.941 |
| test | reference: watershed_prob@tuned | Steel2 (hypothesis: gapped boundaries merge regions) | 0.30 | 0.3110 | 0.7692 | 0.4042 | 1.408 |

## Checks

- PASS marker selected on VAL rows only
- PASS TEST scored exactly the three pre-declared configurations -- watershed_prob@val-best marker, none@tuned, reference: watershed_prob@tuned
- PASS val none@tuned Steel1: attributed FN/tile equals pq_fn -- 13.3579 vs 13.3579
- PASS val none@tuned Steel2: attributed FN/tile equals pq_fn -- 17.9683 vs 17.9683
- PASS val reference: watershed_prob@tuned Steel1: attributed FN/tile equals pq_fn -- 12.1474 vs 12.1474
- PASS val reference: watershed_prob@tuned Steel2: attributed FN/tile equals pq_fn -- 14.8016 vs 14.8016
- PASS test none@tuned Steel1: attributed FN/tile equals pq_fn -- 12.4688 vs 12.4688
- PASS test none@tuned Steel2: attributed FN/tile equals pq_fn -- 23.9127 vs 23.9127
- PASS test reference: watershed_prob@tuned Steel1: attributed FN/tile equals pq_fn -- 11.0625 vs 11.0625
- PASS test reference: watershed_prob@tuned Steel2: attributed FN/tile equals pq_fn -- 18.3968 vs 18.3968
- PASS Steel1: TEST watershed reference reproduces Cell 20 -- max |diff| 0.00e+00
- PASS Steel2: TEST watershed reference reproduces Cell 20 -- max |diff| 0.00e+00
