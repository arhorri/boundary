# False-negative attribution -- fold_steel_combined (colab)

Checkpoint: epoch 21, config hash `067fecc2e06a7d0a`. An unmatched true region (best IoU <= 0.5, exactly PQ's pq_fn) is TINY (area below the dataset's threshold, tested first), MERGED (>50% inside one predicted region that also holds >50% of another true region), SPLIT (no predicted region holds >50%) or OTHER. MERGED gap = true-boundary pixels between the merged regions not within the gap tolerance of the predicted boundary.

## VAL -- none@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 95 | 13.2 | 549 (44%) | 655 (52%) | 30 (2%) | 19 (2%) | 106 | 61 / 188 | 0.36 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 77.0 | 4137 (43%) | 4231 (44%) | 946 (10%) | 394 (4%) | 59 | 0 / 67 | 0.06 |

## VAL -- reference: watershed_prob@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 95 | 12.5 | 556 (47%) | 453 (38%) | 99 (8%) | 80 (7%) | 80 | 16 / 110 | 0.41 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 91.4 | 4210 (37%) | 6377 (55%) | 457 (4%) | 473 (4%) | 69 | 0 / 32 | 0.05 |

## TEST -- none@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 96 | 12.1 | 582 (50%) | 528 (45%) | 33 (3%) | 18 (2%) | 45 | 28 / 176 | 0.21 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 103.8 | 5730 (44%) | 5847 (45%) | 1149 (9%) | 358 (3%) | 58 | 0 / 70 | 0.05 |

## TEST -- reference: watershed_prob@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 96 | 11.3 | 585 (54%) | 366 (34%) | 83 (8%) | 54 (5%) | 31 | 8 / 98 | 0.22 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 114.3 | 5774 (40%) | 7437 (52%) | 660 (5%) | 531 (4%) | 64 | 0 / 38 | 0.05 |

## Watershed marker threshold (default 0.3; swept [0.2, 0.25, 0.3, 0.35, 0.4] on VAL)

| split | config | dataset | marker | PQ | SQ | RQ | over-seg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | watershed_prob@marker0.20 | Steel1 (control -- MISPLACED, a model problem) | 0.20 | 0.2034 | 0.7135 | 0.2862 | 1.025 |
| val | watershed_prob@marker0.20 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.20 | 0.0675 | 0.6884 | 0.0976 | 0.412 |
| val | watershed_prob@marker0.25 | Steel1 (control -- MISPLACED, a model problem) | 0.25 | 0.2165 | 0.7141 | 0.3039 | 1.066 |
| val | watershed_prob@marker0.25 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.25 | 0.0799 | 0.6977 | 0.1143 | 0.422 |
| val | watershed_prob@marker0.30 | Steel1 (control -- MISPLACED, a model problem) | 0.30 | 0.2247 | 0.7197 | 0.3145 | 1.110 |
| val | watershed_prob@marker0.30 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.30 | 0.0911 | 0.7026 | 0.1305 | 0.433 |
| val | watershed_prob@marker0.35 | Steel1 (control -- MISPLACED, a model problem) | 0.35 | 0.2278 | 0.7173 | 0.3192 | 1.176 |
| val | watershed_prob@marker0.35 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.35 | 0.1033 | 0.7042 | 0.1475 | 0.457 |
| val | watershed_prob@marker0.40 **<- selected** | Steel1 (control -- MISPLACED, a model problem) | 0.40 | 0.2325 | 0.7141 | 0.3286 | 1.217 |
| val | watershed_prob@marker0.40 **<- selected** | Steel2 (hypothesis: gapped boundaries merge regions) | 0.40 | 0.1151 | 0.7036 | 0.1646 | 0.474 |
| test | watershed_prob@val-best marker | Steel1 (control -- MISPLACED, a model problem) | 0.40 | 0.3302 | 0.7456 | 0.4389 | 0.997 |
| test | watershed_prob@val-best marker | Steel2 (hypothesis: gapped boundaries merge regions) | 0.40 | 0.1215 | 0.6911 | 0.1750 | 0.416 |
| test | none@tuned | Steel1 (control -- MISPLACED, a model problem) | 0.30 | 0.2720 | 0.7471 | 0.3633 | 0.813 |
| test | none@tuned | Steel2 (hypothesis: gapped boundaries merge regions) | 0.30 | 0.1693 | 0.6621 | 0.2508 | 0.469 |
| test | reference: watershed_prob@tuned | Steel1 (control -- MISPLACED, a model problem) | 0.30 | 0.3025 | 0.7391 | 0.4055 | 0.983 |
| test | reference: watershed_prob@tuned | Steel2 (hypothesis: gapped boundaries merge regions) | 0.30 | 0.0985 | 0.6889 | 0.1417 | 0.389 |

## Checks

- PASS marker selected on VAL rows only
- PASS TEST scored exactly the three pre-declared configurations -- watershed_prob@val-best marker, none@tuned, reference: watershed_prob@tuned
- PASS val none@tuned Steel1: attributed FN/tile equals pq_fn -- 13.1895 vs 13.1895
- PASS val none@tuned Steel2: attributed FN/tile equals pq_fn -- 77.0476 vs 77.0476
- PASS val reference: watershed_prob@tuned Steel1: attributed FN/tile equals pq_fn -- 12.5053 vs 12.5053
- PASS val reference: watershed_prob@tuned Steel2: attributed FN/tile equals pq_fn -- 91.4048 vs 91.4048
- PASS test none@tuned Steel1: attributed FN/tile equals pq_fn -- 12.0938 vs 12.0938
- PASS test none@tuned Steel2: attributed FN/tile equals pq_fn -- 103.8413 vs 103.8413
- PASS test reference: watershed_prob@tuned Steel1: attributed FN/tile equals pq_fn -- 11.3333 vs 11.3333
- PASS test reference: watershed_prob@tuned Steel2: attributed FN/tile equals pq_fn -- 114.3016 vs 114.3016
- PASS Steel1: TEST watershed reference reproduces Cell 20 -- max |diff| 0.00e+00
- PASS Steel2: TEST watershed reference reproduces Cell 20 -- max |diff| 0.00e+00
