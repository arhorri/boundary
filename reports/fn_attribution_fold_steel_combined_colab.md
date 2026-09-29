# False-negative attribution -- fold_steel_combined (colab)

Checkpoint: epoch 21, config hash `067fecc2e06a7d0a`. An unmatched true region (best IoU <= 0.5, exactly PQ's pq_fn) is TINY (area below the dataset's threshold, tested first), MERGED (>50% inside one predicted region that also holds >50% of another true region), SPLIT (no predicted region holds >50%) or OTHER. MERGED gap = true-boundary pixels between the merged regions not within the gap tolerance of the predicted boundary.

## VAL -- none@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 95 | 13.2 | 549 (44%) | 655 (52%) | 30 (2%) | 19 (2%) | 106 | 61 / 188 | 0.36 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 77.1 | 4139 (43%) | 4237 (44%) | 946 (10%) | 393 (4%) | 59 | 0 / 67 | 0.06 |

## VAL -- reference: watershed_prob@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 95 | 12.5 | 556 (47%) | 454 (38%) | 99 (8%) | 80 (7%) | 80 | 16 / 110 | 0.41 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 91.4 | 4210 (37%) | 6375 (55%) | 462 (4%) | 470 (4%) | 69 | 0 / 31 | 0.05 |

## TEST -- none@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 96 | 12.1 | 582 (50%) | 527 (45%) | 33 (3%) | 19 (2%) | 45 | 28 / 176 | 0.21 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 103.9 | 5731 (44%) | 5847 (45%) | 1152 (9%) | 357 (3%) | 58 | 0 / 70 | 0.05 |

## TEST -- reference: watershed_prob@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 96 | 11.3 | 585 (54%) | 366 (34%) | 83 (8%) | 54 (5%) | 31 | 8 / 98 | 0.22 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 114.3 | 5774 (40%) | 7439 (52%) | 657 (5%) | 530 (4%) | 64 | 0 / 38 | 0.05 |

## Watershed marker threshold (default 0.3; swept [0.2, 0.25, 0.3, 0.35, 0.4] on VAL)

| split | config | dataset | marker | PQ | SQ | RQ | over-seg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | watershed_prob@marker0.20 | Steel1 (control -- MISPLACED, a model problem) | 0.20 | 0.2026 | 0.7138 | 0.2850 | 1.029 |
| val | watershed_prob@marker0.20 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.20 | 0.0673 | 0.6883 | 0.0974 | 0.413 |
| val | watershed_prob@marker0.25 | Steel1 (control -- MISPLACED, a model problem) | 0.25 | 0.2171 | 0.7137 | 0.3048 | 1.064 |
| val | watershed_prob@marker0.25 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.25 | 0.0799 | 0.6974 | 0.1144 | 0.422 |
| val | watershed_prob@marker0.30 | Steel1 (control -- MISPLACED, a model problem) | 0.30 | 0.2240 | 0.7200 | 0.3135 | 1.111 |
| val | watershed_prob@marker0.30 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.30 | 0.0913 | 0.7020 | 0.1306 | 0.433 |
| val | watershed_prob@marker0.35 | Steel1 (control -- MISPLACED, a model problem) | 0.35 | 0.2268 | 0.7179 | 0.3177 | 1.177 |
| val | watershed_prob@marker0.35 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.35 | 0.1032 | 0.7040 | 0.1473 | 0.457 |
| val | watershed_prob@marker0.40 **<- selected** | Steel1 (control -- MISPLACED, a model problem) | 0.40 | 0.2325 | 0.7149 | 0.3284 | 1.210 |
| val | watershed_prob@marker0.40 **<- selected** | Steel2 (hypothesis: gapped boundaries merge regions) | 0.40 | 0.1147 | 0.7039 | 0.1639 | 0.474 |
| test | watershed_prob@val-best marker | Steel1 (control -- MISPLACED, a model problem) | 0.40 | 0.3305 | 0.7457 | 0.4392 | 0.995 |
| test | watershed_prob@val-best marker | Steel2 (hypothesis: gapped boundaries merge regions) | 0.40 | 0.1215 | 0.6907 | 0.1752 | 0.417 |
| test | none@tuned | Steel1 (control -- MISPLACED, a model problem) | 0.30 | 0.2721 | 0.7471 | 0.3634 | 0.813 |
| test | none@tuned | Steel2 (hypothesis: gapped boundaries merge regions) | 0.30 | 0.1691 | 0.6622 | 0.2505 | 0.469 |
| test | reference: watershed_prob@tuned | Steel1 (control -- MISPLACED, a model problem) | 0.30 | 0.3027 | 0.7391 | 0.4056 | 0.982 |
| test | reference: watershed_prob@tuned | Steel2 (hypothesis: gapped boundaries merge regions) | 0.30 | 0.0985 | 0.6883 | 0.1419 | 0.389 |

## Checks

- PASS marker selected on VAL rows only
- PASS TEST scored exactly the three pre-declared configurations -- watershed_prob@val-best marker, none@tuned, reference: watershed_prob@tuned
- PASS val none@tuned Steel1: attributed FN/tile equals pq_fn -- 13.1895 vs 13.1895
- PASS val none@tuned Steel2: attributed FN/tile equals pq_fn -- 77.1032 vs 77.1032
- PASS val reference: watershed_prob@tuned Steel1: attributed FN/tile equals pq_fn -- 12.5158 vs 12.5158
- PASS val reference: watershed_prob@tuned Steel2: attributed FN/tile equals pq_fn -- 91.4048 vs 91.4048
- PASS test none@tuned Steel1: attributed FN/tile equals pq_fn -- 12.0938 vs 12.0938
- PASS test none@tuned Steel2: attributed FN/tile equals pq_fn -- 103.8651 vs 103.8651
- PASS test reference: watershed_prob@tuned Steel1: attributed FN/tile equals pq_fn -- 11.3333 vs 11.3333
- PASS test reference: watershed_prob@tuned Steel2: attributed FN/tile equals pq_fn -- 114.2857 vs 114.2857
- PASS Steel1: TEST watershed reference reproduces Cell 20 -- max |diff| 0.00e+00
- PASS Steel2: TEST watershed reference reproduces Cell 20 -- max |diff| 0.00e+00
