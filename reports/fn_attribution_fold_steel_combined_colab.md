# False-negative attribution -- fold_steel_combined (colab)

Checkpoint: epoch 21, config hash `067fecc2e06a7d0a`. An unmatched true region (best IoU <= 0.5, exactly PQ's pq_fn) is TINY (area below the dataset's threshold, tested first), MERGED (>50% inside one predicted region that also holds >50% of another true region), SPLIT (no predicted region holds >50%) or OTHER. MERGED gap = true-boundary pixels between the merged regions not within the gap tolerance of the predicted boundary.

## VAL -- none@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 95 | 13.2 | 549 (44%) | 655 (52%) | 30 (2%) | 19 (2%) | 106 | 61 / 188 | 0.36 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 77.1 | 4137 (43%) | 4232 (44%) | 950 (10%) | 390 (4%) | 59 | 0 / 67 | 0.06 |

## VAL -- reference: watershed_prob@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 95 | 12.5 | 556 (47%) | 453 (38%) | 98 (8%) | 81 (7%) | 80 | 16 / 110 | 0.41 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 91.4 | 4210 (37%) | 6365 (55%) | 465 (4%) | 476 (4%) | 69 | 0 / 32 | 0.05 |

## TEST -- none@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 96 | 12.1 | 582 (50%) | 528 (45%) | 33 (3%) | 18 (2%) | 45 | 28 / 176 | 0.21 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 103.9 | 5730 (44%) | 5849 (45%) | 1150 (9%) | 357 (3%) | 58 | 0 / 70 | 0.05 |

## TEST -- reference: watershed_prob@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 96 | 11.3 | 585 (54%) | 366 (34%) | 83 (8%) | 54 (5%) | 31 | 8 / 98 | 0.22 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 114.3 | 5774 (40%) | 7436 (52%) | 659 (5%) | 534 (4%) | 64 | 0 / 38 | 0.05 |

## Watershed marker threshold (default 0.3; swept [0.2, 0.25, 0.3, 0.35, 0.4] on VAL)

| split | config | dataset | marker | PQ | SQ | RQ | over-seg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | watershed_prob@marker0.20 | Steel1 (control -- MISPLACED, a model problem) | 0.20 | 0.2034 | 0.7136 | 0.2861 | 1.025 |
| val | watershed_prob@marker0.20 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.20 | 0.0676 | 0.6879 | 0.0978 | 0.412 |
| val | watershed_prob@marker0.25 | Steel1 (control -- MISPLACED, a model problem) | 0.25 | 0.2175 | 0.7136 | 0.3054 | 1.066 |
| val | watershed_prob@marker0.25 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.25 | 0.0799 | 0.7022 | 0.1144 | 0.421 |
| val | watershed_prob@marker0.30 | Steel1 (control -- MISPLACED, a model problem) | 0.30 | 0.2248 | 0.7199 | 0.3146 | 1.109 |
| val | watershed_prob@marker0.30 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.30 | 0.0912 | 0.7026 | 0.1307 | 0.433 |
| val | watershed_prob@marker0.35 | Steel1 (control -- MISPLACED, a model problem) | 0.35 | 0.2273 | 0.7176 | 0.3185 | 1.178 |
| val | watershed_prob@marker0.35 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.35 | 0.1035 | 0.7038 | 0.1478 | 0.457 |
| val | watershed_prob@marker0.40 **<- selected** | Steel1 (control -- MISPLACED, a model problem) | 0.40 | 0.2335 | 0.7146 | 0.3299 | 1.212 |
| val | watershed_prob@marker0.40 **<- selected** | Steel2 (hypothesis: gapped boundaries merge regions) | 0.40 | 0.1150 | 0.7035 | 0.1643 | 0.474 |
| test | watershed_prob@val-best marker | Steel1 (control -- MISPLACED, a model problem) | 0.40 | 0.3305 | 0.7460 | 0.4391 | 0.996 |
| test | watershed_prob@val-best marker | Steel2 (hypothesis: gapped boundaries merge regions) | 0.40 | 0.1216 | 0.6908 | 0.1753 | 0.416 |
| test | none@tuned | Steel1 (control -- MISPLACED, a model problem) | 0.30 | 0.2724 | 0.7472 | 0.3637 | 0.811 |
| test | none@tuned | Steel2 (hypothesis: gapped boundaries merge regions) | 0.30 | 0.1693 | 0.6624 | 0.2507 | 0.470 |
| test | reference: watershed_prob@tuned | Steel1 (control -- MISPLACED, a model problem) | 0.30 | 0.3027 | 0.7394 | 0.4056 | 0.982 |
| test | reference: watershed_prob@tuned | Steel2 (hypothesis: gapped boundaries merge regions) | 0.30 | 0.0985 | 0.6888 | 0.1417 | 0.389 |

## Checks

- PASS marker selected on VAL rows only
- PASS TEST scored exactly the three pre-declared configurations -- watershed_prob@val-best marker, none@tuned, reference: watershed_prob@tuned
- PASS val none@tuned Steel1: attributed FN/tile equals pq_fn -- 13.1895 vs 13.1895
- PASS val none@tuned Steel2: attributed FN/tile equals pq_fn -- 77.0556 vs 77.0556
- PASS val reference: watershed_prob@tuned Steel1: attributed FN/tile equals pq_fn -- 12.5053 vs 12.5053
- PASS val reference: watershed_prob@tuned Steel2: attributed FN/tile equals pq_fn -- 91.3968 vs 91.3968
- PASS test none@tuned Steel1: attributed FN/tile equals pq_fn -- 12.0938 vs 12.0938
- PASS test none@tuned Steel2: attributed FN/tile equals pq_fn -- 103.8571 vs 103.8571
- PASS test reference: watershed_prob@tuned Steel1: attributed FN/tile equals pq_fn -- 11.3333 vs 11.3333
- PASS test reference: watershed_prob@tuned Steel2: attributed FN/tile equals pq_fn -- 114.3095 vs 114.3095
- PASS Steel1: TEST watershed reference reproduces Cell 20 -- max |diff| 0.00e+00
- PASS Steel2: TEST watershed reference reproduces Cell 20 -- max |diff| 0.00e+00
