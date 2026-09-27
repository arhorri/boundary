# False-negative attribution -- fold_steel_combined (colab)

Checkpoint: epoch 21, config hash `067fecc2e06a7d0a`. An unmatched true region (best IoU <= 0.5, exactly PQ's pq_fn) is TINY (area below the dataset's threshold, tested first), MERGED (>50% inside one predicted region that also holds >50% of another true region), SPLIT (no predicted region holds >50%) or OTHER. MERGED gap = true-boundary pixels between the merged regions not within the gap tolerance of the predicted boundary.

## VAL -- none@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 95 | 9.1 | 178 (21%) | 641 (74%) | 27 (3%) | 22 (3%) | 278 | 66 / 196 | 0.36 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 67.6 | 2930 (34%) | 4228 (50%) | 924 (11%) | 435 (5%) | 67 | 0 / 67 | 0.06 |

## VAL -- reference: watershed_prob@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 95 | 8.4 | 182 (23%) | 424 (53%) | 101 (13%) | 95 (12%) | 222 | 20 / 120 | 0.41 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 81.9 | 2999 (29%) | 6340 (61%) | 458 (4%) | 519 (5%) | 79 | 0 / 31 | 0.05 |

## TEST -- none@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 96 | 7.5 | 151 (21%) | 522 (72%) | 26 (4%) | 25 (3%) | 299 | 29 / 192 | 0.21 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 89.1 | 3825 (34%) | 5871 (52%) | 1137 (10%) | 395 (4%) | 70 | 0 / 71 | 0.05 |

## TEST -- reference: watershed_prob@tuned

| dataset | tiles | FN/tile | TINY | MERGED | SPLIT | OTHER | FN area p50 | MERGED gap p50 / interface p50 | pooled gap frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 (control -- MISPLACED, a model problem) | 96 | 6.7 | 153 (24%) | 337 (52%) | 85 (13%) | 71 (11%) | 208 | 8 / 118 | 0.22 |
| Steel2 (hypothesis: gapped boundaries merge regions) | 126 | 99.1 | 3864 (31%) | 7340 (59%) | 673 (5%) | 610 (5%) | 77 | 0 / 37 | 0.05 |

## Watershed marker threshold (default 0.3; swept [0.2, 0.25, 0.3, 0.35, 0.4] on VAL)

| split | config | dataset | marker | PQ | SQ | RQ | over-seg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | watershed_prob@marker0.20 | Steel1 (control -- MISPLACED, a model problem) | 0.20 | 0.2258 | 0.7140 | 0.3176 | 1.234 |
| val | watershed_prob@marker0.20 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.20 | 0.0713 | 0.6876 | 0.1031 | 0.457 |
| val | watershed_prob@marker0.25 | Steel1 (control -- MISPLACED, a model problem) | 0.25 | 0.2407 | 0.7145 | 0.3379 | 1.287 |
| val | watershed_prob@marker0.25 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.25 | 0.0843 | 0.7021 | 0.1207 | 0.467 |
| val | watershed_prob@marker0.30 | Steel1 (control -- MISPLACED, a model problem) | 0.30 | 0.2485 | 0.7192 | 0.3483 | 1.335 |
| val | watershed_prob@marker0.30 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.30 | 0.0965 | 0.7024 | 0.1381 | 0.479 |
| val | watershed_prob@marker0.35 | Steel1 (control -- MISPLACED, a model problem) | 0.35 | 0.2499 | 0.7179 | 0.3499 | 1.413 |
| val | watershed_prob@marker0.35 | Steel2 (hypothesis: gapped boundaries merge regions) | 0.35 | 0.1093 | 0.7038 | 0.1560 | 0.505 |
| val | watershed_prob@marker0.40 **<- selected** | Steel1 (control -- MISPLACED, a model problem) | 0.40 | 0.2573 | 0.7140 | 0.3641 | 1.459 |
| val | watershed_prob@marker0.40 **<- selected** | Steel2 (hypothesis: gapped boundaries merge regions) | 0.40 | 0.1216 | 0.7036 | 0.1737 | 0.524 |
| test | watershed_prob@val-best marker | Steel1 (control -- MISPLACED, a model problem) | 0.40 | 0.3661 | 0.7457 | 0.4883 | 1.233 |
| test | watershed_prob@val-best marker | Steel2 (hypothesis: gapped boundaries merge regions) | 0.40 | 0.1307 | 0.6895 | 0.1887 | 0.470 |
| test | none@tuned | Steel1 (control -- MISPLACED, a model problem) | 0.30 | 0.3212 | 0.7469 | 0.4297 | 0.916 |
| test | none@tuned | Steel2 (hypothesis: gapped boundaries merge regions) | 0.30 | 0.1809 | 0.6614 | 0.2684 | 0.511 |
| test | reference: watershed_prob@tuned | Steel1 (control -- MISPLACED, a model problem) | 0.30 | 0.3394 | 0.7388 | 0.4562 | 1.203 |
| test | reference: watershed_prob@tuned | Steel2 (hypothesis: gapped boundaries merge regions) | 0.30 | 0.1059 | 0.6876 | 0.1525 | 0.439 |

## Checks

- PASS marker selected on VAL rows only
- PASS TEST scored exactly the three pre-declared configurations -- watershed_prob@val-best marker, none@tuned, reference: watershed_prob@tuned
- PASS val none@tuned Steel1: attributed FN/tile equals pq_fn -- 9.1368 vs 9.1368
- PASS val none@tuned Steel2: attributed FN/tile equals pq_fn -- 67.5952 vs 67.5952
- PASS val reference: watershed_prob@tuned Steel1: attributed FN/tile equals pq_fn -- 8.4421 vs 8.4421
- PASS val reference: watershed_prob@tuned Steel2: attributed FN/tile equals pq_fn -- 81.8730 vs 81.8730
- PASS test none@tuned Steel1: attributed FN/tile equals pq_fn -- 7.5417 vs 7.5417
- PASS test none@tuned Steel2: attributed FN/tile equals pq_fn -- 89.1111 vs 89.1111
- PASS test reference: watershed_prob@tuned Steel1: attributed FN/tile equals pq_fn -- 6.7292 vs 6.7292
- PASS test reference: watershed_prob@tuned Steel2: attributed FN/tile equals pq_fn -- 99.1032 vs 99.1032
- PASS Steel1: TEST watershed reference reproduces Cell 20 -- max |diff| 0.00e+00
- PASS Steel2: TEST watershed reference reproduces Cell 20 -- max |diff| 0.00e+00
