# Ground-truth region-size profile

Region count and region-area distribution of the boundary masks THEMSELVES -- independent of any checkpoint -- using the same boundary-to-region conversion `evaluate_region_metrics` scores predictions against.

| dataset | tiles | regions/tile (mean/median/min/max) | total regions | p50 area px | p10 area px | < 10px | < 25px | < 50px | < 100px |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 | 904 | 12.3/12.0/2/50 | 11143 | 453.0 | 32.0 | 12 (0.1%) | 871 (7.8%) | 1438 (12.9%) | 2059 (18.5%) |
| Steel2 | 504 | 92.4/95.0/2/189 | 46593 | 91.0 | 31.0 | 19 (0.0%) | 1759 (3.8%) | 12261 (26.3%) | 24772 (53.2%) |

## Area percentiles in full (pixels)

| dataset | p5 | p10 | p25 | p50 | p75 | p90 | p95 | p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 | 20.0 | 32.0 | 162.0 | 453.0 | 1362.0 | 7051.8 | 54463.5 | 60604.5 |
| Steel2 | 26.0 | 31.0 | 48.0 | 91.0 | 218.0 | 610.0 | 1334.0 | 14241.7 |
