# Ground-truth region-size profile

Region count and region-area distribution of the boundary masks THEMSELVES -- independent of any checkpoint -- using the same boundary-to-region conversion `evaluate_region_metrics` scores predictions against.

| dataset | tiles | regions/tile (mean/median/min/max) | total regions | p50 area px | p10 area px | < 10px | < 25px | < 50px | < 100px |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 | 904 | 16.5/14.0/2/73 | 14929 | 237.0 | 13.0 | 1101 (7.4%) | 4413 (29.6%) | 5135 (34.4%) | 5823 (39.0%) |
| Steel2 | 504 | 102.8/104.0/3/207 | 51819 | 78.0 | 21.0 | 984 (1.9%) | 6597 (12.7%) | 17526 (33.8%) | 29997 (57.9%) |

## Area percentiles in full (pixels)

| dataset | p5 | p10 | p25 | p50 | p75 | p90 | p95 | p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 | 8.0 | 13.0 | 21.0 | 237.0 | 869.0 | 3707.2 | 48706.4 | 60030.5 |
| Steel2 | 16.0 | 21.0 | 38.0 | 78.0 | 192.0 | 538.0 | 1169.1 | 11591.9 |
