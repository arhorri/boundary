# Region-level metrics -- fold_steel_combined (colab)

Checkpoint: epoch 21, config hash `067fecc2e06a7d0a`. Ground truth scored at line_width_px=4.0.

Marker-controlled watershed on the probability map (watershed_marker_threshold=0.3), scored against the region partition the ground-truth boundary implies. Values are MEANS over validation tiles -- ARI/VI/PQ are per-tile, not additive the way pixel counts are.

| dataset | tiles | ARI | VI | PQ | SQ | RQ | true regions | pred regions | over-seg factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 | 96 | 0.548 | 0.765 | 0.339 | 0.739 | 0.456 | 12.4 | 13.4 | 1.20 |
| Steel2 | 126 | 0.308 | 2.111 | 0.106 | 0.687 | 0.153 | 112.0 | 49.5 | 0.44 |

## MISPLACED / OVER-DETECTION / THICKNESS decomposition

The pixel/skeleton view (:func:`decompose_error`) this region view complements -- same validation pass, same checkpoint.

| dataset | verdict | pixel Dice | skeleton Dice | curve-length ratio | width ratio | tolerance px |
| --- | --- | --- | --- | --- | --- | --- |
| Steel1 | MISPLACED | 0.5857 | 0.1453 | 1.03 | 1.50 | 4 |
| Steel2 | THICKNESS | 0.7762 | 0.2225 | 0.78 | 1.52 | 4 |

Verdicts, in full:

- **Steel1**: MISPLACED -- 82% of the true centreline has something within tolerance of it, but the two skeletons barely overlap (Dice 0.145) and only 77% of what was drawn is on a boundary. That is coincidental coverage from predicting far too much, not detection of the real curves
- **Steel2**: THICKNESS -- the curves are the right ones in the right places (95% of the drawn centreline is on a true boundary) and are 1.5x too fat
