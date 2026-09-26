# Region-level metrics -- fold_steel_combined (colab)

Checkpoint: epoch 21, config hash `067fecc2e06a7d0a`.

Marker-controlled watershed on the probability map (watershed_marker_threshold=0.3), scored against the region partition the ground-truth boundary implies. Values are MEANS over validation tiles -- ARI/VI/PQ are per-tile, not additive the way pixel counts are.

| dataset | tiles | ARI | VI | PQ | SQ | RQ | true regions | pred regions | over-seg factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 | 96 | 0.549 | 0.765 | 0.339 | 0.739 | 0.456 | 12.4 | 13.4 | 1.20 |
| Steel2 | 126 | 0.307 | 2.112 | 0.106 | 0.688 | 0.152 | 112.0 | 49.5 | 0.44 |

## MISPLACED / OVER-DETECTION / THICKNESS decomposition

The pixel/skeleton view (:func:`decompose_error`) this region view complements -- same validation pass, same checkpoint.

| dataset | verdict | pixel Dice | skeleton Dice | curve-length ratio | width ratio |
| --- | --- | --- | --- | --- | --- |
| Steel1 | MISPLACED | 0.5857 | 0.1452 | 1.03 | 1.50 |
| Steel2 | MISPLACED | 0.7762 | 0.2225 | 0.78 | 1.52 |

Verdicts, in full:

- **Steel1**: MISPLACED -- the true curves were not found, so thickness is not the problem and thinning the prediction would not help
- **Steel2**: MISPLACED -- the true curves were not found, so thickness is not the problem and thinning the prediction would not help
