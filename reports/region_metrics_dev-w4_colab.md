# Region-level metrics -- dev-w4 (colab)

Checkpoint: epoch 22, config hash `067fecc2e06a7d0a`.

Marker-controlled watershed on the probability map (watershed_marker_threshold=0.3), scored against the region partition the ground-truth boundary implies. Values are MEANS over validation tiles -- ARI/VI/PQ are per-tile, not additive the way pixel counts are.

| dataset | tiles | ARI | VI | PQ | SQ | RQ | true regions | pred regions | over-seg factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 | 189 | 0.453 | 0.733 | 0.311 | 0.735 | 0.423 | 9.5 | 16.5 | 1.94 |
| uhcs2 | 265 | 0.206 | 1.943 | 0.041 | 0.645 | 0.059 | 18.7 | 71.3 | 6.15 |

## MISPLACED / OVER-DETECTION / THICKNESS decomposition

The pixel/skeleton view (:func:`decompose_error`) this region view complements -- same validation pass, same checkpoint.

| dataset | verdict | pixel Dice | skeleton Dice | curve-length ratio | width ratio |
| --- | --- | --- | --- | --- | --- |
| Steel1 | MISPLACED | 0.3350 | 0.1267 | 1.04 | 1.91 |
| uhcs2 | MISPLACED | 0.1474 | 0.0612 | 2.58 | 2.51 |

Verdicts, in full:

- **Steel1**: MISPLACED -- the true curves were not found, so thickness is not the problem and thinning the prediction would not help
- **uhcs2**: MISPLACED -- the true curves were not found, so thickness is not the problem and thinning the prediction would not help
