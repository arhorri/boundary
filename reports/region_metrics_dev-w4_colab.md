# Region-level metrics -- dev-w4 (colab)

Checkpoint: epoch 22, config hash `067fecc2e06a7d0a`.

Marker-controlled watershed on the probability map (watershed_marker_threshold=0.3), scored against the region partition the ground-truth boundary implies. Values are MEANS over validation tiles -- ARI/VI/PQ are per-tile, not additive the way pixel counts are.

| dataset | tiles | ARI | VI | PQ | SQ | RQ | true regions | pred regions | over-seg factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 | 189 | 0.486 | 0.724 | 0.319 | 0.734 | 0.434 | 10.9 | 16.4 | 1.70 |
| uhcs2 | 265 | 0.201 | 1.958 | 0.041 | 0.639 | 0.059 | 20.3 | 71.3 | 5.71 |

## MISPLACED / OVER-DETECTION / THICKNESS decomposition

The pixel/skeleton view (:func:`decompose_error`) this region view complements -- same validation pass, same checkpoint.

| dataset | verdict | pixel Dice | skeleton Dice | curve-length ratio | width ratio |
| --- | --- | --- | --- | --- | --- |
| Steel1 | MISPLACED | 0.4695 | 0.1260 | 1.04 | 0.89 |
| uhcs2 | MISPLACED | 0.2591 | 0.0626 | 2.58 | 1.18 |

Verdicts, in full:

- **Steel1**: MISPLACED -- the true curves were not found, so thickness is not the problem and thinning the prediction would not help
- **uhcs2**: MISPLACED -- the true curves were not found, so thickness is not the problem and thinning the prediction would not help
