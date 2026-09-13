# Region-level metrics -- dev (colab)

Marker-controlled watershed on the probability map (watershed_marker_threshold=0.3), scored against the region partition the ground-truth boundary implies. Values are MEANS over validation tiles -- ARI/VI/PQ are per-tile, not additive the way pixel counts are.

| dataset | tiles | ARI | VI | PQ | SQ | RQ | true regions | pred regions | over-seg factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 | 189 | 0.429 | 0.732 | 0.253 | 0.740 | 0.344 | 9.5 | 17.5 | 2.10 |
| uhcs2 | 265 | 0.221 | 1.969 | 0.037 | 0.624 | 0.054 | 18.7 | 80.2 | 7.23 |

## MISPLACED / OVER-DETECTION / THICKNESS decomposition

The pixel/skeleton view (:func:`decompose_error`) this region view complements -- same validation pass, same checkpoint.

| dataset | verdict | pixel Dice | skeleton Dice | curve-length ratio | width ratio |
| --- | --- | --- | --- | --- | --- |
| Steel1 | MISPLACED | 0.3271 | 0.1214 | 0.98 | 1.96 |
| uhcs2 | MISPLACED | 0.1489 | 0.0574 | 2.71 | 2.53 |

Verdicts, in full:

- **Steel1**: MISPLACED -- the true curves were not found, so thickness is not the problem and thinning the prediction would not help
- **uhcs2**: MISPLACED -- the true curves were not found, so thickness is not the problem and thinning the prediction would not help
