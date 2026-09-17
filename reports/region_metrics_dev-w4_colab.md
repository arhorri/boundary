# Region-level metrics -- dev-w4 (colab)

Checkpoint: epoch 20, config hash `067fecc2e06a7d0a`.

Marker-controlled watershed on the probability map (watershed_marker_threshold=0.3), scored against the region partition the ground-truth boundary implies. Values are MEANS over validation tiles -- ARI/VI/PQ are per-tile, not additive the way pixel counts are.

| dataset | tiles | ARI | VI | PQ | SQ | RQ | true regions | pred regions | over-seg factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 | 189 | 0.480 | 0.789 | 0.328 | 0.738 | 0.442 | 10.9 | 12.1 | 1.27 |
| uhcs2 | 265 | 0.175 | 2.469 | 0.035 | 0.622 | 0.049 | 20.3 | 77.5 | 6.56 |

## MISPLACED / OVER-DETECTION / THICKNESS decomposition

The pixel/skeleton view (:func:`decompose_error`) this region view complements -- same validation pass, same checkpoint.

| dataset | verdict | pixel Dice | skeleton Dice | curve-length ratio | width ratio |
| --- | --- | --- | --- | --- | --- |
| Steel1 | MISPLACED | 0.5143 | 0.1272 | 1.10 | 1.33 |
| uhcs2 | MISPLACED | 0.2618 | 0.0628 | 2.70 | 1.56 |

Verdicts, in full:

- **Steel1**: MISPLACED -- the true curves were not found, so thickness is not the problem and thinning the prediction would not help
- **uhcs2**: MISPLACED -- the true curves were not found, so thickness is not the problem and thinning the prediction would not help
