# Region-level metrics -- dev-w4-sched (colab)

Checkpoint: epoch 20, config hash `360884933d0f8082`.

Marker-controlled watershed on the probability map (watershed_marker_threshold=0.3), scored against the region partition the ground-truth boundary implies. Values are MEANS over validation tiles -- ARI/VI/PQ are per-tile, not additive the way pixel counts are.

| dataset | tiles | ARI | VI | PQ | SQ | RQ | true regions | pred regions | over-seg factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 | 189 | 0.558 | 0.646 | 0.324 | 0.752 | 0.432 | 10.9 | 10.5 | 1.09 |
| uhcs2 | 265 | 0.202 | 2.550 | 0.038 | 0.702 | 0.051 | 20.3 | 80.0 | 6.90 |

## MISPLACED / OVER-DETECTION / THICKNESS decomposition

The pixel/skeleton view (:func:`decompose_error`) this region view complements -- same validation pass, same checkpoint.

| dataset | verdict | pixel Dice | skeleton Dice | curve-length ratio | width ratio |
| --- | --- | --- | --- | --- | --- |
| Steel1 | MISPLACED | 0.5189 | 0.1242 | 0.99 | 1.37 |
| uhcs2 | MISPLACED | 0.2705 | 0.0671 | 2.69 | 1.54 |

Verdicts, in full:

- **Steel1**: MISPLACED -- the true curves were not found, so thickness is not the problem and thinning the prediction would not help
- **uhcs2**: MISPLACED -- the true curves were not found, so thickness is not the problem and thinning the prediction would not help
