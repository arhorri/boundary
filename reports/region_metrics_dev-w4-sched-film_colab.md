# Region-level metrics -- dev-w4-sched-film (colab)

Checkpoint: epoch 1, config hash `ffae94fd1f4b8c3e`.

Marker-controlled watershed on the probability map (watershed_marker_threshold=0.3), scored against the region partition the ground-truth boundary implies. Values are MEANS over validation tiles -- ARI/VI/PQ are per-tile, not additive the way pixel counts are.

| dataset | tiles | ARI | VI | PQ | SQ | RQ | true regions | pred regions | over-seg factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 | 189 | 0.316 | 0.857 | 0.120 | 0.715 | 0.163 | 10.9 | 19.7 | 2.16 |
| uhcs2 | 265 | 0.145 | 1.918 | 0.029 | 0.404 | 0.043 | 20.3 | 35.4 | 3.58 |

## MISPLACED / OVER-DETECTION / THICKNESS decomposition

The pixel/skeleton view (:func:`decompose_error`) this region view complements -- same validation pass, same checkpoint.

| dataset | verdict | pixel Dice | skeleton Dice | curve-length ratio | width ratio |
| --- | --- | --- | --- | --- | --- |
| Steel1 | MISPLACED | 0.4106 | 0.0986 | 1.68 | 0.98 |
| uhcs2 | MISPLACED | 0.2516 | 0.0598 | 3.47 | 1.23 |

Verdicts, in full:

- **Steel1**: MISPLACED -- the true curves were not found, so thickness is not the problem and thinning the prediction would not help
- **uhcs2**: MISPLACED -- the true curves were not found, so thickness is not the problem and thinning the prediction would not help
