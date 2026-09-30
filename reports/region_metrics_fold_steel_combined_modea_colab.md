# Region-level metrics -- fold_steel_combined_modea (colab)

Checkpoint: epoch 32, config hash `067fecc2e06a7d0a`. Ground truth scored at line_width_px=4.0.

Marker-controlled watershed on the probability map (watershed_marker_threshold=0.3), scored against the region partition the ground-truth boundary implies. Values are MEANS over validation tiles -- ARI/VI/PQ are per-tile, not additive the way pixel counts are.

| dataset | tiles | ARI | VI | PQ | SQ | RQ | true regions | pred regions | over-seg factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 | 96 | 0.563 | 0.736 | 0.324 | 0.751 | 0.430 | 17.0 | 13.3 | 0.94 |
| Steel2 | 126 | 0.339 | 1.457 | 0.311 | 0.769 | 0.404 | 34.5 | 44.5 | 1.41 |

## MISPLACED / OVER-DETECTION / THICKNESS decomposition

The pixel/skeleton view (:func:`decompose_error`) this region view complements -- same validation pass, same checkpoint.

| dataset | verdict | pixel Dice | skeleton Dice | curve-length ratio | width ratio | tolerance px |
| --- | --- | --- | --- | --- | --- | --- |
| Steel1 | MISPLACED | 0.5934 | 0.1513 | 1.02 | 1.47 | 4 |
| Steel2 | OFFSET | 0.7340 | 0.4096 | 0.82 | 1.33 | 4 |

Verdicts, in full:

- **Steel1**: MISPLACED -- 83% of the true centreline has something within tolerance of it, but the two skeletons barely overlap (Dice 0.151) and only 78% of what was drawn is on a boundary. That is coincidental coverage from predicting far too much, not detection of the real curves
- **Steel2**: OFFSET -- the curves are within 4 px of the truth but barely overlap it exactly (skeleton Dice 0.410). A sub-tolerance registration shift, not a placement failure and not thickness
