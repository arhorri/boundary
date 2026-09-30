# Post-processing sweep -- fold_steel_combined (colab)

Checkpoint: epoch 21, config hash `067fecc2e06a7d0a`. Ground truth line_width_px=4.0; skeleton_redilate target width 4 px; watershed marker threshold 0.3.

Regions for every configuration are the ones its thresholded, post-processed BOUNDARY implies (`binary_boundary`: the same conversion the ground truth goes through). Cell 20's region metrics use a watershed on the RAW probability map, which no threshold or post-processing can move; that partition appears only as the `reference` row. Selection is by PQ on VAL only; TEST is scored once, on configurations fixed before it was looked at.

## VAL -- Steel1 (control -- not expected to help)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference: watershed_prob@tuned | 0.4889 | 0.1099 | 1.4637 | 0.6562 | 0.7207 | MISPLACED | 0.2248 | 0.7199 | 0.3146 | 1.1078 |
| skeleton_redilate@0.70 **<- selected** | 0.4362 | 0.1080 | 0.9964 | 0.6513 | 0.7174 | MISPLACED | 0.2147 | 0.7121 | 0.2973 | 0.7606 |
| skeleton_redilate@tuned | 0.4362 | 0.1080 | 0.9964 | 0.6513 | 0.7174 | MISPLACED | 0.2147 | 0.7121 | 0.2973 | 0.7606 |
| none@0.70 | 0.4889 | 0.1099 | 1.4637 | 0.6562 | 0.7207 | MISPLACED | 0.2088 | 0.7336 | 0.2835 | 0.9023 |
| none@tuned | 0.4889 | 0.1099 | 1.4637 | 0.6562 | 0.7207 | MISPLACED | 0.2088 | 0.7336 | 0.2835 | 0.9023 |
| skeleton_redilate@0.75 | 0.4431 | 0.1087 | 1.0044 | 0.6652 | 0.7117 | MISPLACED | 0.2027 | 0.7383 | 0.2763 | 0.6596 |
| skeleton_redilate@0.80 | 0.4468 | 0.1111 | 1.0181 | 0.6836 | 0.6971 | MISPLACED | 0.1914 | 0.7301 | 0.2593 | 0.5552 |
| none@0.75 | 0.4876 | 0.1099 | 1.3172 | 0.6710 | 0.7155 | MISPLACED | 0.1796 | 0.7390 | 0.2420 | 0.7607 |
| none@0.80 | 0.4783 | 0.1138 | 1.1765 | 0.6895 | 0.7012 | MISPLACED | 0.1719 | 0.7444 | 0.2289 | 0.5802 |
| skeleton_redilate@0.85 | 0.4435 | 0.1081 | 1.0294 | 0.7043 | 0.6670 | MISPLACED | 0.1686 | 0.7402 | 0.2240 | 0.4244 |
| none@0.85 | 0.4551 | 0.1106 | 1.0354 | 0.7120 | 0.6688 | MISPLACED | 0.1633 | 0.7497 | 0.2144 | 0.3777 |
| skeleton_redilate@0.90 | 0.4306 | 0.1050 | 1.0397 | 0.7350 | 0.6140 | MISPLACED | 0.1432 | 0.7606 | 0.1851 | 0.3197 |
| none@0.90 | 0.4035 | 0.1053 | 0.8705 | 0.7438 | 0.6130 | MISPLACED | 0.1395 | 0.7683 | 0.1749 | 0.1623 |

## VAL -- Steel2 (hypothesis)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skeleton_redilate@0.85 **<- selected** | 0.6701 | 0.2343 | 0.9855 | 0.9321 | 0.9085 | OFFSET | 0.2100 | 0.6835 | 0.3053 | 0.5751 |
| skeleton_redilate@tuned | 0.6701 | 0.2343 | 0.9855 | 0.9321 | 0.9085 | OFFSET | 0.2100 | 0.6835 | 0.3053 | 0.5751 |
| none@0.85 | 0.7609 | 0.2328 | 1.5170 | 0.9345 | 0.9096 | THICKNESS | 0.2086 | 0.6812 | 0.3043 | 0.5990 |
| none@tuned | 0.7609 | 0.2328 | 1.5170 | 0.9345 | 0.9096 | THICKNESS | 0.2086 | 0.6812 | 0.3043 | 0.5990 |
| skeleton_redilate@0.90 | 0.6996 | 0.2495 | 0.9970 | 0.9536 | 0.9220 | OFFSET | 0.2074 | 0.6835 | 0.2999 | 0.5124 |
| none@0.90 | 0.7500 | 0.2487 | 1.2588 | 0.9559 | 0.9225 | OFFSET | 0.1946 | 0.6822 | 0.2820 | 0.5228 |
| none@0.80 | 0.7514 | 0.2157 | 1.7203 | 0.9168 | 0.8818 | THICKNESS | 0.1920 | 0.6814 | 0.2802 | 0.5839 |
| skeleton_redilate@0.80 | 0.6375 | 0.2168 | 0.9857 | 0.9141 | 0.8813 | OFFSET | 0.1857 | 0.6797 | 0.2713 | 0.5440 |
| none@0.75 | 0.7392 | 0.2003 | 1.8839 | 0.9001 | 0.8533 | THICKNESS | 0.1818 | 0.6772 | 0.2673 | 0.5838 |
| none@0.70 | 0.7272 | 0.1871 | 2.0149 | 0.8853 | 0.8294 | THICKNESS | 0.1676 | 0.6767 | 0.2468 | 0.5759 |
| skeleton_redilate@0.75 | 0.6095 | 0.2015 | 0.9880 | 0.8976 | 0.8531 | OFFSET | 0.1673 | 0.6755 | 0.2466 | 0.5017 |
| skeleton_redilate@0.70 | 0.5867 | 0.1894 | 0.9893 | 0.8823 | 0.8298 | OFFSET | 0.1491 | 0.6720 | 0.2216 | 0.4815 |
| reference: watershed_prob@tuned | 0.7609 | 0.2328 | 1.5170 | 0.9345 | 0.9096 | THICKNESS | 0.0912 | 0.7025 | 0.1306 | 0.4330 |

## Selected on VAL

- **Steel1** (control -- not expected to help): `skeleton_redilate@0.70` (mode skeleton_redilate, threshold 0.70, val PQ 0.2147)
- **Steel2** (hypothesis): `skeleton_redilate@0.85` (mode skeleton_redilate, threshold 0.85, val PQ 0.2100)

## TEST -- Steel1 (control -- not expected to help)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cell 20 (recorded) | 0.5857 | 0.1453 | 1.4986 | 0.7664 | 0.8202 | MISPLACED | 0.3027 | 0.7394 | 0.4056 | 0.9822 |
| selected (val-best) | 0.5381 | 0.1407 | 0.9890 | 0.7617 | 0.8164 | MISPLACED | 0.2792 | 0.7405 | 0.3748 | 0.6692 |
| none@tuned | 0.5857 | 0.1453 | 1.4986 | 0.7664 | 0.8202 | MISPLACED | 0.2721 | 0.7471 | 0.3634 | 0.8127 |
| reference: watershed_prob@tuned | 0.5857 | 0.1453 | 1.4986 | 0.7664 | 0.8202 | MISPLACED | 0.3027 | 0.7394 | 0.4056 | 0.9822 |

## TEST -- Steel2 (hypothesis)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cell 20 (recorded) | 0.7762 | 0.2226 | 1.5151 | 0.9532 | 0.8977 | THICKNESS | 0.0985 | 0.6883 | 0.1418 | 0.3884 |
| selected (val-best) | 0.6549 | 0.2260 | 0.9881 | 0.9518 | 0.8974 | OFFSET | 0.1724 | 0.6657 | 0.2542 | 0.4556 |
| none@tuned | 0.7762 | 0.2226 | 1.5151 | 0.9532 | 0.8977 | THICKNESS | 0.1691 | 0.6623 | 0.2504 | 0.4700 |
| reference: watershed_prob@tuned | 0.7762 | 0.2226 | 1.5151 | 0.9532 | 0.8977 | THICKNESS | 0.0985 | 0.6883 | 0.1418 | 0.3884 |

## Checks

- PASS selection saw VAL rows only -- splits ['val']
- PASS TEST scored exactly the three pre-declared configurations -- selected (val-best), none@tuned, reference: watershed_prob@tuned
- PASS Steel1: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00, label MISPLACED vs MISPLACED
- PASS Steel1: watershed reference reproduces Cell 20's region metrics -- max |diff| 0.00e+00
- PASS Steel2: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00, label THICKNESS vs THICKNESS
- PASS Steel2: watershed reference reproduces Cell 20's region metrics -- max |diff| 0.00e+00
