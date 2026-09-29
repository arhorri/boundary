# Post-processing sweep -- fold_steel_combined (colab)

Checkpoint: epoch 21, config hash `067fecc2e06a7d0a`. Ground truth line_width_px=4.0; skeleton_redilate target width 4 px; watershed marker threshold 0.3.

Regions for every configuration are the ones its thresholded, post-processed BOUNDARY implies (`binary_boundary`: the same conversion the ground truth goes through). Cell 20's region metrics use a watershed on the RAW probability map, which no threshold or post-processing can move; that partition appears only as the `reference` row. Selection is by PQ on VAL only; TEST is scored once, on configurations fixed before it was looked at.

## VAL -- Steel1 (control -- not expected to help)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference: watershed_prob@tuned | 0.4889 | 0.1098 | 1.4638 | 0.6564 | 0.7207 | MISPLACED | 0.2240 | 0.7200 | 0.3135 | 1.1109 |
| skeleton_redilate@0.70 **<- selected** | 0.4362 | 0.1079 | 0.9965 | 0.6513 | 0.7173 | MISPLACED | 0.2146 | 0.7119 | 0.2972 | 0.7609 |
| skeleton_redilate@tuned | 0.4362 | 0.1079 | 0.9965 | 0.6513 | 0.7173 | MISPLACED | 0.2146 | 0.7119 | 0.2972 | 0.7609 |
| none@0.70 | 0.4889 | 0.1098 | 1.4638 | 0.6564 | 0.7207 | MISPLACED | 0.2088 | 0.7333 | 0.2836 | 0.9015 |
| none@tuned | 0.4889 | 0.1098 | 1.4638 | 0.6564 | 0.7207 | MISPLACED | 0.2088 | 0.7333 | 0.2836 | 0.9015 |
| skeleton_redilate@0.75 | 0.4431 | 0.1087 | 1.0042 | 0.6652 | 0.7118 | MISPLACED | 0.2030 | 0.7380 | 0.2767 | 0.6620 |
| skeleton_redilate@0.80 | 0.4468 | 0.1112 | 1.0180 | 0.6835 | 0.6971 | MISPLACED | 0.1913 | 0.7301 | 0.2590 | 0.5567 |
| none@0.75 | 0.4876 | 0.1099 | 1.3173 | 0.6710 | 0.7156 | MISPLACED | 0.1797 | 0.7392 | 0.2421 | 0.7606 |
| none@0.80 | 0.4783 | 0.1138 | 1.1765 | 0.6894 | 0.7012 | MISPLACED | 0.1719 | 0.7445 | 0.2290 | 0.5818 |
| skeleton_redilate@0.85 | 0.4435 | 0.1081 | 1.0294 | 0.7043 | 0.6669 | MISPLACED | 0.1685 | 0.7402 | 0.2239 | 0.4233 |
| none@0.85 | 0.4551 | 0.1105 | 1.0355 | 0.7120 | 0.6688 | MISPLACED | 0.1633 | 0.7497 | 0.2144 | 0.3777 |
| skeleton_redilate@0.90 | 0.4306 | 0.1049 | 1.0397 | 0.7348 | 0.6141 | MISPLACED | 0.1432 | 0.7606 | 0.1850 | 0.3195 |
| none@0.90 | 0.4035 | 0.1052 | 0.8705 | 0.7437 | 0.6132 | MISPLACED | 0.1392 | 0.7684 | 0.1745 | 0.1610 |

## VAL -- Steel2 (hypothesis)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skeleton_redilate@0.85 **<- selected** | 0.6700 | 0.2344 | 0.9856 | 0.9320 | 0.9083 | OFFSET | 0.2097 | 0.6837 | 0.3048 | 0.5748 |
| skeleton_redilate@tuned | 0.6700 | 0.2344 | 0.9856 | 0.9320 | 0.9083 | OFFSET | 0.2097 | 0.6837 | 0.3048 | 0.5748 |
| none@0.85 | 0.7609 | 0.2328 | 1.5171 | 0.9344 | 0.9095 | THICKNESS | 0.2083 | 0.6815 | 0.3038 | 0.5988 |
| none@tuned | 0.7609 | 0.2328 | 1.5171 | 0.9344 | 0.9095 | THICKNESS | 0.2083 | 0.6815 | 0.3038 | 0.5988 |
| skeleton_redilate@0.90 | 0.6996 | 0.2495 | 0.9970 | 0.9536 | 0.9219 | OFFSET | 0.2076 | 0.6832 | 0.3004 | 0.5134 |
| none@0.90 | 0.7500 | 0.2488 | 1.2587 | 0.9559 | 0.9225 | OFFSET | 0.1944 | 0.6818 | 0.2816 | 0.5227 |
| none@0.80 | 0.7514 | 0.2157 | 1.7202 | 0.9167 | 0.8817 | THICKNESS | 0.1916 | 0.6819 | 0.2795 | 0.5838 |
| skeleton_redilate@0.80 | 0.6374 | 0.2168 | 0.9857 | 0.9139 | 0.8812 | OFFSET | 0.1854 | 0.6798 | 0.2709 | 0.5443 |
| none@0.75 | 0.7392 | 0.2003 | 1.8838 | 0.9000 | 0.8533 | THICKNESS | 0.1818 | 0.6772 | 0.2673 | 0.5842 |
| none@0.70 | 0.7272 | 0.1872 | 2.0149 | 0.8854 | 0.8295 | THICKNESS | 0.1678 | 0.6764 | 0.2472 | 0.5756 |
| skeleton_redilate@0.75 | 0.6095 | 0.2016 | 0.9881 | 0.8976 | 0.8531 | OFFSET | 0.1673 | 0.6754 | 0.2465 | 0.5025 |
| skeleton_redilate@0.70 | 0.5868 | 0.1895 | 0.9893 | 0.8823 | 0.8299 | OFFSET | 0.1493 | 0.6719 | 0.2219 | 0.4820 |
| reference: watershed_prob@tuned | 0.7609 | 0.2328 | 1.5171 | 0.9344 | 0.9095 | THICKNESS | 0.0913 | 0.7020 | 0.1306 | 0.4331 |

## Selected on VAL

- **Steel1** (control -- not expected to help): `skeleton_redilate@0.70` (mode skeleton_redilate, threshold 0.70, val PQ 0.2146)
- **Steel2** (hypothesis): `skeleton_redilate@0.85` (mode skeleton_redilate, threshold 0.85, val PQ 0.2097)

## TEST -- Steel1 (control -- not expected to help)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cell 20 (recorded) | 0.5857 | 0.1453 | 1.4985 | 0.7663 | 0.8203 | MISPLACED | 0.3027 | 0.7391 | 0.4056 | 0.9822 |
| selected (val-best) | 0.5381 | 0.1406 | 0.9890 | 0.7617 | 0.8164 | MISPLACED | 0.2796 | 0.7405 | 0.3753 | 0.6672 |
| none@tuned | 0.5857 | 0.1453 | 1.4985 | 0.7663 | 0.8203 | MISPLACED | 0.2721 | 0.7471 | 0.3634 | 0.8132 |
| reference: watershed_prob@tuned | 0.5857 | 0.1453 | 1.4985 | 0.7663 | 0.8203 | MISPLACED | 0.3027 | 0.7391 | 0.4056 | 0.9822 |

## TEST -- Steel2 (hypothesis)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cell 20 (recorded) | 0.7762 | 0.2225 | 1.5154 | 0.9532 | 0.8976 | THICKNESS | 0.0985 | 0.6883 | 0.1419 | 0.3886 |
| selected (val-best) | 0.6548 | 0.2260 | 0.9881 | 0.9517 | 0.8974 | OFFSET | 0.1719 | 0.6659 | 0.2534 | 0.4559 |
| none@tuned | 0.7762 | 0.2225 | 1.5154 | 0.9532 | 0.8976 | THICKNESS | 0.1691 | 0.6622 | 0.2505 | 0.4690 |
| reference: watershed_prob@tuned | 0.7762 | 0.2225 | 1.5154 | 0.9532 | 0.8976 | THICKNESS | 0.0985 | 0.6883 | 0.1419 | 0.3886 |

## Checks

- PASS selection saw VAL rows only -- splits ['val']
- PASS TEST scored exactly the three pre-declared configurations -- selected (val-best), none@tuned, reference: watershed_prob@tuned
- PASS Steel1: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00, label MISPLACED vs MISPLACED
- PASS Steel1: watershed reference reproduces Cell 20's region metrics -- max |diff| 0.00e+00
- PASS Steel2: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00, label THICKNESS vs THICKNESS
- PASS Steel2: watershed reference reproduces Cell 20's region metrics -- max |diff| 0.00e+00
