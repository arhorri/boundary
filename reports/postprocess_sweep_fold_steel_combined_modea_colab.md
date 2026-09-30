# Post-processing sweep -- fold_steel_combined_modea (colab)

Checkpoint: epoch 32, config hash `067fecc2e06a7d0a`. Ground truth line_width_px=4.0; skeleton_redilate target width 4 px; watershed marker threshold 0.3.

Regions for every configuration are the ones its thresholded, post-processed BOUNDARY implies (`binary_boundary`: the same conversion the ground truth goes through). Cell 20's region metrics use a watershed on the RAW probability map, which no threshold or post-processing can move; that partition appears only as the `reference` row. Selection is by PQ on VAL only; TEST is scored once, on configurations fixed before it was looked at.

## VAL -- Steel1 (control -- not expected to help)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference: watershed_prob@tuned | 0.4946 | 0.1105 | 1.4744 | 0.6681 | 0.7301 | MISPLACED | 0.2432 | 0.7137 | 0.3408 | 1.0601 |
| skeleton_redilate@0.70 **<- selected** | 0.4362 | 0.1063 | 0.9919 | 0.6463 | 0.7275 | MISPLACED | 0.2366 | 0.7325 | 0.3252 | 0.8069 |
| none@0.70 | 0.4916 | 0.1075 | 1.6380 | 0.6521 | 0.7302 | MISPLACED | 0.2321 | 0.7325 | 0.3185 | 0.9571 |
| skeleton_redilate@0.75 | 0.4449 | 0.1099 | 1.0003 | 0.6630 | 0.7260 | MISPLACED | 0.2313 | 0.7253 | 0.3162 | 0.7362 |
| skeleton_redilate@tuned | 0.4449 | 0.1099 | 1.0003 | 0.6630 | 0.7260 | MISPLACED | 0.2313 | 0.7253 | 0.3162 | 0.7362 |
| skeleton_redilate@0.80 | 0.4506 | 0.1123 | 1.0100 | 0.6787 | 0.7153 | MISPLACED | 0.2169 | 0.7313 | 0.2976 | 0.6034 |
| none@0.75 | 0.4946 | 0.1105 | 1.4744 | 0.6681 | 0.7301 | MISPLACED | 0.2063 | 0.7440 | 0.2804 | 0.8744 |
| none@tuned | 0.4946 | 0.1105 | 1.4744 | 0.6681 | 0.7301 | MISPLACED | 0.2063 | 0.7440 | 0.2804 | 0.8744 |
| skeleton_redilate@0.85 | 0.4528 | 0.1118 | 1.0251 | 0.6985 | 0.6923 | MISPLACED | 0.1962 | 0.7365 | 0.2634 | 0.4365 |
| none@0.80 | 0.4917 | 0.1138 | 1.3050 | 0.6839 | 0.7180 | MISPLACED | 0.1895 | 0.7325 | 0.2570 | 0.6788 |
| none@0.85 | 0.4766 | 0.1132 | 1.1243 | 0.7049 | 0.6952 | MISPLACED | 0.1753 | 0.7476 | 0.2319 | 0.4853 |
| skeleton_redilate@0.90 | 0.4449 | 0.1095 | 1.0397 | 0.7287 | 0.6473 | MISPLACED | 0.1587 | 0.7520 | 0.2062 | 0.3163 |
| none@0.90 | 0.4332 | 0.1106 | 0.9357 | 0.7368 | 0.6461 | MISPLACED | 0.1535 | 0.7651 | 0.1951 | 0.2109 |

## VAL -- Steel2 (hypothesis)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skeleton_redilate@0.70 **<- selected** | 0.7573 | 0.3972 | 1.0131 | 0.9148 | 0.9352 | OFFSET | 0.3890 | 0.8065 | 0.4808 | 0.9420 |
| skeleton_redilate@0.75 | 0.7585 | 0.3989 | 1.0175 | 0.9245 | 0.9298 | OFFSET | 0.3873 | 0.8099 | 0.4767 | 0.8528 |
| none@0.70 | 0.7233 | 0.3862 | 1.4759 | 0.9208 | 0.9334 | THICKNESS | 0.3730 | 0.7930 | 0.4708 | 0.9300 |
| skeleton_redilate@0.80 | 0.7564 | 0.3977 | 1.0230 | 0.9345 | 0.9215 | OFFSET | 0.3674 | 0.8086 | 0.4531 | 0.7296 |
| skeleton_redilate@tuned | 0.7564 | 0.3977 | 1.0230 | 0.9345 | 0.9215 | OFFSET | 0.3674 | 0.8086 | 0.4531 | 0.7296 |
| none@0.75 | 0.7277 | 0.3856 | 1.4210 | 0.9299 | 0.9272 | THICKNESS | 0.3582 | 0.7920 | 0.4514 | 0.7849 |
| none@0.80 | 0.7280 | 0.3833 | 1.3593 | 0.9397 | 0.9183 | OFFSET | 0.3399 | 0.7926 | 0.4265 | 0.6729 |
| none@tuned | 0.7280 | 0.3833 | 1.3593 | 0.9397 | 0.9183 | OFFSET | 0.3399 | 0.7926 | 0.4265 | 0.6729 |
| skeleton_redilate@0.85 | 0.7490 | 0.3921 | 1.0309 | 0.9442 | 0.9070 | OFFSET | 0.3214 | 0.7979 | 0.3976 | 0.5906 |
| none@0.85 | 0.7197 | 0.3770 | 1.2834 | 0.9486 | 0.9025 | OFFSET | 0.2920 | 0.7885 | 0.3654 | 0.5323 |
| reference: watershed_prob@tuned | 0.7280 | 0.3833 | 1.3593 | 0.9397 | 0.9183 | OFFSET | 0.2786 | 0.7688 | 0.3630 | 1.5927 |
| skeleton_redilate@0.90 | 0.7252 | 0.3703 | 1.0453 | 0.9532 | 0.8770 | OFFSET | 0.2494 | 0.7897 | 0.3051 | 0.4215 |
| none@0.90 | 0.6903 | 0.3550 | 1.1803 | 0.9561 | 0.8696 | OFFSET | 0.2154 | 0.7748 | 0.2663 | 0.3736 |

## Selected on VAL

- **Steel1** (control -- not expected to help): `skeleton_redilate@0.70` (mode skeleton_redilate, threshold 0.70, val PQ 0.2366)
- **Steel2** (hypothesis): `skeleton_redilate@0.70` (mode skeleton_redilate, threshold 0.70, val PQ 0.3890)

## TEST -- Steel1 (control -- not expected to help)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cell 20 (recorded) | 0.5934 | 0.1513 | 1.4718 | 0.7823 | 0.8278 | MISPLACED | 0.3238 | 0.7513 | 0.4298 | 0.9412 |
| selected (val-best) | 0.5417 | 0.1437 | 0.9843 | 0.7633 | 0.8264 | MISPLACED | 0.2980 | 0.7401 | 0.4007 | 0.7097 |
| none@tuned | 0.5934 | 0.1513 | 1.4718 | 0.7823 | 0.8278 | MISPLACED | 0.2563 | 0.7481 | 0.3421 | 0.7298 |
| reference: watershed_prob@tuned | 0.5934 | 0.1513 | 1.4718 | 0.7823 | 0.8278 | MISPLACED | 0.3238 | 0.7513 | 0.4298 | 0.9412 |

## TEST -- Steel2 (hypothesis)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cell 20 (recorded) | 0.7340 | 0.4096 | 1.3343 | 0.9612 | 0.9171 | OFFSET | 0.3110 | 0.7692 | 0.4042 | 1.4078 |
| selected (val-best) | 0.7794 | 0.4252 | 1.0157 | 0.9445 | 0.9360 | OFFSET | 0.3643 | 0.8038 | 0.4529 | 0.7937 |
| none@tuned | 0.7340 | 0.4096 | 1.3343 | 0.9612 | 0.9171 | OFFSET | 0.2899 | 0.7890 | 0.3669 | 0.5552 |
| reference: watershed_prob@tuned | 0.7340 | 0.4096 | 1.3343 | 0.9612 | 0.9171 | OFFSET | 0.3110 | 0.7692 | 0.4042 | 1.4078 |

## Checks

- PASS selection saw VAL rows only -- splits ['val']
- PASS TEST scored exactly the three pre-declared configurations -- selected (val-best), none@tuned, reference: watershed_prob@tuned
- PASS Steel1: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00, label MISPLACED vs MISPLACED
- PASS Steel1: watershed reference reproduces Cell 20's region metrics -- max |diff| 0.00e+00
- PASS Steel2: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00, label OFFSET vs OFFSET
- PASS Steel2: watershed reference reproduces Cell 20's region metrics -- max |diff| 0.00e+00
