# Post-processing sweep -- fold_steel_combined_modea (colab)

Checkpoint: epoch 32, config hash `067fecc2e06a7d0a`. Ground truth line_width_px=4.0; skeleton_redilate target width 4 px; watershed marker threshold 0.3.

Regions for every configuration are the ones its thresholded, post-processed BOUNDARY implies (`binary_boundary`: the same conversion the ground truth goes through). Cell 20's region metrics use a watershed on the RAW probability map, which no threshold or post-processing can move; that partition appears only as the `reference` row. Selection is by PQ on VAL only; TEST is scored once, on configurations fixed before it was looked at.

## VAL -- Steel1 (control -- not expected to help)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference: watershed_prob@tuned | 0.4946 | 0.1105 | 1.4748 | 0.6682 | 0.7301 | MISPLACED | 0.2427 | 0.7138 | 0.3402 | 1.0600 |
| skeleton_redilate@0.70 **<- selected** | 0.4362 | 0.1063 | 0.9919 | 0.6464 | 0.7276 | MISPLACED | 0.2371 | 0.7324 | 0.3260 | 0.8052 |
| none@0.70 | 0.4916 | 0.1075 | 1.6386 | 0.6522 | 0.7301 | MISPLACED | 0.2322 | 0.7324 | 0.3187 | 0.9574 |
| skeleton_redilate@0.75 | 0.4449 | 0.1099 | 1.0002 | 0.6629 | 0.7260 | MISPLACED | 0.2306 | 0.7263 | 0.3150 | 0.7375 |
| skeleton_redilate@tuned | 0.4449 | 0.1099 | 1.0002 | 0.6629 | 0.7260 | MISPLACED | 0.2306 | 0.7263 | 0.3150 | 0.7375 |
| skeleton_redilate@0.80 | 0.4507 | 0.1123 | 1.0101 | 0.6787 | 0.7153 | MISPLACED | 0.2179 | 0.7302 | 0.2991 | 0.6048 |
| none@0.75 | 0.4946 | 0.1105 | 1.4748 | 0.6682 | 0.7301 | MISPLACED | 0.2068 | 0.7440 | 0.2811 | 0.8685 |
| none@tuned | 0.4946 | 0.1105 | 1.4748 | 0.6682 | 0.7301 | MISPLACED | 0.2068 | 0.7440 | 0.2811 | 0.8685 |
| skeleton_redilate@0.85 | 0.4528 | 0.1119 | 1.0251 | 0.6984 | 0.6924 | MISPLACED | 0.1961 | 0.7367 | 0.2632 | 0.4380 |
| none@0.80 | 0.4917 | 0.1138 | 1.3054 | 0.6839 | 0.7180 | MISPLACED | 0.1898 | 0.7319 | 0.2575 | 0.6816 |
| none@0.85 | 0.4767 | 0.1133 | 1.1245 | 0.7048 | 0.6953 | MISPLACED | 0.1754 | 0.7477 | 0.2319 | 0.4850 |
| skeleton_redilate@0.90 | 0.4450 | 0.1095 | 1.0396 | 0.7287 | 0.6475 | MISPLACED | 0.1576 | 0.7550 | 0.2044 | 0.3157 |
| none@0.90 | 0.4332 | 0.1106 | 0.9353 | 0.7369 | 0.6463 | MISPLACED | 0.1534 | 0.7652 | 0.1950 | 0.2124 |

## VAL -- Steel2 (hypothesis)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skeleton_redilate@0.70 **<- selected** | 0.7572 | 0.3972 | 1.0130 | 0.9148 | 0.9352 | OFFSET | 0.3890 | 0.8062 | 0.4810 | 0.9398 |
| skeleton_redilate@0.75 | 0.7585 | 0.3987 | 1.0175 | 0.9246 | 0.9298 | OFFSET | 0.3879 | 0.8101 | 0.4773 | 0.8498 |
| none@0.70 | 0.7233 | 0.3861 | 1.4758 | 0.9208 | 0.9334 | THICKNESS | 0.3728 | 0.7930 | 0.4705 | 0.9311 |
| skeleton_redilate@0.80 | 0.7565 | 0.3977 | 1.0231 | 0.9346 | 0.9216 | OFFSET | 0.3678 | 0.8086 | 0.4536 | 0.7290 |
| skeleton_redilate@tuned | 0.7565 | 0.3977 | 1.0231 | 0.9346 | 0.9216 | OFFSET | 0.3678 | 0.8086 | 0.4536 | 0.7290 |
| none@0.75 | 0.7276 | 0.3854 | 1.4210 | 0.9299 | 0.9271 | THICKNESS | 0.3582 | 0.7922 | 0.4514 | 0.7849 |
| none@0.80 | 0.7280 | 0.3832 | 1.3590 | 0.9398 | 0.9184 | OFFSET | 0.3396 | 0.7925 | 0.4262 | 0.6727 |
| none@tuned | 0.7280 | 0.3832 | 1.3590 | 0.9398 | 0.9184 | OFFSET | 0.3396 | 0.7925 | 0.4262 | 0.6727 |
| skeleton_redilate@0.85 | 0.7490 | 0.3920 | 1.0309 | 0.9442 | 0.9071 | OFFSET | 0.3202 | 0.7979 | 0.3961 | 0.5898 |
| none@0.85 | 0.7197 | 0.3768 | 1.2835 | 0.9486 | 0.9025 | OFFSET | 0.2914 | 0.7885 | 0.3648 | 0.5315 |
| reference: watershed_prob@tuned | 0.7280 | 0.3832 | 1.3590 | 0.9398 | 0.9184 | OFFSET | 0.2788 | 0.7692 | 0.3630 | 1.5932 |
| skeleton_redilate@0.90 | 0.7253 | 0.3703 | 1.0453 | 0.9531 | 0.8771 | OFFSET | 0.2501 | 0.7903 | 0.3060 | 0.4222 |
| none@0.90 | 0.6903 | 0.3550 | 1.1801 | 0.9561 | 0.8698 | OFFSET | 0.2154 | 0.7743 | 0.2666 | 0.3748 |

## Selected on VAL

- **Steel1** (control -- not expected to help): `skeleton_redilate@0.70` (mode skeleton_redilate, threshold 0.70, val PQ 0.2371)
- **Steel2** (hypothesis): `skeleton_redilate@0.70` (mode skeleton_redilate, threshold 0.70, val PQ 0.3890)

## TEST -- Steel1 (control -- not expected to help)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cell 20 (recorded) | 0.5934 | 0.1513 | 1.4715 | 0.7822 | 0.8279 | MISPLACED | 0.3242 | 0.7507 | 0.4306 | 0.9430 |
| selected (val-best) | 0.5417 | 0.1437 | 0.9842 | 0.7633 | 0.8264 | MISPLACED | 0.2981 | 0.7400 | 0.4009 | 0.7088 |
| none@tuned | 0.5934 | 0.1513 | 1.4715 | 0.7822 | 0.8279 | MISPLACED | 0.2561 | 0.7481 | 0.3419 | 0.7304 |
| reference: watershed_prob@tuned | 0.5934 | 0.1513 | 1.4715 | 0.7822 | 0.8279 | MISPLACED | 0.3242 | 0.7507 | 0.4306 | 0.9430 |

## TEST -- Steel2 (hypothesis)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cell 20 (recorded) | 0.7340 | 0.4096 | 1.3341 | 0.9612 | 0.9171 | OFFSET | 0.3113 | 0.7691 | 0.4045 | 1.4076 |
| selected (val-best) | 0.7794 | 0.4253 | 1.0157 | 0.9446 | 0.9361 | OFFSET | 0.3657 | 0.8030 | 0.4549 | 0.7933 |
| none@tuned | 0.7340 | 0.4096 | 1.3341 | 0.9612 | 0.9171 | OFFSET | 0.2902 | 0.7890 | 0.3674 | 0.5552 |
| reference: watershed_prob@tuned | 0.7340 | 0.4096 | 1.3341 | 0.9612 | 0.9171 | OFFSET | 0.3113 | 0.7691 | 0.4045 | 1.4076 |

## Checks

- PASS selection saw VAL rows only -- splits ['val']
- PASS TEST scored exactly the three pre-declared configurations -- selected (val-best), none@tuned, reference: watershed_prob@tuned
- PASS Steel1: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00, label MISPLACED vs MISPLACED
- PASS Steel1: watershed reference reproduces Cell 20's region metrics -- max |diff| 0.00e+00
- PASS Steel2: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00, label OFFSET vs OFFSET
- PASS Steel2: watershed reference reproduces Cell 20's region metrics -- max |diff| 0.00e+00
