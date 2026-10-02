# Post-processing sweep -- fold_steel_combined_modea (colab)

Checkpoint: epoch 32, config hash `067fecc2e06a7d0a`. Ground truth line_width_px=4.0; skeleton_redilate target width 4 px; watershed marker threshold 0.3.

Regions for every configuration are the ones its thresholded, post-processed BOUNDARY implies (`binary_boundary`: the same conversion the ground truth goes through). Cell 20's region metrics use a watershed on the RAW probability map, which no threshold or post-processing can move; that partition appears only as the `reference` row. Selection is by PQ on VAL only; TEST is scored once, on configurations fixed before it was looked at.

## VAL -- Steel1 (control -- not expected to help)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference: watershed_prob@tuned | 0.4946 | 0.1106 | 1.4748 | 0.6683 | 0.7300 | MISPLACED | 0.2433 | 0.7133 | 0.3410 | 1.0620 |
| skeleton_redilate@0.70 **<- selected** | 0.4363 | 0.1063 | 0.9920 | 0.6464 | 0.7275 | MISPLACED | 0.2369 | 0.7325 | 0.3256 | 0.8046 |
| none@0.70 | 0.4916 | 0.1076 | 1.6383 | 0.6523 | 0.7301 | MISPLACED | 0.2320 | 0.7323 | 0.3187 | 0.9604 |
| skeleton_redilate@0.75 | 0.4450 | 0.1099 | 1.0003 | 0.6633 | 0.7260 | MISPLACED | 0.2317 | 0.7251 | 0.3170 | 0.7357 |
| skeleton_redilate@tuned | 0.4450 | 0.1099 | 1.0003 | 0.6633 | 0.7260 | MISPLACED | 0.2317 | 0.7251 | 0.3170 | 0.7357 |
| skeleton_redilate@0.80 | 0.4507 | 0.1123 | 1.0101 | 0.6788 | 0.7153 | MISPLACED | 0.2165 | 0.7312 | 0.2970 | 0.6032 |
| none@0.75 | 0.4946 | 0.1106 | 1.4748 | 0.6683 | 0.7300 | MISPLACED | 0.2063 | 0.7440 | 0.2804 | 0.8753 |
| none@tuned | 0.4946 | 0.1106 | 1.4748 | 0.6683 | 0.7300 | MISPLACED | 0.2063 | 0.7440 | 0.2804 | 0.8753 |
| skeleton_redilate@0.85 | 0.4529 | 0.1120 | 1.0250 | 0.6984 | 0.6924 | MISPLACED | 0.1961 | 0.7365 | 0.2632 | 0.4379 |
| none@0.80 | 0.4917 | 0.1138 | 1.3055 | 0.6840 | 0.7180 | MISPLACED | 0.1892 | 0.7324 | 0.2564 | 0.6773 |
| none@0.85 | 0.4766 | 0.1133 | 1.1244 | 0.7049 | 0.6953 | MISPLACED | 0.1758 | 0.7461 | 0.2326 | 0.4845 |
| skeleton_redilate@0.90 | 0.4450 | 0.1097 | 1.0396 | 0.7286 | 0.6475 | MISPLACED | 0.1575 | 0.7550 | 0.2043 | 0.3167 |
| none@0.90 | 0.4332 | 0.1106 | 0.9356 | 0.7369 | 0.6463 | MISPLACED | 0.1535 | 0.7652 | 0.1951 | 0.2118 |

## VAL -- Steel2 (hypothesis)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skeleton_redilate@0.70 **<- selected** | 0.7572 | 0.3971 | 1.0130 | 0.9148 | 0.9353 | OFFSET | 0.3887 | 0.8060 | 0.4807 | 0.9416 |
| skeleton_redilate@0.75 | 0.7585 | 0.3988 | 1.0175 | 0.9245 | 0.9299 | OFFSET | 0.3878 | 0.8101 | 0.4773 | 0.8518 |
| none@0.70 | 0.7233 | 0.3860 | 1.4755 | 0.9208 | 0.9335 | THICKNESS | 0.3726 | 0.7932 | 0.4701 | 0.9313 |
| skeleton_redilate@0.80 | 0.7564 | 0.3976 | 1.0231 | 0.9345 | 0.9216 | OFFSET | 0.3676 | 0.8091 | 0.4531 | 0.7290 |
| skeleton_redilate@tuned | 0.7564 | 0.3976 | 1.0231 | 0.9345 | 0.9216 | OFFSET | 0.3676 | 0.8091 | 0.4531 | 0.7290 |
| none@0.75 | 0.7276 | 0.3855 | 1.4210 | 0.9298 | 0.9272 | THICKNESS | 0.3584 | 0.7924 | 0.4515 | 0.7835 |
| none@0.80 | 0.7280 | 0.3833 | 1.3591 | 0.9397 | 0.9184 | OFFSET | 0.3404 | 0.7926 | 0.4272 | 0.6728 |
| none@tuned | 0.7280 | 0.3833 | 1.3591 | 0.9397 | 0.9184 | OFFSET | 0.3404 | 0.7926 | 0.4272 | 0.6728 |
| skeleton_redilate@0.85 | 0.7490 | 0.3921 | 1.0309 | 0.9442 | 0.9070 | OFFSET | 0.3208 | 0.7986 | 0.3965 | 0.5901 |
| none@0.85 | 0.7197 | 0.3770 | 1.2836 | 0.9486 | 0.9024 | OFFSET | 0.2916 | 0.7881 | 0.3652 | 0.5323 |
| reference: watershed_prob@tuned | 0.7280 | 0.3833 | 1.3591 | 0.9397 | 0.9184 | OFFSET | 0.2790 | 0.7693 | 0.3633 | 1.5931 |
| skeleton_redilate@0.90 | 0.7252 | 0.3705 | 1.0451 | 0.9531 | 0.8770 | OFFSET | 0.2501 | 0.7894 | 0.3061 | 0.4235 |
| none@0.90 | 0.6903 | 0.3550 | 1.1802 | 0.9561 | 0.8696 | OFFSET | 0.2157 | 0.7807 | 0.2668 | 0.3740 |

## Selected on VAL

- **Steel1** (control -- not expected to help): `skeleton_redilate@0.70` (mode skeleton_redilate, threshold 0.70, val PQ 0.2369)
- **Steel2** (hypothesis): `skeleton_redilate@0.70` (mode skeleton_redilate, threshold 0.70, val PQ 0.3887)

## TEST -- Steel1 (control -- not expected to help)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cell 20 (recorded) | 0.5934 | 0.1513 | 1.4722 | 0.7825 | 0.8279 | MISPLACED | 0.3235 | 0.7512 | 0.4294 | 0.9424 |
| selected (val-best) | 0.5416 | 0.1437 | 0.9843 | 0.7631 | 0.8264 | MISPLACED | 0.2984 | 0.7398 | 0.4013 | 0.7042 |
| none@tuned | 0.5934 | 0.1513 | 1.4722 | 0.7825 | 0.8279 | MISPLACED | 0.2562 | 0.7481 | 0.3421 | 0.7297 |
| reference: watershed_prob@tuned | 0.5934 | 0.1513 | 1.4722 | 0.7825 | 0.8279 | MISPLACED | 0.3235 | 0.7512 | 0.4294 | 0.9424 |

## TEST -- Steel2 (hypothesis)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cell 20 (recorded) | 0.7340 | 0.4096 | 1.3341 | 0.9612 | 0.9170 | OFFSET | 0.3109 | 0.7690 | 0.4041 | 1.4087 |
| selected (val-best) | 0.7794 | 0.4252 | 1.0157 | 0.9445 | 0.9360 | OFFSET | 0.3636 | 0.8036 | 0.4521 | 0.7942 |
| none@tuned | 0.7340 | 0.4096 | 1.3341 | 0.9612 | 0.9170 | OFFSET | 0.2903 | 0.7890 | 0.3675 | 0.5544 |
| reference: watershed_prob@tuned | 0.7340 | 0.4096 | 1.3341 | 0.9612 | 0.9170 | OFFSET | 0.3109 | 0.7690 | 0.4041 | 1.4087 |

## Checks

- PASS selection saw VAL rows only -- splits ['val']
- PASS TEST scored exactly the three pre-declared configurations -- selected (val-best), none@tuned, reference: watershed_prob@tuned
- PASS Steel1: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00, label MISPLACED vs MISPLACED
- PASS Steel1: watershed reference reproduces Cell 20's region metrics -- max |diff| 0.00e+00
- PASS Steel2: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00, label OFFSET vs OFFSET
- PASS Steel2: watershed reference reproduces Cell 20's region metrics -- max |diff| 0.00e+00
