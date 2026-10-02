# Post-processing sweep -- fold_steel_combined_modea (colab)

Checkpoint: epoch 32, config hash `067fecc2e06a7d0a`. Ground truth line_width_px=4.0; skeleton_redilate target width 4 px; watershed marker threshold 0.3.

Regions for every configuration are the ones its thresholded, post-processed BOUNDARY implies (`binary_boundary`: the same conversion the ground truth goes through). Cell 20's region metrics use a watershed on the RAW probability map, which no threshold or post-processing can move; that partition appears only as the `reference` row. Selection is by PQ on VAL only; TEST is scored once, on configurations fixed before it was looked at.

## VAL -- Steel1 (control -- not expected to help)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference: watershed_prob@tuned | 0.4946 | 0.1105 | 1.4745 | 0.6682 | 0.7300 | MISPLACED | 0.2429 | 0.7138 | 0.3404 | 1.0635 |
| skeleton_redilate@0.70 **<- selected** | 0.4362 | 0.1061 | 0.9921 | 0.6463 | 0.7276 | MISPLACED | 0.2382 | 0.7323 | 0.3274 | 0.8040 |
| none@0.70 | 0.4916 | 0.1074 | 1.6378 | 0.6521 | 0.7302 | MISPLACED | 0.2316 | 0.7325 | 0.3178 | 0.9600 |
| skeleton_redilate@0.75 | 0.4449 | 0.1099 | 1.0002 | 0.6630 | 0.7259 | MISPLACED | 0.2315 | 0.7253 | 0.3166 | 0.7378 |
| skeleton_redilate@tuned | 0.4449 | 0.1099 | 1.0002 | 0.6630 | 0.7259 | MISPLACED | 0.2315 | 0.7253 | 0.3166 | 0.7378 |
| skeleton_redilate@0.80 | 0.4508 | 0.1124 | 1.0101 | 0.6789 | 0.7155 | MISPLACED | 0.2169 | 0.7303 | 0.2977 | 0.6041 |
| none@0.75 | 0.4946 | 0.1105 | 1.4745 | 0.6682 | 0.7300 | MISPLACED | 0.2066 | 0.7441 | 0.2807 | 0.8737 |
| none@tuned | 0.4946 | 0.1105 | 1.4745 | 0.6682 | 0.7300 | MISPLACED | 0.2066 | 0.7441 | 0.2807 | 0.8737 |
| skeleton_redilate@0.85 | 0.4529 | 0.1119 | 1.0251 | 0.6985 | 0.6923 | MISPLACED | 0.1960 | 0.7365 | 0.2631 | 0.4392 |
| none@0.80 | 0.4917 | 0.1139 | 1.3052 | 0.6840 | 0.7182 | MISPLACED | 0.1892 | 0.7319 | 0.2567 | 0.6826 |
| none@0.85 | 0.4766 | 0.1132 | 1.1247 | 0.7050 | 0.6952 | MISPLACED | 0.1755 | 0.7476 | 0.2321 | 0.4840 |
| skeleton_redilate@0.90 | 0.4450 | 0.1095 | 1.0397 | 0.7286 | 0.6474 | MISPLACED | 0.1579 | 0.7532 | 0.2050 | 0.3164 |
| none@0.90 | 0.4332 | 0.1106 | 0.9357 | 0.7368 | 0.6462 | MISPLACED | 0.1534 | 0.7652 | 0.1951 | 0.2119 |

## VAL -- Steel2 (hypothesis)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skeleton_redilate@0.70 **<- selected** | 0.7573 | 0.3972 | 1.0130 | 0.9149 | 0.9353 | OFFSET | 0.3892 | 0.8062 | 0.4812 | 0.9413 |
| skeleton_redilate@0.75 | 0.7585 | 0.3987 | 1.0176 | 0.9245 | 0.9298 | OFFSET | 0.3882 | 0.8101 | 0.4777 | 0.8525 |
| none@0.70 | 0.7233 | 0.3861 | 1.4757 | 0.9208 | 0.9335 | THICKNESS | 0.3730 | 0.7931 | 0.4707 | 0.9320 |
| skeleton_redilate@0.80 | 0.7565 | 0.3977 | 1.0231 | 0.9345 | 0.9217 | OFFSET | 0.3687 | 0.8088 | 0.4547 | 0.7287 |
| skeleton_redilate@tuned | 0.7565 | 0.3977 | 1.0231 | 0.9345 | 0.9217 | OFFSET | 0.3687 | 0.8088 | 0.4547 | 0.7287 |
| none@0.75 | 0.7276 | 0.3855 | 1.4210 | 0.9298 | 0.9271 | THICKNESS | 0.3581 | 0.7923 | 0.4511 | 0.7838 |
| none@0.80 | 0.7280 | 0.3833 | 1.3590 | 0.9398 | 0.9185 | OFFSET | 0.3402 | 0.7926 | 0.4270 | 0.6729 |
| none@tuned | 0.7280 | 0.3833 | 1.3590 | 0.9398 | 0.9185 | OFFSET | 0.3402 | 0.7926 | 0.4270 | 0.6729 |
| skeleton_redilate@0.85 | 0.7490 | 0.3921 | 1.0308 | 0.9442 | 0.9071 | OFFSET | 0.3210 | 0.7981 | 0.3970 | 0.5912 |
| none@0.85 | 0.7197 | 0.3769 | 1.2834 | 0.9486 | 0.9025 | OFFSET | 0.2921 | 0.7883 | 0.3658 | 0.5320 |
| reference: watershed_prob@tuned | 0.7280 | 0.3833 | 1.3590 | 0.9398 | 0.9185 | OFFSET | 0.2786 | 0.7687 | 0.3629 | 1.5924 |
| skeleton_redilate@0.90 | 0.7252 | 0.3704 | 1.0452 | 0.9531 | 0.8771 | OFFSET | 0.2495 | 0.7899 | 0.3052 | 0.4226 |
| none@0.90 | 0.6903 | 0.3550 | 1.1800 | 0.9561 | 0.8697 | OFFSET | 0.2154 | 0.7749 | 0.2664 | 0.3733 |

## Selected on VAL

- **Steel1** (control -- not expected to help): `skeleton_redilate@0.70` (mode skeleton_redilate, threshold 0.70, val PQ 0.2382)
- **Steel2** (hypothesis): `skeleton_redilate@0.70` (mode skeleton_redilate, threshold 0.70, val PQ 0.3892)

## TEST -- Steel1 (control -- not expected to help)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cell 20 (recorded) | 0.5934 | 0.1512 | 1.4720 | 0.7823 | 0.8277 | MISPLACED | 0.3237 | 0.7513 | 0.4296 | 0.9436 |
| selected (val-best) | 0.5417 | 0.1437 | 0.9843 | 0.7631 | 0.8262 | MISPLACED | 0.2982 | 0.7397 | 0.4013 | 0.7082 |
| none@tuned | 0.5934 | 0.1512 | 1.4720 | 0.7823 | 0.8277 | MISPLACED | 0.2558 | 0.7480 | 0.3416 | 0.7335 |
| reference: watershed_prob@tuned | 0.5934 | 0.1512 | 1.4720 | 0.7823 | 0.8277 | MISPLACED | 0.3237 | 0.7513 | 0.4296 | 0.9436 |

## TEST -- Steel2 (hypothesis)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cell 20 (recorded) | 0.7340 | 0.4095 | 1.3342 | 0.9612 | 0.9171 | OFFSET | 0.3107 | 0.7696 | 0.4037 | 1.4080 |
| selected (val-best) | 0.7793 | 0.4253 | 1.0157 | 0.9445 | 0.9360 | OFFSET | 0.3640 | 0.8032 | 0.4526 | 0.7955 |
| none@tuned | 0.7340 | 0.4095 | 1.3342 | 0.9612 | 0.9171 | OFFSET | 0.2902 | 0.7891 | 0.3673 | 0.5545 |
| reference: watershed_prob@tuned | 0.7340 | 0.4095 | 1.3342 | 0.9612 | 0.9171 | OFFSET | 0.3107 | 0.7696 | 0.4037 | 1.4080 |

## Checks

- PASS selection saw VAL rows only -- splits ['val']
- PASS TEST scored exactly the three pre-declared configurations -- selected (val-best), none@tuned, reference: watershed_prob@tuned
- PASS Steel1: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00, label MISPLACED vs MISPLACED
- PASS Steel1: watershed reference reproduces Cell 20's region metrics -- max |diff| 0.00e+00
- PASS Steel2: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00, label OFFSET vs OFFSET
- PASS Steel2: watershed reference reproduces Cell 20's region metrics -- max |diff| 0.00e+00
