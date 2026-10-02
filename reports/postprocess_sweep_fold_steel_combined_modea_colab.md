# Post-processing sweep -- fold_steel_combined_modea (colab)

Checkpoint: epoch 32, config hash `067fecc2e06a7d0a`. Ground truth line_width_px=4.0; skeleton_redilate target width 4 px; watershed marker threshold 0.3.

Regions for every configuration are the ones its thresholded, post-processed BOUNDARY implies (`binary_boundary`: the same conversion the ground truth goes through). Cell 20's region metrics use a watershed on the RAW probability map, which no threshold or post-processing can move; that partition appears only as the `reference` row. Selection is by PQ on VAL only; TEST is scored once, on configurations fixed before it was looked at.

## VAL -- Steel1 (control -- not expected to help)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference: watershed_prob@tuned | 0.4946 | 0.1106 | 1.4746 | 0.6683 | 0.7301 | MISPLACED | 0.2431 | 0.7137 | 0.3408 | 1.0605 |
| skeleton_redilate@0.70 **<- selected** | 0.4363 | 0.1063 | 0.9920 | 0.6465 | 0.7278 | MISPLACED | 0.2367 | 0.7327 | 0.3252 | 0.8055 |
| skeleton_redilate@0.75 | 0.4449 | 0.1099 | 1.0003 | 0.6631 | 0.7260 | MISPLACED | 0.2327 | 0.7344 | 0.3179 | 0.7403 |
| skeleton_redilate@tuned | 0.4449 | 0.1099 | 1.0003 | 0.6631 | 0.7260 | MISPLACED | 0.2327 | 0.7344 | 0.3179 | 0.7403 |
| none@0.70 | 0.4916 | 0.1075 | 1.6380 | 0.6522 | 0.7303 | MISPLACED | 0.2325 | 0.7321 | 0.3194 | 0.9586 |
| skeleton_redilate@0.80 | 0.4507 | 0.1124 | 1.0100 | 0.6788 | 0.7154 | MISPLACED | 0.2176 | 0.7303 | 0.2986 | 0.6022 |
| none@0.75 | 0.4946 | 0.1106 | 1.4746 | 0.6683 | 0.7301 | MISPLACED | 0.2061 | 0.7440 | 0.2802 | 0.8760 |
| none@tuned | 0.4946 | 0.1106 | 1.4746 | 0.6683 | 0.7301 | MISPLACED | 0.2061 | 0.7440 | 0.2802 | 0.8760 |
| skeleton_redilate@0.85 | 0.4529 | 0.1120 | 1.0251 | 0.6985 | 0.6923 | MISPLACED | 0.1961 | 0.7365 | 0.2632 | 0.4378 |
| none@0.80 | 0.4917 | 0.1139 | 1.3051 | 0.6840 | 0.7180 | MISPLACED | 0.1897 | 0.7319 | 0.2574 | 0.6808 |
| none@0.85 | 0.4766 | 0.1133 | 1.1244 | 0.7049 | 0.6952 | MISPLACED | 0.1755 | 0.7476 | 0.2321 | 0.4840 |
| skeleton_redilate@0.90 | 0.4450 | 0.1096 | 1.0396 | 0.7286 | 0.6475 | MISPLACED | 0.1586 | 0.7520 | 0.2061 | 0.3174 |
| none@0.90 | 0.4332 | 0.1105 | 0.9355 | 0.7367 | 0.6463 | MISPLACED | 0.1534 | 0.7651 | 0.1950 | 0.2127 |

## VAL -- Steel2 (hypothesis)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skeleton_redilate@0.70 **<- selected** | 0.7573 | 0.3971 | 1.0130 | 0.9148 | 0.9353 | OFFSET | 0.3886 | 0.8061 | 0.4805 | 0.9423 |
| skeleton_redilate@0.75 | 0.7585 | 0.3988 | 1.0175 | 0.9245 | 0.9298 | OFFSET | 0.3883 | 0.8101 | 0.4777 | 0.8521 |
| none@0.70 | 0.7233 | 0.3861 | 1.4754 | 0.9208 | 0.9335 | THICKNESS | 0.3728 | 0.7931 | 0.4705 | 0.9317 |
| skeleton_redilate@0.80 | 0.7564 | 0.3976 | 1.0232 | 0.9345 | 0.9216 | OFFSET | 0.3682 | 0.8090 | 0.4539 | 0.7298 |
| skeleton_redilate@tuned | 0.7564 | 0.3976 | 1.0232 | 0.9345 | 0.9216 | OFFSET | 0.3682 | 0.8090 | 0.4539 | 0.7298 |
| none@0.75 | 0.7276 | 0.3855 | 1.4210 | 0.9299 | 0.9272 | THICKNESS | 0.3580 | 0.7924 | 0.4509 | 0.7840 |
| none@0.80 | 0.7280 | 0.3832 | 1.3590 | 0.9397 | 0.9184 | OFFSET | 0.3403 | 0.7925 | 0.4272 | 0.6734 |
| none@tuned | 0.7280 | 0.3832 | 1.3590 | 0.9397 | 0.9184 | OFFSET | 0.3403 | 0.7925 | 0.4272 | 0.6734 |
| skeleton_redilate@0.85 | 0.7490 | 0.3919 | 1.0309 | 0.9442 | 0.9070 | OFFSET | 0.3207 | 0.7986 | 0.3963 | 0.5902 |
| none@0.85 | 0.7197 | 0.3769 | 1.2836 | 0.9487 | 0.9024 | OFFSET | 0.2917 | 0.7885 | 0.3652 | 0.5320 |
| reference: watershed_prob@tuned | 0.7280 | 0.3832 | 1.3590 | 0.9397 | 0.9184 | OFFSET | 0.2785 | 0.7693 | 0.3627 | 1.5941 |
| skeleton_redilate@0.90 | 0.7252 | 0.3703 | 1.0452 | 0.9531 | 0.8770 | OFFSET | 0.2504 | 0.7900 | 0.3063 | 0.4230 |
| none@0.90 | 0.6903 | 0.3551 | 1.1804 | 0.9561 | 0.8696 | OFFSET | 0.2165 | 0.7809 | 0.2677 | 0.3736 |

## Selected on VAL

- **Steel1** (control -- not expected to help): `skeleton_redilate@0.70` (mode skeleton_redilate, threshold 0.70, val PQ 0.2367)
- **Steel2** (hypothesis): `skeleton_redilate@0.70` (mode skeleton_redilate, threshold 0.70, val PQ 0.3886)

## TEST -- Steel1 (control -- not expected to help)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cell 20 (recorded) | 0.5934 | 0.1513 | 1.4717 | 0.7823 | 0.8279 | MISPLACED | 0.3238 | 0.7511 | 0.4299 | 0.9427 |
| selected (val-best) | 0.5416 | 0.1437 | 0.9843 | 0.7631 | 0.8263 | MISPLACED | 0.2985 | 0.7397 | 0.4016 | 0.7089 |
| none@tuned | 0.5934 | 0.1513 | 1.4717 | 0.7823 | 0.8279 | MISPLACED | 0.2563 | 0.7481 | 0.3422 | 0.7286 |
| reference: watershed_prob@tuned | 0.5934 | 0.1513 | 1.4717 | 0.7823 | 0.8279 | MISPLACED | 0.3238 | 0.7511 | 0.4299 | 0.9427 |

## TEST -- Steel2 (hypothesis)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cell 20 (recorded) | 0.7340 | 0.4096 | 1.3340 | 0.9612 | 0.9171 | OFFSET | 0.3106 | 0.7695 | 0.4036 | 1.4095 |
| selected (val-best) | 0.7793 | 0.4251 | 1.0157 | 0.9445 | 0.9360 | OFFSET | 0.3637 | 0.8034 | 0.4522 | 0.7949 |
| none@tuned | 0.7340 | 0.4096 | 1.3340 | 0.9612 | 0.9171 | OFFSET | 0.2902 | 0.7889 | 0.3674 | 0.5548 |
| reference: watershed_prob@tuned | 0.7340 | 0.4096 | 1.3340 | 0.9612 | 0.9171 | OFFSET | 0.3106 | 0.7695 | 0.4036 | 1.4095 |

## Checks

- PASS selection saw VAL rows only -- splits ['val']
- PASS TEST scored exactly the three pre-declared configurations -- selected (val-best), none@tuned, reference: watershed_prob@tuned
- PASS Steel1: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00, label MISPLACED vs MISPLACED
- PASS Steel1: watershed reference reproduces Cell 20's region metrics -- max |diff| 0.00e+00
- PASS Steel2: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00, label OFFSET vs OFFSET
- PASS Steel2: watershed reference reproduces Cell 20's region metrics -- max |diff| 0.00e+00
