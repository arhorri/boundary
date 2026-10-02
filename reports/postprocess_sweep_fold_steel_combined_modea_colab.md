# Post-processing sweep -- fold_steel_combined_modea (colab)

Checkpoint: epoch 32, config hash `067fecc2e06a7d0a`. Ground truth line_width_px=4.0; skeleton_redilate target width 4 px; watershed marker threshold 0.3.

Regions for every configuration are the ones its thresholded, post-processed BOUNDARY implies (`binary_boundary`: the same conversion the ground truth goes through). Cell 20's region metrics use a watershed on the RAW probability map, which no threshold or post-processing can move; that partition appears only as the `reference` row. Selection is by PQ on VAL only; TEST is scored once, on configurations fixed before it was looked at.

## VAL -- Steel1 (control -- not expected to help)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference: watershed_prob@tuned | 0.4946 | 0.1105 | 1.4749 | 0.6682 | 0.7300 | MISPLACED | 0.2432 | 0.7134 | 0.3410 | 1.0628 |
| skeleton_redilate@0.70 **<- selected** | 0.4363 | 0.1063 | 0.9918 | 0.6465 | 0.7277 | MISPLACED | 0.2366 | 0.7329 | 0.3250 | 0.8045 |
| skeleton_redilate@0.75 | 0.4449 | 0.1098 | 1.0005 | 0.6631 | 0.7259 | MISPLACED | 0.2329 | 0.7345 | 0.3182 | 0.7372 |
| skeleton_redilate@tuned | 0.4449 | 0.1098 | 1.0005 | 0.6631 | 0.7259 | MISPLACED | 0.2329 | 0.7345 | 0.3182 | 0.7372 |
| none@0.70 | 0.4916 | 0.1076 | 1.6380 | 0.6523 | 0.7302 | MISPLACED | 0.2318 | 0.7325 | 0.3182 | 0.9605 |
| skeleton_redilate@0.80 | 0.4507 | 0.1124 | 1.0100 | 0.6788 | 0.7154 | MISPLACED | 0.2174 | 0.7301 | 0.2985 | 0.6025 |
| none@0.75 | 0.4946 | 0.1105 | 1.4749 | 0.6682 | 0.7300 | MISPLACED | 0.2066 | 0.7440 | 0.2808 | 0.8764 |
| none@tuned | 0.4946 | 0.1105 | 1.4749 | 0.6682 | 0.7300 | MISPLACED | 0.2066 | 0.7440 | 0.2808 | 0.8764 |
| skeleton_redilate@0.85 | 0.4529 | 0.1120 | 1.0250 | 0.6984 | 0.6923 | MISPLACED | 0.1960 | 0.7364 | 0.2631 | 0.4394 |
| none@0.80 | 0.4917 | 0.1138 | 1.3053 | 0.6839 | 0.7181 | MISPLACED | 0.1899 | 0.7320 | 0.2577 | 0.6789 |
| none@0.85 | 0.4766 | 0.1133 | 1.1245 | 0.7050 | 0.6952 | MISPLACED | 0.1755 | 0.7476 | 0.2321 | 0.4836 |
| skeleton_redilate@0.90 | 0.4449 | 0.1096 | 1.0399 | 0.7285 | 0.6475 | MISPLACED | 0.1587 | 0.7520 | 0.2062 | 0.3163 |
| none@0.90 | 0.4332 | 0.1106 | 0.9355 | 0.7367 | 0.6464 | MISPLACED | 0.1534 | 0.7652 | 0.1950 | 0.2127 |

## VAL -- Steel2 (hypothesis)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skeleton_redilate@0.70 **<- selected** | 0.7572 | 0.3971 | 1.0131 | 0.9148 | 0.9352 | OFFSET | 0.3887 | 0.8061 | 0.4807 | 0.9415 |
| skeleton_redilate@0.75 | 0.7584 | 0.3988 | 1.0175 | 0.9245 | 0.9298 | OFFSET | 0.3878 | 0.8095 | 0.4775 | 0.8506 |
| none@0.70 | 0.7233 | 0.3860 | 1.4756 | 0.9208 | 0.9334 | THICKNESS | 0.3727 | 0.7930 | 0.4703 | 0.9322 |
| skeleton_redilate@0.80 | 0.7564 | 0.3976 | 1.0231 | 0.9345 | 0.9216 | OFFSET | 0.3674 | 0.8090 | 0.4529 | 0.7286 |
| skeleton_redilate@tuned | 0.7564 | 0.3976 | 1.0231 | 0.9345 | 0.9216 | OFFSET | 0.3674 | 0.8090 | 0.4529 | 0.7286 |
| none@0.75 | 0.7276 | 0.3855 | 1.4213 | 0.9298 | 0.9271 | THICKNESS | 0.3581 | 0.7920 | 0.4514 | 0.7849 |
| none@0.80 | 0.7280 | 0.3833 | 1.3591 | 0.9397 | 0.9184 | OFFSET | 0.3399 | 0.7926 | 0.4264 | 0.6733 |
| none@tuned | 0.7280 | 0.3833 | 1.3591 | 0.9397 | 0.9184 | OFFSET | 0.3399 | 0.7926 | 0.4264 | 0.6733 |
| skeleton_redilate@0.85 | 0.7490 | 0.3920 | 1.0309 | 0.9443 | 0.9071 | OFFSET | 0.3215 | 0.7984 | 0.3974 | 0.5911 |
| none@0.85 | 0.7197 | 0.3769 | 1.2834 | 0.9486 | 0.9025 | OFFSET | 0.2921 | 0.7883 | 0.3658 | 0.5325 |
| reference: watershed_prob@tuned | 0.7280 | 0.3833 | 1.3591 | 0.9397 | 0.9184 | OFFSET | 0.2784 | 0.7688 | 0.3626 | 1.5922 |
| skeleton_redilate@0.90 | 0.7252 | 0.3705 | 1.0453 | 0.9532 | 0.8770 | OFFSET | 0.2496 | 0.7897 | 0.3054 | 0.4231 |
| none@0.90 | 0.6903 | 0.3550 | 1.1803 | 0.9561 | 0.8697 | OFFSET | 0.2154 | 0.7747 | 0.2664 | 0.3733 |

## Selected on VAL

- **Steel1** (control -- not expected to help): `skeleton_redilate@0.70` (mode skeleton_redilate, threshold 0.70, val PQ 0.2366)
- **Steel2** (hypothesis): `skeleton_redilate@0.70` (mode skeleton_redilate, threshold 0.70, val PQ 0.3887)

## TEST -- Steel1 (control -- not expected to help)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cell 20 (recorded) | 0.5934 | 0.1513 | 1.4719 | 0.7823 | 0.8278 | MISPLACED | 0.3239 | 0.7513 | 0.4300 | 0.9417 |
| selected (val-best) | 0.5417 | 0.1437 | 0.9842 | 0.7632 | 0.8264 | MISPLACED | 0.2979 | 0.7400 | 0.4007 | 0.7097 |
| none@tuned | 0.5934 | 0.1513 | 1.4719 | 0.7823 | 0.8278 | MISPLACED | 0.2562 | 0.7479 | 0.3421 | 0.7304 |
| reference: watershed_prob@tuned | 0.5934 | 0.1513 | 1.4719 | 0.7823 | 0.8278 | MISPLACED | 0.3239 | 0.7513 | 0.4300 | 0.9417 |

## TEST -- Steel2 (hypothesis)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cell 20 (recorded) | 0.7340 | 0.4096 | 1.3342 | 0.9612 | 0.9171 | OFFSET | 0.3109 | 0.7692 | 0.4040 | 1.4082 |
| selected (val-best) | 0.7793 | 0.4252 | 1.0157 | 0.9445 | 0.9360 | OFFSET | 0.3649 | 0.8031 | 0.4537 | 0.7924 |
| none@tuned | 0.7340 | 0.4096 | 1.3342 | 0.9612 | 0.9171 | OFFSET | 0.2901 | 0.7891 | 0.3671 | 0.5544 |
| reference: watershed_prob@tuned | 0.7340 | 0.4096 | 1.3342 | 0.9612 | 0.9171 | OFFSET | 0.3109 | 0.7692 | 0.4040 | 1.4082 |

## Checks

- PASS selection saw VAL rows only -- splits ['val']
- PASS TEST scored exactly the three pre-declared configurations -- selected (val-best), none@tuned, reference: watershed_prob@tuned
- PASS Steel1: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00, label MISPLACED vs MISPLACED
- PASS Steel1: watershed reference reproduces Cell 20's region metrics -- max |diff| 0.00e+00
- PASS Steel2: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00, label OFFSET vs OFFSET
- PASS Steel2: watershed reference reproduces Cell 20's region metrics -- max |diff| 0.00e+00
