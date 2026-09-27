# Post-processing sweep -- fold_steel_combined (colab)

Checkpoint: epoch 21, config hash `067fecc2e06a7d0a`. Ground truth line_width_px=4.0; skeleton_redilate target width 4 px; watershed marker threshold 0.3.

Regions for every configuration are the ones its thresholded, post-processed BOUNDARY implies (`binary_boundary`: the same conversion the ground truth goes through). Cell 20's region metrics use a watershed on the RAW probability map, which no threshold or post-processing can move; that partition appears only as the `reference` row. Selection is by PQ on VAL only; TEST is scored once, on configurations fixed before it was looked at.

## VAL -- Steel1 (control -- not expected to help)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference: watershed_prob@tuned | 0.4889 | 0.1099 | 1.4641 | 0.6563 | 0.7207 | MISPLACED | 0.2485 | 0.7192 | 0.3483 | 1.3345 |
| none@0.70 **<- selected** | 0.4889 | 0.1099 | 1.4641 | 0.6563 | 0.7207 | MISPLACED | 0.2452 | 0.7326 | 0.3331 | 0.9821 |
| none@tuned | 0.4889 | 0.1099 | 1.4641 | 0.6563 | 0.7207 | MISPLACED | 0.2452 | 0.7326 | 0.3331 | 0.9821 |
| skeleton_redilate@0.70 | 0.4363 | 0.1080 | 0.9964 | 0.6512 | 0.7175 | MISPLACED | 0.2440 | 0.7150 | 0.3366 | 0.8722 |
| skeleton_redilate@tuned | 0.4363 | 0.1080 | 0.9964 | 0.6512 | 0.7175 | MISPLACED | 0.2440 | 0.7150 | 0.3366 | 0.8722 |
| skeleton_redilate@0.75 | 0.4431 | 0.1085 | 1.0043 | 0.6652 | 0.7118 | MISPLACED | 0.2333 | 0.7410 | 0.3171 | 0.7530 |
| skeleton_redilate@0.80 | 0.4468 | 0.1113 | 1.0178 | 0.6836 | 0.6971 | MISPLACED | 0.2235 | 0.7296 | 0.3035 | 0.6055 |
| none@0.75 | 0.4876 | 0.1099 | 1.3171 | 0.6710 | 0.7155 | MISPLACED | 0.2158 | 0.7391 | 0.2914 | 0.8024 |
| skeleton_redilate@0.85 | 0.4436 | 0.1081 | 1.0294 | 0.7045 | 0.6672 | MISPLACED | 0.2077 | 0.7397 | 0.2768 | 0.3929 |
| none@0.80 | 0.4783 | 0.1138 | 1.1769 | 0.6895 | 0.7012 | MISPLACED | 0.2052 | 0.7434 | 0.2741 | 0.6006 |
| none@0.85 | 0.4551 | 0.1107 | 1.0353 | 0.7121 | 0.6690 | MISPLACED | 0.1934 | 0.7538 | 0.2533 | 0.3675 |
| skeleton_redilate@0.90 | 0.4305 | 0.1050 | 1.0397 | 0.7349 | 0.6139 | MISPLACED | 0.1725 | 0.7703 | 0.2200 | 0.2420 |
| none@0.90 | 0.4034 | 0.1053 | 0.8707 | 0.7438 | 0.6130 | MISPLACED | 0.1635 | 0.7759 | 0.2032 | 0.1746 |

## VAL -- Steel2 (hypothesis)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skeleton_redilate@0.85 **<- selected** | 0.6701 | 0.2344 | 0.9856 | 0.9320 | 0.9085 | OFFSET | 0.2250 | 0.6839 | 0.3271 | 0.6133 |
| skeleton_redilate@tuned | 0.6701 | 0.2344 | 0.9856 | 0.9320 | 0.9085 | OFFSET | 0.2250 | 0.6839 | 0.3271 | 0.6133 |
| none@0.85 | 0.7609 | 0.2329 | 1.5168 | 0.9344 | 0.9096 | THICKNESS | 0.2241 | 0.6817 | 0.3266 | 0.6401 |
| none@tuned | 0.7609 | 0.2329 | 1.5168 | 0.9344 | 0.9096 | THICKNESS | 0.2241 | 0.6817 | 0.3266 | 0.6401 |
| skeleton_redilate@0.90 | 0.6995 | 0.2494 | 0.9969 | 0.9535 | 0.9220 | OFFSET | 0.2240 | 0.6831 | 0.3243 | 0.5450 |
| none@0.90 | 0.7500 | 0.2487 | 1.2585 | 0.9559 | 0.9225 | OFFSET | 0.2095 | 0.6807 | 0.3039 | 0.5533 |
| none@0.80 | 0.7514 | 0.2157 | 1.7200 | 0.9167 | 0.8818 | THICKNESS | 0.2055 | 0.6817 | 0.2999 | 0.6274 |
| skeleton_redilate@0.80 | 0.6374 | 0.2168 | 0.9857 | 0.9139 | 0.8812 | OFFSET | 0.1994 | 0.6796 | 0.2914 | 0.5801 |
| none@0.75 | 0.7392 | 0.2003 | 1.8834 | 0.9000 | 0.8533 | THICKNESS | 0.1943 | 0.6777 | 0.2855 | 0.6296 |
| skeleton_redilate@0.75 | 0.6095 | 0.2015 | 0.9879 | 0.8975 | 0.8531 | OFFSET | 0.1793 | 0.6754 | 0.2643 | 0.5355 |
| none@0.70 | 0.7272 | 0.1872 | 2.0148 | 0.8853 | 0.8294 | THICKNESS | 0.1793 | 0.6770 | 0.2640 | 0.6237 |
| skeleton_redilate@0.70 | 0.5867 | 0.1895 | 0.9895 | 0.8823 | 0.8298 | OFFSET | 0.1599 | 0.6718 | 0.2378 | 0.5112 |
| reference: watershed_prob@tuned | 0.7609 | 0.2329 | 1.5168 | 0.9344 | 0.9096 | THICKNESS | 0.0965 | 0.7024 | 0.1381 | 0.4786 |

## Selected on VAL

- **Steel1** (control -- not expected to help): `none@0.70` (mode none, threshold 0.70, val PQ 0.2452)
- **Steel2** (hypothesis): `skeleton_redilate@0.85` (mode skeleton_redilate, threshold 0.85, val PQ 0.2250)

## TEST -- Steel1 (control -- not expected to help)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cell 20 (recorded) | 0.5857 | 0.1453 | 1.4989 | 0.7663 | 0.8200 | MISPLACED | 0.3394 | 0.7388 | 0.4562 | 1.2027 |
| selected (val-best) | 0.5857 | 0.1453 | 1.4989 | 0.7663 | 0.8200 | MISPLACED | 0.3212 | 0.7469 | 0.4297 | 0.9160 |
| none@tuned | 0.5857 | 0.1453 | 1.4989 | 0.7663 | 0.8200 | MISPLACED | 0.3212 | 0.7469 | 0.4297 | 0.9160 |
| reference: watershed_prob@tuned | 0.5857 | 0.1453 | 1.4989 | 0.7663 | 0.8200 | MISPLACED | 0.3394 | 0.7388 | 0.4562 | 1.2027 |

## TEST -- Steel2 (hypothesis)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cell 20 (recorded) | 0.7762 | 0.2226 | 1.5152 | 0.9532 | 0.8976 | THICKNESS | 0.1059 | 0.6876 | 0.1525 | 0.4394 |
| selected (val-best) | 0.6549 | 0.2261 | 0.9880 | 0.9517 | 0.8974 | OFFSET | 0.1838 | 0.6657 | 0.2710 | 0.4984 |
| none@tuned | 0.7762 | 0.2226 | 1.5152 | 0.9532 | 0.8976 | THICKNESS | 0.1809 | 0.6614 | 0.2684 | 0.5112 |
| reference: watershed_prob@tuned | 0.7762 | 0.2226 | 1.5152 | 0.9532 | 0.8976 | THICKNESS | 0.1059 | 0.6876 | 0.1525 | 0.4394 |

## Checks

- PASS selection saw VAL rows only -- splits ['val']
- PASS TEST scored exactly the three pre-declared configurations -- selected (val-best), none@tuned, reference: watershed_prob@tuned
- PASS Steel1: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00, label MISPLACED vs MISPLACED
- PASS Steel1: watershed reference reproduces Cell 20's region metrics -- max |diff| 0.00e+00
- PASS Steel2: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00, label THICKNESS vs THICKNESS
- PASS Steel2: watershed reference reproduces Cell 20's region metrics -- max |diff| 0.00e+00
