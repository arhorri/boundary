# Post-processing sweep -- fold_steel_combined (colab)

Checkpoint: epoch 21, config hash `067fecc2e06a7d0a`. Ground truth line_width_px=4.0; skeleton_redilate target width 4 px; watershed marker threshold 0.3.

Regions for every configuration are the ones its thresholded, post-processed BOUNDARY implies (`binary_boundary`: the same conversion the ground truth goes through). Cell 20's region metrics use a watershed on the RAW probability map, which no threshold or post-processing can move; that partition appears only as the `reference` row. Selection is by PQ on VAL only; TEST is scored once, on configurations fixed before it was looked at.

## VAL -- Steel1 (control -- not expected to help)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference: watershed_prob@tuned | 0.4889 | 0.1098 | 1.4639 | 0.6563 | 0.7206 | MISPLACED | 0.2249 | 0.7197 | 0.3148 | 1.1059 |
| skeleton_redilate@0.70 **<- selected** | 0.4362 | 0.1080 | 0.9964 | 0.6512 | 0.7173 | MISPLACED | 0.2144 | 0.7121 | 0.2969 | 0.7644 |
| skeleton_redilate@tuned | 0.4362 | 0.1080 | 0.9964 | 0.6512 | 0.7173 | MISPLACED | 0.2144 | 0.7121 | 0.2969 | 0.7644 |
| none@0.70 | 0.4889 | 0.1098 | 1.4639 | 0.6563 | 0.7206 | MISPLACED | 0.2088 | 0.7336 | 0.2835 | 0.9032 |
| none@tuned | 0.4889 | 0.1098 | 1.4639 | 0.6563 | 0.7206 | MISPLACED | 0.2088 | 0.7336 | 0.2835 | 0.9032 |
| skeleton_redilate@0.75 | 0.4431 | 0.1086 | 1.0042 | 0.6652 | 0.7119 | MISPLACED | 0.2028 | 0.7376 | 0.2766 | 0.6616 |
| skeleton_redilate@0.80 | 0.4468 | 0.1112 | 1.0181 | 0.6835 | 0.6971 | MISPLACED | 0.1910 | 0.7308 | 0.2584 | 0.5536 |
| none@0.75 | 0.4876 | 0.1099 | 1.3172 | 0.6710 | 0.7155 | MISPLACED | 0.1796 | 0.7389 | 0.2421 | 0.7609 |
| none@0.80 | 0.4783 | 0.1138 | 1.1768 | 0.6895 | 0.7011 | MISPLACED | 0.1718 | 0.7444 | 0.2287 | 0.5818 |
| skeleton_redilate@0.85 | 0.4436 | 0.1082 | 1.0293 | 0.7044 | 0.6673 | MISPLACED | 0.1688 | 0.7402 | 0.2242 | 0.4222 |
| none@0.85 | 0.4551 | 0.1106 | 1.0353 | 0.7121 | 0.6691 | MISPLACED | 0.1634 | 0.7497 | 0.2145 | 0.3764 |
| skeleton_redilate@0.90 | 0.4305 | 0.1050 | 1.0396 | 0.7348 | 0.6139 | MISPLACED | 0.1432 | 0.7606 | 0.1850 | 0.3185 |
| none@0.90 | 0.4035 | 0.1053 | 0.8706 | 0.7438 | 0.6129 | MISPLACED | 0.1391 | 0.7684 | 0.1745 | 0.1615 |

## VAL -- Steel2 (hypothesis)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skeleton_redilate@0.85 **<- selected** | 0.6701 | 0.2344 | 0.9855 | 0.9320 | 0.9084 | OFFSET | 0.2103 | 0.6838 | 0.3056 | 0.5744 |
| skeleton_redilate@tuned | 0.6701 | 0.2344 | 0.9855 | 0.9320 | 0.9084 | OFFSET | 0.2103 | 0.6838 | 0.3056 | 0.5744 |
| none@0.85 | 0.7609 | 0.2328 | 1.5169 | 0.9344 | 0.9095 | THICKNESS | 0.2085 | 0.6810 | 0.3042 | 0.5992 |
| none@tuned | 0.7609 | 0.2328 | 1.5169 | 0.9344 | 0.9095 | THICKNESS | 0.2085 | 0.6810 | 0.3042 | 0.5992 |
| skeleton_redilate@0.90 | 0.6994 | 0.2494 | 0.9969 | 0.9536 | 0.9220 | OFFSET | 0.2074 | 0.6836 | 0.3000 | 0.5130 |
| none@0.90 | 0.7500 | 0.2486 | 1.2589 | 0.9559 | 0.9224 | OFFSET | 0.1946 | 0.6820 | 0.2821 | 0.5222 |
| none@0.80 | 0.7514 | 0.2157 | 1.7204 | 0.9167 | 0.8817 | THICKNESS | 0.1920 | 0.6814 | 0.2803 | 0.5837 |
| skeleton_redilate@0.80 | 0.6373 | 0.2167 | 0.9858 | 0.9139 | 0.8812 | OFFSET | 0.1855 | 0.6798 | 0.2711 | 0.5447 |
| none@0.75 | 0.7392 | 0.2002 | 1.8840 | 0.9000 | 0.8532 | THICKNESS | 0.1819 | 0.6772 | 0.2674 | 0.5838 |
| none@0.70 | 0.7272 | 0.1872 | 2.0146 | 0.8854 | 0.8295 | THICKNESS | 0.1678 | 0.6764 | 0.2472 | 0.5759 |
| skeleton_redilate@0.75 | 0.6095 | 0.2014 | 0.9881 | 0.8975 | 0.8530 | OFFSET | 0.1674 | 0.6756 | 0.2467 | 0.5019 |
| skeleton_redilate@0.70 | 0.5867 | 0.1895 | 0.9894 | 0.8823 | 0.8299 | OFFSET | 0.1494 | 0.6719 | 0.2221 | 0.4819 |
| reference: watershed_prob@tuned | 0.7609 | 0.2328 | 1.5169 | 0.9344 | 0.9095 | THICKNESS | 0.0914 | 0.7017 | 0.1309 | 0.4329 |

## Selected on VAL

- **Steel1** (control -- not expected to help): `skeleton_redilate@0.70` (mode skeleton_redilate, threshold 0.70, val PQ 0.2144)
- **Steel2** (hypothesis): `skeleton_redilate@0.85` (mode skeleton_redilate, threshold 0.85, val PQ 0.2103)

## TEST -- Steel1 (control -- not expected to help)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cell 20 (recorded) | 0.5857 | 0.1453 | 1.4986 | 0.7663 | 0.8201 | MISPLACED | 0.3025 | 0.7394 | 0.4054 | 0.9831 |
| selected (val-best) | 0.5380 | 0.1406 | 0.9890 | 0.7617 | 0.8162 | MISPLACED | 0.2783 | 0.7411 | 0.3732 | 0.6668 |
| none@tuned | 0.5857 | 0.1453 | 1.4986 | 0.7663 | 0.8201 | MISPLACED | 0.2719 | 0.7471 | 0.3631 | 0.8146 |
| reference: watershed_prob@tuned | 0.5857 | 0.1453 | 1.4986 | 0.7663 | 0.8201 | MISPLACED | 0.3025 | 0.7394 | 0.4054 | 0.9831 |

## TEST -- Steel2 (hypothesis)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cell 20 (recorded) | 0.7762 | 0.2226 | 1.5151 | 0.9532 | 0.8977 | THICKNESS | 0.0985 | 0.6882 | 0.1419 | 0.3889 |
| selected (val-best) | 0.6549 | 0.2261 | 0.9880 | 0.9517 | 0.8974 | OFFSET | 0.1720 | 0.6661 | 0.2536 | 0.4559 |
| none@tuned | 0.7762 | 0.2226 | 1.5151 | 0.9532 | 0.8977 | THICKNESS | 0.1692 | 0.6622 | 0.2507 | 0.4695 |
| reference: watershed_prob@tuned | 0.7762 | 0.2226 | 1.5151 | 0.9532 | 0.8977 | THICKNESS | 0.0985 | 0.6882 | 0.1419 | 0.3889 |

## Checks

- PASS selection saw VAL rows only -- splits ['val']
- PASS TEST scored exactly the three pre-declared configurations -- selected (val-best), none@tuned, reference: watershed_prob@tuned
- PASS Steel1: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00, label MISPLACED vs MISPLACED
- PASS Steel1: watershed reference reproduces Cell 20's region metrics -- max |diff| 0.00e+00
- PASS Steel2: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00, label THICKNESS vs THICKNESS
- PASS Steel2: watershed reference reproduces Cell 20's region metrics -- max |diff| 0.00e+00
