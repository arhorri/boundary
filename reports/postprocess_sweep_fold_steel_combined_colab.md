# Post-processing sweep -- fold_steel_combined (colab)

Checkpoint: epoch 21, config hash `067fecc2e06a7d0a`. Ground truth line_width_px=4.0; skeleton_redilate target width 4 px; watershed marker threshold 0.3.

Regions for every configuration are the ones its thresholded, post-processed BOUNDARY implies (`binary_boundary`: the same conversion the ground truth goes through). Cell 20's region metrics use a watershed on the RAW probability map, which no threshold or post-processing can move; that partition appears only as the `reference` row. Selection is by PQ on VAL only; TEST is scored once, on configurations fixed before it was looked at.

## VAL -- Steel1 (control -- not expected to help)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference: watershed_prob@tuned | 0.4889 | 0.1099 | 1.4635 | 0.6561 | 0.7207 | MISPLACED | 0.2247 | 0.7197 | 0.3145 | 1.1097 |
| skeleton_redilate@0.70 **<- selected** | 0.4362 | 0.1080 | 0.9963 | 0.6511 | 0.7175 | MISPLACED | 0.2141 | 0.7122 | 0.2965 | 0.7669 |
| skeleton_redilate@tuned | 0.4362 | 0.1080 | 0.9963 | 0.6511 | 0.7175 | MISPLACED | 0.2141 | 0.7122 | 0.2965 | 0.7669 |
| none@0.70 | 0.4889 | 0.1099 | 1.4635 | 0.6561 | 0.7207 | MISPLACED | 0.2087 | 0.7336 | 0.2834 | 0.9043 |
| none@tuned | 0.4889 | 0.1099 | 1.4635 | 0.6561 | 0.7207 | MISPLACED | 0.2087 | 0.7336 | 0.2834 | 0.9043 |
| skeleton_redilate@0.75 | 0.4431 | 0.1086 | 1.0043 | 0.6653 | 0.7118 | MISPLACED | 0.2028 | 0.7375 | 0.2766 | 0.6621 |
| skeleton_redilate@0.80 | 0.4468 | 0.1112 | 1.0180 | 0.6836 | 0.6971 | MISPLACED | 0.1910 | 0.7308 | 0.2583 | 0.5541 |
| none@0.75 | 0.4876 | 0.1099 | 1.3171 | 0.6710 | 0.7155 | MISPLACED | 0.1793 | 0.7388 | 0.2417 | 0.7637 |
| none@0.80 | 0.4783 | 0.1138 | 1.1767 | 0.6895 | 0.7011 | MISPLACED | 0.1717 | 0.7445 | 0.2287 | 0.5817 |
| skeleton_redilate@0.85 | 0.4436 | 0.1081 | 1.0293 | 0.7044 | 0.6673 | MISPLACED | 0.1688 | 0.7402 | 0.2243 | 0.4217 |
| none@0.85 | 0.4551 | 0.1106 | 1.0350 | 0.7121 | 0.6691 | MISPLACED | 0.1632 | 0.7497 | 0.2142 | 0.3798 |
| skeleton_redilate@0.90 | 0.4305 | 0.1050 | 1.0397 | 0.7350 | 0.6139 | MISPLACED | 0.1433 | 0.7606 | 0.1852 | 0.3198 |
| none@0.90 | 0.4034 | 0.1053 | 0.8705 | 0.7438 | 0.6130 | MISPLACED | 0.1391 | 0.7684 | 0.1745 | 0.1615 |

## VAL -- Steel2 (hypothesis)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skeleton_redilate@0.85 **<- selected** | 0.6701 | 0.2343 | 0.9856 | 0.9320 | 0.9086 | OFFSET | 0.2098 | 0.6835 | 0.3051 | 0.5750 |
| skeleton_redilate@tuned | 0.6701 | 0.2343 | 0.9856 | 0.9320 | 0.9086 | OFFSET | 0.2098 | 0.6835 | 0.3051 | 0.5750 |
| none@0.85 | 0.7609 | 0.2327 | 1.5166 | 0.9344 | 0.9097 | THICKNESS | 0.2087 | 0.6812 | 0.3044 | 0.5989 |
| none@tuned | 0.7609 | 0.2327 | 1.5166 | 0.9344 | 0.9097 | THICKNESS | 0.2087 | 0.6812 | 0.3044 | 0.5989 |
| skeleton_redilate@0.90 | 0.6995 | 0.2495 | 0.9969 | 0.9536 | 0.9220 | OFFSET | 0.2076 | 0.6835 | 0.3003 | 0.5126 |
| none@0.90 | 0.7500 | 0.2487 | 1.2586 | 0.9559 | 0.9225 | OFFSET | 0.1947 | 0.6822 | 0.2821 | 0.5241 |
| none@0.80 | 0.7514 | 0.2157 | 1.7206 | 0.9168 | 0.8817 | THICKNESS | 0.1920 | 0.6815 | 0.2802 | 0.5825 |
| skeleton_redilate@0.80 | 0.6374 | 0.2168 | 0.9857 | 0.9141 | 0.8811 | OFFSET | 0.1861 | 0.6796 | 0.2720 | 0.5433 |
| none@0.75 | 0.7392 | 0.2002 | 1.8837 | 0.9000 | 0.8533 | THICKNESS | 0.1819 | 0.6771 | 0.2675 | 0.5837 |
| none@0.70 | 0.7272 | 0.1872 | 2.0149 | 0.8853 | 0.8295 | THICKNESS | 0.1677 | 0.6764 | 0.2470 | 0.5759 |
| skeleton_redilate@0.75 | 0.6095 | 0.2014 | 0.9880 | 0.8976 | 0.8531 | OFFSET | 0.1671 | 0.6756 | 0.2463 | 0.5023 |
| skeleton_redilate@0.70 | 0.5867 | 0.1894 | 0.9893 | 0.8823 | 0.8299 | OFFSET | 0.1494 | 0.6716 | 0.2221 | 0.4819 |
| reference: watershed_prob@tuned | 0.7609 | 0.2327 | 1.5166 | 0.9344 | 0.9097 | THICKNESS | 0.0911 | 0.7026 | 0.1305 | 0.4332 |

## Selected on VAL

- **Steel1** (control -- not expected to help): `skeleton_redilate@0.70` (mode skeleton_redilate, threshold 0.70, val PQ 0.2141)
- **Steel2** (hypothesis): `skeleton_redilate@0.85` (mode skeleton_redilate, threshold 0.85, val PQ 0.2098)

## TEST -- Steel1 (control -- not expected to help)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cell 20 (recorded) | 0.5857 | 0.1452 | 1.4986 | 0.7664 | 0.8203 | MISPLACED | 0.3025 | 0.7391 | 0.4055 | 0.9829 |
| selected (val-best) | 0.5381 | 0.1406 | 0.9891 | 0.7618 | 0.8164 | MISPLACED | 0.2786 | 0.7413 | 0.3735 | 0.6655 |
| none@tuned | 0.5857 | 0.1452 | 1.4986 | 0.7664 | 0.8203 | MISPLACED | 0.2720 | 0.7471 | 0.3633 | 0.8125 |
| reference: watershed_prob@tuned | 0.5857 | 0.1452 | 1.4986 | 0.7664 | 0.8203 | MISPLACED | 0.3025 | 0.7391 | 0.4055 | 0.9829 |

## TEST -- Steel2 (hypothesis)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cell 20 (recorded) | 0.7762 | 0.2225 | 1.5153 | 0.9533 | 0.8976 | THICKNESS | 0.0985 | 0.6889 | 0.1417 | 0.3886 |
| selected (val-best) | 0.6549 | 0.2260 | 0.9881 | 0.9518 | 0.8974 | OFFSET | 0.1720 | 0.6660 | 0.2536 | 0.4553 |
| none@tuned | 0.7762 | 0.2225 | 1.5153 | 0.9533 | 0.8976 | THICKNESS | 0.1693 | 0.6621 | 0.2508 | 0.4692 |
| reference: watershed_prob@tuned | 0.7762 | 0.2225 | 1.5153 | 0.9533 | 0.8976 | THICKNESS | 0.0985 | 0.6889 | 0.1417 | 0.3886 |

## Checks

- PASS selection saw VAL rows only -- splits ['val']
- PASS TEST scored exactly the three pre-declared configurations -- selected (val-best), none@tuned, reference: watershed_prob@tuned
- PASS Steel1: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00, label MISPLACED vs MISPLACED
- PASS Steel1: watershed reference reproduces Cell 20's region metrics -- max |diff| 0.00e+00
- PASS Steel2: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00, label THICKNESS vs THICKNESS
- PASS Steel2: watershed reference reproduces Cell 20's region metrics -- max |diff| 0.00e+00
