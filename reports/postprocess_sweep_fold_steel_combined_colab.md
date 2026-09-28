# Post-processing sweep -- fold_steel_combined (colab)

Checkpoint: epoch 21, config hash `067fecc2e06a7d0a`. Ground truth line_width_px=4.0; skeleton_redilate target width 4 px; watershed marker threshold 0.3.

Regions for every configuration are the ones its thresholded, post-processed BOUNDARY implies (`binary_boundary`: the same conversion the ground truth goes through). Cell 20's region metrics use a watershed on the RAW probability map, which no threshold or post-processing can move; that partition appears only as the `reference` row. Selection is by PQ on VAL only; TEST is scored once, on configurations fixed before it was looked at.

## VAL -- Steel1 (control -- not expected to help)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference: watershed_prob@tuned | 0.4889 | 0.1099 | 1.4633 | 0.6562 | 0.7208 | MISPLACED | 0.2249 | 0.7198 | 0.3148 | 1.1074 |
| skeleton_redilate@0.70 **<- selected** | 0.4363 | 0.1080 | 0.9963 | 0.6510 | 0.7174 | MISPLACED | 0.2152 | 0.7198 | 0.2979 | 0.7635 |
| skeleton_redilate@tuned | 0.4363 | 0.1080 | 0.9963 | 0.6510 | 0.7174 | MISPLACED | 0.2152 | 0.7198 | 0.2979 | 0.7635 |
| none@0.70 | 0.4889 | 0.1099 | 1.4633 | 0.6562 | 0.7208 | MISPLACED | 0.2090 | 0.7335 | 0.2838 | 0.9009 |
| none@tuned | 0.4889 | 0.1099 | 1.4633 | 0.6562 | 0.7208 | MISPLACED | 0.2090 | 0.7335 | 0.2838 | 0.9009 |
| skeleton_redilate@0.75 | 0.4432 | 0.1086 | 1.0045 | 0.6655 | 0.7121 | MISPLACED | 0.2039 | 0.7377 | 0.2780 | 0.6574 |
| skeleton_redilate@0.80 | 0.4467 | 0.1112 | 1.0180 | 0.6836 | 0.6970 | MISPLACED | 0.1910 | 0.7301 | 0.2587 | 0.5586 |
| none@0.75 | 0.4876 | 0.1098 | 1.3171 | 0.6711 | 0.7157 | MISPLACED | 0.1794 | 0.7391 | 0.2418 | 0.7633 |
| none@0.80 | 0.4783 | 0.1138 | 1.1767 | 0.6894 | 0.7009 | MISPLACED | 0.1718 | 0.7445 | 0.2288 | 0.5822 |
| skeleton_redilate@0.85 | 0.4435 | 0.1081 | 1.0293 | 0.7042 | 0.6671 | MISPLACED | 0.1686 | 0.7402 | 0.2239 | 0.4227 |
| none@0.85 | 0.4551 | 0.1105 | 1.0352 | 0.7119 | 0.6690 | MISPLACED | 0.1634 | 0.7497 | 0.2145 | 0.3770 |
| skeleton_redilate@0.90 | 0.4305 | 0.1049 | 1.0397 | 0.7349 | 0.6140 | MISPLACED | 0.1433 | 0.7606 | 0.1852 | 0.3198 |
| none@0.90 | 0.4035 | 0.1053 | 0.8706 | 0.7437 | 0.6130 | MISPLACED | 0.1391 | 0.7684 | 0.1744 | 0.1620 |

## VAL -- Steel2 (hypothesis)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skeleton_redilate@0.85 **<- selected** | 0.6701 | 0.2345 | 0.9856 | 0.9321 | 0.9085 | OFFSET | 0.2096 | 0.6836 | 0.3047 | 0.5755 |
| skeleton_redilate@tuned | 0.6701 | 0.2345 | 0.9856 | 0.9321 | 0.9085 | OFFSET | 0.2096 | 0.6836 | 0.3047 | 0.5755 |
| none@0.85 | 0.7609 | 0.2327 | 1.5169 | 0.9344 | 0.9097 | THICKNESS | 0.2084 | 0.6814 | 0.3039 | 0.5987 |
| none@tuned | 0.7609 | 0.2327 | 1.5169 | 0.9344 | 0.9097 | THICKNESS | 0.2084 | 0.6814 | 0.3039 | 0.5987 |
| skeleton_redilate@0.90 | 0.6995 | 0.2495 | 0.9969 | 0.9536 | 0.9221 | OFFSET | 0.2076 | 0.6835 | 0.3004 | 0.5130 |
| none@0.90 | 0.7500 | 0.2487 | 1.2590 | 0.9558 | 0.9226 | OFFSET | 0.1946 | 0.6820 | 0.2821 | 0.5220 |
| none@0.80 | 0.7514 | 0.2157 | 1.7204 | 0.9167 | 0.8817 | THICKNESS | 0.1916 | 0.6818 | 0.2795 | 0.5833 |
| skeleton_redilate@0.80 | 0.6374 | 0.2168 | 0.9858 | 0.9140 | 0.8811 | OFFSET | 0.1855 | 0.6797 | 0.2712 | 0.5439 |
| none@0.75 | 0.7392 | 0.2001 | 1.8837 | 0.8999 | 0.8533 | THICKNESS | 0.1817 | 0.6771 | 0.2671 | 0.5840 |
| none@0.70 | 0.7272 | 0.1872 | 2.0151 | 0.8853 | 0.8294 | THICKNESS | 0.1676 | 0.6768 | 0.2467 | 0.5755 |
| skeleton_redilate@0.75 | 0.6094 | 0.2013 | 0.9882 | 0.8975 | 0.8531 | OFFSET | 0.1672 | 0.6756 | 0.2463 | 0.5016 |
| skeleton_redilate@0.70 | 0.5867 | 0.1895 | 0.9895 | 0.8823 | 0.8298 | OFFSET | 0.1494 | 0.6720 | 0.2221 | 0.4814 |
| reference: watershed_prob@tuned | 0.7609 | 0.2327 | 1.5169 | 0.9344 | 0.9097 | THICKNESS | 0.0911 | 0.7028 | 0.1304 | 0.4329 |

## Selected on VAL

- **Steel1** (control -- not expected to help): `skeleton_redilate@0.70` (mode skeleton_redilate, threshold 0.70, val PQ 0.2152)
- **Steel2** (hypothesis): `skeleton_redilate@0.85` (mode skeleton_redilate, threshold 0.85, val PQ 0.2096)

## TEST -- Steel1 (control -- not expected to help)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cell 20 (recorded) | 0.5857 | 0.1452 | 1.4990 | 0.7664 | 0.8201 | MISPLACED | 0.3025 | 0.7391 | 0.4055 | 0.9835 |
| selected (val-best) | 0.5380 | 0.1406 | 0.9890 | 0.7617 | 0.8163 | MISPLACED | 0.2786 | 0.7412 | 0.3736 | 0.6653 |
| none@tuned | 0.5857 | 0.1452 | 1.4990 | 0.7664 | 0.8201 | MISPLACED | 0.2723 | 0.7471 | 0.3637 | 0.8108 |
| reference: watershed_prob@tuned | 0.5857 | 0.1452 | 1.4990 | 0.7664 | 0.8201 | MISPLACED | 0.3025 | 0.7391 | 0.4055 | 0.9835 |

## TEST -- Steel2 (hypothesis)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cell 20 (recorded) | 0.7762 | 0.2226 | 1.5155 | 0.9532 | 0.8976 | THICKNESS | 0.0986 | 0.6886 | 0.1419 | 0.3887 |
| selected (val-best) | 0.6549 | 0.2261 | 0.9881 | 0.9518 | 0.8974 | OFFSET | 0.1722 | 0.6662 | 0.2538 | 0.4555 |
| none@tuned | 0.7762 | 0.2226 | 1.5155 | 0.9532 | 0.8976 | THICKNESS | 0.1694 | 0.6620 | 0.2511 | 0.4697 |
| reference: watershed_prob@tuned | 0.7762 | 0.2226 | 1.5155 | 0.9532 | 0.8976 | THICKNESS | 0.0986 | 0.6886 | 0.1419 | 0.3887 |

## Checks

- PASS selection saw VAL rows only -- splits ['val']
- PASS TEST scored exactly the three pre-declared configurations -- selected (val-best), none@tuned, reference: watershed_prob@tuned
- PASS Steel1: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00, label MISPLACED vs MISPLACED
- PASS Steel1: watershed reference reproduces Cell 20's region metrics -- max |diff| 0.00e+00
- PASS Steel2: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00, label THICKNESS vs THICKNESS
- PASS Steel2: watershed reference reproduces Cell 20's region metrics -- max |diff| 0.00e+00
