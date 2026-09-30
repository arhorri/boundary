# Post-processing sweep -- fold_steel_combined (colab)

Checkpoint: epoch 21, config hash `067fecc2e06a7d0a`. Ground truth line_width_px=4.0; skeleton_redilate target width 4 px; watershed marker threshold 0.3.

Regions for every configuration are the ones its thresholded, post-processed BOUNDARY implies (`binary_boundary`: the same conversion the ground truth goes through). Cell 20's region metrics use a watershed on the RAW probability map, which no threshold or post-processing can move; that partition appears only as the `reference` row. Selection is by PQ on VAL only; TEST is scored once, on configurations fixed before it was looked at.

## VAL -- Steel1 (control -- not expected to help)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference: watershed_prob@tuned | 0.4889 | 0.1098 | 1.4637 | 0.6562 | 0.7207 | MISPLACED | 0.2248 | 0.7199 | 0.3146 | 1.1088 |
| skeleton_redilate@0.70 **<- selected** | 0.4362 | 0.1080 | 0.9964 | 0.6513 | 0.7173 | MISPLACED | 0.2147 | 0.7121 | 0.2973 | 0.7606 |
| skeleton_redilate@tuned | 0.4362 | 0.1080 | 0.9964 | 0.6513 | 0.7173 | MISPLACED | 0.2147 | 0.7121 | 0.2973 | 0.7606 |
| none@0.70 | 0.4889 | 0.1098 | 1.4637 | 0.6562 | 0.7207 | MISPLACED | 0.2088 | 0.7336 | 0.2835 | 0.9023 |
| none@tuned | 0.4889 | 0.1098 | 1.4637 | 0.6562 | 0.7207 | MISPLACED | 0.2088 | 0.7336 | 0.2835 | 0.9023 |
| skeleton_redilate@0.75 | 0.4431 | 0.1087 | 1.0044 | 0.6652 | 0.7117 | MISPLACED | 0.2027 | 0.7383 | 0.2763 | 0.6596 |
| skeleton_redilate@0.80 | 0.4468 | 0.1111 | 1.0181 | 0.6836 | 0.6971 | MISPLACED | 0.1915 | 0.7301 | 0.2593 | 0.5548 |
| none@0.75 | 0.4876 | 0.1099 | 1.3172 | 0.6710 | 0.7155 | MISPLACED | 0.1796 | 0.7390 | 0.2420 | 0.7607 |
| none@0.80 | 0.4783 | 0.1138 | 1.1765 | 0.6895 | 0.7011 | MISPLACED | 0.1719 | 0.7445 | 0.2289 | 0.5806 |
| skeleton_redilate@0.85 | 0.4435 | 0.1081 | 1.0294 | 0.7043 | 0.6670 | MISPLACED | 0.1687 | 0.7403 | 0.2242 | 0.4236 |
| none@0.85 | 0.4551 | 0.1105 | 1.0354 | 0.7120 | 0.6688 | MISPLACED | 0.1633 | 0.7497 | 0.2144 | 0.3777 |
| skeleton_redilate@0.90 | 0.4305 | 0.1051 | 1.0397 | 0.7350 | 0.6139 | MISPLACED | 0.1432 | 0.7606 | 0.1851 | 0.3197 |
| none@0.90 | 0.4035 | 0.1054 | 0.8706 | 0.7439 | 0.6129 | MISPLACED | 0.1395 | 0.7683 | 0.1749 | 0.1623 |

## VAL -- Steel2 (hypothesis)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skeleton_redilate@0.85 **<- selected** | 0.6700 | 0.2344 | 0.9855 | 0.9321 | 0.9085 | OFFSET | 0.2098 | 0.6833 | 0.3052 | 0.5755 |
| skeleton_redilate@tuned | 0.6700 | 0.2344 | 0.9855 | 0.9321 | 0.9085 | OFFSET | 0.2098 | 0.6833 | 0.3052 | 0.5755 |
| none@0.85 | 0.7609 | 0.2328 | 1.5170 | 0.9344 | 0.9096 | THICKNESS | 0.2087 | 0.6812 | 0.3044 | 0.5990 |
| none@tuned | 0.7609 | 0.2328 | 1.5170 | 0.9344 | 0.9096 | THICKNESS | 0.2087 | 0.6812 | 0.3044 | 0.5990 |
| skeleton_redilate@0.90 | 0.6996 | 0.2496 | 0.9971 | 0.9536 | 0.9220 | OFFSET | 0.2073 | 0.6836 | 0.2998 | 0.5130 |
| none@0.90 | 0.7500 | 0.2487 | 1.2587 | 0.9558 | 0.9226 | OFFSET | 0.1948 | 0.6820 | 0.2823 | 0.5229 |
| none@0.80 | 0.7514 | 0.2157 | 1.7203 | 0.9167 | 0.8818 | THICKNESS | 0.1919 | 0.6815 | 0.2801 | 0.5838 |
| skeleton_redilate@0.80 | 0.6374 | 0.2167 | 0.9857 | 0.9140 | 0.8812 | OFFSET | 0.1854 | 0.6797 | 0.2709 | 0.5447 |
| none@0.75 | 0.7392 | 0.2002 | 1.8837 | 0.9000 | 0.8533 | THICKNESS | 0.1819 | 0.6768 | 0.2675 | 0.5845 |
| none@0.70 | 0.7272 | 0.1872 | 2.0149 | 0.8853 | 0.8294 | THICKNESS | 0.1674 | 0.6768 | 0.2465 | 0.5760 |
| skeleton_redilate@0.75 | 0.6094 | 0.2014 | 0.9881 | 0.8975 | 0.8530 | OFFSET | 0.1671 | 0.6756 | 0.2462 | 0.5021 |
| skeleton_redilate@0.70 | 0.5867 | 0.1895 | 0.9894 | 0.8824 | 0.8298 | OFFSET | 0.1491 | 0.6721 | 0.2216 | 0.4813 |
| reference: watershed_prob@tuned | 0.7609 | 0.2328 | 1.5170 | 0.9344 | 0.9096 | THICKNESS | 0.0912 | 0.7026 | 0.1307 | 0.4329 |

## Selected on VAL

- **Steel1** (control -- not expected to help): `skeleton_redilate@0.70` (mode skeleton_redilate, threshold 0.70, val PQ 0.2147)
- **Steel2** (hypothesis): `skeleton_redilate@0.85` (mode skeleton_redilate, threshold 0.85, val PQ 0.2098)

## TEST -- Steel1 (control -- not expected to help)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cell 20 (recorded) | 0.5857 | 0.1453 | 1.4985 | 0.7664 | 0.8203 | MISPLACED | 0.3027 | 0.7394 | 0.4056 | 0.9822 |
| selected (val-best) | 0.5381 | 0.1407 | 0.9890 | 0.7617 | 0.8165 | MISPLACED | 0.2792 | 0.7405 | 0.3748 | 0.6692 |
| none@tuned | 0.5857 | 0.1453 | 1.4985 | 0.7664 | 0.8203 | MISPLACED | 0.2724 | 0.7472 | 0.3637 | 0.8113 |
| reference: watershed_prob@tuned | 0.5857 | 0.1453 | 1.4985 | 0.7664 | 0.8203 | MISPLACED | 0.3027 | 0.7394 | 0.4056 | 0.9822 |

## TEST -- Steel2 (hypothesis)

| config | pixel_dice | skeleton_dice | width_ratio | real | found | label | pq | sq | rq | over_segmentation_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cell 20 (recorded) | 0.7762 | 0.2225 | 1.5154 | 0.9533 | 0.8976 | THICKNESS | 0.0985 | 0.6888 | 0.1417 | 0.3886 |
| selected (val-best) | 0.6549 | 0.2260 | 0.9880 | 0.9518 | 0.8973 | OFFSET | 0.1721 | 0.6661 | 0.2538 | 0.4555 |
| none@tuned | 0.7762 | 0.2225 | 1.5154 | 0.9533 | 0.8976 | THICKNESS | 0.1693 | 0.6624 | 0.2507 | 0.4695 |
| reference: watershed_prob@tuned | 0.7762 | 0.2225 | 1.5154 | 0.9533 | 0.8976 | THICKNESS | 0.0985 | 0.6888 | 0.1417 | 0.3886 |

## Checks

- PASS selection saw VAL rows only -- splits ['val']
- PASS TEST scored exactly the three pre-declared configurations -- selected (val-best), none@tuned, reference: watershed_prob@tuned
- PASS Steel1: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00, label MISPLACED vs MISPLACED
- PASS Steel1: watershed reference reproduces Cell 20's region metrics -- max |diff| 0.00e+00
- PASS Steel2: none@tuned reproduces Cell 20's decomposition -- max |diff| 0.00e+00, label THICKNESS vs THICKNESS
- PASS Steel2: watershed reference reproduces Cell 20's region metrics -- max |diff| 0.00e+00
