# Training report -- dev-w4-sched on colab

Held-out dataset: **uhcs2**. Generated 2026-09-17T08:06:46Z on colab (Tesla T4).

This file is keyed by run and host: `train_dev-w4-sched_colab.md`. The same fold trained on another host writes its own file beside this one rather than overwriting it -- two runs of one fold are two measurements, and they differ in GPU, worker count and I/O path.

- training mixture: the fold's full set, nothing excluded
- config hash `360884933d0f8082`, seed 0 (statistically reproducible; cudnn.benchmark picks algorithms by timing, so bitwise equality across runs is not claimed)
- 80 epochs, batch 64, 4 workers (configs/dataloader.yaml hosts.colab.num_workers, measured 2026-09-06T07:14:10Z)
- lr 0.0003 (encoder 8.999999999999999e-05), weight decay 0.0001, warmup 2 epochs, grad clip 1.0
- pos_weight 7.111000061035156 (configs/fold_stats.yaml folds.dev.pos_weight)
- best epoch 20 by best-threshold Dice on `uhcs2` = 0.2705 at threshold 0.65
- validation threshold swept over 0.05..0.95 (19 points); fixed reference 0.50
- FiLM: disabled
- patience 10 (stopped early at epoch 30)
- boundary_gt.line_width_px 4 (the ground truth this run trained on -- NOT directly comparable to a report written under a different value: pos_weight and every boundary fraction move with it)

## Per-dataset validation metrics (the headline)

Validation is a mixture. The pooled row is a footnote; the held-out dataset's row is the measurement this fold exists to make.

Every row appears twice: at the fixed `train.threshold` and at the threshold that maximised Dice for that dataset. A single fixed threshold measures the model and the operating point together and reports the sum as if it were the model. `best.pt` is selected on the best-threshold Dice of the held-out dataset.

| epoch | dataset | thr | tiles | IoU | Dice | Precision | Recall | boundary-F | pred frac | true frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20 (best) | Steel1 fixed | 0.50 | 189 | 0.3228 | 0.4881 | 0.3537 | 0.7872 | 0.6809 | 0.1859 | 0.0835 |
| 20 (best) | Steel1 **best** | 0.75 | 189 | 0.3504 | 0.5189 | 0.4496 | 0.6134 | 0.7100 | 0.1139 | 0.0835 |
| 20 (best) | uhcs2 (held out) fixed | 0.50 | 265 | 0.1530 | 0.2654 | 0.1604 | 0.7695 | 0.3994 | 0.5542 | 0.1155 |
| 20 (best) | uhcs2 (held out) **best** | 0.65 | 265 | 0.1564 | 0.2705 | 0.1679 | 0.6947 | 0.4007 | 0.4778 | 0.1155 |
| 20 (best) | _pooled (footnote)_ fixed | 0.50 | 454 | 0.1870 | 0.3151 | 0.1977 | 0.7755 | 0.4647 | 0.4009 | 0.1022 |
| 20 (best) | _pooled (footnote)_ **best** | 0.70 | 454 | 0.1925 | 0.3228 | 0.2138 | 0.6584 | 0.4663 | 0.3146 | 0.1022 |
| 30 (final) | Steel1 fixed | 0.50 | 189 | 0.3090 | 0.4722 | 0.3261 | 0.8554 | 0.6640 | 0.2191 | 0.0835 |
| 30 (final) | Steel1 **best** | 0.80 | 189 | 0.3600 | 0.5294 | 0.4483 | 0.6463 | 0.7254 | 0.1204 | 0.0835 |
| 30 (final) | uhcs2 (held out) fixed | 0.50 | 265 | 0.1485 | 0.2586 | 0.1559 | 0.7589 | 0.3892 | 0.5624 | 0.1155 |
| 30 (final) | uhcs2 (held out) **best** | 0.60 | 265 | 0.1498 | 0.2606 | 0.1595 | 0.7118 | 0.3888 | 0.5153 | 0.1155 |
| 30 (final) | _pooled (footnote)_ fixed | 0.50 | 454 | 0.1836 | 0.3102 | 0.1929 | 0.7918 | 0.4581 | 0.4195 | 0.1022 |
| 30 (final) | _pooled (footnote)_ **best** | 0.70 | 454 | 0.1894 | 0.3185 | 0.2081 | 0.6783 | 0.4609 | 0.3330 | 0.1022 |

## The chosen threshold, per epoch, per dataset

This table is a measurement, not bookkeeping. If the held-out dataset's optimal threshold sits far from the training-side datasets', that gap IS the domain shift, expressed in the units of the decision the downstream watershed has to make -- and one global threshold will not serve both.

| epoch | Steel1 | uhcs2 | spread |
| --- | --- | --- | --- |
| 0 | 0.70 | 0.65 | 0.05 |
| 1 | 0.75 | 0.80 | 0.05 |
| 2 | 0.60 | 0.65 | 0.05 |
| 3 | 0.65 | 0.45 | 0.20 |
| 4 | 0.75 | 0.35 | 0.40 |
| 5 | 0.75 | 0.50 | 0.25 |
| 6 | 0.70 | 0.60 | 0.10 |
| 7 | 0.75 | 0.65 | 0.10 |
| 8 | 0.70 | 0.55 | 0.15 |
| 9 | 0.75 | 0.60 | 0.15 |
| 10 | 0.70 | 0.55 | 0.15 |
| 11 | 0.75 | 0.60 | 0.15 |
| 12 | 0.80 | 0.60 | 0.20 |
| 13 | 0.75 | 0.65 | 0.10 |
| 14 | 0.80 | 0.60 | 0.20 |
| 15 | 0.70 | 0.65 | 0.05 |
| 16 | 0.80 | 0.65 | 0.15 |
| 17 | 0.75 | 0.60 | 0.15 |
| 18 | 0.75 | 0.60 | 0.15 |
| 19 | 0.75 | 0.55 | 0.20 |
| 20 | 0.75 | 0.65 | 0.10 |
| 21 | 0.80 | 0.60 | 0.20 |
| 22 | 0.70 | 0.55 | 0.15 |
| 23 | 0.75 | 0.60 | 0.15 |
| 24 | 0.75 | 0.60 | 0.15 |
| 25 | 0.80 | 0.55 | 0.25 |
| 26 | 0.75 | 0.60 | 0.15 |
| 27 | 0.80 | 0.60 | 0.20 |
| 28 | 0.75 | 0.50 | 0.25 |
| 29 | 0.80 | 0.45 | 0.35 |
| 30 | 0.80 | 0.60 | 0.20 |

**Final epoch.** uhcs2 (held out) wants 0.60; the others want Steel1 0.80 -- a gap of 0.20. That is a domain-shift finding: the held-out microscope needs a materially different operating point, so step 7 should set the threshold PER DATASET rather than globally.

## Loss terms, separately

They differ by orders of magnitude, so the total alone does not say which one moved.

| epoch | train total | train BCE | train Dice | train clDice | val total | val BCE | val Dice | val clDice |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.4330 | 1.5183 | 0.6140 | 0.6013 | 2.6099 | 1.3901 | 0.8189 | 0.8017 |
| 1 | 1.9605 | 1.2007 | 0.5617 | 0.3962 | 2.6704 | 1.5242 | 0.7964 | 0.6996 |
| 2 | 1.7011 | 1.0503 | 0.5288 | 0.2438 | 2.4938 | 1.3870 | 0.7983 | 0.6170 |
| 3 | 1.5977 | 0.9843 | 0.5032 | 0.2206 | 2.4426 | 1.3391 | 0.7951 | 0.6167 |
| 4 | 1.4772 | 0.8987 | 0.4714 | 0.2142 | 2.3156 | 1.2263 | 0.7724 | 0.6337 |
| 5 | 1.3897 | 0.8408 | 0.4417 | 0.2145 | 2.2992 | 1.2424 | 0.7430 | 0.6276 |
| 6 | 1.3302 | 0.8024 | 0.4221 | 0.2116 | 2.2464 | 1.2023 | 0.7303 | 0.6276 |
| 7 | 1.2783 | 0.7771 | 0.3996 | 0.2033 | 2.2482 | 1.2198 | 0.7158 | 0.6254 |
| 8 | 1.2594 | 0.7803 | 0.3816 | 0.1951 | 2.2500 | 1.2430 | 0.7023 | 0.6094 |
| 9 | 1.2371 | 0.7615 | 0.3790 | 0.1930 | 2.2349 | 1.2327 | 0.6986 | 0.6074 |
| 10 | 1.2238 | 0.7557 | 0.3739 | 0.1884 | 2.2125 | 1.2173 | 0.6962 | 0.5980 |
| 11 | 1.1960 | 0.7378 | 0.3671 | 0.1820 | 2.2037 | 1.2004 | 0.6984 | 0.6097 |
| 12 | 1.1806 | 0.7335 | 0.3588 | 0.1765 | 2.2121 | 1.2149 | 0.6983 | 0.5978 |
| 13 | 1.1890 | 0.7410 | 0.3603 | 0.1755 | 2.1820 | 1.1835 | 0.6970 | 0.6030 |
| 14 | 1.1740 | 0.7261 | 0.3601 | 0.1758 | 2.2116 | 1.2154 | 0.7028 | 0.5867 |
| 15 | 1.1741 | 0.7262 | 0.3607 | 0.1743 | 2.1906 | 1.2134 | 0.6815 | 0.5912 |
| 16 | 1.1390 | 0.7095 | 0.3480 | 0.1630 | 2.2138 | 1.2310 | 0.6897 | 0.5862 |
| 17 | 1.1324 | 0.7059 | 0.3463 | 0.1604 | 2.1872 | 1.2161 | 0.6806 | 0.5811 |
| 18 | 1.1318 | 0.7013 | 0.3492 | 0.1626 | 2.2176 | 1.2423 | 0.6862 | 0.5783 |
| 19 | 1.1251 | 0.6967 | 0.3473 | 0.1621 | 2.1705 | 1.2074 | 0.6786 | 0.5689 |
| 20 | 1.1364 | 0.7074 | 0.3481 | 0.1617 | 2.1867 | 1.2215 | 0.6755 | 0.5793 |
| 21 | 1.1229 | 0.6960 | 0.3475 | 0.1588 | 2.2608 | 1.2777 | 0.6931 | 0.5799 |
| 22 | 1.1141 | 0.6919 | 0.3437 | 0.1569 | 2.2323 | 1.2695 | 0.6761 | 0.5735 |
| 23 | 1.0970 | 0.6865 | 0.3361 | 0.1487 | 2.1422 | 1.1853 | 0.6744 | 0.5649 |
| 24 | 1.1039 | 0.6893 | 0.3387 | 0.1518 | 2.1689 | 1.2027 | 0.6785 | 0.5754 |
| 25 | 1.0894 | 0.6763 | 0.3377 | 0.1508 | 2.1604 | 1.1989 | 0.6792 | 0.5646 |
| 26 | 1.0930 | 0.6833 | 0.3358 | 0.1477 | 2.1365 | 1.1769 | 0.6754 | 0.5683 |
| 27 | 1.0941 | 0.6894 | 0.3320 | 0.1456 | 2.2113 | 1.2466 | 0.6830 | 0.5634 |
| 28 | 1.0687 | 0.6595 | 0.3357 | 0.1470 | 2.2465 | 1.2881 | 0.6759 | 0.5650 |
| 29 | 1.0551 | 0.6592 | 0.3261 | 0.1397 | 2.2546 | 1.2951 | 0.6830 | 0.5530 |
| 30 | 1.0798 | 0.6735 | 0.3331 | 0.1464 | 2.2234 | 1.2556 | 0.6874 | 0.5608 |

## clDice diagnostic on real predictions

Step 5 measured clDice against perturbed ground truth. This measures it against what the model actually produces, which is soft and thicker than the target. `skeleton_delta` is `mean |soft_skeleton(sigmoid(logits)) - sigmoid(logits)|`: near zero means the soft skeleton is returning its input and the term is inert.

| epoch | skel(pred) | skel(true) | skel(pred) on true | t_prec | t_rec | skeleton delta | Dice | clDice | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 3167 | 1318 | 341 | 0.1057 | 0.6785 | 0.591884 | 0.8296 | 0.8195 | active |
| 1 | 1004 | 1318 | 168 | 0.1699 | 0.7874 | 0.653219 | 0.8121 | 0.7267 | active |
| 2 | 705 | 1318 | 152 | 0.2412 | 0.7110 | 0.592815 | 0.8161 | 0.6591 | active |
| 3 | 701 | 1318 | 146 | 0.2464 | 0.7036 | 0.574060 | 0.8111 | 0.6571 | active |
| 4 | 839 | 1318 | 160 | 0.2226 | 0.6902 | 0.513986 | 0.7927 | 0.6806 | active |
| 5 | 811 | 1318 | 154 | 0.2279 | 0.7035 | 0.478844 | 0.7684 | 0.6741 | active |
| 6 | 851 | 1318 | 162 | 0.2324 | 0.7010 | 0.455544 | 0.7583 | 0.6688 | active |
| 7 | 880 | 1318 | 167 | 0.2332 | 0.7179 | 0.448322 | 0.7447 | 0.6664 | active |
| 8 | 871 | 1318 | 170 | 0.2427 | 0.6851 | 0.418029 | 0.7364 | 0.6628 | active |
| 9 | 825 | 1318 | 161 | 0.2391 | 0.7065 | 0.427525 | 0.7365 | 0.6626 | active |
| 10 | 831 | 1318 | 164 | 0.2541 | 0.6759 | 0.390332 | 0.7279 | 0.6524 | active |
| 11 | 835 | 1318 | 147 | 0.2518 | 0.7097 | 0.424819 | 0.7254 | 0.6567 | active |
| 12 | 869 | 1318 | 158 | 0.2507 | 0.7115 | 0.406781 | 0.7276 | 0.6531 | active |
| 13 | 939 | 1318 | 165 | 0.2497 | 0.7221 | 0.413600 | 0.7290 | 0.6560 | active |
| 14 | 828 | 1318 | 160 | 0.2551 | 0.7270 | 0.436603 | 0.7336 | 0.6478 | active |
| 15 | 883 | 1318 | 162 | 0.2684 | 0.6986 | 0.391896 | 0.7131 | 0.6430 | active |
| 16 | 829 | 1318 | 150 | 0.2556 | 0.7308 | 0.426378 | 0.7244 | 0.6514 | active |
| 17 | 837 | 1318 | 159 | 0.2708 | 0.6745 | 0.374779 | 0.7145 | 0.6418 | active |
| 18 | 855 | 1318 | 169 | 0.2802 | 0.7173 | 0.411315 | 0.7139 | 0.6276 | active |
| 19 | 839 | 1318 | 172 | 0.2786 | 0.7093 | 0.405255 | 0.7125 | 0.6328 | active |
| 20 | 829 | 1318 | 153 | 0.2752 | 0.7303 | 0.406097 | 0.7058 | 0.6362 | active |
| 21 | 918 | 1318 | 185 | 0.2649 | 0.6967 | 0.403726 | 0.7302 | 0.6418 | active |
| 22 | 832 | 1318 | 159 | 0.2907 | 0.6791 | 0.370087 | 0.7043 | 0.6223 | active |
| 23 | 829 | 1318 | 162 | 0.2853 | 0.6977 | 0.385208 | 0.7067 | 0.6277 | active |
| 24 | 855 | 1318 | 164 | 0.2778 | 0.7223 | 0.398441 | 0.7094 | 0.6321 | active |
| 25 | 826 | 1318 | 164 | 0.2788 | 0.7274 | 0.412414 | 0.7116 | 0.6303 | active |
| 26 | 800 | 1318 | 150 | 0.2826 | 0.7172 | 0.404113 | 0.7062 | 0.6314 | active |
| 27 | 871 | 1318 | 162 | 0.2777 | 0.7132 | 0.404837 | 0.7157 | 0.6304 | active |
| 28 | 913 | 1318 | 179 | 0.2848 | 0.7073 | 0.404745 | 0.7090 | 0.6315 | active |
| 29 | 881 | 1318 | 172 | 0.2879 | 0.7058 | 0.399741 | 0.7168 | 0.6219 | active |
| 30 | 794 | 1318 | 157 | 0.2728 | 0.7314 | 0.418612 | 0.7199 | 0.6355 | active |

**Finding.** 0 of 31 epochs came back degenerate. Final verdict: soft skeleton differs from the prediction -- clDice is measuring a real centreline.

If this says degenerate, `loss.w_cldice` has been buying nothing and the honest response is to record that here, not to retune the weight. The levers that would change it are a thicker `boundary_gt.line_width_px` (re-running steps 2 and 3) or an explicit connectivity metric at evaluation time.

