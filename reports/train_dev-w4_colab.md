# Training report -- dev-w4 on colab

Held-out dataset: **uhcs2**. Generated 2026-09-17T06:56:11Z on colab (Tesla T4).

This file is keyed by run and host: `train_dev-w4_colab.md`. The same fold trained on another host writes its own file beside this one rather than overwriting it -- two runs of one fold are two measurements, and they differ in GPU, worker count and I/O path.

- training mixture: the fold's full set, nothing excluded
- config hash `067fecc2e06a7d0a`, seed 0 (statistically reproducible; cudnn.benchmark picks algorithms by timing, so bitwise equality across runs is not claimed)
- 40 epochs, batch 64, 4 workers (configs/dataloader.yaml hosts.colab.num_workers, measured 2026-09-06T07:14:10Z)
- lr 0.0003 (encoder 2.9999999999999997e-05), weight decay 0.0001, warmup 2 epochs, grad clip 1.0
- pos_weight 7.111000061035156 (configs/fold_stats.yaml folds.dev.pos_weight)
- best epoch 20 by best-threshold Dice on `uhcs2` = 0.2618 at threshold 0.55
- validation threshold swept over 0.05..0.95 (19 points); fixed reference 0.50
- FiLM: disabled
- patience None
- boundary_gt.line_width_px 4 (the ground truth this run trained on -- NOT directly comparable to a report written under a different value: pos_weight and every boundary fraction move with it)

## Per-dataset validation metrics (the headline)

Validation is a mixture. The pooled row is a footnote; the held-out dataset's row is the measurement this fold exists to make.

Every row appears twice: at the fixed `train.threshold` and at the threshold that maximised Dice for that dataset. A single fixed threshold measures the model and the operating point together and reports the sum as if it were the model. `best.pt` is selected on the best-threshold Dice of the held-out dataset.

| epoch | dataset | thr | tiles | IoU | Dice | Precision | Recall | boundary-F | pred frac | true frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20 (best) | Steel1 fixed | 0.50 | 189 | 0.3108 | 0.4742 | 0.3343 | 0.8155 | 0.6647 | 0.2037 | 0.0835 |
| 20 (best) | Steel1 **best** | 0.75 | 189 | 0.3462 | 0.5143 | 0.4327 | 0.6339 | 0.7055 | 0.1224 | 0.0835 |
| 20 (best) | uhcs2 (held out) fixed | 0.50 | 265 | 0.1503 | 0.2613 | 0.1605 | 0.7029 | 0.3894 | 0.5058 | 0.1155 |
| 20 (best) | uhcs2 (held out) **best** | 0.55 | 265 | 0.1506 | 0.2618 | 0.1620 | 0.6807 | 0.3885 | 0.4852 | 0.1155 |
| 20 (best) | _pooled (footnote)_ fixed | 0.50 | 454 | 0.1863 | 0.3141 | 0.1993 | 0.7413 | 0.4604 | 0.3800 | 0.1022 |
| 20 (best) | _pooled (footnote)_ **best** | 0.65 | 454 | 0.1892 | 0.3182 | 0.2094 | 0.6620 | 0.4604 | 0.3230 | 0.1022 |
| 39 (final) | Steel1 fixed | 0.50 | 189 | 0.3171 | 0.4816 | 0.3414 | 0.8167 | 0.6731 | 0.1998 | 0.0835 |
| 39 (final) | Steel1 **best** | 0.75 | 189 | 0.3519 | 0.5206 | 0.4327 | 0.6532 | 0.7123 | 0.1261 | 0.0835 |
| 39 (final) | uhcs2 (held out) fixed | 0.50 | 265 | 0.1439 | 0.2516 | 0.1550 | 0.6685 | 0.3753 | 0.4982 | 0.1155 |
| 39 (final) | uhcs2 (held out) **best** | 0.40 | 265 | 0.1444 | 0.2523 | 0.1532 | 0.7148 | 0.3786 | 0.5390 | 0.1155 |
| 39 (final) | _pooled (footnote)_ fixed | 0.50 | 454 | 0.1824 | 0.3085 | 0.1964 | 0.7189 | 0.4524 | 0.3740 | 0.1022 |
| 39 (final) | _pooled (footnote)_ **best** | 0.60 | 454 | 0.1832 | 0.3096 | 0.2016 | 0.6675 | 0.4509 | 0.3384 | 0.1022 |

## The chosen threshold, per epoch, per dataset

This table is a measurement, not bookkeeping. If the held-out dataset's optimal threshold sits far from the training-side datasets', that gap IS the domain shift, expressed in the units of the decision the downstream watershed has to make -- and one global threshold will not serve both.

| epoch | Steel1 | uhcs2 | spread |
| --- | --- | --- | --- |
| 0 | 0.65 | 0.65 | 0.00 |
| 1 | 0.80 | 0.55 | 0.25 |
| 2 | 0.65 | 0.50 | 0.15 |
| 3 | 0.60 | 0.50 | 0.10 |
| 4 | 0.70 | 0.40 | 0.30 |
| 5 | 0.75 | 0.40 | 0.35 |
| 6 | 0.70 | 0.50 | 0.20 |
| 7 | 0.70 | 0.55 | 0.15 |
| 8 | 0.70 | 0.55 | 0.15 |
| 9 | 0.75 | 0.50 | 0.25 |
| 10 | 0.65 | 0.45 | 0.20 |
| 11 | 0.70 | 0.45 | 0.25 |
| 12 | 0.75 | 0.45 | 0.30 |
| 13 | 0.75 | 0.55 | 0.20 |
| 14 | 0.75 | 0.45 | 0.30 |
| 15 | 0.75 | 0.50 | 0.25 |
| 16 | 0.80 | 0.50 | 0.30 |
| 17 | 0.75 | 0.50 | 0.25 |
| 18 | 0.80 | 0.50 | 0.30 |
| 19 | 0.75 | 0.50 | 0.25 |
| 20 | 0.75 | 0.55 | 0.20 |
| 21 | 0.80 | 0.50 | 0.30 |
| 22 | 0.70 | 0.45 | 0.25 |
| 23 | 0.75 | 0.45 | 0.30 |
| 24 | 0.75 | 0.50 | 0.25 |
| 25 | 0.80 | 0.40 | 0.40 |
| 26 | 0.75 | 0.50 | 0.25 |
| 27 | 0.80 | 0.45 | 0.35 |
| 28 | 0.75 | 0.45 | 0.30 |
| 29 | 0.75 | 0.45 | 0.30 |
| 30 | 0.75 | 0.45 | 0.30 |
| 31 | 0.80 | 0.45 | 0.35 |
| 32 | 0.75 | 0.40 | 0.35 |
| 33 | 0.75 | 0.45 | 0.30 |
| 34 | 0.80 | 0.45 | 0.35 |
| 35 | 0.75 | 0.45 | 0.30 |
| 36 | 0.75 | 0.45 | 0.30 |
| 37 | 0.75 | 0.40 | 0.35 |
| 38 | 0.75 | 0.40 | 0.35 |
| 39 | 0.75 | 0.40 | 0.35 |

**Final epoch.** uhcs2 (held out) wants 0.40; the others want Steel1 0.75 -- a gap of 0.35. That is a domain-shift finding: the held-out microscope needs a materially different operating point, so step 7 should set the threshold PER DATASET rather than globally.

## Loss terms, separately

They differ by orders of magnitude, so the total alone does not say which one moved.

| epoch | train total | train BCE | train Dice | train clDice | val total | val BCE | val Dice | val clDice |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.4317 | 1.5174 | 0.6140 | 0.6004 | 2.5829 | 1.3653 | 0.8172 | 0.8009 |
| 1 | 1.9637 | 1.2048 | 0.5615 | 0.3948 | 2.6091 | 1.4605 | 0.8079 | 0.6813 |
| 2 | 1.7123 | 1.0578 | 0.5309 | 0.2472 | 2.4895 | 1.3719 | 0.8027 | 0.6297 |
| 3 | 1.6266 | 1.0041 | 0.5101 | 0.2249 | 2.4369 | 1.3292 | 0.7975 | 0.6205 |
| 4 | 1.5230 | 0.9297 | 0.4842 | 0.2184 | 2.3983 | 1.2964 | 0.7865 | 0.6308 |
| 5 | 1.4315 | 0.8672 | 0.4546 | 0.2196 | 2.3045 | 1.2327 | 0.7552 | 0.6332 |
| 6 | 1.3683 | 0.8249 | 0.4345 | 0.2177 | 2.2967 | 1.2423 | 0.7401 | 0.6286 |
| 7 | 1.3151 | 0.7973 | 0.4131 | 0.2094 | 2.3092 | 1.2743 | 0.7259 | 0.6182 |
| 8 | 1.2821 | 0.7889 | 0.3936 | 0.1992 | 2.2290 | 1.2031 | 0.7154 | 0.6209 |
| 9 | 1.2552 | 0.7682 | 0.3883 | 0.1973 | 2.2558 | 1.2383 | 0.7095 | 0.6161 |
| 10 | 1.2483 | 0.7677 | 0.3829 | 0.1953 | 2.2868 | 1.2715 | 0.7093 | 0.6119 |
| 11 | 1.2215 | 0.7513 | 0.3756 | 0.1892 | 2.2528 | 1.2437 | 0.7045 | 0.6091 |
| 12 | 1.2078 | 0.7480 | 0.3672 | 0.1851 | 2.2345 | 1.2283 | 0.7034 | 0.6057 |
| 13 | 1.2169 | 0.7564 | 0.3686 | 0.1839 | 2.2331 | 1.2228 | 0.7077 | 0.6054 |
| 14 | 1.1974 | 0.7382 | 0.3676 | 0.1831 | 2.2669 | 1.2666 | 0.6996 | 0.6013 |
| 15 | 1.1961 | 0.7385 | 0.3664 | 0.1824 | 2.2398 | 1.2398 | 0.7015 | 0.5972 |
| 16 | 1.1755 | 0.7313 | 0.3577 | 0.1730 | 2.3036 | 1.3007 | 0.7057 | 0.5944 |
| 17 | 1.1596 | 0.7202 | 0.3544 | 0.1700 | 2.2137 | 1.2210 | 0.6950 | 0.5955 |
| 18 | 1.1603 | 0.7170 | 0.3572 | 0.1723 | 2.2502 | 1.2536 | 0.7011 | 0.5910 |
| 19 | 1.1661 | 0.7222 | 0.3571 | 0.1736 | 2.2410 | 1.2520 | 0.6901 | 0.5978 |
| 20 | 1.1572 | 0.7173 | 0.3549 | 0.1699 | 2.2953 | 1.3100 | 0.6907 | 0.5892 |
| 21 | 1.1552 | 0.7136 | 0.3565 | 0.1703 | 2.2826 | 1.2870 | 0.6992 | 0.5926 |
| 22 | 1.1480 | 0.7129 | 0.3515 | 0.1673 | 2.2700 | 1.2890 | 0.6869 | 0.5883 |
| 23 | 1.1347 | 0.7090 | 0.3459 | 0.1598 | 2.2535 | 1.2759 | 0.6850 | 0.5852 |
| 24 | 1.1366 | 0.7074 | 0.3478 | 0.1629 | 2.2392 | 1.2563 | 0.6899 | 0.5859 |
| 25 | 1.1299 | 0.6988 | 0.3493 | 0.1637 | 2.2686 | 1.2837 | 0.6924 | 0.5850 |
| 26 | 1.1226 | 0.6972 | 0.3456 | 0.1594 | 2.2093 | 1.2293 | 0.6872 | 0.5857 |
| 27 | 1.1334 | 0.7134 | 0.3410 | 0.1580 | 2.3174 | 1.3358 | 0.6917 | 0.5798 |
| 28 | 1.1162 | 0.6866 | 0.3489 | 0.1615 | 2.2349 | 1.2575 | 0.6882 | 0.5782 |
| 29 | 1.1067 | 0.6895 | 0.3395 | 0.1554 | 2.2735 | 1.2937 | 0.6906 | 0.5783 |
| 30 | 1.1317 | 0.7041 | 0.3466 | 0.1621 | 2.2850 | 1.3062 | 0.6891 | 0.5795 |
| 31 | 1.1123 | 0.6842 | 0.3474 | 0.1614 | 2.2503 | 1.2711 | 0.6895 | 0.5794 |
| 32 | 1.1195 | 0.6967 | 0.3434 | 0.1589 | 2.3188 | 1.3452 | 0.6853 | 0.5766 |
| 33 | 1.1041 | 0.6853 | 0.3413 | 0.1551 | 2.2470 | 1.2710 | 0.6871 | 0.5778 |
| 34 | 1.1080 | 0.6873 | 0.3420 | 0.1573 | 2.2599 | 1.2819 | 0.6888 | 0.5782 |
| 35 | 1.1078 | 0.6833 | 0.3450 | 0.1590 | 2.3025 | 1.3284 | 0.6856 | 0.5770 |
| 36 | 1.0913 | 0.6698 | 0.3431 | 0.1569 | 2.2751 | 1.2995 | 0.6868 | 0.5774 |
| 37 | 1.0932 | 0.6802 | 0.3371 | 0.1518 | 2.2711 | 1.2958 | 0.6864 | 0.5780 |
| 38 | 1.0959 | 0.6850 | 0.3355 | 0.1509 | 2.2511 | 1.2734 | 0.6887 | 0.5780 |
| 39 | 1.1188 | 0.6927 | 0.3461 | 0.1600 | 2.3100 | 1.3343 | 0.6876 | 0.5762 |

## clDice diagnostic on real predictions

Step 5 measured clDice against perturbed ground truth. This measures it against what the model actually produces, which is soft and thicker than the target. `skeleton_delta` is `mean |soft_skeleton(sigmoid(logits)) - sigmoid(logits)|`: near zero means the soft skeleton is returning its input and the term is inert.

| epoch | skel(pred) | skel(true) | skel(pred) on true | t_prec | t_rec | skeleton delta | Dice | clDice | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2996 | 1318 | 323 | 0.1072 | 0.6862 | 0.597540 | 0.8284 | 0.8169 | active |
| 1 | 1060 | 1318 | 193 | 0.1884 | 0.7111 | 0.623966 | 0.8243 | 0.7080 | active |
| 2 | 790 | 1318 | 166 | 0.2337 | 0.7073 | 0.602222 | 0.8187 | 0.6656 | active |
| 3 | 713 | 1318 | 145 | 0.2455 | 0.7055 | 0.574163 | 0.8112 | 0.6613 | active |
| 4 | 864 | 1318 | 169 | 0.2267 | 0.6929 | 0.542754 | 0.8044 | 0.6746 | active |
| 5 | 808 | 1318 | 155 | 0.2254 | 0.6851 | 0.478845 | 0.7790 | 0.6760 | active |
| 6 | 837 | 1318 | 162 | 0.2338 | 0.6876 | 0.464792 | 0.7655 | 0.6694 | active |
| 7 | 830 | 1318 | 167 | 0.2425 | 0.6853 | 0.441442 | 0.7536 | 0.6593 | active |
| 8 | 924 | 1318 | 178 | 0.2401 | 0.6785 | 0.418469 | 0.7436 | 0.6646 | active |
| 9 | 892 | 1318 | 170 | 0.2412 | 0.6918 | 0.410313 | 0.7363 | 0.6610 | active |
| 10 | 860 | 1318 | 169 | 0.2503 | 0.6547 | 0.391638 | 0.7382 | 0.6579 | active |
| 11 | 900 | 1318 | 171 | 0.2583 | 0.6591 | 0.384121 | 0.7297 | 0.6521 | active |
| 12 | 884 | 1318 | 166 | 0.2500 | 0.6792 | 0.396395 | 0.7329 | 0.6563 | active |
| 13 | 906 | 1318 | 178 | 0.2498 | 0.7054 | 0.426785 | 0.7373 | 0.6541 | active |
| 14 | 916 | 1318 | 173 | 0.2585 | 0.6940 | 0.401819 | 0.7293 | 0.6478 | active |
| 15 | 905 | 1318 | 170 | 0.2585 | 0.6977 | 0.418430 | 0.7318 | 0.6488 | active |
| 16 | 879 | 1318 | 164 | 0.2581 | 0.7024 | 0.420474 | 0.7356 | 0.6452 | active |
| 17 | 912 | 1318 | 169 | 0.2639 | 0.6862 | 0.403860 | 0.7243 | 0.6478 | active |
| 18 | 887 | 1318 | 170 | 0.2628 | 0.7058 | 0.424427 | 0.7328 | 0.6418 | active |
| 19 | 965 | 1318 | 186 | 0.2619 | 0.6738 | 0.383026 | 0.7227 | 0.6501 | active |
| 20 | 912 | 1318 | 171 | 0.2665 | 0.6883 | 0.385330 | 0.7210 | 0.6431 | active |
| 21 | 966 | 1318 | 179 | 0.2596 | 0.6823 | 0.398630 | 0.7326 | 0.6484 | active |
| 22 | 954 | 1318 | 180 | 0.2746 | 0.6426 | 0.351773 | 0.7185 | 0.6420 | active |
| 23 | 957 | 1318 | 187 | 0.2763 | 0.6790 | 0.376726 | 0.7145 | 0.6355 | active |
| 24 | 927 | 1318 | 175 | 0.2754 | 0.6960 | 0.392047 | 0.7186 | 0.6331 | active |
| 25 | 954 | 1318 | 190 | 0.2724 | 0.6775 | 0.378334 | 0.7229 | 0.6353 | active |
| 26 | 947 | 1318 | 182 | 0.2779 | 0.6927 | 0.390741 | 0.7183 | 0.6326 | active |
| 27 | 924 | 1318 | 181 | 0.2764 | 0.6774 | 0.383606 | 0.7226 | 0.6319 | active |
| 28 | 887 | 1318 | 177 | 0.2814 | 0.6855 | 0.387629 | 0.7188 | 0.6282 | active |
| 29 | 892 | 1318 | 173 | 0.2803 | 0.6758 | 0.382304 | 0.7210 | 0.6299 | active |
| 30 | 900 | 1318 | 173 | 0.2784 | 0.6714 | 0.381323 | 0.7211 | 0.6328 | active |
| 31 | 942 | 1318 | 187 | 0.2794 | 0.6702 | 0.370569 | 0.7206 | 0.6305 | active |
| 32 | 963 | 1318 | 189 | 0.2824 | 0.6558 | 0.361163 | 0.7183 | 0.6303 | active |
| 33 | 938 | 1318 | 183 | 0.2813 | 0.6813 | 0.383738 | 0.7190 | 0.6293 | active |
| 34 | 938 | 1318 | 183 | 0.2783 | 0.6810 | 0.380412 | 0.7207 | 0.6308 | active |
| 35 | 939 | 1318 | 185 | 0.2808 | 0.6752 | 0.375187 | 0.7180 | 0.6301 | active |
| 36 | 940 | 1318 | 186 | 0.2814 | 0.6769 | 0.377964 | 0.7185 | 0.6293 | active |
| 37 | 935 | 1318 | 183 | 0.2812 | 0.6752 | 0.378439 | 0.7177 | 0.6300 | active |
| 38 | 943 | 1318 | 185 | 0.2810 | 0.6747 | 0.378134 | 0.7206 | 0.6293 | active |
| 39 | 947 | 1318 | 187 | 0.2803 | 0.6741 | 0.377705 | 0.7206 | 0.6298 | active |

**Finding.** 0 of 40 epochs came back degenerate. Final verdict: soft skeleton differs from the prediction -- clDice is measuring a real centreline.

If this says degenerate, `loss.w_cldice` has been buying nothing and the honest response is to record that here, not to retune the weight. The levers that would change it are a thicker `boundary_gt.line_width_px` (re-running steps 2 and 3) or an explicit connectivity metric at evaluation time.

