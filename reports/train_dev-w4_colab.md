# Training report -- dev-w4 on colab

Held-out dataset: **uhcs2**. Generated 2026-09-16T13:18:00Z on colab (Tesla T4).

This file is keyed by run and host: `train_dev-w4_colab.md`. The same fold trained on another host writes its own file beside this one rather than overwriting it -- two runs of one fold are two measurements, and they differ in GPU, worker count and I/O path.

- training mixture: the fold's full set, nothing excluded
- config hash `067fecc2e06a7d0a`, seed 0 (statistically reproducible; cudnn.benchmark picks algorithms by timing, so bitwise equality across runs is not claimed)
- 40 epochs, batch 64, 4 workers (configs/dataloader.yaml hosts.colab.num_workers, measured 2026-09-06T07:14:10Z)
- lr 0.0003 (encoder 2.9999999999999997e-05), weight decay 0.0001, warmup 2 epochs, grad clip 1.0
- pos_weight 7.111000061035156 (configs/fold_stats.yaml folds.dev.pos_weight)
- best epoch 22 by best-threshold Dice on `uhcs2` = 0.1474 at threshold 0.45
- validation threshold swept over 0.05..0.95 (19 points); fixed reference 0.50
- FiLM: disabled
- patience None
- boundary_gt.line_width_px 4 (the ground truth this run trained on -- NOT directly comparable to a report written under a different value: pos_weight and every boundary fraction move with it)

## Per-dataset validation metrics (the headline)

Validation is a mixture. The pooled row is a footnote; the held-out dataset's row is the measurement this fold exists to make.

Every row appears twice: at the fixed `train.threshold` and at the threshold that maximised Dice for that dataset. A single fixed threshold measures the model and the operating point together and reports the sum as if it were the model. `best.pt` is selected on the best-threshold Dice of the held-out dataset.

| epoch | dataset | thr | tiles | IoU | Dice | Precision | Recall | boundary-F | pred frac | true frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 22 (best) | Steel1 fixed | 0.50 | 189 | 0.1790 | 0.3036 | 0.1934 | 0.7057 | 0.5936 | 0.1417 | 0.0389 |
| 22 (best) | Steel1 **best** | 0.70 | 189 | 0.2012 | 0.3350 | 0.2521 | 0.4993 | 0.6278 | 0.0770 | 0.0389 |
| 22 (best) | uhcs2 (held out) fixed | 0.50 | 265 | 0.0793 | 0.1470 | 0.0857 | 0.5182 | 0.3143 | 0.3286 | 0.0543 |
| 22 (best) | uhcs2 (held out) **best** | 0.45 | 265 | 0.0795 | 0.1474 | 0.0851 | 0.5507 | 0.3171 | 0.3517 | 0.0543 |
| 22 (best) | _pooled (footnote)_ fixed | 0.50 | 454 | 0.1028 | 0.1864 | 0.1110 | 0.5815 | 0.3890 | 0.2508 | 0.0479 |
| 22 (best) | _pooled (footnote)_ **best** | 0.55 | 454 | 0.1033 | 0.1873 | 0.1132 | 0.5436 | 0.3878 | 0.2300 | 0.0479 |
| 39 (final) | Steel1 fixed | 0.50 | 189 | 0.1964 | 0.3283 | 0.2169 | 0.6748 | 0.6265 | 0.1209 | 0.0389 |
| 39 (final) | Steel1 **best** | 0.70 | 189 | 0.2121 | 0.3499 | 0.2722 | 0.4897 | 0.6438 | 0.0699 | 0.0389 |
| 39 (final) | uhcs2 (held out) fixed | 0.50 | 265 | 0.0748 | 0.1391 | 0.0810 | 0.4929 | 0.3010 | 0.3306 | 0.0543 |
| 39 (final) | uhcs2 (held out) **best** | 0.30 | 265 | 0.0759 | 0.1411 | 0.0797 | 0.6142 | 0.3103 | 0.4187 | 0.0543 |
| 39 (final) | _pooled (footnote)_ fixed | 0.50 | 454 | 0.1003 | 0.1823 | 0.1091 | 0.5544 | 0.3818 | 0.2433 | 0.0479 |
| 39 (final) | _pooled (footnote)_ **best** | 0.50 | 454 | 0.1003 | 0.1823 | 0.1091 | 0.5544 | 0.3818 | 0.2433 | 0.0479 |

## The chosen threshold, per epoch, per dataset

This table is a measurement, not bookkeeping. If the held-out dataset's optimal threshold sits far from the training-side datasets', that gap IS the domain shift, expressed in the units of the decision the downstream watershed has to make -- and one global threshold will not serve both.

| epoch | Steel1 | uhcs2 | spread |
| --- | --- | --- | --- |
| 0 | 0.65 | 0.05 | 0.60 |
| 1 | 0.65 | 0.55 | 0.10 |
| 2 | 0.55 | 0.50 | 0.05 |
| 3 | 0.50 | 0.55 | 0.05 |
| 4 | 0.55 | 0.45 | 0.10 |
| 5 | 0.65 | 0.35 | 0.30 |
| 6 | 0.60 | 0.40 | 0.20 |
| 7 | 0.75 | 0.35 | 0.40 |
| 8 | 0.70 | 0.35 | 0.35 |
| 9 | 0.70 | 0.45 | 0.25 |
| 10 | 0.65 | 0.40 | 0.25 |
| 11 | 0.60 | 0.40 | 0.20 |
| 12 | 0.75 | 0.40 | 0.35 |
| 13 | 0.70 | 0.40 | 0.30 |
| 14 | 0.65 | 0.40 | 0.25 |
| 15 | 0.75 | 0.40 | 0.35 |
| 16 | 0.65 | 0.40 | 0.25 |
| 17 | 0.60 | 0.35 | 0.25 |
| 18 | 0.70 | 0.35 | 0.35 |
| 19 | 0.75 | 0.40 | 0.35 |
| 20 | 0.70 | 0.35 | 0.35 |
| 21 | 0.65 | 0.35 | 0.30 |
| 22 | 0.70 | 0.45 | 0.25 |
| 23 | 0.70 | 0.35 | 0.35 |
| 24 | 0.60 | 0.35 | 0.25 |
| 25 | 0.65 | 0.30 | 0.35 |
| 26 | 0.65 | 0.35 | 0.30 |
| 27 | 0.65 | 0.35 | 0.30 |
| 28 | 0.65 | 0.35 | 0.30 |
| 29 | 0.70 | 0.30 | 0.40 |
| 30 | 0.70 | 0.35 | 0.35 |
| 31 | 0.70 | 0.35 | 0.35 |
| 32 | 0.65 | 0.35 | 0.30 |
| 33 | 0.65 | 0.35 | 0.30 |
| 34 | 0.70 | 0.35 | 0.35 |
| 35 | 0.70 | 0.35 | 0.35 |
| 36 | 0.70 | 0.30 | 0.40 |
| 37 | 0.65 | 0.30 | 0.35 |
| 38 | 0.70 | 0.30 | 0.40 |
| 39 | 0.70 | 0.30 | 0.40 |

**Final epoch.** uhcs2 (held out) wants 0.30; the others want Steel1 0.70 -- a gap of 0.40. That is a domain-shift finding: the held-out microscope needs a materially different operating point, so step 7 should set the threshold PER DATASET rather than globally.

## Loss terms, separately

They differ by orders of magnitude, so the total alone does not say which one moved.

| epoch | train total | train BCE | train Dice | train clDice | val total | val BCE | val Dice | val clDice |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.5009 | 1.3286 | 0.7882 | 0.7680 | 2.4045 | 1.0402 | 0.9173 | 0.8941 |
| 1 | 2.1217 | 1.0899 | 0.7567 | 0.5503 | 2.3071 | 1.0049 | 0.9042 | 0.7960 |
| 2 | 1.8990 | 0.9672 | 0.7247 | 0.4143 | 2.2676 | 0.9672 | 0.9076 | 0.7857 |
| 3 | 1.8100 | 0.9078 | 0.7052 | 0.3939 | 2.2133 | 0.9177 | 0.9040 | 0.7831 |
| 4 | 1.7088 | 0.8347 | 0.6841 | 0.3799 | 2.1679 | 0.8766 | 0.8923 | 0.7981 |
| 5 | 1.6012 | 0.7599 | 0.6516 | 0.3794 | 2.1115 | 0.8400 | 0.8769 | 0.7892 |
| 6 | 1.4889 | 0.6886 | 0.6111 | 0.3783 | 2.0465 | 0.8040 | 0.8511 | 0.7828 |
| 7 | 1.4279 | 0.6639 | 0.5797 | 0.3687 | 2.0567 | 0.8144 | 0.8509 | 0.7829 |
| 8 | 1.4052 | 0.6659 | 0.5590 | 0.3608 | 2.0139 | 0.7858 | 0.8404 | 0.7756 |
| 9 | 1.3738 | 0.6455 | 0.5498 | 0.3569 | 2.0306 | 0.8026 | 0.8390 | 0.7780 |
| 10 | 1.3671 | 0.6466 | 0.5435 | 0.3538 | 2.0082 | 0.7969 | 0.8287 | 0.7651 |
| 11 | 1.3442 | 0.6362 | 0.5346 | 0.3467 | 2.0087 | 0.8032 | 0.8236 | 0.7639 |
| 12 | 1.3285 | 0.6324 | 0.5264 | 0.3396 | 2.0599 | 0.8255 | 0.8444 | 0.7798 |
| 13 | 1.3394 | 0.6396 | 0.5287 | 0.3422 | 2.0405 | 0.8213 | 0.8345 | 0.7694 |
| 14 | 1.3196 | 0.6235 | 0.5260 | 0.3401 | 2.0257 | 0.8161 | 0.8300 | 0.7590 |
| 15 | 1.3129 | 0.6201 | 0.5236 | 0.3384 | 2.0588 | 0.8342 | 0.8402 | 0.7689 |
| 16 | 1.2906 | 0.6146 | 0.5131 | 0.3257 | 2.0259 | 0.8285 | 0.8209 | 0.7531 |
| 17 | 1.2891 | 0.6154 | 0.5114 | 0.3246 | 2.0546 | 0.8599 | 0.8171 | 0.7551 |
| 18 | 1.2898 | 0.6115 | 0.5140 | 0.3288 | 2.0174 | 0.8240 | 0.8187 | 0.7494 |
| 19 | 1.2842 | 0.6082 | 0.5124 | 0.3273 | 2.0339 | 0.8291 | 0.8277 | 0.7542 |
| 20 | 1.2813 | 0.6084 | 0.5104 | 0.3249 | 2.0501 | 0.8461 | 0.8277 | 0.7527 |
| 21 | 1.2832 | 0.6070 | 0.5128 | 0.3270 | 2.0271 | 0.8385 | 0.8139 | 0.7493 |
| 22 | 1.2786 | 0.6062 | 0.5103 | 0.3244 | 2.0324 | 0.8349 | 0.8230 | 0.7489 |
| 23 | 1.2661 | 0.6059 | 0.5028 | 0.3146 | 2.0480 | 0.8561 | 0.8177 | 0.7482 |
| 24 | 1.2706 | 0.6059 | 0.5054 | 0.3188 | 2.0344 | 0.8471 | 0.8131 | 0.7484 |
| 25 | 1.2588 | 0.5945 | 0.5047 | 0.3192 | 2.0508 | 0.8630 | 0.8139 | 0.7476 |
| 26 | 1.2550 | 0.5962 | 0.5017 | 0.3140 | 2.0258 | 0.8373 | 0.8146 | 0.7478 |
| 27 | 1.2551 | 0.6037 | 0.4961 | 0.3107 | 2.0636 | 0.8789 | 0.8140 | 0.7416 |
| 28 | 1.2494 | 0.5867 | 0.5039 | 0.3176 | 2.0428 | 0.8543 | 0.8154 | 0.7462 |
| 29 | 1.2411 | 0.5903 | 0.4961 | 0.3097 | 2.0417 | 0.8502 | 0.8194 | 0.7442 |
| 30 | 1.2580 | 0.5987 | 0.5011 | 0.3163 | 2.0378 | 0.8498 | 0.8170 | 0.7418 |
| 31 | 1.2363 | 0.5807 | 0.4993 | 0.3126 | 2.0494 | 0.8628 | 0.8159 | 0.7416 |
| 32 | 1.2455 | 0.5925 | 0.4971 | 0.3116 | 2.0597 | 0.8774 | 0.8118 | 0.7410 |
| 33 | 1.2492 | 0.5935 | 0.4990 | 0.3134 | 2.0362 | 0.8495 | 0.8150 | 0.7434 |
| 34 | 1.2468 | 0.5914 | 0.4988 | 0.3133 | 2.0357 | 0.8481 | 0.8163 | 0.7427 |
| 35 | 1.2398 | 0.5835 | 0.4993 | 0.3141 | 2.0508 | 0.8662 | 0.8142 | 0.7409 |
| 36 | 1.2303 | 0.5768 | 0.4980 | 0.3111 | 2.0420 | 0.8563 | 0.8146 | 0.7423 |
| 37 | 1.2234 | 0.5805 | 0.4915 | 0.3028 | 2.0406 | 0.8542 | 0.8149 | 0.7431 |
| 38 | 1.2300 | 0.5873 | 0.4911 | 0.3031 | 2.0375 | 0.8500 | 0.8157 | 0.7436 |
| 39 | 1.2507 | 0.5917 | 0.5010 | 0.3161 | 2.0523 | 0.8681 | 0.8136 | 0.7413 |

## clDice diagnostic on real predictions

Step 5 measured clDice against perturbed ground truth. This measures it against what the model actually produces, which is soft and thicker than the target. `skeleton_delta` is `mean |soft_skeleton(sigmoid(logits)) - sigmoid(logits)|`: near zero means the soft skeleton is returning its input and the term is inert.

| epoch | skel(pred) | skel(true) | skel(pred) on true | t_prec | t_rec | skeleton delta | Dice | clDice | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 3222 | 1072 | 184 | 0.0559 | 0.4938 | 0.474951 | 0.9249 | 0.9008 | active |
| 1 | 1069 | 1072 | 108 | 0.1133 | 0.6041 | 0.530580 | 0.9127 | 0.8145 | active |
| 2 | 930 | 1072 | 90 | 0.1245 | 0.5382 | 0.489114 | 0.9165 | 0.8093 | active |
| 3 | 944 | 1072 | 91 | 0.1309 | 0.5303 | 0.460337 | 0.9123 | 0.8039 | active |
| 4 | 1269 | 1072 | 113 | 0.1156 | 0.5310 | 0.418674 | 0.9037 | 0.8198 | active |
| 5 | 1261 | 1072 | 123 | 0.1191 | 0.5446 | 0.377765 | 0.8902 | 0.8124 | active |
| 6 | 1217 | 1072 | 121 | 0.1227 | 0.5116 | 0.304009 | 0.8672 | 0.8107 | active |
| 7 | 1431 | 1072 | 147 | 0.1205 | 0.5131 | 0.292478 | 0.8694 | 0.8111 | active |
| 8 | 1418 | 1072 | 145 | 0.1277 | 0.5114 | 0.272191 | 0.8593 | 0.8038 | active |
| 9 | 1469 | 1072 | 155 | 0.1268 | 0.5225 | 0.269426 | 0.8575 | 0.8033 | active |
| 10 | 1354 | 1072 | 146 | 0.1413 | 0.4965 | 0.247962 | 0.8474 | 0.7897 | active |
| 11 | 1394 | 1072 | 153 | 0.1479 | 0.4971 | 0.250098 | 0.8424 | 0.7860 | active |
| 12 | 1475 | 1072 | 155 | 0.1255 | 0.5114 | 0.258977 | 0.8599 | 0.8042 | active |
| 13 | 1468 | 1072 | 154 | 0.1327 | 0.5261 | 0.269915 | 0.8541 | 0.7965 | active |
| 14 | 1396 | 1072 | 143 | 0.1444 | 0.5261 | 0.281338 | 0.8503 | 0.7863 | active |
| 15 | 1420 | 1072 | 147 | 0.1337 | 0.5274 | 0.260187 | 0.8550 | 0.7939 | active |
| 16 | 1471 | 1072 | 158 | 0.1485 | 0.5203 | 0.252344 | 0.8409 | 0.7815 | active |
| 17 | 1349 | 1072 | 147 | 0.1582 | 0.4849 | 0.234816 | 0.8346 | 0.7779 | active |
| 18 | 1466 | 1072 | 160 | 0.1524 | 0.5231 | 0.256991 | 0.8387 | 0.7771 | active |
| 19 | 1456 | 1072 | 160 | 0.1424 | 0.5387 | 0.266581 | 0.8473 | 0.7848 | active |
| 20 | 1434 | 1072 | 157 | 0.1462 | 0.5318 | 0.268330 | 0.8472 | 0.7815 | active |
| 21 | 1446 | 1072 | 161 | 0.1553 | 0.4967 | 0.238472 | 0.8348 | 0.7787 | active |
| 22 | 1373 | 1072 | 152 | 0.1479 | 0.5186 | 0.249987 | 0.8438 | 0.7806 | active |
| 23 | 1487 | 1072 | 162 | 0.1524 | 0.5156 | 0.241758 | 0.8380 | 0.7774 | active |
| 24 | 1463 | 1072 | 162 | 0.1595 | 0.5160 | 0.248819 | 0.8321 | 0.7733 | active |
| 25 | 1520 | 1072 | 166 | 0.1559 | 0.5073 | 0.241121 | 0.8344 | 0.7761 | active |
| 26 | 1492 | 1072 | 163 | 0.1549 | 0.5253 | 0.253956 | 0.8349 | 0.7757 | active |
| 27 | 1475 | 1072 | 162 | 0.1584 | 0.5122 | 0.238477 | 0.8329 | 0.7716 | active |
| 28 | 1452 | 1072 | 156 | 0.1601 | 0.5168 | 0.253073 | 0.8341 | 0.7721 | active |
| 29 | 1449 | 1072 | 162 | 0.1576 | 0.5274 | 0.253232 | 0.8379 | 0.7704 | active |
| 30 | 1446 | 1072 | 159 | 0.1596 | 0.5165 | 0.241663 | 0.8356 | 0.7691 | active |
| 31 | 1448 | 1072 | 162 | 0.1588 | 0.5275 | 0.252344 | 0.8351 | 0.7699 | active |
| 32 | 1477 | 1072 | 165 | 0.1625 | 0.5141 | 0.245851 | 0.8318 | 0.7691 | active |
| 33 | 1511 | 1072 | 166 | 0.1590 | 0.5288 | 0.255472 | 0.8343 | 0.7712 | active |
| 34 | 1480 | 1072 | 162 | 0.1555 | 0.5340 | 0.257192 | 0.8365 | 0.7728 | active |
| 35 | 1506 | 1072 | 167 | 0.1594 | 0.5243 | 0.245229 | 0.8337 | 0.7692 | active |
| 36 | 1515 | 1072 | 169 | 0.1594 | 0.5220 | 0.243232 | 0.8334 | 0.7697 | active |
| 37 | 1512 | 1072 | 165 | 0.1602 | 0.5233 | 0.248545 | 0.8337 | 0.7697 | active |
| 38 | 1525 | 1072 | 166 | 0.1590 | 0.5245 | 0.250285 | 0.8346 | 0.7707 | active |
| 39 | 1498 | 1072 | 167 | 0.1605 | 0.5211 | 0.245565 | 0.8330 | 0.7691 | active |

**Finding.** 0 of 40 epochs came back degenerate. Final verdict: soft skeleton differs from the prediction -- clDice is measuring a real centreline.

If this says degenerate, `loss.w_cldice` has been buying nothing and the honest response is to record that here, not to retune the weight. The levers that would change it are a thicker `boundary_gt.line_width_px` (re-running steps 2 and 3) or an explicit connectivity metric at evaluation time.

