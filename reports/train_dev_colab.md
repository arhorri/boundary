# Training report -- dev on colab

Held-out dataset: **uhcs2**. Generated 2026-09-09T13:27:16Z on colab (Tesla T4).

This file is keyed by run and host: `train_dev_colab.md`. The same fold trained on another host writes its own file beside this one rather than overwriting it -- two runs of one fold are two measurements, and they differ in GPU, worker count and I/O path.

- training mixture: the fold's full set, nothing excluded
- config hash `6c4ff95a34f95e2e`, seed 0 (statistically reproducible; cudnn.benchmark picks algorithms by timing, so bitwise equality across runs is not claimed)
- 40 epochs, batch 64, 4 workers (configs/dataloader.yaml hosts.colab.num_workers, measured 2026-09-06T07:14:10Z)
- lr 0.0003 (encoder 2.9999999999999997e-05), weight decay 0.0001, warmup 2 epochs, grad clip 1.0
- pos_weight 7.111000061035156 (configs/fold_stats.yaml folds.dev.pos_weight)
- best epoch 22 by best-threshold Dice on `uhcs2` = 0.1493 at threshold 0.4
- validation threshold swept over 0.05..0.95 (19 points); fixed reference 0.50

## Per-dataset validation metrics (the headline)

Validation is a mixture. The pooled row is a footnote; the held-out dataset's row is the measurement this fold exists to make.

Every row appears twice: at the fixed `train.threshold` and at the threshold that maximised Dice for that dataset. A single fixed threshold measures the model and the operating point together and reports the sum as if it were the model. `best.pt` is selected on the best-threshold Dice of the held-out dataset.

| epoch | dataset | thr | tiles | IoU | Dice | Precision | Recall | boundary-F | pred frac | true frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 22 (best) | Steel1 fixed | 0.50 | 189 | 0.1894 | 0.3184 | 0.2092 | 0.6669 | 0.6099 | 0.1239 | 0.0389 |
| 22 (best) | Steel1 **best** | 0.70 | 189 | 0.2037 | 0.3385 | 0.2574 | 0.4941 | 0.6283 | 0.0746 | 0.0389 |
| 22 (best) | uhcs2 (held out) fixed | 0.50 | 265 | 0.0799 | 0.1480 | 0.0864 | 0.5140 | 0.3176 | 0.3231 | 0.0543 |
| 22 (best) | uhcs2 (held out) **best** | 0.40 | 265 | 0.0807 | 0.1493 | 0.0855 | 0.5878 | 0.3241 | 0.3734 | 0.0543 |
| 22 (best) | _pooled (footnote)_ fixed | 0.50 | 454 | 0.1038 | 0.1881 | 0.1128 | 0.5657 | 0.3916 | 0.2402 | 0.0479 |
| 22 (best) | _pooled (footnote)_ **best** | 0.55 | 454 | 0.1039 | 0.1883 | 0.1146 | 0.5276 | 0.3894 | 0.2204 | 0.0479 |
| 39 (final) | Steel1 fixed | 0.50 | 189 | 0.1960 | 0.3278 | 0.2168 | 0.6721 | 0.6247 | 0.1205 | 0.0389 |
| 39 (final) | Steel1 **best** | 0.70 | 189 | 0.2121 | 0.3500 | 0.2690 | 0.5006 | 0.6434 | 0.0723 | 0.0389 |
| 39 (final) | uhcs2 (held out) fixed | 0.50 | 265 | 0.0762 | 0.1417 | 0.0830 | 0.4842 | 0.3045 | 0.3170 | 0.0543 |
| 39 (final) | uhcs2 (held out) **best** | 0.35 | 265 | 0.0778 | 0.1444 | 0.0823 | 0.5876 | 0.3152 | 0.3878 | 0.0543 |
| 39 (final) | _pooled (footnote)_ fixed | 0.50 | 454 | 0.1021 | 0.1853 | 0.1115 | 0.5477 | 0.3860 | 0.2352 | 0.0479 |
| 39 (final) | _pooled (footnote)_ **best** | 0.50 | 454 | 0.1021 | 0.1853 | 0.1115 | 0.5477 | 0.3860 | 0.2352 | 0.0479 |

## The chosen threshold, per epoch, per dataset

This table is a measurement, not bookkeeping. If the held-out dataset's optimal threshold sits far from the training-side datasets', that gap IS the domain shift, expressed in the units of the decision the downstream watershed has to make -- and one global threshold will not serve both.

| epoch | Steel1 | uhcs2 | spread |
| --- | --- | --- | --- |
| 0 | 0.65 | 0.05 | 0.60 |
| 1 | 0.65 | 0.65 | 0.00 |
| 2 | 0.55 | 0.55 | 0.00 |
| 3 | 0.50 | 0.55 | 0.05 |
| 4 | 0.65 | 0.45 | 0.20 |
| 5 | 0.65 | 0.45 | 0.20 |
| 6 | 0.70 | 0.45 | 0.25 |
| 7 | 0.70 | 0.40 | 0.30 |
| 8 | 0.60 | 0.40 | 0.20 |
| 9 | 0.65 | 0.45 | 0.20 |
| 10 | 0.65 | 0.40 | 0.25 |
| 11 | 0.65 | 0.45 | 0.20 |
| 12 | 0.65 | 0.45 | 0.20 |
| 13 | 0.70 | 0.35 | 0.35 |
| 14 | 0.65 | 0.40 | 0.25 |
| 15 | 0.65 | 0.45 | 0.20 |
| 16 | 0.70 | 0.40 | 0.30 |
| 17 | 0.65 | 0.40 | 0.25 |
| 18 | 0.70 | 0.50 | 0.20 |
| 19 | 0.70 | 0.35 | 0.35 |
| 20 | 0.70 | 0.40 | 0.30 |
| 21 | 0.70 | 0.40 | 0.30 |
| 22 | 0.70 | 0.40 | 0.30 |
| 23 | 0.70 | 0.40 | 0.30 |
| 24 | 0.70 | 0.35 | 0.35 |
| 25 | 0.70 | 0.35 | 0.35 |
| 26 | 0.70 | 0.35 | 0.35 |
| 27 | 0.70 | 0.35 | 0.35 |
| 28 | 0.65 | 0.35 | 0.30 |
| 29 | 0.70 | 0.35 | 0.35 |
| 30 | 0.70 | 0.35 | 0.35 |
| 31 | 0.70 | 0.35 | 0.35 |
| 32 | 0.70 | 0.35 | 0.35 |
| 33 | 0.70 | 0.35 | 0.35 |
| 34 | 0.70 | 0.35 | 0.35 |
| 35 | 0.70 | 0.35 | 0.35 |
| 36 | 0.70 | 0.35 | 0.35 |
| 37 | 0.70 | 0.35 | 0.35 |
| 38 | 0.70 | 0.35 | 0.35 |
| 39 | 0.70 | 0.35 | 0.35 |

**Final epoch.** uhcs2 (held out) wants 0.35; the others want Steel1 0.70 -- a gap of 0.35. That is a domain-shift finding: the held-out microscope needs a materially different operating point, so step 7 should set the threshold PER DATASET rather than globally.

## Loss terms, separately

They differ by orders of magnitude, so the total alone does not say which one moved.

| epoch | train total | train BCE | train Dice | train clDice | val total | val BCE | val Dice | val clDice |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.4987 | 1.3274 | 0.7880 | 0.7668 | 2.4282 | 1.0609 | 0.9190 | 0.8964 |
| 1 | 2.1164 | 1.0862 | 0.7556 | 0.5491 | 2.3259 | 1.0204 | 0.9037 | 0.8036 |
| 2 | 1.8901 | 0.9604 | 0.7230 | 0.4134 | 2.2566 | 0.9643 | 0.9044 | 0.7758 |
| 3 | 1.8019 | 0.9023 | 0.7028 | 0.3935 | 2.2174 | 0.9276 | 0.9028 | 0.7741 |
| 4 | 1.6881 | 0.8228 | 0.6762 | 0.3781 | 2.1890 | 0.9054 | 0.8943 | 0.7786 |
| 5 | 1.5539 | 0.7338 | 0.6310 | 0.3780 | 2.0865 | 0.8236 | 0.8722 | 0.7814 |
| 6 | 1.4654 | 0.6801 | 0.5980 | 0.3746 | 2.0557 | 0.8080 | 0.8588 | 0.7779 |
| 7 | 1.4200 | 0.6630 | 0.5746 | 0.3648 | 2.0266 | 0.7951 | 0.8454 | 0.7721 |
| 8 | 1.4005 | 0.6651 | 0.5567 | 0.3575 | 2.0067 | 0.7866 | 0.8353 | 0.7695 |
| 9 | 1.3812 | 0.6508 | 0.5519 | 0.3570 | 2.0053 | 0.7861 | 0.8352 | 0.7680 |
| 10 | 1.3678 | 0.6466 | 0.5453 | 0.3519 | 1.9930 | 0.7810 | 0.8311 | 0.7616 |
| 11 | 1.3469 | 0.6375 | 0.5368 | 0.3454 | 1.9923 | 0.7814 | 0.8325 | 0.7568 |
| 12 | 1.3327 | 0.6348 | 0.5285 | 0.3387 | 1.9805 | 0.7709 | 0.8303 | 0.7585 |
| 13 | 1.3319 | 0.6348 | 0.5276 | 0.3391 | 2.0118 | 0.8062 | 0.8271 | 0.7568 |
| 14 | 1.3176 | 0.6228 | 0.5261 | 0.3375 | 1.9965 | 0.7975 | 0.8213 | 0.7553 |
| 15 | 1.3164 | 0.6228 | 0.5250 | 0.3373 | 2.0054 | 0.8041 | 0.8242 | 0.7544 |
| 16 | 1.2969 | 0.6191 | 0.5147 | 0.3263 | 2.0245 | 0.8219 | 0.8259 | 0.7535 |
| 17 | 1.2911 | 0.6149 | 0.5135 | 0.3254 | 2.0327 | 0.8349 | 0.8201 | 0.7554 |
| 18 | 1.2908 | 0.6110 | 0.5154 | 0.3288 | 2.0083 | 0.8120 | 0.8215 | 0.7495 |
| 19 | 1.2828 | 0.6081 | 0.5117 | 0.3260 | 2.0138 | 0.8178 | 0.8204 | 0.7512 |
| 20 | 1.2819 | 0.6092 | 0.5108 | 0.3238 | 2.0073 | 0.8124 | 0.8200 | 0.7497 |
| 21 | 1.2803 | 0.6045 | 0.5132 | 0.3254 | 2.0089 | 0.8165 | 0.8183 | 0.7482 |
| 22 | 1.2805 | 0.6076 | 0.5106 | 0.3244 | 2.0046 | 0.8155 | 0.8154 | 0.7471 |
| 23 | 1.2702 | 0.6083 | 0.5041 | 0.3157 | 2.0026 | 0.8196 | 0.8118 | 0.7423 |
| 24 | 1.2728 | 0.6060 | 0.5072 | 0.3192 | 2.0048 | 0.8143 | 0.8179 | 0.7454 |
| 25 | 1.2549 | 0.5914 | 0.5045 | 0.3181 | 2.0211 | 0.8257 | 0.8203 | 0.7500 |
| 26 | 1.2610 | 0.6005 | 0.5029 | 0.3153 | 2.0009 | 0.8127 | 0.8163 | 0.7439 |
| 27 | 1.2616 | 0.6082 | 0.4978 | 0.3113 | 2.0137 | 0.8261 | 0.8164 | 0.7425 |
| 28 | 1.2501 | 0.5864 | 0.5052 | 0.3169 | 2.0073 | 0.8182 | 0.8163 | 0.7455 |
| 29 | 1.2412 | 0.5906 | 0.4963 | 0.3088 | 2.0120 | 0.8253 | 0.8160 | 0.7414 |
| 30 | 1.2559 | 0.5974 | 0.5011 | 0.3149 | 2.0201 | 0.8343 | 0.8144 | 0.7428 |
| 31 | 1.2401 | 0.5816 | 0.5016 | 0.3138 | 2.0202 | 0.8352 | 0.8151 | 0.7398 |
| 32 | 1.2546 | 0.5981 | 0.4995 | 0.3141 | 2.0302 | 0.8492 | 0.8122 | 0.7378 |
| 33 | 1.2462 | 0.5905 | 0.4996 | 0.3123 | 2.0166 | 0.8300 | 0.8158 | 0.7417 |
| 34 | 1.2447 | 0.5895 | 0.4991 | 0.3122 | 2.0166 | 0.8304 | 0.8159 | 0.7405 |
| 35 | 1.2339 | 0.5789 | 0.4990 | 0.3119 | 2.0267 | 0.8431 | 0.8137 | 0.7399 |
| 36 | 1.2311 | 0.5755 | 0.4999 | 0.3113 | 2.0178 | 0.8306 | 0.8160 | 0.7423 |
| 37 | 1.2345 | 0.5877 | 0.4938 | 0.3059 | 2.0115 | 0.8261 | 0.8147 | 0.7413 |
| 38 | 1.2219 | 0.5818 | 0.4898 | 0.3005 | 2.0063 | 0.8193 | 0.8162 | 0.7415 |
| 39 | 1.2484 | 0.5889 | 0.5018 | 0.3153 | 2.0291 | 0.8446 | 0.8141 | 0.7406 |

## clDice diagnostic on real predictions

Step 5 measured clDice against perturbed ground truth. This measures it against what the model actually produces, which is soft and thicker than the target. `skeleton_delta` is `mean |soft_skeleton(sigmoid(logits)) - sigmoid(logits)|`: near zero means the soft skeleton is returning its input and the term is inert.

| epoch | skel(pred) | skel(true) | skel(pred) on true | t_prec | t_rec | skeleton delta | Dice | clDice | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 3084 | 1072 | 169 | 0.0542 | 0.4769 | 0.462475 | 0.9262 | 0.9038 | active |
| 1 | 1147 | 1072 | 105 | 0.1056 | 0.6099 | 0.534340 | 0.9126 | 0.8253 | active |
| 2 | 961 | 1072 | 97 | 0.1236 | 0.5610 | 0.496835 | 0.9137 | 0.8071 | active |
| 3 | 988 | 1072 | 96 | 0.1380 | 0.5601 | 0.472757 | 0.9100 | 0.7944 | active |
| 4 | 1211 | 1072 | 120 | 0.1244 | 0.5643 | 0.444080 | 0.9045 | 0.8064 | active |
| 5 | 1289 | 1072 | 127 | 0.1251 | 0.5249 | 0.345475 | 0.8860 | 0.8080 | active |
| 6 | 1482 | 1072 | 143 | 0.1259 | 0.5433 | 0.324455 | 0.8753 | 0.8045 | active |
| 7 | 1365 | 1072 | 146 | 0.1330 | 0.5170 | 0.277334 | 0.8622 | 0.7973 | active |
| 8 | 1460 | 1072 | 147 | 0.1416 | 0.5010 | 0.263146 | 0.8522 | 0.7933 | active |
| 9 | 1409 | 1072 | 148 | 0.1350 | 0.5255 | 0.275795 | 0.8553 | 0.7956 | active |
| 10 | 1319 | 1072 | 139 | 0.1443 | 0.5043 | 0.254601 | 0.8485 | 0.7872 | active |
| 11 | 1341 | 1072 | 141 | 0.1443 | 0.5383 | 0.283455 | 0.8512 | 0.7850 | active |
| 12 | 1338 | 1072 | 139 | 0.1436 | 0.5309 | 0.266300 | 0.8487 | 0.7855 | active |
| 13 | 1483 | 1072 | 153 | 0.1421 | 0.5169 | 0.247829 | 0.8466 | 0.7871 | active |
| 14 | 1424 | 1072 | 146 | 0.1475 | 0.5030 | 0.239991 | 0.8394 | 0.7847 | active |
| 15 | 1369 | 1072 | 136 | 0.1420 | 0.5195 | 0.254224 | 0.8434 | 0.7889 | active |
| 16 | 1455 | 1072 | 146 | 0.1407 | 0.5399 | 0.271502 | 0.8472 | 0.7876 | active |
| 17 | 1474 | 1072 | 146 | 0.1438 | 0.5075 | 0.249572 | 0.8390 | 0.7888 | active |
| 18 | 1434 | 1072 | 147 | 0.1420 | 0.5456 | 0.278214 | 0.8435 | 0.7864 | active |
| 19 | 1514 | 1072 | 162 | 0.1464 | 0.5046 | 0.243368 | 0.8407 | 0.7844 | active |
| 20 | 1441 | 1072 | 152 | 0.1450 | 0.5394 | 0.266636 | 0.8409 | 0.7830 | active |
| 21 | 1494 | 1072 | 155 | 0.1479 | 0.5182 | 0.247367 | 0.8394 | 0.7836 | active |
| 22 | 1453 | 1072 | 156 | 0.1475 | 0.5166 | 0.243508 | 0.8377 | 0.7830 | active |
| 23 | 1436 | 1072 | 156 | 0.1543 | 0.5183 | 0.244888 | 0.8316 | 0.7761 | active |
| 24 | 1459 | 1072 | 155 | 0.1536 | 0.5242 | 0.254260 | 0.8365 | 0.7759 | active |
| 25 | 1569 | 1072 | 167 | 0.1488 | 0.5108 | 0.242072 | 0.8408 | 0.7811 | active |
| 26 | 1511 | 1072 | 159 | 0.1562 | 0.5148 | 0.245381 | 0.8357 | 0.7750 | active |
| 27 | 1475 | 1072 | 160 | 0.1548 | 0.5193 | 0.243484 | 0.8354 | 0.7740 | active |
| 28 | 1501 | 1072 | 159 | 0.1556 | 0.5068 | 0.241756 | 0.8360 | 0.7758 | active |
| 29 | 1467 | 1072 | 155 | 0.1563 | 0.5132 | 0.239483 | 0.8347 | 0.7738 | active |
| 30 | 1534 | 1072 | 163 | 0.1559 | 0.5124 | 0.236716 | 0.8333 | 0.7746 | active |
| 31 | 1503 | 1072 | 164 | 0.1564 | 0.5253 | 0.247255 | 0.8347 | 0.7721 | active |
| 32 | 1480 | 1072 | 163 | 0.1590 | 0.5131 | 0.236444 | 0.8318 | 0.7703 | active |
| 33 | 1509 | 1072 | 164 | 0.1586 | 0.5123 | 0.243067 | 0.8353 | 0.7716 | active |
| 34 | 1490 | 1072 | 161 | 0.1549 | 0.5255 | 0.245303 | 0.8359 | 0.7731 | active |
| 35 | 1528 | 1072 | 167 | 0.1577 | 0.5132 | 0.233124 | 0.8333 | 0.7714 | active |
| 36 | 1541 | 1072 | 168 | 0.1555 | 0.5172 | 0.240747 | 0.8358 | 0.7731 | active |
| 37 | 1516 | 1072 | 163 | 0.1572 | 0.5156 | 0.240381 | 0.8342 | 0.7724 | active |
| 38 | 1508 | 1072 | 162 | 0.1569 | 0.5156 | 0.246230 | 0.8363 | 0.7727 | active |
| 39 | 1519 | 1072 | 165 | 0.1565 | 0.5160 | 0.237481 | 0.8339 | 0.7724 | active |

**Finding.** 0 of 40 epochs came back degenerate. Final verdict: soft skeleton differs from the prediction -- clDice is measuring a real centreline.

If this says degenerate, `loss.w_cldice` has been buying nothing and the honest response is to record that here, not to retune the weight. The levers that would change it are a thicker `boundary_gt.line_width_px` (re-running steps 2 and 3) or an explicit connectivity metric at evaluation time.

