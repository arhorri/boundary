# Training report -- dev on colab

Held-out dataset: **uhcs2**. Generated 2026-09-10T07:57:09Z on colab (Tesla T4).

This file is keyed by run and host: `train_dev_colab.md`. The same fold trained on another host writes its own file beside this one rather than overwriting it -- two runs of one fold are two measurements, and they differ in GPU, worker count and I/O path.

- training mixture: the fold's full set, nothing excluded
- config hash `6c4ff95a34f95e2e`, seed 0 (statistically reproducible; cudnn.benchmark picks algorithms by timing, so bitwise equality across runs is not claimed)
- 40 epochs, batch 64, 4 workers (configs/dataloader.yaml hosts.colab.num_workers, measured 2026-09-06T07:14:10Z)
- lr 0.0003 (encoder 2.9999999999999997e-05), weight decay 0.0001, warmup 2 epochs, grad clip 1.0
- pos_weight 7.111000061035156 (configs/fold_stats.yaml folds.dev.pos_weight)
- best epoch 12 by best-threshold Dice on `uhcs2` = 0.1489 at threshold 0.4
- validation threshold swept over 0.05..0.95 (19 points); fixed reference 0.50

## Per-dataset validation metrics (the headline)

Validation is a mixture. The pooled row is a footnote; the held-out dataset's row is the measurement this fold exists to make.

Every row appears twice: at the fixed `train.threshold` and at the threshold that maximised Dice for that dataset. A single fixed threshold measures the model and the operating point together and reports the sum as if it were the model. `best.pt` is selected on the best-threshold Dice of the held-out dataset.

| epoch | dataset | thr | tiles | IoU | Dice | Precision | Recall | boundary-F | pred frac | true frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12 (best) | Steel1 fixed | 0.50 | 189 | 0.1797 | 0.3046 | 0.1965 | 0.6775 | 0.5940 | 0.1340 | 0.0389 |
| 12 (best) | Steel1 **best** | 0.70 | 189 | 0.1955 | 0.3271 | 0.2488 | 0.4774 | 0.6131 | 0.0746 | 0.0389 |
| 12 (best) | uhcs2 (held out) fixed | 0.50 | 265 | 0.0798 | 0.1479 | 0.0865 | 0.5078 | 0.3182 | 0.3188 | 0.0543 |
| 12 (best) | uhcs2 (held out) **best** | 0.40 | 265 | 0.0804 | 0.1489 | 0.0853 | 0.5851 | 0.3238 | 0.3726 | 0.0543 |
| 12 (best) | _pooled (footnote)_ fixed | 0.50 | 454 | 0.1030 | 0.1868 | 0.1119 | 0.5651 | 0.3916 | 0.2418 | 0.0479 |
| 12 (best) | _pooled (footnote)_ **best** | 0.55 | 454 | 0.1033 | 0.1872 | 0.1140 | 0.5227 | 0.3899 | 0.2195 | 0.0479 |
| 39 (final) | Steel1 fixed | 0.50 | 189 | 0.1945 | 0.3257 | 0.2139 | 0.6823 | 0.6233 | 0.1240 | 0.0389 |
| 39 (final) | Steel1 **best** | 0.70 | 189 | 0.2118 | 0.3496 | 0.2673 | 0.5052 | 0.6444 | 0.0734 | 0.0389 |
| 39 (final) | uhcs2 (held out) fixed | 0.50 | 265 | 0.0751 | 0.1396 | 0.0812 | 0.4996 | 0.3014 | 0.3343 | 0.0543 |
| 39 (final) | uhcs2 (held out) **best** | 0.35 | 265 | 0.0761 | 0.1414 | 0.0802 | 0.5951 | 0.3101 | 0.4030 | 0.0543 |
| 39 (final) | _pooled (footnote)_ fixed | 0.50 | 454 | 0.1004 | 0.1824 | 0.1089 | 0.5613 | 0.3818 | 0.2468 | 0.0479 |
| 39 (final) | _pooled (footnote)_ **best** | 0.50 | 454 | 0.1004 | 0.1824 | 0.1089 | 0.5613 | 0.3818 | 0.2468 | 0.0479 |

## The chosen threshold, per epoch, per dataset

This table is a measurement, not bookkeeping. If the held-out dataset's optimal threshold sits far from the training-side datasets', that gap IS the domain shift, expressed in the units of the decision the downstream watershed has to make -- and one global threshold will not serve both.

| epoch | Steel1 | uhcs2 | spread |
| --- | --- | --- | --- |
| 0 | 0.65 | 0.05 | 0.60 |
| 1 | 0.60 | 0.60 | 0.00 |
| 2 | 0.55 | 0.60 | 0.05 |
| 3 | 0.50 | 0.50 | 0.00 |
| 4 | 0.55 | 0.45 | 0.10 |
| 5 | 0.65 | 0.30 | 0.35 |
| 6 | 0.70 | 0.35 | 0.35 |
| 7 | 0.65 | 0.35 | 0.30 |
| 8 | 0.75 | 0.40 | 0.35 |
| 9 | 0.70 | 0.40 | 0.30 |
| 10 | 0.70 | 0.35 | 0.35 |
| 11 | 0.65 | 0.40 | 0.25 |
| 12 | 0.70 | 0.40 | 0.30 |
| 13 | 0.70 | 0.40 | 0.30 |
| 14 | 0.55 | 0.45 | 0.10 |
| 15 | 0.70 | 0.45 | 0.25 |
| 16 | 0.70 | 0.45 | 0.25 |
| 17 | 0.60 | 0.40 | 0.20 |
| 18 | 0.70 | 0.45 | 0.25 |
| 19 | 0.70 | 0.35 | 0.35 |
| 20 | 0.70 | 0.45 | 0.25 |
| 21 | 0.60 | 0.45 | 0.15 |
| 22 | 0.70 | 0.45 | 0.25 |
| 23 | 0.70 | 0.45 | 0.25 |
| 24 | 0.60 | 0.35 | 0.25 |
| 25 | 0.70 | 0.35 | 0.35 |
| 26 | 0.70 | 0.40 | 0.30 |
| 27 | 0.65 | 0.40 | 0.25 |
| 28 | 0.65 | 0.35 | 0.30 |
| 29 | 0.70 | 0.40 | 0.30 |
| 30 | 0.70 | 0.40 | 0.30 |
| 31 | 0.70 | 0.40 | 0.30 |
| 32 | 0.65 | 0.35 | 0.30 |
| 33 | 0.65 | 0.35 | 0.30 |
| 34 | 0.70 | 0.40 | 0.30 |
| 35 | 0.70 | 0.40 | 0.30 |
| 36 | 0.70 | 0.35 | 0.35 |
| 37 | 0.70 | 0.35 | 0.35 |
| 38 | 0.70 | 0.40 | 0.30 |
| 39 | 0.70 | 0.35 | 0.35 |

**Final epoch.** uhcs2 (held out) wants 0.35; the others want Steel1 0.70 -- a gap of 0.35. That is a domain-shift finding: the held-out microscope needs a materially different operating point, so step 7 should set the threshold PER DATASET rather than globally.

## Loss terms, separately

They differ by orders of magnitude, so the total alone does not say which one moved.

| epoch | train total | train BCE | train Dice | train clDice | val total | val BCE | val Dice | val clDice |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.5013 | 1.3292 | 0.7882 | 0.7678 | 2.4000 | 1.0357 | 0.9167 | 0.8951 |
| 1 | 2.1202 | 1.0875 | 0.7562 | 0.5531 | 2.3241 | 1.0092 | 0.9054 | 0.8190 |
| 2 | 1.8975 | 0.9646 | 0.7242 | 0.4174 | 2.2706 | 0.9754 | 0.9041 | 0.7823 |
| 3 | 1.7981 | 0.8994 | 0.7028 | 0.3918 | 2.1908 | 0.8933 | 0.8961 | 0.8029 |
| 4 | 1.6836 | 0.8164 | 0.6772 | 0.3799 | 2.1301 | 0.8412 | 0.8893 | 0.7992 |
| 5 | 1.5394 | 0.7231 | 0.6260 | 0.3805 | 2.1064 | 0.8359 | 0.8725 | 0.7958 |
| 6 | 1.4589 | 0.6774 | 0.5927 | 0.3775 | 2.0437 | 0.7884 | 0.8593 | 0.7918 |
| 7 | 1.4161 | 0.6620 | 0.5702 | 0.3677 | 2.0543 | 0.8244 | 0.8407 | 0.7784 |
| 8 | 1.3965 | 0.6638 | 0.5527 | 0.3600 | 2.0391 | 0.8111 | 0.8402 | 0.7756 |
| 9 | 1.3713 | 0.6456 | 0.5472 | 0.3570 | 2.0354 | 0.8152 | 0.8321 | 0.7762 |
| 10 | 1.3637 | 0.6459 | 0.5413 | 0.3530 | 2.0363 | 0.8177 | 0.8339 | 0.7695 |
| 11 | 1.3402 | 0.6336 | 0.5339 | 0.3455 | 2.0022 | 0.7876 | 0.8325 | 0.7641 |
| 12 | 1.3329 | 0.6357 | 0.5268 | 0.3407 | 1.9969 | 0.7910 | 0.8244 | 0.7629 |
| 13 | 1.3418 | 0.6415 | 0.5291 | 0.3424 | 2.0292 | 0.8153 | 0.8323 | 0.7632 |
| 14 | 1.3151 | 0.6222 | 0.5242 | 0.3374 | 2.0724 | 0.8629 | 0.8236 | 0.7718 |
| 15 | 1.3139 | 0.6211 | 0.5239 | 0.3378 | 2.0467 | 0.8279 | 0.8323 | 0.7728 |
| 16 | 1.2950 | 0.6184 | 0.5134 | 0.3263 | 2.0386 | 0.8347 | 0.8256 | 0.7566 |
| 17 | 1.3032 | 0.6236 | 0.5150 | 0.3290 | 2.0635 | 0.8632 | 0.8204 | 0.7599 |
| 18 | 1.2934 | 0.6126 | 0.5161 | 0.3295 | 2.0211 | 0.8223 | 0.8227 | 0.7524 |
| 19 | 1.2905 | 0.6118 | 0.5141 | 0.3290 | 2.0380 | 0.8304 | 0.8267 | 0.7617 |
| 20 | 1.2834 | 0.6096 | 0.5114 | 0.3249 | 2.0539 | 0.8492 | 0.8265 | 0.7563 |
| 21 | 1.2796 | 0.6041 | 0.5126 | 0.3258 | 2.0231 | 0.8307 | 0.8167 | 0.7515 |
| 22 | 1.2769 | 0.6055 | 0.5096 | 0.3236 | 2.0282 | 0.8351 | 0.8187 | 0.7488 |
| 23 | 1.2584 | 0.6011 | 0.5010 | 0.3126 | 2.0289 | 0.8420 | 0.8138 | 0.7463 |
| 24 | 1.2672 | 0.6030 | 0.5052 | 0.3181 | 2.0353 | 0.8449 | 0.8156 | 0.7498 |
| 25 | 1.2587 | 0.5943 | 0.5047 | 0.3195 | 2.0468 | 0.8510 | 0.8201 | 0.7514 |
| 26 | 1.2509 | 0.5942 | 0.5004 | 0.3126 | 2.0141 | 0.8265 | 0.8145 | 0.7463 |
| 27 | 1.2636 | 0.6091 | 0.4981 | 0.3127 | 2.0390 | 0.8551 | 0.8113 | 0.7452 |
| 28 | 1.2497 | 0.5872 | 0.5036 | 0.3176 | 2.0294 | 0.8411 | 0.8143 | 0.7479 |
| 29 | 1.2335 | 0.5858 | 0.4943 | 0.3069 | 2.0388 | 0.8501 | 0.8162 | 0.7448 |
| 30 | 1.2507 | 0.5945 | 0.4995 | 0.3134 | 2.0401 | 0.8515 | 0.8165 | 0.7441 |
| 31 | 1.2375 | 0.5808 | 0.4996 | 0.3141 | 2.0481 | 0.8571 | 0.8185 | 0.7450 |
| 32 | 1.2491 | 0.5956 | 0.4974 | 0.3121 | 2.0581 | 0.8737 | 0.8124 | 0.7438 |
| 33 | 1.2477 | 0.5928 | 0.4989 | 0.3120 | 2.0321 | 0.8442 | 0.8157 | 0.7445 |
| 34 | 1.2424 | 0.5886 | 0.4978 | 0.3120 | 2.0378 | 0.8484 | 0.8172 | 0.7443 |
| 35 | 1.2425 | 0.5853 | 0.4998 | 0.3150 | 2.0480 | 0.8621 | 0.8142 | 0.7432 |
| 36 | 1.2262 | 0.5739 | 0.4976 | 0.3094 | 2.0339 | 0.8468 | 0.8151 | 0.7441 |
| 37 | 1.2301 | 0.5849 | 0.4926 | 0.3053 | 2.0354 | 0.8480 | 0.8152 | 0.7445 |
| 38 | 1.2286 | 0.5866 | 0.4907 | 0.3025 | 2.0263 | 0.8377 | 0.8160 | 0.7453 |
| 39 | 1.2443 | 0.5875 | 0.4997 | 0.3142 | 2.0487 | 0.8626 | 0.8146 | 0.7430 |

## clDice diagnostic on real predictions

Step 5 measured clDice against perturbed ground truth. This measures it against what the model actually produces, which is soft and thicker than the target. `skeleton_delta` is `mean |soft_skeleton(sigmoid(logits)) - sigmoid(logits)|`: near zero means the soft skeleton is returning its input and the term is inert.

| epoch | skel(pred) | skel(true) | skel(pred) on true | t_prec | t_rec | skeleton delta | Dice | clDice | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 3284 | 1072 | 185 | 0.0551 | 0.5021 | 0.470464 | 0.9233 | 0.9016 | active |
| 1 | 1141 | 1072 | 100 | 0.0955 | 0.5947 | 0.529813 | 0.9137 | 0.8387 | active |
| 2 | 1040 | 1072 | 99 | 0.1216 | 0.5629 | 0.497409 | 0.9135 | 0.8105 | active |
| 3 | 892 | 1072 | 88 | 0.1151 | 0.5683 | 0.460907 | 0.9060 | 0.8143 | active |
| 4 | 1126 | 1072 | 105 | 0.1139 | 0.5303 | 0.401882 | 0.8996 | 0.8216 | active |
| 5 | 1437 | 1072 | 136 | 0.1152 | 0.5158 | 0.337132 | 0.8865 | 0.8178 | active |
| 6 | 1476 | 1072 | 140 | 0.1177 | 0.5151 | 0.293794 | 0.8727 | 0.8138 | active |
| 7 | 1430 | 1072 | 145 | 0.1303 | 0.5023 | 0.272389 | 0.8583 | 0.8024 | active |
| 8 | 1321 | 1072 | 136 | 0.1280 | 0.5194 | 0.280329 | 0.8595 | 0.8012 | active |
| 9 | 1424 | 1072 | 148 | 0.1256 | 0.5129 | 0.262258 | 0.8524 | 0.8067 | active |
| 10 | 1386 | 1072 | 142 | 0.1326 | 0.4903 | 0.242386 | 0.8522 | 0.7988 | active |
| 11 | 1382 | 1072 | 140 | 0.1389 | 0.5151 | 0.267503 | 0.8510 | 0.7925 | active |
| 12 | 1439 | 1072 | 141 | 0.1371 | 0.5121 | 0.244059 | 0.8439 | 0.7939 | active |
| 13 | 1420 | 1072 | 143 | 0.1343 | 0.5195 | 0.264010 | 0.8535 | 0.7962 | active |
| 14 | 1313 | 1072 | 125 | 0.1434 | 0.4754 | 0.255394 | 0.8438 | 0.8008 | active |
| 15 | 1478 | 1072 | 138 | 0.1274 | 0.5073 | 0.264624 | 0.8546 | 0.8065 | active |
| 16 | 1417 | 1072 | 142 | 0.1413 | 0.5318 | 0.270545 | 0.8463 | 0.7885 | active |
| 17 | 1361 | 1072 | 133 | 0.1492 | 0.4684 | 0.234484 | 0.8387 | 0.7893 | active |
| 18 | 1461 | 1072 | 152 | 0.1435 | 0.5274 | 0.261705 | 0.8443 | 0.7849 | active |
| 19 | 1496 | 1072 | 151 | 0.1384 | 0.5021 | 0.258438 | 0.8484 | 0.7950 | active |
| 20 | 1470 | 1072 | 153 | 0.1429 | 0.5149 | 0.256028 | 0.8481 | 0.7879 | active |
| 21 | 1480 | 1072 | 152 | 0.1546 | 0.5009 | 0.248308 | 0.8368 | 0.7812 | active |
| 22 | 1412 | 1072 | 151 | 0.1454 | 0.5164 | 0.253087 | 0.8417 | 0.7848 | active |
| 23 | 1412 | 1072 | 151 | 0.1508 | 0.5168 | 0.250341 | 0.8354 | 0.7803 | active |
| 24 | 1522 | 1072 | 158 | 0.1550 | 0.5105 | 0.249242 | 0.8348 | 0.7787 | active |
| 25 | 1581 | 1072 | 167 | 0.1461 | 0.5251 | 0.252898 | 0.8409 | 0.7834 | active |
| 26 | 1475 | 1072 | 154 | 0.1516 | 0.5201 | 0.248962 | 0.8346 | 0.7797 | active |
| 27 | 1478 | 1072 | 155 | 0.1565 | 0.4966 | 0.232452 | 0.8305 | 0.7775 | active |
| 28 | 1506 | 1072 | 153 | 0.1510 | 0.5066 | 0.244652 | 0.8353 | 0.7818 | active |
| 29 | 1516 | 1072 | 157 | 0.1515 | 0.5157 | 0.246008 | 0.8361 | 0.7792 | active |
| 30 | 1445 | 1072 | 151 | 0.1528 | 0.5107 | 0.246016 | 0.8373 | 0.7781 | active |
| 31 | 1497 | 1072 | 159 | 0.1508 | 0.5238 | 0.253753 | 0.8389 | 0.7783 | active |
| 32 | 1488 | 1072 | 157 | 0.1560 | 0.5040 | 0.242807 | 0.8327 | 0.7768 | active |
| 33 | 1529 | 1072 | 160 | 0.1553 | 0.5160 | 0.247743 | 0.8356 | 0.7757 | active |
| 34 | 1518 | 1072 | 159 | 0.1507 | 0.5271 | 0.253029 | 0.8382 | 0.7780 | active |
| 35 | 1527 | 1072 | 162 | 0.1534 | 0.5176 | 0.245455 | 0.8352 | 0.7768 | active |
| 36 | 1533 | 1072 | 162 | 0.1533 | 0.5165 | 0.246140 | 0.8358 | 0.7769 | active |
| 37 | 1522 | 1072 | 159 | 0.1530 | 0.5220 | 0.251453 | 0.8359 | 0.7769 | active |
| 38 | 1539 | 1072 | 161 | 0.1528 | 0.5186 | 0.248103 | 0.8366 | 0.7770 | active |
| 39 | 1517 | 1072 | 161 | 0.1528 | 0.5161 | 0.245509 | 0.8358 | 0.7771 | active |

**Finding.** 0 of 40 epochs came back degenerate. Final verdict: soft skeleton differs from the prediction -- clDice is measuring a real centreline.

If this says degenerate, `loss.w_cldice` has been buying nothing and the honest response is to record that here, not to retune the weight. The levers that would change it are a thicker `boundary_gt.line_width_px` (re-running steps 2 and 3) or an explicit connectivity metric at evaluation time.

