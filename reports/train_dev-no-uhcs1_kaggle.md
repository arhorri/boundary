# Training report -- dev-no-uhcs1 on kaggle (TRAINED WITHOUT uhcs1)

Held-out dataset: **uhcs2**. Generated 2026-09-07T22:37:42Z on kaggle (Tesla T4).

This file is keyed by run and host: `train_dev-no-uhcs1_kaggle.md`. The same fold trained on another host writes its own file beside this one rather than overwriting it -- two runs of one fold are two measurements, and they differ in GPU, worker count and I/O path.

- **training mixture: uhcs1 EXCLUDED.** Trained on ['MetalDam', 'Steel1'], validated unchanged on ['Steel1', 'uhcs2']. pos_weight is held at the fold's recorded 7.111 in both arms (this split alone would imply 6.897) so that the training data is the only thing that differs.
- config hash `4ddeffe0ba13f352`, seed 0 (statistically reproducible; cudnn.benchmark picks algorithms by timing, so bitwise equality across runs is not claimed)
- 40 epochs, batch 64, 2 workers (configs/dataloader.yaml hosts.kaggle.num_workers, NOT YET MEASURED on this host -- run notebooks/04 here)
- lr 0.0003 (encoder 2.9999999999999997e-05), weight decay 0.0001, warmup 2 epochs, grad clip 1.0
- pos_weight 7.111000061035156 (configs/fold_stats.yaml folds.dev.pos_weight)
- best epoch 19 by best-threshold Dice on `uhcs2` = 0.1405 at threshold 0.45
- validation threshold swept over 0.05..0.95 (19 points); fixed reference 0.50

## Per-dataset validation metrics (the headline)

Validation is a mixture. The pooled row is a footnote; the held-out dataset's row is the measurement this fold exists to make.

Every row appears twice: at the fixed `train.threshold` and at the threshold that maximised Dice for that dataset. A single fixed threshold measures the model and the operating point together and reports the sum as if it were the model. `best.pt` is selected on the best-threshold Dice of the held-out dataset.

| epoch | dataset | thr | tiles | IoU | Dice | Precision | Recall | boundary-F | pred frac | true frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 19 (best) | Steel1 fixed | 0.50 | 189 | 0.1795 | 0.3043 | 0.1937 | 0.7092 | 0.5972 | 0.1422 | 0.0389 |
| 19 (best) | Steel1 **best** | 0.75 | 189 | 0.2020 | 0.3360 | 0.2639 | 0.4626 | 0.6273 | 0.0681 | 0.0389 |
| 19 (best) | uhcs2 (held out) fixed | 0.50 | 265 | 0.0755 | 0.1403 | 0.0810 | 0.5263 | 0.3146 | 0.3531 | 0.0543 |
| 19 (best) | uhcs2 (held out) **best** | 0.45 | 265 | 0.0756 | 0.1405 | 0.0802 | 0.5658 | 0.3162 | 0.3830 | 0.0543 |
| 19 (best) | _pooled (footnote)_ fixed | 0.50 | 454 | 0.0988 | 0.1798 | 0.1061 | 0.5881 | 0.3876 | 0.2653 | 0.0479 |
| 19 (best) | _pooled (footnote)_ **best** | 0.60 | 454 | 0.1000 | 0.1819 | 0.1110 | 0.5029 | 0.3876 | 0.2169 | 0.0479 |
| 39 (final) | Steel1 fixed | 0.50 | 189 | 0.1969 | 0.3290 | 0.2180 | 0.6708 | 0.6278 | 0.1196 | 0.0389 |
| 39 (final) | Steel1 **best** | 0.70 | 189 | 0.2123 | 0.3502 | 0.2735 | 0.4867 | 0.6420 | 0.0691 | 0.0389 |
| 39 (final) | uhcs2 (held out) fixed | 0.50 | 265 | 0.0714 | 0.1332 | 0.0769 | 0.4988 | 0.3025 | 0.3525 | 0.0543 |
| 39 (final) | uhcs2 (held out) **best** | 0.35 | 265 | 0.0719 | 0.1342 | 0.0755 | 0.6035 | 0.3066 | 0.4344 | 0.0543 |
| 39 (final) | _pooled (footnote)_ fixed | 0.50 | 454 | 0.0964 | 0.1758 | 0.1044 | 0.5569 | 0.3799 | 0.2555 | 0.0479 |
| 39 (final) | _pooled (footnote)_ **best** | 0.55 | 454 | 0.0966 | 0.1761 | 0.1060 | 0.5195 | 0.3788 | 0.2345 | 0.0479 |

## The chosen threshold, per epoch, per dataset

This table is a measurement, not bookkeeping. If the held-out dataset's optimal threshold sits far from the training-side datasets', that gap IS the domain shift, expressed in the units of the decision the downstream watershed has to make -- and one global threshold will not serve both.

| epoch | Steel1 | uhcs2 | spread |
| --- | --- | --- | --- |
| 0 | 0.60 | 0.85 | 0.25 |
| 1 | 0.60 | 0.05 | 0.55 |
| 2 | 0.55 | 0.65 | 0.10 |
| 3 | 0.75 | 0.55 | 0.20 |
| 4 | 0.65 | 0.35 | 0.30 |
| 5 | 0.55 | 0.35 | 0.20 |
| 6 | 0.60 | 0.30 | 0.30 |
| 7 | 0.65 | 0.35 | 0.30 |
| 8 | 0.60 | 0.35 | 0.25 |
| 9 | 0.75 | 0.35 | 0.40 |
| 10 | 0.75 | 0.45 | 0.30 |
| 11 | 0.60 | 0.35 | 0.25 |
| 12 | 0.75 | 0.30 | 0.45 |
| 13 | 0.70 | 0.45 | 0.25 |
| 14 | 0.70 | 0.45 | 0.25 |
| 15 | 0.70 | 0.50 | 0.20 |
| 16 | 0.55 | 0.50 | 0.05 |
| 17 | 0.50 | 0.40 | 0.10 |
| 18 | 0.65 | 0.40 | 0.25 |
| 19 | 0.75 | 0.45 | 0.30 |
| 20 | 0.70 | 0.35 | 0.35 |
| 21 | 0.70 | 0.40 | 0.30 |
| 22 | 0.70 | 0.40 | 0.30 |
| 23 | 0.65 | 0.45 | 0.20 |
| 24 | 0.70 | 0.40 | 0.30 |
| 25 | 0.70 | 0.40 | 0.30 |
| 26 | 0.70 | 0.35 | 0.35 |
| 27 | 0.70 | 0.40 | 0.30 |
| 28 | 0.65 | 0.35 | 0.30 |
| 29 | 0.70 | 0.40 | 0.30 |
| 30 | 0.70 | 0.35 | 0.35 |
| 31 | 0.70 | 0.40 | 0.30 |
| 32 | 0.70 | 0.40 | 0.30 |
| 33 | 0.65 | 0.40 | 0.25 |
| 34 | 0.70 | 0.35 | 0.35 |
| 35 | 0.70 | 0.40 | 0.30 |
| 36 | 0.70 | 0.35 | 0.35 |
| 37 | 0.70 | 0.35 | 0.35 |
| 38 | 0.70 | 0.35 | 0.35 |
| 39 | 0.70 | 0.35 | 0.35 |

**Final epoch.** uhcs2 (held out) wants 0.35; the others want Steel1 0.70 -- a gap of 0.35. That is a domain-shift finding: the held-out microscope needs a materially different operating point, so step 7 should set the threshold PER DATASET rather than globally.

## Loss terms, separately

They differ by orders of magnitude, so the total alone does not say which one moved.

| epoch | train total | train BCE | train Dice | train clDice | val total | val BCE | val Dice | val clDice |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.5285 | 1.3746 | 0.7748 | 0.7583 | 2.4965 | 1.1357 | 0.9104 | 0.9007 |
| 1 | 2.1790 | 1.1385 | 0.7444 | 0.5921 | 2.3396 | 0.9880 | 0.9142 | 0.8750 |
| 2 | 1.9047 | 0.9756 | 0.7044 | 0.4495 | 2.2558 | 0.9361 | 0.9003 | 0.8389 |
| 3 | 1.7587 | 0.8791 | 0.6746 | 0.4100 | 2.2171 | 0.9046 | 0.8936 | 0.8378 |
| 4 | 1.5735 | 0.7570 | 0.6231 | 0.3867 | 2.1618 | 0.8680 | 0.8837 | 0.8202 |
| 5 | 1.5262 | 0.7268 | 0.6009 | 0.3970 | 2.1032 | 0.8378 | 0.8619 | 0.8071 |
| 6 | 1.4660 | 0.6992 | 0.5773 | 0.3790 | 2.0926 | 0.8419 | 0.8538 | 0.7937 |
| 7 | 1.4289 | 0.6837 | 0.5611 | 0.3682 | 2.0473 | 0.8004 | 0.8517 | 0.7904 |
| 8 | 1.4242 | 0.6726 | 0.5637 | 0.3758 | 2.0852 | 0.8475 | 0.8424 | 0.7905 |
| 9 | 1.4033 | 0.6727 | 0.5490 | 0.3632 | 2.1195 | 0.8779 | 0.8466 | 0.7898 |
| 10 | 1.3890 | 0.6725 | 0.5393 | 0.3544 | 2.0409 | 0.8149 | 0.8358 | 0.7804 |
| 11 | 1.3725 | 0.6594 | 0.5372 | 0.3518 | 2.0414 | 0.8215 | 0.8319 | 0.7761 |
| 12 | 1.3719 | 0.6605 | 0.5354 | 0.3519 | 2.0650 | 0.8415 | 0.8371 | 0.7729 |
| 13 | 1.3554 | 0.6529 | 0.5306 | 0.3438 | 2.0429 | 0.8293 | 0.8300 | 0.7672 |
| 14 | 1.3472 | 0.6520 | 0.5251 | 0.3403 | 2.0451 | 0.8325 | 0.8295 | 0.7662 |
| 15 | 1.3340 | 0.6429 | 0.5227 | 0.3367 | 2.0569 | 0.8406 | 0.8285 | 0.7757 |
| 16 | 1.3321 | 0.6406 | 0.5231 | 0.3367 | 2.0142 | 0.8100 | 0.8222 | 0.7641 |
| 17 | 1.3233 | 0.6425 | 0.5162 | 0.3294 | 2.0537 | 0.8508 | 0.8182 | 0.7693 |
| 18 | 1.3201 | 0.6380 | 0.5167 | 0.3307 | 1.9890 | 0.7893 | 0.8207 | 0.7580 |
| 19 | 1.3214 | 0.6418 | 0.5152 | 0.3291 | 1.9873 | 0.7874 | 0.8217 | 0.7565 |
| 20 | 1.3156 | 0.6291 | 0.5197 | 0.3335 | 2.0193 | 0.8105 | 0.8293 | 0.7590 |
| 21 | 1.3090 | 0.6280 | 0.5162 | 0.3296 | 2.0331 | 0.8360 | 0.8182 | 0.7578 |
| 22 | 1.2905 | 0.6251 | 0.5061 | 0.3185 | 2.0330 | 0.8282 | 0.8242 | 0.7611 |
| 23 | 1.2830 | 0.6130 | 0.5088 | 0.3223 | 2.0112 | 0.8184 | 0.8153 | 0.7551 |
| 24 | 1.2945 | 0.6219 | 0.5102 | 0.3247 | 2.0153 | 0.8114 | 0.8252 | 0.7574 |
| 25 | 1.3011 | 0.6280 | 0.5102 | 0.3260 | 2.0368 | 0.8393 | 0.8195 | 0.7559 |
| 26 | 1.2890 | 0.6208 | 0.5075 | 0.3214 | 2.0448 | 0.8432 | 0.8228 | 0.7577 |
| 27 | 1.2767 | 0.6184 | 0.5013 | 0.3140 | 2.0124 | 0.8196 | 0.8166 | 0.7522 |
| 28 | 1.2813 | 0.6197 | 0.5031 | 0.3171 | 2.0437 | 0.8488 | 0.8176 | 0.7545 |
| 29 | 1.2791 | 0.6183 | 0.5030 | 0.3157 | 2.0102 | 0.8145 | 0.8198 | 0.7519 |
| 30 | 1.2863 | 0.6085 | 0.5129 | 0.3297 | 2.0240 | 0.8258 | 0.8212 | 0.7540 |
| 31 | 1.2651 | 0.6089 | 0.5005 | 0.3115 | 2.0132 | 0.8173 | 0.8201 | 0.7515 |
| 32 | 1.2651 | 0.6042 | 0.5030 | 0.3159 | 2.0299 | 0.8325 | 0.8201 | 0.7545 |
| 33 | 1.2644 | 0.6073 | 0.5010 | 0.3123 | 2.0087 | 0.8179 | 0.8151 | 0.7513 |
| 34 | 1.2791 | 0.6129 | 0.5058 | 0.3209 | 2.0367 | 0.8446 | 0.8153 | 0.7534 |
| 35 | 1.2613 | 0.6087 | 0.4978 | 0.3096 | 2.0142 | 0.8236 | 0.8152 | 0.7507 |
| 36 | 1.2713 | 0.6097 | 0.5031 | 0.3169 | 2.0277 | 0.8333 | 0.8182 | 0.7525 |
| 37 | 1.2649 | 0.6055 | 0.5024 | 0.3140 | 2.0154 | 0.8223 | 0.8173 | 0.7516 |
| 38 | 1.2811 | 0.6157 | 0.5050 | 0.3208 | 2.0245 | 0.8311 | 0.8172 | 0.7525 |
| 39 | 1.2635 | 0.6131 | 0.4964 | 0.3078 | 2.0223 | 0.8302 | 0.8163 | 0.7516 |

## clDice diagnostic on real predictions

Step 5 measured clDice against perturbed ground truth. This measures it against what the model actually produces, which is soft and thicker than the target. `skeleton_delta` is `mean |soft_skeleton(sigmoid(logits)) - sigmoid(logits)|`: near zero means the soft skeleton is returning its input and the term is inert.

| epoch | skel(pred) | skel(true) | skel(pred) on true | t_prec | t_rec | skeleton delta | Dice | clDice | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 4109 | 1072 | 208 | 0.0501 | 0.6062 | 0.521762 | 0.9162 | 0.9082 | active |
| 1 | 2181 | 1072 | 131 | 0.0636 | 0.4577 | 0.434200 | 0.9232 | 0.8901 | active |
| 2 | 1562 | 1072 | 104 | 0.0828 | 0.5418 | 0.456682 | 0.9101 | 0.8618 | active |
| 3 | 1643 | 1072 | 111 | 0.0828 | 0.5657 | 0.441865 | 0.9048 | 0.8588 | active |
| 4 | 1877 | 1072 | 134 | 0.0939 | 0.5157 | 0.361223 | 0.8966 | 0.8458 | active |
| 5 | 1869 | 1072 | 134 | 0.1104 | 0.4511 | 0.281570 | 0.8806 | 0.8357 | active |
| 6 | 1837 | 1072 | 147 | 0.1236 | 0.4825 | 0.294631 | 0.8730 | 0.8183 | active |
| 7 | 1808 | 1072 | 146 | 0.1208 | 0.4914 | 0.286055 | 0.8710 | 0.8172 | active |
| 8 | 1942 | 1072 | 152 | 0.1295 | 0.4725 | 0.264599 | 0.8635 | 0.8159 | active |
| 9 | 1982 | 1072 | 155 | 0.1185 | 0.5254 | 0.296475 | 0.8663 | 0.8153 | active |
| 10 | 1806 | 1072 | 148 | 0.1203 | 0.5435 | 0.295072 | 0.8571 | 0.8139 | active |
| 11 | 1803 | 1072 | 147 | 0.1377 | 0.5150 | 0.277650 | 0.8507 | 0.8029 | active |
| 12 | 1885 | 1072 | 154 | 0.1318 | 0.5454 | 0.292974 | 0.8555 | 0.8000 | active |
| 13 | 1818 | 1072 | 147 | 0.1355 | 0.5458 | 0.277634 | 0.8480 | 0.7977 | active |
| 14 | 1885 | 1072 | 158 | 0.1337 | 0.5491 | 0.283164 | 0.8488 | 0.7989 | active |
| 15 | 1886 | 1072 | 157 | 0.1296 | 0.5371 | 0.274200 | 0.8469 | 0.8052 | active |
| 16 | 1694 | 1072 | 147 | 0.1486 | 0.5191 | 0.265298 | 0.8396 | 0.7926 | active |
| 17 | 1878 | 1072 | 152 | 0.1549 | 0.4804 | 0.238820 | 0.8357 | 0.7947 | active |
| 18 | 1738 | 1072 | 150 | 0.1462 | 0.5237 | 0.255749 | 0.8389 | 0.7895 | active |
| 19 | 1715 | 1072 | 150 | 0.1432 | 0.5478 | 0.261753 | 0.8387 | 0.7875 | active |
| 20 | 1779 | 1072 | 152 | 0.1399 | 0.5500 | 0.274947 | 0.8471 | 0.7918 | active |
| 21 | 1860 | 1072 | 158 | 0.1463 | 0.5378 | 0.257516 | 0.8362 | 0.7900 | active |
| 22 | 1870 | 1072 | 160 | 0.1412 | 0.5348 | 0.265002 | 0.8424 | 0.7924 | active |
| 23 | 1766 | 1072 | 155 | 0.1502 | 0.5346 | 0.254766 | 0.8320 | 0.7856 | active |
| 24 | 1760 | 1072 | 154 | 0.1412 | 0.5434 | 0.262102 | 0.8434 | 0.7905 | active |
| 25 | 1875 | 1072 | 162 | 0.1451 | 0.5426 | 0.263316 | 0.8382 | 0.7881 | active |
| 26 | 1912 | 1072 | 162 | 0.1472 | 0.5257 | 0.258884 | 0.8407 | 0.7873 | active |
| 27 | 1807 | 1072 | 156 | 0.1487 | 0.5442 | 0.260055 | 0.8333 | 0.7846 | active |
| 28 | 1919 | 1072 | 161 | 0.1512 | 0.5294 | 0.257171 | 0.8358 | 0.7849 | active |
| 29 | 1781 | 1072 | 155 | 0.1494 | 0.5447 | 0.265487 | 0.8377 | 0.7838 | active |
| 30 | 1881 | 1072 | 158 | 0.1444 | 0.5410 | 0.263546 | 0.8405 | 0.7892 | active |
| 31 | 1808 | 1072 | 157 | 0.1451 | 0.5546 | 0.266408 | 0.8384 | 0.7864 | active |
| 32 | 1900 | 1072 | 161 | 0.1451 | 0.5442 | 0.263842 | 0.8386 | 0.7876 | active |
| 33 | 1804 | 1072 | 156 | 0.1550 | 0.5351 | 0.256551 | 0.8321 | 0.7816 | active |
| 34 | 1942 | 1072 | 166 | 0.1514 | 0.5258 | 0.247456 | 0.8326 | 0.7843 | active |
| 35 | 1816 | 1072 | 159 | 0.1525 | 0.5335 | 0.250548 | 0.8326 | 0.7818 | active |
| 36 | 1877 | 1072 | 161 | 0.1494 | 0.5397 | 0.257177 | 0.8360 | 0.7842 | active |
| 37 | 1823 | 1072 | 160 | 0.1502 | 0.5435 | 0.260449 | 0.8351 | 0.7832 | active |
| 38 | 1871 | 1072 | 161 | 0.1506 | 0.5377 | 0.257573 | 0.8349 | 0.7836 | active |
| 39 | 1855 | 1072 | 161 | 0.1520 | 0.5384 | 0.255232 | 0.8335 | 0.7820 | active |

**Finding.** 0 of 40 epochs came back degenerate. Final verdict: soft skeleton differs from the prediction -- clDice is measuring a real centreline.

If this says degenerate, `loss.w_cldice` has been buying nothing and the honest response is to record that here, not to retune the weight. The levers that would change it are a thicker `boundary_gt.line_width_px` (re-running steps 2 and 3) or an explicit connectivity metric at evaluation time.

