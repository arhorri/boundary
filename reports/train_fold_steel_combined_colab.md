# Training report -- fold_steel_combined on colab

Held-out dataset: **mixed(Steel1+Steel2)**. Generated 2026-09-26T07:36:29Z on colab (Tesla T4).

This file is keyed by run and host: `train_fold_steel_combined_colab.md`. The same fold trained on another host writes its own file beside this one rather than overwriting it -- two runs of one fold are two measurements, and they differ in GPU, worker count and I/O path.

- training mixture: the fold's full set, nothing excluded
- config hash `067fecc2e06a7d0a`, seed 0 (statistically reproducible; cudnn.benchmark picks algorithms by timing, so bitwise equality across runs is not claimed)
- 40 epochs, batch 64, 4 workers (configs/dataloader.yaml hosts.colab.num_workers, measured 2026-09-06T07:14:10Z)
- lr 0.0003 (encoder 2.9999999999999997e-05), weight decay 0.0001, warmup 2 epochs, grad clip 1.0
- pos_weight 6.515999794006348 (configs/fold_stats.yaml folds.fold_steel_combined.pos_weight)
- best epoch 21 by best-threshold Dice on `pooled (held-out dataset absent)` = 0.7065 at threshold 0.85
- validation threshold swept over 0.05..0.95 (19 points); fixed reference 0.50
- FiLM: disabled
- patience None
- boundary_gt.line_width_px 4 (the ground truth this run trained on -- NOT directly comparable to a report written under a different value: pos_weight and every boundary fraction move with it)

## Per-dataset validation metrics (the headline)

Validation is a mixture. The pooled row is a footnote; the held-out dataset's row is the measurement this fold exists to make.

Every row appears twice: at the fixed `train.threshold` and at the threshold that maximised Dice for that dataset. A single fixed threshold measures the model and the operating point together and reports the sum as if it were the model. `best.pt` is selected on the best-threshold Dice of the held-out dataset.

| epoch | dataset | thr | tiles | IoU | Dice | Precision | Recall | boundary-F | pred frac | true frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 21 (best) | Steel1 fixed | 0.50 | 95 | 0.3019 | 0.4638 | 0.3314 | 0.7724 | 0.6561 | 0.2202 | 0.0945 |
| 21 (best) | Steel1 **best** | 0.70 | 95 | 0.3236 | 0.4889 | 0.4020 | 0.6237 | 0.6846 | 0.1466 | 0.0945 |
| 21 (best) | Steel2 fixed | 0.50 | 126 | 0.5184 | 0.6828 | 0.5210 | 0.9901 | 0.8942 | 0.5149 | 0.2710 |
| 21 (best) | Steel2 **best** | 0.85 | 126 | 0.6141 | 0.7609 | 0.6705 | 0.8796 | 0.9450 | 0.3555 | 0.2710 |
| 21 (best) | _pooled (footnote)_ fixed | 0.50 | 221 | 0.4620 | 0.6320 | 0.4748 | 0.9448 | 0.8406 | 0.3882 | 0.1951 |
| 21 (best) | _pooled (footnote)_ **best** | 0.85 | 221 | 0.5462 | 0.7065 | 0.6413 | 0.7866 | 0.8936 | 0.2393 | 0.1951 |
| 39 (final) | Steel1 fixed | 0.50 | 95 | 0.3086 | 0.4717 | 0.3414 | 0.7625 | 0.6638 | 0.2110 | 0.0945 |
| 39 (final) | Steel1 **best** | 0.70 | 95 | 0.3277 | 0.4936 | 0.4087 | 0.6229 | 0.6879 | 0.1440 | 0.0945 |
| 39 (final) | Steel2 fixed | 0.50 | 126 | 0.5907 | 0.7427 | 0.6337 | 0.8969 | 0.9258 | 0.3835 | 0.2710 |
| 39 (final) | Steel2 **best** | 0.55 | 126 | 0.5908 | 0.7428 | 0.6432 | 0.8789 | 0.9262 | 0.3703 | 0.2710 |
| 39 (final) | _pooled (footnote)_ fixed | 0.50 | 221 | 0.5062 | 0.6721 | 0.5480 | 0.8689 | 0.8579 | 0.3094 | 0.1951 |
| 39 (final) | _pooled (footnote)_ **best** | 0.65 | 221 | 0.5146 | 0.6795 | 0.5911 | 0.7990 | 0.8677 | 0.2637 | 0.1951 |

## The chosen threshold, per epoch, per dataset

This table is a measurement, not bookkeeping. If the held-out dataset's optimal threshold sits far from the training-side datasets', that gap IS the domain shift, expressed in the units of the decision the downstream watershed has to make -- and one global threshold will not serve both.

| epoch | Steel1 | Steel2 | spread |
| --- | --- | --- | --- |
| 0 | 0.65 | 0.60 | 0.05 |
| 1 | 0.75 | 0.95 | 0.20 |
| 2 | 0.70 | 0.80 | 0.10 |
| 3 | 0.65 | 0.55 | 0.10 |
| 4 | 0.65 | 0.50 | 0.15 |
| 5 | 0.65 | 0.70 | 0.05 |
| 6 | 0.70 | 0.75 | 0.05 |
| 7 | 0.75 | 0.75 | 0.00 |
| 8 | 0.70 | 0.45 | 0.25 |
| 9 | 0.75 | 0.10 | 0.65 |
| 10 | 0.75 | 0.55 | 0.20 |
| 11 | 0.70 | 0.50 | 0.20 |
| 12 | 0.70 | 0.35 | 0.35 |
| 13 | 0.75 | 0.75 | 0.00 |
| 14 | 0.75 | 0.65 | 0.10 |
| 15 | 0.75 | 0.20 | 0.55 |
| 16 | 0.70 | 0.80 | 0.10 |
| 17 | 0.75 | 0.80 | 0.05 |
| 18 | 0.65 | 0.70 | 0.05 |
| 19 | 0.80 | 0.35 | 0.45 |
| 20 | 0.75 | 0.55 | 0.20 |
| 21 | 0.70 | 0.85 | 0.15 |
| 22 | 0.70 | 0.75 | 0.05 |
| 23 | 0.70 | 0.70 | 0.00 |
| 24 | 0.70 | 0.65 | 0.05 |
| 25 | 0.70 | 0.40 | 0.30 |
| 26 | 0.70 | 0.30 | 0.40 |
| 27 | 0.70 | 0.35 | 0.35 |
| 28 | 0.70 | 0.45 | 0.25 |
| 29 | 0.70 | 0.70 | 0.00 |
| 30 | 0.70 | 0.75 | 0.05 |
| 31 | 0.70 | 0.50 | 0.20 |
| 32 | 0.70 | 0.75 | 0.05 |
| 33 | 0.70 | 0.45 | 0.25 |
| 34 | 0.70 | 0.40 | 0.30 |
| 35 | 0.70 | 0.50 | 0.20 |
| 36 | 0.70 | 0.55 | 0.15 |
| 37 | 0.70 | 0.45 | 0.25 |
| 38 | 0.70 | 0.45 | 0.25 |
| 39 | 0.70 | 0.55 | 0.15 |

## Loss terms, separately

They differ by orders of magnitude, so the total alone does not say which one moved.

| epoch | train total | train BCE | train Dice | train clDice | val total | val BCE | val Dice | val clDice |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.5295 | 1.3162 | 0.8094 | 0.8078 | 4.7844 | 3.7507 | 0.6864 | 0.6946 |
| 1 | 2.3043 | 1.1212 | 0.8008 | 0.7646 | 2.6539 | 1.6237 | 0.6661 | 0.7282 |
| 2 | 2.0819 | 0.9985 | 0.7814 | 0.6040 | 2.0628 | 1.1263 | 0.6292 | 0.6145 |
| 3 | 1.9613 | 0.9394 | 0.7659 | 0.5120 | 2.0106 | 1.1267 | 0.6311 | 0.5057 |
| 4 | 1.8727 | 0.8986 | 0.7424 | 0.4635 | 2.1383 | 1.2628 | 0.6346 | 0.4817 |
| 5 | 1.8335 | 0.8627 | 0.7393 | 0.4630 | 1.7999 | 0.9541 | 0.6148 | 0.4618 |
| 6 | 1.7644 | 0.8204 | 0.7191 | 0.4498 | 1.6514 | 0.8556 | 0.5719 | 0.4480 |
| 7 | 1.7152 | 0.7763 | 0.7121 | 0.4535 | 1.5690 | 0.8049 | 0.5519 | 0.4246 |
| 8 | 1.6486 | 0.7294 | 0.6978 | 0.4429 | 1.6629 | 0.9012 | 0.5520 | 0.4195 |
| 9 | 1.6237 | 0.7136 | 0.6904 | 0.4394 | 2.2154 | 1.3964 | 0.5820 | 0.4740 |
| 10 | 1.5733 | 0.6846 | 0.6764 | 0.4246 | 1.5035 | 0.7915 | 0.5168 | 0.3902 |
| 11 | 1.5466 | 0.6719 | 0.6615 | 0.4266 | 1.5249 | 0.8088 | 0.5243 | 0.3837 |
| 12 | 1.5047 | 0.6451 | 0.6534 | 0.4125 | 1.5899 | 0.8878 | 0.5111 | 0.3820 |
| 13 | 1.4868 | 0.6336 | 0.6481 | 0.4101 | 1.3812 | 0.7035 | 0.5022 | 0.3510 |
| 14 | 1.4606 | 0.6209 | 0.6374 | 0.4044 | 1.4082 | 0.7352 | 0.4883 | 0.3692 |
| 15 | 1.4428 | 0.6110 | 0.6316 | 0.4006 | 1.7536 | 1.0507 | 0.5100 | 0.3856 |
| 16 | 1.4402 | 0.6129 | 0.6255 | 0.4037 | 1.3465 | 0.6828 | 0.4927 | 0.3420 |
| 17 | 1.4052 | 0.5999 | 0.6112 | 0.3883 | 1.3479 | 0.6892 | 0.4944 | 0.3286 |
| 18 | 1.3863 | 0.5969 | 0.6002 | 0.3784 | 1.3215 | 0.6786 | 0.4840 | 0.3178 |
| 19 | 1.3644 | 0.5797 | 0.5964 | 0.3766 | 1.4968 | 0.8539 | 0.4723 | 0.3411 |
| 20 | 1.3763 | 0.5946 | 0.5933 | 0.3768 | 1.3692 | 0.7415 | 0.4722 | 0.3111 |
| 21 | 1.3498 | 0.5691 | 0.5930 | 0.3756 | 1.3146 | 0.6724 | 0.4803 | 0.3239 |
| 22 | 1.3469 | 0.5850 | 0.5790 | 0.3658 | 1.2840 | 0.6661 | 0.4659 | 0.3041 |
| 23 | 1.3289 | 0.5690 | 0.5784 | 0.3630 | 1.2945 | 0.6786 | 0.4581 | 0.3156 |
| 24 | 1.3199 | 0.5610 | 0.5781 | 0.3616 | 1.3101 | 0.6985 | 0.4587 | 0.3058 |
| 25 | 1.3043 | 0.5568 | 0.5709 | 0.3531 | 1.3827 | 0.7684 | 0.4567 | 0.3152 |
| 26 | 1.3205 | 0.5655 | 0.5733 | 0.3632 | 1.5529 | 0.9184 | 0.4693 | 0.3304 |
| 27 | 1.2987 | 0.5557 | 0.5674 | 0.3512 | 1.4478 | 0.8212 | 0.4654 | 0.3223 |
| 28 | 1.2995 | 0.5587 | 0.5652 | 0.3510 | 1.3617 | 0.7522 | 0.4552 | 0.3088 |
| 29 | 1.2918 | 0.5571 | 0.5623 | 0.3448 | 1.2816 | 0.6859 | 0.4477 | 0.2960 |
| 30 | 1.2812 | 0.5586 | 0.5528 | 0.3397 | 1.2584 | 0.6626 | 0.4510 | 0.2896 |
| 31 | 1.2903 | 0.5560 | 0.5611 | 0.3464 | 1.3213 | 0.7283 | 0.4469 | 0.2922 |
| 32 | 1.2886 | 0.5525 | 0.5621 | 0.3479 | 1.2451 | 0.6559 | 0.4456 | 0.2874 |
| 33 | 1.2842 | 0.5529 | 0.5595 | 0.3436 | 1.3618 | 0.7609 | 0.4504 | 0.3010 |
| 34 | 1.2724 | 0.5416 | 0.5580 | 0.3455 | 1.4006 | 0.7938 | 0.4521 | 0.3092 |
| 35 | 1.2878 | 0.5606 | 0.5548 | 0.3448 | 1.3269 | 0.7345 | 0.4457 | 0.2934 |
| 36 | 1.2763 | 0.5504 | 0.5554 | 0.3409 | 1.3081 | 0.7167 | 0.4463 | 0.2902 |
| 37 | 1.2696 | 0.5325 | 0.5639 | 0.3464 | 1.3431 | 0.7474 | 0.4466 | 0.2983 |
| 38 | 1.2841 | 0.5582 | 0.5539 | 0.3440 | 1.3530 | 0.7575 | 0.4461 | 0.2989 |
| 39 | 1.2919 | 0.5540 | 0.5620 | 0.3517 | 1.3123 | 0.7221 | 0.4437 | 0.2929 |

## clDice diagnostic on real predictions

Step 5 measured clDice against perturbed ground truth. This measures it against what the model actually produces, which is soft and thicker than the target. `skeleton_delta` is `mean |soft_skeleton(sigmoid(logits)) - sigmoid(logits)|`: near zero means the soft skeleton is returning its input and the term is inert.

| epoch | skel(pred) | skel(true) | skel(pred) on true | t_prec | t_rec | skeleton delta | Dice | clDice | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1256 | 2381 | 177 | 0.2009 | 0.8433 | 0.810412 | 0.6825 | 0.6841 | active |
| 1 | 1431 | 2381 | 213 | 0.1578 | 0.8534 | 0.716808 | 0.6579 | 0.7428 | active |
| 2 | 939 | 2381 | 221 | 0.2352 | 0.8498 | 0.612908 | 0.6149 | 0.6419 | active |
| 3 | 781 | 2381 | 321 | 0.3750 | 0.7048 | 0.455238 | 0.6151 | 0.5252 | active |
| 4 | 500 | 2381 | 231 | 0.4302 | 0.6670 | 0.417665 | 0.6216 | 0.4987 | active |
| 5 | 395 | 2381 | 163 | 0.3958 | 0.7899 | 0.536532 | 0.6058 | 0.4828 | active |
| 6 | 435 | 2381 | 185 | 0.4067 | 0.8295 | 0.492791 | 0.5547 | 0.4675 | active |
| 7 | 467 | 2381 | 216 | 0.4441 | 0.8367 | 0.458242 | 0.5327 | 0.4345 | active |
| 8 | 490 | 2381 | 251 | 0.4904 | 0.7585 | 0.377156 | 0.5331 | 0.4236 | active |
| 9 | 515 | 2381 | 268 | 0.4892 | 0.6325 | 0.291936 | 0.5703 | 0.4850 | active |
| 10 | 501 | 2381 | 265 | 0.5059 | 0.8250 | 0.386935 | 0.4953 | 0.3905 | active |
| 11 | 510 | 2381 | 284 | 0.5267 | 0.7959 | 0.370204 | 0.5074 | 0.3849 | active |
| 12 | 558 | 2381 | 322 | 0.5443 | 0.7796 | 0.329979 | 0.4907 | 0.3791 | active |
| 13 | 529 | 2381 | 303 | 0.5415 | 0.8527 | 0.396964 | 0.4845 | 0.3550 | active |
| 14 | 450 | 2381 | 239 | 0.5078 | 0.8627 | 0.384200 | 0.4713 | 0.3765 | active |
| 15 | 507 | 2381 | 302 | 0.5625 | 0.7415 | 0.291564 | 0.4918 | 0.3838 | active |
| 16 | 472 | 2381 | 267 | 0.5404 | 0.8632 | 0.403822 | 0.4818 | 0.3466 | active |
| 17 | 561 | 2381 | 341 | 0.5640 | 0.8652 | 0.404675 | 0.4867 | 0.3320 | active |
| 18 | 574 | 2381 | 381 | 0.6112 | 0.8183 | 0.351571 | 0.4725 | 0.3147 | active |
| 19 | 467 | 2381 | 293 | 0.5847 | 0.8253 | 0.318218 | 0.4531 | 0.3369 | active |
| 20 | 603 | 2381 | 420 | 0.6370 | 0.8161 | 0.318193 | 0.4557 | 0.3066 | active |
| 21 | 494 | 2381 | 296 | 0.5609 | 0.8750 | 0.401130 | 0.4729 | 0.3273 | active |
| 22 | 552 | 2381 | 373 | 0.6175 | 0.8513 | 0.355244 | 0.4556 | 0.2978 | active |
| 23 | 481 | 2381 | 302 | 0.5915 | 0.8563 | 0.343115 | 0.4418 | 0.3136 | active |
| 24 | 498 | 2381 | 338 | 0.6245 | 0.8411 | 0.329159 | 0.4395 | 0.2984 | active |
| 25 | 537 | 2381 | 365 | 0.6236 | 0.8158 | 0.300996 | 0.4395 | 0.3088 | active |
| 26 | 529 | 2381 | 375 | 0.6399 | 0.7666 | 0.270712 | 0.4503 | 0.3223 | active |
| 27 | 539 | 2381 | 366 | 0.6257 | 0.8062 | 0.298695 | 0.4478 | 0.3133 | active |
| 28 | 522 | 2381 | 358 | 0.6338 | 0.8204 | 0.306657 | 0.4376 | 0.3011 | active |
| 29 | 500 | 2381 | 341 | 0.6335 | 0.8472 | 0.326418 | 0.4303 | 0.2883 | active |
| 30 | 561 | 2381 | 394 | 0.6406 | 0.8515 | 0.334206 | 0.4372 | 0.2821 | active |
| 31 | 571 | 2381 | 414 | 0.6602 | 0.8325 | 0.303795 | 0.4270 | 0.2803 | active |
| 32 | 518 | 2381 | 358 | 0.6403 | 0.8577 | 0.333094 | 0.4282 | 0.2798 | active |
| 33 | 506 | 2381 | 357 | 0.6494 | 0.8215 | 0.300112 | 0.4310 | 0.2899 | active |
| 34 | 501 | 2381 | 347 | 0.6375 | 0.8174 | 0.297127 | 0.4339 | 0.2991 | active |
| 35 | 559 | 2381 | 402 | 0.6560 | 0.8317 | 0.302473 | 0.4266 | 0.2819 | active |
| 36 | 550 | 2381 | 396 | 0.6574 | 0.8375 | 0.309027 | 0.4275 | 0.2787 | active |
| 37 | 508 | 2381 | 358 | 0.6461 | 0.8338 | 0.304923 | 0.4264 | 0.2872 | active |
| 38 | 514 | 2381 | 365 | 0.6508 | 0.8243 | 0.298024 | 0.4260 | 0.2874 | active |
| 39 | 522 | 2381 | 372 | 0.6522 | 0.8344 | 0.305483 | 0.4235 | 0.2819 | active |

**Finding.** 0 of 40 epochs came back degenerate. Final verdict: soft skeleton differs from the prediction -- clDice is measuring a real centreline.

If this says degenerate, `loss.w_cldice` has been buying nothing and the honest response is to record that here, not to retune the weight. The levers that would change it are a thicker `boundary_gt.line_width_px` (re-running steps 2 and 3) or an explicit connectivity metric at evaluation time.

