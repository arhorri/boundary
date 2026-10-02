# Training report -- fold_steel_combined_modea on colab

Held-out dataset: **mixed(Steel1+Steel2)**. Generated 2026-10-02T19:44:58Z on colab (Tesla T4).

This file is keyed by run and host: `train_fold_steel_combined_modea_colab.md`. The same fold trained on another host writes its own file beside this one rather than overwriting it -- two runs of one fold are two measurements, and they differ in GPU, worker count and I/O path.

- training mixture: the fold's full set, nothing excluded
- config hash `067fecc2e06a7d0a`, seed 0 (statistically reproducible; cudnn.benchmark picks algorithms by timing, so bitwise equality across runs is not claimed)
- 40 epochs, batch 64, 4 workers (configs/dataloader.yaml hosts.colab.num_workers, measured 2026-09-06T07:14:10Z)
- lr 0.0003 (encoder 2.9999999999999997e-05), weight decay 0.0001, warmup 2 epochs, grad clip 1.0
- pos_weight 7.692999839782715 (configs/fold_stats.yaml folds.fold_steel_combined_modea.pos_weight)
- best epoch 32 by best-threshold Dice on `pooled (held-out dataset absent)` = 0.6606 at threshold 0.8
- validation threshold swept over 0.05..0.95 (19 points); fixed reference 0.50
- FiLM: disabled
- patience None
- boundary_gt.line_width_px 4 (the ground truth this run trained on -- NOT directly comparable to a report written under a different value: pos_weight and every boundary fraction move with it)

## Per-dataset validation metrics (the headline)

Validation is a mixture. The pooled row is a footnote; the held-out dataset's row is the measurement this fold exists to make.

Every row appears twice: at the fixed `train.threshold` and at the threshold that maximised Dice for that dataset. A single fixed threshold measures the model and the operating point together and reports the sum as if it were the model. `best.pt` is selected on the best-threshold Dice of the held-out dataset.

| epoch | dataset | thr | tiles | IoU | Dice | Precision | Recall | boundary-F | pred frac | true frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 (best) | Steel1 fixed | 0.50 | 95 | 0.2970 | 0.4580 | 0.3196 | 0.8074 | 0.6495 | 0.2387 | 0.0945 |
| 32 (best) | Steel1 **best** | 0.75 | 95 | 0.3285 | 0.4946 | 0.4070 | 0.6303 | 0.6914 | 0.1464 | 0.0945 |
| 32 (best) | Steel2 fixed | 0.50 | 126 | 0.5152 | 0.6800 | 0.5350 | 0.9328 | 0.8800 | 0.3303 | 0.1894 |
| 32 (best) | Steel2 **best** | 0.80 | 126 | 0.5724 | 0.7280 | 0.6744 | 0.7910 | 0.9162 | 0.2222 | 0.1894 |
| 32 (best) | _pooled (footnote)_ fixed | 0.50 | 221 | 0.4364 | 0.6077 | 0.4591 | 0.8986 | 0.8061 | 0.2909 | 0.1486 |
| 32 (best) | _pooled (footnote)_ **best** | 0.80 | 221 | 0.4932 | 0.6606 | 0.6032 | 0.7299 | 0.8521 | 0.1798 | 0.1486 |
| 39 (final) | Steel1 fixed | 0.50 | 95 | 0.3013 | 0.4630 | 0.3257 | 0.8006 | 0.6553 | 0.2323 | 0.0945 |
| 39 (final) | Steel1 **best** | 0.75 | 95 | 0.3291 | 0.4952 | 0.4108 | 0.6234 | 0.6917 | 0.1434 | 0.0945 |
| 39 (final) | Steel2 fixed | 0.50 | 126 | 0.5395 | 0.7009 | 0.5768 | 0.8931 | 0.8918 | 0.2933 | 0.1894 |
| 39 (final) | Steel2 **best** | 0.70 | 126 | 0.5705 | 0.7265 | 0.6677 | 0.7966 | 0.9138 | 0.2260 | 0.1894 |
| 39 (final) | _pooled (footnote)_ fixed | 0.50 | 221 | 0.4498 | 0.6205 | 0.4829 | 0.8678 | 0.8122 | 0.2671 | 0.1486 |
| 39 (final) | _pooled (footnote)_ **best** | 0.75 | 221 | 0.4841 | 0.6524 | 0.5937 | 0.7238 | 0.8430 | 0.1812 | 0.1486 |

## The chosen threshold, per epoch, per dataset

This table is a measurement, not bookkeeping. If the held-out dataset's optimal threshold sits far from the training-side datasets', that gap IS the domain shift, expressed in the units of the decision the downstream watershed has to make -- and one global threshold will not serve both.

| epoch | Steel1 | Steel2 | spread |
| --- | --- | --- | --- |
| 0 | 0.70 | 0.65 | 0.05 |
| 1 | 0.80 | 0.80 | 0.00 |
| 2 | 0.75 | 0.90 | 0.15 |
| 3 | 0.70 | 0.70 | 0.00 |
| 4 | 0.60 | 0.60 | 0.00 |
| 5 | 0.65 | 0.60 | 0.05 |
| 6 | 0.75 | 0.75 | 0.00 |
| 7 | 0.80 | 0.65 | 0.15 |
| 8 | 0.75 | 0.60 | 0.15 |
| 9 | 0.80 | 0.85 | 0.05 |
| 10 | 0.75 | 0.70 | 0.05 |
| 11 | 0.75 | 0.55 | 0.20 |
| 12 | 0.75 | 0.60 | 0.15 |
| 13 | 0.75 | 0.80 | 0.05 |
| 14 | 0.75 | 0.80 | 0.05 |
| 15 | 0.75 | 0.55 | 0.20 |
| 16 | 0.75 | 0.85 | 0.10 |
| 17 | 0.75 | 0.80 | 0.05 |
| 18 | 0.75 | 0.70 | 0.05 |
| 19 | 0.75 | 0.60 | 0.15 |
| 20 | 0.80 | 0.65 | 0.15 |
| 21 | 0.75 | 0.70 | 0.05 |
| 22 | 0.75 | 0.85 | 0.10 |
| 23 | 0.75 | 0.80 | 0.05 |
| 24 | 0.75 | 0.85 | 0.10 |
| 25 | 0.75 | 0.50 | 0.25 |
| 26 | 0.75 | 0.70 | 0.05 |
| 27 | 0.75 | 0.70 | 0.05 |
| 28 | 0.75 | 0.70 | 0.05 |
| 29 | 0.75 | 0.75 | 0.00 |
| 30 | 0.75 | 0.70 | 0.05 |
| 31 | 0.75 | 0.65 | 0.10 |
| 32 | 0.75 | 0.80 | 0.05 |
| 33 | 0.75 | 0.70 | 0.05 |
| 34 | 0.75 | 0.70 | 0.05 |
| 35 | 0.75 | 0.70 | 0.05 |
| 36 | 0.75 | 0.70 | 0.05 |
| 37 | 0.75 | 0.70 | 0.05 |
| 38 | 0.75 | 0.70 | 0.05 |
| 39 | 0.75 | 0.70 | 0.05 |

## Loss terms, separately

They differ by orders of magnitude, so the total alone does not say which one moved.

| epoch | train total | train BCE | train Dice | train clDice | val total | val BCE | val Dice | val clDice |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.5851 | 1.3577 | 0.8187 | 0.8175 | 4.4874 | 3.3675 | 0.7448 | 0.7501 |
| 1 | 2.3514 | 1.1566 | 0.8080 | 0.7734 | 2.2660 | 1.2045 | 0.6995 | 0.7239 |
| 2 | 2.1105 | 1.0218 | 0.7845 | 0.6083 | 2.0571 | 1.0837 | 0.6660 | 0.6149 |
| 3 | 2.0073 | 0.9715 | 0.7720 | 0.5277 | 1.8037 | 0.9330 | 0.6375 | 0.4666 |
| 4 | 1.9014 | 0.9167 | 0.7485 | 0.4724 | 1.8572 | 1.0071 | 0.6410 | 0.4184 |
| 5 | 1.8493 | 0.8804 | 0.7412 | 0.4555 | 1.7170 | 0.8999 | 0.6133 | 0.4077 |
| 6 | 1.7901 | 0.8422 | 0.7243 | 0.4471 | 1.5795 | 0.7848 | 0.5900 | 0.4093 |
| 7 | 1.7312 | 0.7957 | 0.7144 | 0.4421 | 1.5298 | 0.7852 | 0.5580 | 0.3732 |
| 8 | 1.6749 | 0.7548 | 0.7019 | 0.4364 | 1.5480 | 0.8127 | 0.5515 | 0.3674 |
| 9 | 1.6518 | 0.7419 | 0.6925 | 0.4349 | 1.5101 | 0.7385 | 0.5655 | 0.4120 |
| 10 | 1.6085 | 0.7126 | 0.6813 | 0.4290 | 1.4718 | 0.7352 | 0.5517 | 0.3697 |
| 11 | 1.5670 | 0.6921 | 0.6643 | 0.4213 | 1.4953 | 0.7924 | 0.5253 | 0.3552 |
| 12 | 1.5309 | 0.6679 | 0.6567 | 0.4126 | 1.4515 | 0.7537 | 0.5219 | 0.3519 |
| 13 | 1.5093 | 0.6545 | 0.6485 | 0.4128 | 1.4122 | 0.6875 | 0.5354 | 0.3786 |
| 14 | 1.4804 | 0.6412 | 0.6355 | 0.4074 | 1.3793 | 0.6798 | 0.5194 | 0.3601 |
| 15 | 1.4573 | 0.6284 | 0.6270 | 0.4038 | 1.5149 | 0.8176 | 0.5247 | 0.3454 |
| 16 | 1.4543 | 0.6336 | 0.6190 | 0.4033 | 1.3955 | 0.6738 | 0.5324 | 0.3786 |
| 17 | 1.4256 | 0.6190 | 0.6086 | 0.3959 | 1.3385 | 0.6530 | 0.5076 | 0.3558 |
| 18 | 1.3983 | 0.6132 | 0.5949 | 0.3804 | 1.3564 | 0.6760 | 0.5112 | 0.3384 |
| 19 | 1.3825 | 0.5993 | 0.5932 | 0.3798 | 1.4564 | 0.8091 | 0.4829 | 0.3288 |
| 20 | 1.3824 | 0.6107 | 0.5864 | 0.3706 | 1.3357 | 0.6841 | 0.4894 | 0.3244 |
| 21 | 1.3660 | 0.5883 | 0.5895 | 0.3764 | 1.3115 | 0.6543 | 0.4921 | 0.3300 |
| 22 | 1.3657 | 0.6071 | 0.5769 | 0.3635 | 1.3347 | 0.6519 | 0.5063 | 0.3529 |
| 23 | 1.3463 | 0.5895 | 0.5761 | 0.3614 | 1.3093 | 0.6466 | 0.4952 | 0.3349 |
| 24 | 1.3355 | 0.5811 | 0.5755 | 0.3580 | 1.3357 | 0.6557 | 0.5055 | 0.3491 |
| 25 | 1.3196 | 0.5778 | 0.5672 | 0.3491 | 1.4033 | 0.7710 | 0.4763 | 0.3119 |
| 26 | 1.3266 | 0.5804 | 0.5691 | 0.3543 | 1.2921 | 0.6631 | 0.4744 | 0.3091 |
| 27 | 1.3196 | 0.5794 | 0.5655 | 0.3492 | 1.3001 | 0.6684 | 0.4766 | 0.3103 |
| 28 | 1.3166 | 0.5778 | 0.5654 | 0.3469 | 1.3062 | 0.6795 | 0.4740 | 0.3054 |
| 29 | 1.3195 | 0.5839 | 0.5640 | 0.3431 | 1.2822 | 0.6423 | 0.4812 | 0.3172 |
| 30 | 1.2991 | 0.5778 | 0.5543 | 0.3339 | 1.2870 | 0.6604 | 0.4730 | 0.3073 |
| 31 | 1.3182 | 0.5844 | 0.5620 | 0.3436 | 1.3450 | 0.7338 | 0.4624 | 0.2976 |
| 32 | 1.3164 | 0.5789 | 0.5650 | 0.3451 | 1.2754 | 0.6433 | 0.4760 | 0.3121 |
| 33 | 1.3055 | 0.5754 | 0.5607 | 0.3387 | 1.3306 | 0.7114 | 0.4688 | 0.3008 |
| 34 | 1.2982 | 0.5668 | 0.5599 | 0.3430 | 1.3160 | 0.7046 | 0.4631 | 0.2968 |
| 35 | 1.2922 | 0.5731 | 0.5530 | 0.3322 | 1.2848 | 0.6716 | 0.4635 | 0.2992 |
| 36 | 1.2995 | 0.5733 | 0.5576 | 0.3373 | 1.2976 | 0.6819 | 0.4663 | 0.2989 |
| 37 | 1.2933 | 0.5557 | 0.5661 | 0.3428 | 1.3080 | 0.6936 | 0.4654 | 0.2981 |
| 38 | 1.3120 | 0.5865 | 0.5555 | 0.3401 | 1.3045 | 0.6919 | 0.4634 | 0.2983 |
| 39 | 1.3040 | 0.5728 | 0.5606 | 0.3412 | 1.2853 | 0.6707 | 0.4649 | 0.2995 |

## clDice diagnostic on real predictions

Step 5 measured clDice against perturbed ground truth. This measures it against what the model actually produces, which is soft and thicker than the target. `skeleton_delta` is `mean |soft_skeleton(sigmoid(logits)) - sigmoid(logits)|`: near zero means the soft skeleton is returning its input and the term is inert.

| epoch | skel(pred) | skel(true) | skel(pred) on true | t_prec | t_rec | skeleton delta | Dice | clDice | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1306 | 1795 | 177 | 0.1471 | 0.8538 | 0.812220 | 0.7515 | 0.7555 | active |
| 1 | 2169 | 1795 | 374 | 0.1645 | 0.8081 | 0.569581 | 0.6989 | 0.7367 | active |
| 2 | 1004 | 1795 | 233 | 0.2295 | 0.8792 | 0.568241 | 0.6619 | 0.6461 | active |
| 3 | 850 | 1795 | 378 | 0.4051 | 0.8009 | 0.445561 | 0.6285 | 0.4802 | active |
| 4 | 702 | 1795 | 401 | 0.5005 | 0.7248 | 0.388175 | 0.6305 | 0.4318 | active |
| 5 | 710 | 1795 | 404 | 0.4983 | 0.7530 | 0.373000 | 0.6031 | 0.4192 | active |
| 6 | 698 | 1795 | 355 | 0.4561 | 0.8442 | 0.408544 | 0.5786 | 0.4239 | active |
| 7 | 770 | 1795 | 458 | 0.5244 | 0.8262 | 0.349837 | 0.5426 | 0.3811 | active |
| 8 | 806 | 1795 | 505 | 0.5516 | 0.7906 | 0.313363 | 0.5367 | 0.3756 | active |
| 9 | 655 | 1795 | 304 | 0.4319 | 0.8769 | 0.402846 | 0.5630 | 0.4365 | active |
| 10 | 761 | 1795 | 444 | 0.5248 | 0.8201 | 0.336729 | 0.5436 | 0.3807 | active |
| 11 | 798 | 1795 | 527 | 0.5767 | 0.7863 | 0.278408 | 0.5122 | 0.3618 | active |
| 12 | 837 | 1795 | 520 | 0.5522 | 0.8143 | 0.295263 | 0.5124 | 0.3640 | active |
| 13 | 688 | 1795 | 355 | 0.4763 | 0.8683 | 0.356427 | 0.5357 | 0.4011 | active |
| 14 | 745 | 1795 | 421 | 0.5118 | 0.8613 | 0.330481 | 0.5153 | 0.3778 | active |
| 15 | 771 | 1795 | 522 | 0.6007 | 0.7749 | 0.267570 | 0.5190 | 0.3510 | active |
| 16 | 682 | 1795 | 347 | 0.4753 | 0.8655 | 0.349532 | 0.5351 | 0.4002 | active |
| 17 | 731 | 1795 | 411 | 0.5118 | 0.8660 | 0.319946 | 0.5075 | 0.3751 | active |
| 18 | 731 | 1795 | 448 | 0.5530 | 0.8368 | 0.303363 | 0.5114 | 0.3525 | active |
| 19 | 821 | 1795 | 591 | 0.6192 | 0.7828 | 0.231783 | 0.4755 | 0.3343 | active |
| 20 | 789 | 1795 | 519 | 0.5792 | 0.8437 | 0.281556 | 0.4848 | 0.3347 | active |
| 21 | 758 | 1795 | 472 | 0.5559 | 0.8463 | 0.289299 | 0.4914 | 0.3461 | active |
| 22 | 640 | 1795 | 342 | 0.5009 | 0.8738 | 0.331643 | 0.5105 | 0.3775 | active |
| 23 | 711 | 1795 | 414 | 0.5346 | 0.8665 | 0.309607 | 0.4967 | 0.3539 | active |
| 24 | 651 | 1795 | 357 | 0.5074 | 0.8732 | 0.331162 | 0.5095 | 0.3713 | active |
| 25 | 811 | 1795 | 589 | 0.6330 | 0.7936 | 0.233672 | 0.4720 | 0.3195 | active |
| 26 | 723 | 1795 | 481 | 0.5909 | 0.8455 | 0.271343 | 0.4702 | 0.3211 | active |
| 27 | 763 | 1795 | 511 | 0.5916 | 0.8463 | 0.272358 | 0.4733 | 0.3226 | active |
| 28 | 758 | 1795 | 523 | 0.6089 | 0.8344 | 0.262580 | 0.4690 | 0.3142 | active |
| 29 | 718 | 1795 | 450 | 0.5632 | 0.8622 | 0.291610 | 0.4811 | 0.3339 | active |
| 30 | 740 | 1795 | 496 | 0.5921 | 0.8475 | 0.270172 | 0.4707 | 0.3206 | active |
| 31 | 779 | 1795 | 573 | 0.6390 | 0.8204 | 0.238986 | 0.4545 | 0.3027 | active |
| 32 | 713 | 1795 | 455 | 0.5703 | 0.8626 | 0.286921 | 0.4745 | 0.3284 | active |
| 33 | 779 | 1795 | 560 | 0.6266 | 0.8218 | 0.247120 | 0.4630 | 0.3085 | active |
| 34 | 774 | 1795 | 558 | 0.6278 | 0.8300 | 0.246909 | 0.4568 | 0.3042 | active |
| 35 | 761 | 1795 | 527 | 0.6087 | 0.8445 | 0.258814 | 0.4591 | 0.3101 | active |
| 36 | 748 | 1795 | 525 | 0.6142 | 0.8394 | 0.257786 | 0.4617 | 0.3087 | active |
| 37 | 753 | 1795 | 535 | 0.6202 | 0.8363 | 0.254506 | 0.4598 | 0.3071 | active |
| 38 | 754 | 1795 | 537 | 0.6218 | 0.8329 | 0.250499 | 0.4575 | 0.3059 | active |
| 39 | 750 | 1795 | 519 | 0.6094 | 0.8432 | 0.259842 | 0.4601 | 0.3093 | active |

**Finding.** 0 of 40 epochs came back degenerate. Final verdict: soft skeleton differs from the prediction -- clDice is measuring a real centreline.

If this says degenerate, `loss.w_cldice` has been buying nothing and the honest response is to record that here, not to retune the weight. The levers that would change it are a thicker `boundary_gt.line_width_px` (re-running steps 2 and 3) or an explicit connectivity metric at evaluation time.

