# Training report -- dev-w4-sched-film on colab

Held-out dataset: **uhcs2**. Generated 2026-09-17T09:00:14Z on colab (Tesla T4).

This file is keyed by run and host: `train_dev-w4-sched-film_colab.md`. The same fold trained on another host writes its own file beside this one rather than overwriting it -- two runs of one fold are two measurements, and they differ in GPU, worker count and I/O path.

- training mixture: the fold's full set, nothing excluded
- config hash `ffae94fd1f4b8c3e`, seed 0 (statistically reproducible; cudnn.benchmark picks algorithms by timing, so bitwise equality across runs is not claimed)
- 80 epochs, batch 64, 4 workers (configs/dataloader.yaml hosts.colab.num_workers, measured 2026-09-06T07:14:10Z)
- lr 0.0003 (encoder 8.999999999999999e-05), weight decay 0.0001, warmup 2 epochs, grad clip 1.0
- pos_weight 7.111000061035156 (configs/fold_stats.yaml folds.dev.pos_weight)
- best epoch 1 by best-threshold Dice on `uhcs2` = 0.2516 at threshold 0.7
- validation threshold swept over 0.05..0.95 (19 points); fixed reference 0.50
- FiLM: enabled, vocabulary ['MetalDam', 'Steel1', 'uhcs1']
- patience 10 (stopped early at epoch 11)
- boundary_gt.line_width_px 4 (the ground truth this run trained on -- NOT directly comparable to a report written under a different value: pos_weight and every boundary fraction move with it)

## Per-dataset validation metrics (the headline)

Validation is a mixture. The pooled row is a footnote; the held-out dataset's row is the measurement this fold exists to make.

Every row appears twice: at the fixed `train.threshold` and at the threshold that maximised Dice for that dataset. A single fixed threshold measures the model and the operating point together and reports the sum as if it were the model. `best.pt` is selected on the best-threshold Dice of the held-out dataset.

| epoch | dataset | thr | tiles | IoU | Dice | Precision | Recall | boundary-F | pred frac | true frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (best) | Steel1 fixed | 0.50 | 189 | 0.1725 | 0.2942 | 0.1762 | 0.8919 | 0.4486 | 0.4228 | 0.0835 |
| 1 (best) | Steel1 **best** | 0.75 | 189 | 0.2583 | 0.4106 | 0.3305 | 0.5419 | 0.5978 | 0.1369 | 0.0835 |
| 1 (best) | uhcs2 (held out) fixed | 0.50 | 265 | 0.1326 | 0.2342 | 0.1352 | 0.8767 | 0.3652 | 0.7491 | 0.1155 |
| 1 (best) | uhcs2 (held out) **best** | 0.70 | 265 | 0.1439 | 0.2516 | 0.1553 | 0.6636 | 0.3837 | 0.4936 | 0.1155 |
| 1 (best) | _pooled (footnote)_ fixed | 0.50 | 454 | 0.1441 | 0.2519 | 0.1469 | 0.8819 | 0.3900 | 0.6133 | 0.1022 |
| 1 (best) | _pooled (footnote)_ **best** | 0.70 | 454 | 0.1674 | 0.2868 | 0.1844 | 0.6452 | 0.4298 | 0.3575 | 0.1022 |
| 11 (final) | Steel1 fixed | 0.50 | 189 | 0.3178 | 0.4823 | 0.3545 | 0.7541 | 0.6750 | 0.1776 | 0.0835 |
| 11 (final) | Steel1 **best** | 0.70 | 189 | 0.3401 | 0.5076 | 0.4336 | 0.6119 | 0.6983 | 0.1179 | 0.0835 |
| 11 (final) | uhcs2 (held out) fixed | 0.50 | 265 | 0.1382 | 0.2428 | 0.1470 | 0.6970 | 0.3618 | 0.5477 | 0.1155 |
| 11 (final) | uhcs2 (held out) **best** | 0.50 | 265 | 0.1382 | 0.2428 | 0.1470 | 0.6970 | 0.3618 | 0.5477 | 0.1155 |
| 11 (final) | _pooled (footnote)_ fixed | 0.50 | 454 | 0.1732 | 0.2953 | 0.1860 | 0.7164 | 0.4338 | 0.3936 | 0.1022 |
| 11 (final) | _pooled (footnote)_ **best** | 0.60 | 454 | 0.1740 | 0.2965 | 0.1912 | 0.6594 | 0.4313 | 0.3523 | 0.1022 |

## The chosen threshold, per epoch, per dataset

This table is a measurement, not bookkeeping. If the held-out dataset's optimal threshold sits far from the training-side datasets', that gap IS the domain shift, expressed in the units of the decision the downstream watershed has to make -- and one global threshold will not serve both.

| epoch | Steel1 | uhcs2 | spread |
| --- | --- | --- | --- |
| 0 | 0.70 | 0.65 | 0.05 |
| 1 | 0.75 | 0.70 | 0.05 |
| 2 | 0.65 | 0.55 | 0.10 |
| 3 | 0.65 | 0.45 | 0.20 |
| 4 | 0.75 | 0.35 | 0.40 |
| 5 | 0.80 | 0.25 | 0.55 |
| 6 | 0.75 | 0.55 | 0.20 |
| 7 | 0.80 | 0.55 | 0.25 |
| 8 | 0.80 | 0.50 | 0.30 |
| 9 | 0.80 | 0.50 | 0.30 |
| 10 | 0.70 | 0.40 | 0.30 |
| 11 | 0.70 | 0.50 | 0.20 |

**Final epoch.** uhcs2 (held out) wants 0.50; the others want Steel1 0.70 -- a gap of 0.20. That is a domain-shift finding: the held-out microscope needs a materially different operating point, so step 7 should set the threshold PER DATASET rather than globally.

## Loss terms, separately

They differ by orders of magnitude, so the total alone does not say which one moved.

| epoch | train total | train BCE | train Dice | train clDice | val total | val BCE | val Dice | val clDice |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.4300 | 1.5158 | 0.6141 | 0.6001 | 2.5878 | 1.3703 | 0.8182 | 0.7988 |
| 1 | 1.9438 | 1.1917 | 0.5604 | 0.3835 | 2.5234 | 1.4000 | 0.7967 | 0.6533 |
| 2 | 1.6825 | 1.0387 | 0.5278 | 0.2321 | 2.4100 | 1.3073 | 0.7994 | 0.6066 |
| 3 | 1.5808 | 0.9738 | 0.5012 | 0.2115 | 2.4456 | 1.3364 | 0.7994 | 0.6197 |
| 4 | 1.4813 | 0.9029 | 0.4733 | 0.2101 | 2.2866 | 1.1915 | 0.7834 | 0.6233 |
| 5 | 1.3787 | 0.8321 | 0.4405 | 0.2124 | 2.2701 | 1.2102 | 0.7506 | 0.6187 |
| 6 | 1.3183 | 0.7937 | 0.4199 | 0.2094 | 2.1985 | 1.1586 | 0.7318 | 0.6161 |
| 7 | 1.2750 | 0.7724 | 0.4013 | 0.2028 | 2.2144 | 1.1786 | 0.7275 | 0.6165 |
| 8 | 1.2508 | 0.7715 | 0.3829 | 0.1929 | 2.2045 | 1.1935 | 0.7129 | 0.5963 |
| 9 | 1.2181 | 0.7458 | 0.3775 | 0.1896 | 2.1799 | 1.1696 | 0.7092 | 0.6021 |
| 10 | 1.2057 | 0.7416 | 0.3707 | 0.1869 | 2.2100 | 1.2152 | 0.6967 | 0.5962 |
| 11 | 1.1825 | 0.7285 | 0.3638 | 0.1803 | 2.2016 | 1.2102 | 0.6938 | 0.5952 |

## clDice diagnostic on real predictions

Step 5 measured clDice against perturbed ground truth. This measures it against what the model actually produces, which is soft and thicker than the target. `skeleton_delta` is `mean |soft_skeleton(sigmoid(logits)) - sigmoid(logits)|`: near zero means the soft skeleton is returning its input and the term is inert.

| epoch | skel(pred) | skel(true) | skel(pred) on true | t_prec | t_rec | skeleton delta | Dice | clDice | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2972 | 1318 | 322 | 0.1069 | 0.6981 | 0.611049 | 0.8291 | 0.8173 | active |
| 1 | 929 | 1318 | 178 | 0.2036 | 0.7703 | 0.638720 | 0.8133 | 0.6880 | active |
| 2 | 662 | 1318 | 150 | 0.2551 | 0.7107 | 0.589458 | 0.8148 | 0.6424 | active |
| 3 | 552 | 1318 | 114 | 0.2529 | 0.6769 | 0.558241 | 0.8136 | 0.6558 | active |
| 4 | 674 | 1318 | 139 | 0.2430 | 0.6607 | 0.502096 | 0.8011 | 0.6629 | active |
| 5 | 701 | 1318 | 144 | 0.2321 | 0.6672 | 0.461107 | 0.7766 | 0.6682 | active |
| 6 | 716 | 1318 | 141 | 0.2337 | 0.6756 | 0.434865 | 0.7588 | 0.6662 | active |
| 7 | 762 | 1318 | 155 | 0.2353 | 0.6776 | 0.427050 | 0.7535 | 0.6648 | active |
| 8 | 774 | 1318 | 166 | 0.2491 | 0.6637 | 0.399995 | 0.7455 | 0.6520 | active |
| 9 | 759 | 1318 | 152 | 0.2430 | 0.6874 | 0.425157 | 0.7446 | 0.6575 | active |
| 10 | 825 | 1318 | 167 | 0.2581 | 0.6311 | 0.357964 | 0.7317 | 0.6520 | active |
| 11 | 749 | 1318 | 145 | 0.2625 | 0.6733 | 0.403651 | 0.7249 | 0.6449 | active |

**Finding.** 0 of 12 epochs came back degenerate. Final verdict: soft skeleton differs from the prediction -- clDice is measuring a real centreline.

If this says degenerate, `loss.w_cldice` has been buying nothing and the honest response is to record that here, not to retune the weight. The levers that would change it are a thicker `boundary_gt.line_width_px` (re-running steps 2 and 3) or an explicit connectivity metric at evaluation time.

