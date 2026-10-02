# Steel GT / region-path step -- final

Every number below is copied from a committed report; the file is named beside it, and `reports/steel_step_final.json` records the exact key path. `tests/test_steel_step.py` re-reads each one and asserts it is identical.

- **fold**: `fold_steel_combined_modea` (configs/default.yaml `steel_step.default_fold`)
- **checkpoint**: best.pt, epoch 32, config hash `067fecc2e06a7d0a`, trained on ground truth `gt_extraction_sha256` `53155b40c2b6ebf8` (`region_metrics_fold_steel_combined_modea_colab.json`, `modea_retrain_comparison_colab.json`)
- **region path**: `watershed_prob`, marker 0.6 for Steel1 and Steel2, no post-processing, `min_region_area_px` off (configs/default.yaml `steel_step.region_path`)

## Final TEST numbers

| dataset | tiles | pixel Dice | skeleton Dice | verdict | marker | PQ | SQ | RQ | over-seg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steel1 | 96 | 0.5934 | 0.1512 | MISPLACED | 0.60 | 0.3408 | 0.7413 | 0.4571 | 0.937 |
| Steel2 | 126 | 0.7340 | 0.4095 | OFFSET | 0.60 | 0.3605 | 0.7652 | 0.4699 | 1.071 |

Sources: tiles, pixel Dice, skeleton Dice, verdict -- `region_metrics_fold_steel_combined_modea_colab.json` (`decomposition`, at each dataset's tuned threshold); marker, PQ/SQ/RQ, over-seg -- `gap_closing_fold_steel_combined_modea_colab.json` (`marker.test_rows`, `watershed@val-best marker (extended)`).

Repeat measurement, same checkpoint, same marker 0.6, a later session (`modea_retrain_comparison_colab.json`): PQ Steel1 0.3405, Steel2 0.3602. It differs from the table in the fourth decimal; the cause was not investigated.
The Run-all session of 2026-10-02 re-measured the same checkpoint and re-committed `region_metrics_fold_steel_combined_modea_colab.json`, `postprocess_sweep_...` and `fn_attribution_...`; their numbers moved in the third to fourth decimal (Steel1 skeleton Dice 0.1513 @ b6de92e -> 0.1512; pixel Dice 0.5934 -> 0.5934). The figures in this report are those of the files as committed now. Evaluation of one checkpoint is not bit-reproducible on the host.

## Decisions and the evidence for each

- **Steel2 MODE A (`mode_a_line_class: 8`)** -- Steel2 GT regions/tile 102.8 -> 28.7 (`steel2_gt_mode_check_forced.json`); retrained model on the new GT PQ 0.3602 vs the old model on the same GT 0.0955 (`modea_retrain_comparison_colab.json`).
- **connectivity=1 background labelling** -- a diagonal-only boundary leaked two regions into one (`test_true_regions_from_boundary_does_not_leak_through_a_diagonal_only_boundary`); fixing it raised GT regions/tile Steel1 12.40 -> 16.99, Steel2 111.97 -> 127.40 (`region_metrics_fold_steel_combined_colab.json` @ 881d37e -> commit 0ce89c5).
- **watershed marker 0.6 over gap closing** -- TEST PQ Steel1 0.3408 vs 0.2979; Steel2 0.3605 vs 0.3649, a tie on one test parent (`gap_closing_fold_steel_combined_modea_colab.json`). Marker VAL-selected: Steel1 0.60, Steel2 0.60.
- **post-processing rejected** -- the VAL-selected `skeleton_redilate@0.70` gives TEST PQ Steel1 0.2982, Steel2 0.3640 (`postprocess_sweep_fold_steel_combined_modea_colab.json`): below the watershed on Steel1 (0.3408), level with it on Steel2.
- **`min_region_area_px` off** -- recorded setting `{}` (`gt_extraction.json`). Cleanup at 50 px would take Steel2 GT regions/tile 28.7 -> 22.0 (`steel2_mode_a_regeneration.json`), but no model was trained on that GT, so it is not adopted.

## Known limitations

- **Steel1 is MISPLACED** -- verdict MISPLACED, skeleton Dice 0.1512 (`region_metrics_fold_steel_combined_modea_colab.json`). The region path does not fix where the model puts boundaries.
- **Steel2 open contours and TINY misses** -- the line-class GT has 126.9 interior skeleton endpoints per tile on average, and 0.0% of tiles have none (`steel2_mode_a_regeneration.json`). TEST misses (none@tuned): 23.9 per tile, TINY 855 (28%), MERGED 2096 (70%) (`fn_attribution_fold_steel_combined_modea_colab.json`).
- **single seed, small test splits** -- one training run per fold; test parents Steel1 2, Steel2 1 (`modea_retrain_comparison_colab.json`). Treat Steel1 differences under 0.02 as noise (new - old PQ -0.0024).
- **clDice skeleton iterations vs the 4 px ground truth -- a candidate for the model round, NOT changed.** `loss.cldice_iters` stays 3, the value in the hashed config of the trained checkpoint (`tests/test_steel_step.py` pins it); changing it changes training. The thickness-invariance test (`tests/test_losses.py`) was written for 2 px ground truth and failed in the Colab pytest run at the default 3 on the 4 px ground truth (which assertion failed was not captured). It is now width-aware: its iterations come from `losses.thickness_invariance_iters(line_width)`.
  - **Measured: nothing yet.** The measurement the model round needs -- soft-skeleton thickness and length for iterations 3-6 on straight and 45-degree lines of width 2 and 4 and on the densest MetalDam tile, and from which iteration count the invariance property holds -- needs execution, so it is `06d_steel_combined.ipynb` Cell 33 (`RUN_DIAGNOSTICS = True`), which writes `reports/cldice_iters_measurement.{md,json}`. Paste its table here before the model round.
  - **Derived by hand from `soft_skeletonize` (NOT measured; `tests/test_losses.py::test_soft_skeleton_of_a_straight_even_width_line_is_a_two_pixel_central_band` and `::test_thickness_invariance_on_a_straight_line_needs_iterations_of_about_half_the_width` pin it, and run on the host):** on a STRAIGHT line of even width W the skeleton is a 2 px central band (thickness 2, not 1) and is identical for iterations 3, 4 and 5, because erosion consumes the line after about W/2 iterations; clDice is invariant to a one-pixel dilation from `W/2` iterations up (1 at 2 px, 2 at 4 px), and below that the dilated line's skeleton is EMPTY, which scores clDice ~0 for the wrong reason (the test's real-skeleton checks catch it). So 3 iterations should suffice on straight lines at 4 px; the open question is DIAGONAL lines, whose axis-aligned thickness is sqrt(2) times their width, giving `ceil((W + 2) / sqrt(2))` = 3 at 2 px (the value that passed) and 5 at 4 px. That formula is a prediction; its only measured point is 2 px.
  - **Candidate:** `loss.cldice_iters = 5` for 4 px ground truth, to be adopted only if Cell 33 shows the property holding on the dense tile at 5 and not at 3.
- **Width read-out bias, fixed, no recorded number changed.** `evaluate_checkpoint` read even-width lines one pixel low when called without `width_hint` (a 2 px band 1.0, a 4 px band 3.0). It now takes an optional `width_hint` (default unchanged), which Cells 15 and 17 pass from `line_width`; `measured_line_width` returns `None` for a tile with no background instead of an overflowing value. No committed report stores these widths, and `decompose_error`'s `width_ratio` is untouched. `06_train.ipynb`'s call does not know the ground-truth width at that point and still uses the default.
- **watershed connectivity inconsistency, left unfixed on purpose** -- `train.watershed_regions` labels marker seeds 8-connected (`skimage.measure.label` default) while `skimage.segmentation.watershed` floods 4-connected. Fixing it would move every reported watershed number for every checkpoint.
- **old fold kept** -- `fold_steel_combined` (MODE B Steel2 GT) is the superseded control: its manifests, fold_stats entry, reports and checkpoint are kept, not deleted, and never rebuilt from the new GT.

## Next step (stated, not started)

1. One time-boxed model round on `fold_steel_combined_modea`: Arms C + D plus a seed control, judged on Steel1 skeleton Dice.
2. Then Phase 2: watershed segmentation on the region path above.
