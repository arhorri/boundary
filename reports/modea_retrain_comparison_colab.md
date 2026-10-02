# fold_steel_combined_modea vs fold_steel_combined: the retrain on line-class Steel2 ground truth

Both checkpoints scored on the identical test tiles. Pixel Dice at each checkpoint's own tuned threshold; PQ/SQ/RQ of the watershed at the marker that VAL selected for that checkpoint. TEST is read once per checkpoint.

## generated_utc

- 2026-10-02T20:06:07Z

## folds

- **new**: fold_steel_combined_modea
- **old**: fold_steel_combined

## checkpoints

- **new**:
  - **path**: /content/drive/MyDrive/phase11-persistent/checkpoints/fold_steel_combined_modea/best.pt
  - **epoch**: 32
  - **config_hash**: 067fecc2e06a7d0a
  - **gt_fingerprint**:
    - **gt_extraction_sha256**: 53155b40c2b6ebf8
    - **gt_extraction_generated_utc**: 2026-09-30T12:12:48Z
    - **boundary_gt_settings**:
      - **artifact_colours**:
        - **MetalDam**:
          - [0]
            - 75, 176, 40
        - **uhcs2**:
          - [0]
            - 75, 176, 40
      - **close_kernel**: 3
      - **exclusions**:
        - **uhcs2**:
          - uhcs0596.png
      - **fold_artifact_colours**: False
      - **hsv_percentiles**:
        - 2, 98
      - **hsv_sample_files**: 12
      - **line_width_px**: 4
      - **max_unsnapped_fraction**: 0.02
      - **min_palette_share**: 0.002
      - **min_region_area_px**: {}
      - **mode_a_line_class**:
        - **Steel2**: 8
      - **open_kernel**: 3
      - **open_mode**: speckle
      - **output_subdir**: gt_boundaries
      - **palette_merge_distance**: 48
      - **size_tolerance_px**: 2
    - **mode_a_line_class**:
      - **Steel2**:
        - **class_index**: 1
        - **colour**:
          - 0, 0, 0
        - **blob_thickness_px**: 8
- **old**:
  - **path**: /content/drive/MyDrive/phase11-persistent/checkpoints/fold_steel_combined/best.pt
  - **epoch**: 21
  - **config_hash**: 067fecc2e06a7d0a
  - **gt_fingerprint**: None

## how_to_read

- **Steel1**: same tiles, same ground truth: new vs old is a direct comparison
- **Steel2**: new model vs the NEW ground truth is the score of interest; the OLD model scored against the same NEW ground truth is the control, so 'delta' is what retraining adds beyond changing the ground truth under the same tiles. Nothing here scores the NEW model against the OLD Steel2 ground truth.

## test_tiles

- **Steel1**:
  - **tiles**: 96
  - **parents**: 2
- **Steel2**:
  - **tiles**: 126
  - **parents**: 1

## marker_grid_selected_on_val

- 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7

## default_marker

- 0.3

## selected_markers

- **new**:
  - **Steel1**: 0.6
  - **Steel2**: 0.6
- **old**:
  - **Steel1**: 0.55
  - **Steel2**: 0.7

## tuned_thresholds

- **new**:
  - **Steel1**: 0.75
  - **Steel2**: 0.8
- **old**:
  - **Steel1**: 0.7
  - **Steel2**: 0.85

## comparison

- **Steel1**:
  - **pixel_dice**:
    - **new**: 0.593443
    - **old**: 0.585725
    - **delta**: 0.00771721
  - **pq**:
    - **new**: 0.340503
    - **old**: 0.342858
    - **delta**: -0.00235481
  - **sq**:
    - **new**: 0.741439
    - **old**: 0.744224
    - **delta**: -0.00278496
  - **rq**:
    - **new**: 0.456673
    - **old**: 0.457418
    - **delta**: -0.000744711
- **Steel2**:
  - **pixel_dice**:
    - **new**: 0.734013
    - **old**: 0.596434
    - **delta**: 0.137579
  - **pq**:
    - **new**: 0.360107
    - **old**: 0.0954852
    - **delta**: 0.264622
  - **sq**:
    - **new**: 0.765908
    - **old**: 0.640014
    - **delta**: 0.125894
  - **rq**:
    - **new**: 0.469074
    - **old**: 0.145271
    - **delta**: 0.323803

## summaries

- **new**:
  - **Steel1**:
    - **tiles**: 96
    - **threshold**: 0.75
    - **pixel_dice**: 0.593443
    - **marker_threshold**: 0.6
    - **pq**: 0.340503
    - **sq**: 0.741439
    - **rq**: 0.456673
    - **over_segmentation_factor**: 0.935757
    - **reference_default_marker**:
      - **marker_threshold**: 0.3
      - **pq**: 0.323494
      - **sq**: 0.751222
      - **rq**: 0.429406
  - **Steel2**:
    - **tiles**: 126
    - **threshold**: 0.8
    - **pixel_dice**: 0.734013
    - **marker_threshold**: 0.6
    - **pq**: 0.360107
    - **sq**: 0.765908
    - **rq**: 0.469074
    - **over_segmentation_factor**: 1.07198
    - **reference_default_marker**:
      - **marker_threshold**: 0.3
      - **pq**: 0.31089
      - **sq**: 0.768985
      - **rq**: 0.404104
- **old**:
  - **Steel1**:
    - **tiles**: 96
    - **threshold**: 0.7
    - **pixel_dice**: 0.585725
    - **marker_threshold**: 0.55
    - **pq**: 0.342858
    - **sq**: 0.744224
    - **rq**: 0.457418
    - **over_segmentation_factor**: 0.992552
    - **reference_default_marker**:
      - **marker_threshold**: 0.3
      - **pq**: 0.302177
      - **sq**: 0.739575
      - **rq**: 0.404744
  - **Steel2**:
    - **tiles**: 126
    - **threshold**: 0.85
    - **pixel_dice**: 0.596434
    - **marker_threshold**: 0.7
    - **pq**: 0.0954852
    - **sq**: 0.640014
    - **rq**: 0.145271
    - **over_segmentation_factor**: 1.92551
    - **reference_default_marker**:
      - **marker_threshold**: 0.3
      - **pq**: 0.0640264
      - **sq**: 0.598974
      - **rq**: 0.0983746

## checks

- [0]
  - **check**: test tile membership identical in both folds (train, val and test)
  - **ok**: True
  - **detail**: train 965, val 221, test 222
- [1]
  - **check**: both checkpoints scored exactly the same test tiles, in the same order
  - **ok**: True
  - **detail**: 222 tiles
- [2]
  - **check**: Steel1 test tiles: same ground-truth files and boundary fractions in both folds
  - **ok**: True
  - **detail**: 
- [3]
  - **check**: Steel2 test tiles: same tiles, different ground truth
  - **ok**: True
  - **detail**: 
- [4]
  - **check**: the NEW checkpoint was trained on the ground truth on disk
  - **ok**: True
  - **detail**: 53155b40c2b6ebf8
- [5]
  - **check**: the OLD checkpoint was NOT (it is the control)
  - **ok**: True
  - **detail**: None
- [6]
  - **check**: both runs share one config hash (only the ground truth and fold name differ)
  - **ok**: True
  - **detail**: 067fecc2e06a7d0a
- [7]
  - **check**: markers were selected on VAL rows only, for both
  - **ok**: True
  - **detail**: 
- [8]
  - **check**: TEST scored exactly the three pre-declared configurations, for both
  - **ok**: True
  - **detail**: watershed_prob@val-best marker, none@tuned, reference: watershed_prob@tuned
- [9]
  - **check**: the pipeline reproduces the committed OLD-fold report on Steel1 (same model, tiles, GT, default marker)
  - **ok**: True
  - **detail**: 0.3022 vs committed 0.3027 (region_metrics_fold_steel_combined_colab.json)
