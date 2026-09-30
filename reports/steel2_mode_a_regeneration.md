# Steel2 ground truth regenerated with mode_a_line_class

Steel2 only. Steel1 and every other dataset's boundary maps are checked byte-identical before/after. Nothing was retrained; the original fold_steel_combined is untouched.

## generated_utc

- 2026-09-30T12:14:55Z

## decision

- adopt boundary_gt.mode_a_line_class for Steel2 only (Steel1 unchanged)

## settings

- **mode_a_line_class**:
  - **Steel2**: 8
- **min_region_area_px**: {}
- **line_width_px**: 4
- **note**: configs/default.yaml still lists min_region_area_px Steel2: 50, never applied to the ground truth on disk; this regeneration forced it off

## mode_a_line_class_record

- **class_index**: 1
- **colour**:
  - 0, 0, 0
- **blob_thickness_px**: 8

## gt_fingerprint

- **before**:
  - **gt_extraction_sha256**: b6266b7480ad4db1
  - **gt_extraction_generated_utc**: 2026-09-16T07:38:12Z
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
    - **open_kernel**: 3
    - **open_mode**: speckle
    - **output_subdir**: gt_boundaries
    - **palette_merge_distance**: 48
    - **size_tolerance_px**: 2
- **after**:
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

## steel2_boundary_fraction_median

- **before**: 0.288498
- **after**: 0.201797

## profile_in_memory_cell29

- **mode_b**:
  - **tiles_without_any_boundary**: 0
  - **regions**:
    - **n_tiles**: 504
    - **regions_per_tile**: 102.815
    - **area_percentiles**:
      - **p25**: 38
      - **p50**: 78
      - **p75**: 192
    - **share_lt_50**: 0.338216
    - **share_lt_100**: 0.57888
  - **dangling_ends**:
    - **n_tiles**: 504
    - **endpoints_per_tile**:
      - **mean**: 116.847
      - **median**: 116
      - **p95**: 161.85
      - **max**: 222
    - **interior_endpoints_per_tile**:
      - **mean**: 83.5992
      - **median**: 80
      - **p95**: 129
      - **max**: 198
    - **share_tiles_without_interior_endpoint**: 0
    - **interior_endpoints_per_1000_skeleton_px**: 22.641
- **mode_a**:
  - **tiles_without_any_boundary**: 0
  - **regions**:
    - **n_tiles**: 504
    - **regions_per_tile**: 28.7381
    - **area_percentiles**:
      - **p25**: 64
      - **p50**: 146
      - **p75**: 395
    - **share_lt_50**: 0.193869
    - **share_lt_100**: 0.378279
  - **dangling_ends**:
    - **n_tiles**: 504
    - **endpoints_per_tile**:
      - **mean**: 149.804
      - **median**: 153
      - **p95**: 193
      - **max**: 235
    - **interior_endpoints_per_tile**:
      - **mean**: 126.867
      - **median**: 130
      - **p95**: 164
      - **max**: 211
    - **share_tiles_without_interior_endpoint**: 0
    - **interior_endpoints_per_1000_skeleton_px**: 48.0889
- **mode_a_cleanup50**:
  - **tiles_without_any_boundary**: 0
  - **regions**:
    - **n_tiles**: 504
    - **regions_per_tile**: 21.9563
    - **area_percentiles**:
      - **p25**: 65
      - **p50**: 184
      - **p75**: 532
    - **share_lt_50**: 0.202241
    - **share_lt_100**: 0.342852
  - **dangling_ends**:
    - **n_tiles**: 504
    - **endpoints_per_tile**:
      - **mean**: 114.347
      - **median**: 121
      - **p95**: 148
      - **max**: 183
    - **interior_endpoints_per_tile**:
      - **mean**: 98.746
      - **median**: 103
      - **p95**: 129
      - **max**: 164
    - **share_tiles_without_interior_endpoint**: 0
    - **interior_endpoints_per_1000_skeleton_px**: 42.5593
- **mode_a_cleanup100**:
  - **tiles_without_any_boundary**: 0
  - **regions**:
    - **n_tiles**: 504
    - **regions_per_tile**: 18.7996
    - **area_percentiles**:
      - **p25**: 64
      - **p50**: 199
      - **p75**: 650
    - **share_lt_50**: 0.204855
    - **share_lt_100**: 0.349129
  - **dangling_ends**:
    - **n_tiles**: 504
    - **endpoints_per_tile**:
      - **mean**: 98.8948
      - **median**: 107
      - **p95**: 138
      - **max**: 173
    - **interior_endpoints_per_tile**:
      - **mean**: 86.1706
      - **median**: 92
      - **p95**: 119.85
      - **max**: 155
    - **share_tiles_without_interior_endpoint**: 0
    - **interior_endpoints_per_1000_skeleton_px**: 40.4704

## regenerated_on_disk

- **regions**:
  - **n_tiles**: 504
  - **regions_per_tile**: 28.7381
  - **area_percentiles**:
    - **p25**: 64
    - **p50**: 146
    - **p75**: 395
  - **share_lt_50**: 0.193869
  - **share_lt_100**: 0.378279
- **dangling_ends**:
  - **n_tiles**: 504
  - **endpoints_per_tile**:
    - **mean**: 149.804
    - **median**: 153
    - **p95**: 193
    - **max**: 235
  - **interior_endpoints_per_tile**:
    - **mean**: 126.867
    - **median**: 130
    - **p95**: 164
    - **max**: 211
  - **share_tiles_without_interior_endpoint**: 0
  - **interior_endpoints_per_1000_skeleton_px**: 48.0889

## interaction_with_min_region_area_px

- Both act on the palette-snapped LABEL map, cleanup first, then the line class is drawn from what is left -- so MODE A plus cleanup is applicable. Measured, not enabled: see mode_a_cleanup50 / mode_a_cleanup100 in profile_in_memory_cell29.

## backup_of_mode_b_steel2_gt

- /content/drive/MyDrive/phase11-persistent/gt_boundaries/_backup_mode_b/Steel2

## checks

- [0]
  - **check**: Steel1 ground truth byte-identical before/after
  - **ok**: True
  - **detail**: 907 files
- [1]
  - **check**: no other dataset's ground truth changed
  - **ok**: True
  - **detail**: MetalDam, uhcs1, uhcs2
- [2]
  - **check**: Steel1 record in gt_extraction.json unchanged
  - **ok**: True
  - **detail**: 
- [3]
  - **check**: every dataset other than Steel2 has an unchanged record
  - **ok**: True
  - **detail**: 
- [4]
  - **check**: Steel2 file set unchanged (same names, none missing, none stale)
  - **ok**: True
  - **detail**: 504 files
- [5]
  - **check**: Steel2 record carries the mode and LINE/BLOB threshold
  - **ok**: True
  - **detail**: {'class_index': 1, 'colour': [0, 0, 0], 'blob_thickness_px': 8.0}
- [6]
  - **check**: cleanup is off in what was recorded
  - **ok**: True
  - **detail**: {}
- [7]
  - **check**: gt_fingerprint differs from the MODE B ground truth's
  - **ok**: True
  - **detail**: b6266b7480ad4db1 -> 53155b40c2b6ebf8
- [8]
  - **check**: regions/tile on disk equals Cell 29's in-memory MODE A, all tiles
  - **ok**: True
  - **detail**: 28.74 regions/tile, 37.8% under 100 px
- [9]
  - **check**: gallery tile 87450661_tile_2560_512.png: regions equal the forced diagnostic's MODE A count
  - **ok**: True
  - **detail**: on disk 66 vs diagnostic 66 (MODE B was 207)
- [10]
  - **check**: gallery tile 87450661_tile_2048_2048.png: regions equal the forced diagnostic's MODE A count
  - **ok**: True
  - **detail**: on disk 61 vs diagnostic 61 (MODE B was 205)
- [11]
  - **check**: gallery tile 87450741_tile_1024_1536.png: regions equal the forced diagnostic's MODE A count
  - **ok**: True
  - **detail**: on disk 54 vs diagnostic 54 (MODE B was 201)
- [12]
  - **check**: gallery tile 87450661_tile_1280_1792.png: regions equal the forced diagnostic's MODE A count
  - **ok**: True
  - **detail**: on disk 64 vs diagnostic 64 (MODE B was 200)
