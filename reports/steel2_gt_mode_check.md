# GT mode check: is a MODE B palette class actually a painted line?

- generated: 2026-09-30T11:54:21Z
- settings: {'line_width_px': 4.0, 'blob_thickness_px': 8.0, 'thin_width_px': 8.0, 'sample_files': 60}

MODE B assumes every palette class is a PHASE (an area with an inside) and draws `find_boundaries` around it. Nothing checks that assumption. This report measures, per dataset, whether the class actually looks like a thin painted line instead -- which `find_boundaries` would still happily outline on both edges, trapping the line's own pixels as a spurious strip "region".

## Steel1

- K = 2 palette classes: `[0, 0, 0]`, `[255, 255, 255]`
- sampled 60 mask files fresh from disk

Raw label alphabet (top values, exact-match pixel share, before any palette-merge snapping):

| value | pixel share |
| --- | --- |
| `[0, 0, 0]` | 0.8563 |
| `[255, 255, 255]` | 0.1437 |

Per-palette-class shape/intensity profile:

| class idx | colour | pixel share | width p50 | width p95 | share thin | components | largest CC share | mean inside | mean outside |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `[0, 0, 0]` | 0.8563 | 40.36 | 110.28 | 6.27% | 112 | 98.04% | 135.2 | 200.8 |
| 1 | `[255, 255, 255]` | 0.1437 | 10.39 | 34.00 | 29.74% | 531 | 48.27% | 200.8 | 135.2 |

## Steel2

- K = 2 palette classes: `[255, 255, 255]`, `[0, 0, 0]`
- sampled 60 mask files fresh from disk

Raw label alphabet (top values, exact-match pixel share, before any palette-merge snapping):

| value | pixel share |
| --- | --- |
| `[255, 255, 255]` | 0.6321 |
| `[0, 0, 0]` | 0.1084 |
| `[254, 254, 254]` | 0.0264 |
| `[253, 253, 253]` | 0.0256 |
| `[252, 252, 252]` | 0.0227 |
| `[251, 251, 251]` | 0.0195 |
| `[250, 250, 250]` | 0.0170 |
| `[249, 249, 249]` | 0.0138 |
| `[1, 1, 1]` | 0.0136 |
| `[2, 2, 2]` | 0.0128 |

Per-palette-class shape/intensity profile:

| class idx | colour | pixel share | width p50 | width p95 | share thin | components | largest CC share | mean inside | mean outside |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `[255, 255, 255]` | 0.8040 | 8.79 | 30.76 | 43.19% | 4146 | 95.99% | 130.6 | 94.8 |
| 1 | `[0, 0, 0]` | 0.1960 | 4.00 | 8.40 | 90.22% | 11843 | 23.58% | 94.8 | 130.6 |

## Verdict

- **Steel2** class 1 (colour `[0, 0, 0]`): not line-like
- share of skeleton thinner than 8.0 px = 0.9021516393442623  (gate >= 0.7); largest connected component = 0.23580131111725197 of the class's own area  (gate >= 0.5) -- at least one gate not cleared: not confirmed as a painted line here.

No MODE B/A comparison was run (the class did not verify as line-like enough to warrant one).

## What this report does NOT do

- does not regenerate any boundary PNG under `GT_BOUNDARIES_ROOT`
- does not rebuild `fold_steel_combined`'s manifests or `configs/fold_stats.yaml`
- does not retrain or touch any checkpoint
- proposes a per-dataset MODE option; does not turn it on anywhere
