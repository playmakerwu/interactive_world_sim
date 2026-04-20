# HSV Calibration Summary

Frames: 20 (stratified across 5 train episodes, 4 per ep)
Seed: 42
Theoretical T-block area at 128 canvas: 471.8 px

## The three mandatory numbers (design doc §1.3)

| variant | drop rate | median area (frac of theoretical) | area recovery vs REAL (pct pts) |
|---|---|---|---|
| REAL | 0.0% | 513.6 px (108.9%) | +0.00 |
| WM | 0.0% | 513.6 px (108.9%) | +0.00 |
| REAL+H_wide5 | 0.0% | 513.6 px (108.9%) | +0.00 |
| REAL+S_lo0 | 0.0% | 513.6 px (108.9%) | +0.00 |
| REAL+V_hi255 | 0.0% | 513.6 px (108.9%) | +0.00 |

REAL-vs-WM agreement on 20 frames: **100.0% pixel-identical**
Agreement breakdown:
- identical: 20

## ICP residual distribution (successful frames only)

| variant | mean | p95 |
|---|---|---|
| REAL | 0.356 | 0.436 |
| WM | 0.356 | 0.436 |
| REAL+H_wide5 | 0.356 | 0.436 |
| REAL+S_lo0 | 0.356 | 0.436 |
| REAL+V_hi255 | 0.356 | 0.436 |

## Decision

**Proposed preset: `REAL`**

Pass against the design-doc §1.2 decision criteria:

| variant | drop rate <= 10%% | median area >= 90%% theoretical |
|---|---|---|
| REAL | pass | pass |
| WM | pass | pass |
| REAL+H_wide5 | pass | pass |
| REAL+S_lo0 | pass | pass |
| REAL+V_hi255 | pass | pass |

False-positive check (criterion 3) requires visual review of the grid PNGs in calibration_outputs/ — flagged to the user at Checkpoint 3.
