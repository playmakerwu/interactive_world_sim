# Inference-time camera usage — file by file audit

`grep -nE "camera_[01]_color|OBS_KEY|obs_key"` over every script that reads from
the HDF5 files. Two groups emerge:

## Group A — reads `camera_1_color` (matches WM training; correct)

| file | line | code |
|------|------|------|
| `scripts/label_replay_buffer.py` | 53 | `OBS_KEY = "camera_1_color"` |
| `scripts/calibrate_hsv.py` | 67 | `OBS_KEY = "camera_1_color"` |
| `tests/goal_selection/encode_goal.py` | 22 | `OBS_KEY = "camera_1_color"` |

These were written during the **probe / state-estimator** work (Phase 3-A on
`state-probe-training-cloud`). They correctly use the WM's training camera.

## Group B — reads `camera_0_color` (mismatched — BUG)

| file | line | code |
|------|------|------|
| **`scripts/run_mppi.py`** | 58 | `OBS_KEY = "camera_0_color"` |
| **`scripts/run_mppi_pairs.py`** | 32 | `OBS_KEY = "camera_0_color"` |
| **`scripts/compute_state_goal_offaxis.py`** | 46 | `OBS_KEY = "camera_0_color"` |

These were written during the **MPPI** work (Step 5 + follow-ups). They all use
the wrong camera.

## Notes on indirect paths

- `scripts/wm_interactive_replay.py` does NOT read RGB directly — it consumes
  `initial_latent.pt` and `action_history.pt` produced by `run_mppi.py`. The
  latents it replays are therefore already encoded from the wrong-camera RGB.
- `tests/goal_selection/z_goal.pt` was produced by `tests/goal_selection/encode_goal.py`
  — that one uses **camera_1_color**, so the goal latent is correct. But the
  *initial* latents that MPPI starts from are wrong.
- `compute_state_goal.py` (the original axis-aligned goal generator) doesn't appear
  in either list because it operates on `z_goal.pt` directly, not raw HDF5 frames.

## Conclusion

**The MPPI pipeline encodes wrong-camera RGB into the WM's latent space.** Every
v1–v10 run we ran started from a `camera_0_color`-derived latent, while the goal
latent was correctly derived from `camera_1_color`. The controller has been
trying to drive a wrong-distribution latent toward a right-distribution goal.
