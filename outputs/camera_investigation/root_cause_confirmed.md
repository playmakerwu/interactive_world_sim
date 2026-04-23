# Root cause CONFIRMED — camera mismatch

## The two pieces of evidence

1. **WM was trained on `camera_1_color`** — see
   [wm_training_camera.md](wm_training_camera.md). Direct evidence in the
   checkpoint's hydra config (`outputs/pusht_cam1/.hydra/config.yaml:70-71`).

2. **MPPI / off-axis goal / pair sweep all read `camera_0_color`** — see
   [inference_camera.md](inference_camera.md). Three scripts hard-code
   `OBS_KEY = "camera_0_color"`:
   - `scripts/run_mppi.py:58`
   - `scripts/run_mppi_pairs.py:32`
   - `scripts/compute_state_goal_offaxis.py:46`

The visual confirmation in
[camera_comparison.png](camera_comparison.png) and
[camera_comparison_t0.png](camera_comparison_t0.png) makes the consequence concrete:
**these are physically different camera views**. `camera_1_color` is the top-down
overhead view that the WM was trained to encode/decode. `camera_0_color` is a
side/front view that the WM has literally never seen during training.

## Does this explain the observed symptoms?

Yes — every failure pattern fits this single root cause:

| Symptom | How camera mismatch causes it |
|---------|------------------------------|
| **Arms disappearing in MPPI rollouts** | The encoded `z0` is from a side view; the WM decoder, trained only on top views, has no idea what the rollout latents represent and outputs blurred/missing-arm reconstructions |
| **180° CV flips on decoded frames** | The decoded RGB is a hallucinated top-down view of a side-view scene; the T-block geometry is malformed, so HSV+ICP has no stable solution and flips between branches |
| **MPPI never converges (10–30 px)** | The controller is operating in a latent space that is not the WM's true manifold. Goal latent (correct camera) and current latent (wrong camera) are not directly comparable. Reward gradients steer toward something that isn't physically meaningful |
| **WM rollout drift varies wildly across seeds** | Off-distribution latents are highly noise-sensitive — small noise pushes them in inconsistent directions because the dynamics model has no trained behavior in this region |
| **Probe sin/cos collapse (on probe branch)** | `scripts/label_replay_buffer.py` correctly reads camera_1, so the probe LABELS are correct. But the PROBE DECODES through the WM to get latent inputs — and if anything in that pipeline confused cameras, latents fed to the probe would also be off-distribution. (Needs separate verification on the probe branch.) |

## Visual proof

[camera_comparison.png](camera_comparison.png) (val/0 t=100) and
[camera_comparison_t0.png](camera_comparison_t0.png) (the v1 initial state)
show side-by-side renders. The two cameras have different:
- Vantage point (overhead vs side)
- Field of view
- Apparent T-block geometry (flat-from-above vs 3D-with-depth)
- Lighting / exposure
- Arms visible at completely different angles

There is no way the WM's encoder, trained only on the left image, can produce
a sane latent from the right image. Yet that is exactly what `run_mppi.py`
has been doing for every Step 5 run (v1–v10) and every pair experiment.

## Recommended fix (NOT EXECUTED — diagnostic only)

Single-line change in three files:

```bash
# Change all of these from "camera_0_color" to "camera_1_color":
scripts/run_mppi.py:58              OBS_KEY = "camera_1_color"
scripts/run_mppi_pairs.py:32        OBS_KEY = "camera_1_color"
scripts/compute_state_goal_offaxis.py:46  OBS_KEY = "camera_1_color"
```

After the fix, all previous MPPI conclusions need to be re-validated:
- Re-run v1 with the correct camera. The arm-disappearance + 180-flip pattern
  should largely vanish (because the encoded latent will now be in the WM's
  training distribution).
- Re-run v2 (sym-aware) similarly — its 11.9 px result may improve dramatically
  because the underlying CV signal will be cleaner.
- The 10-pair sweep should be redone — a new pair_00 with the correct camera
  may actually hit `success_strict`.
- The keyboard-MPPI v10 conclusions remain valid (the hypothesis was about
  action-distribution shape, independent of which camera was used).

## What to do BEFORE the re-runs

The user asked for diagnostic only — do not execute fixes. Decisions to make:
1. Confirm the fix on a single re-run (e.g., v1 with `camera_1_color`) before
   redoing all 10+ experiments.
2. Decide whether to revisit the probe-training failure on the
   `state-probe-training-cloud` branch with the same audit. The probe code in
   `scripts/label_replay_buffer.py` already used the correct camera, so the
   probe failure may have a different root cause — but worth double-checking.
3. Update the MPPI_NOTES.md to flag every numerical result as preliminary
   pending camera-fix re-verification.

## Why this happened

Looking at git history: the probe-training scripts (`label_replay_buffer.py`,
`calibrate_hsv.py`, `encode_goal.py`) were written first, on the
`state-probe-training-cloud` branch, and they correctly read `camera_1_color`.

The MPPI scripts (`run_mppi.py`, `run_mppi_pairs.py`,
`compute_state_goal_offaxis.py`) were written later, on `mppi-baseline`,
and silently default to `camera_0_color`. The first time `OBS_KEY` appeared
in `run_mppi.py` (Step 5 commit `bac9f62`) it was hard-coded to
`"camera_0_color"` without justification — almost certainly a typo or
copy-from-wrong-source-file. None of the subsequent commits revisited it,
because there was no test that would catch a camera mismatch (the WM still
runs and produces outputs; they're just wrong-distribution outputs).
