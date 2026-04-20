# State Estimator Research — Phase 1

Branch: `state-probe-training` (off `dreamer-rl`)
Date: 2026-04-19
Author: Phase 1 investigation per the v5 task brief

---

## 1. TL;DR

The supervisor's repo at `~/Documents/aloha` does not contain a neural state
estimator. What it contains is a **classical CV pose estimator for the T-block
only**, used in `analyze.py` to score policy rollouts via IoU. We confirmed
this with the supervisor and revised scope per §1 of the v5 task brief:

- **Scope**: T-block pose only (cx, cy, θ). Drop arm/gripper requirements.
- **Approach**: offline-label replay buffer with the CV pipeline → train a
  differentiable `latent → (cx, cy, sin θ, cos θ)` MLP probe.
- **Decoder-in-the-loop POC**: skipped. The CV pipeline is non-differentiable
  (cv2.findContours + ICP), so a decode-based reward path has no analytic
  gradient. Probe is the only viable Dreamer-compatible path.

This document closes Phase 1: it records what's actually in the supervisor's
repo, the API we'll consume, the decisions, and the open empirical question
(do the preset HSV thresholds work on our decoder outputs) that the sanity
check is built to answer.

---

## 2. What the supervisor's repo contains

### 2.1 Repository type
`~/Documents/aloha` is a Python package for **ALOHA bimanual robot control,
data collection, and world-model evaluation**. It is not a state-estimator
package. There are no NN checkpoints for state estimation; the only `.pt`
file in the entire tree is `aloha/world_model/vendor/algorithms/common/metrics/i3d_torchscript.pt`
(an Inception-3D classifier used for FVD/FID, unrelated to our task).

The pose estimation that exists lives in
[`~/Documents/aloha/aloha/world_model/eval/analyze.py`](../../Documents/aloha/aloha/world_model/eval/analyze.py)
and is purely a classical CV pipeline.

### 2.2 The CV pipeline (the actual "estimator" we're consuming)

`estimate_current_pose(frame, template_contour, scale, hsv_lower, hsv_upper)`
at `analyze.py:156-208`. Pipeline:

1. `cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)` (input must be BGR uint8)
2. `cv2.inRange` with hard-coded HSV bounds
3. Morphological close + open with a 3×3 kernel
4. `cv2.findContours(RETR_EXTERNAL)` → take largest by area; reject if area < 100
5. Sample 200 points uniformly from the contour
6. **Trimmed ICP** (50% trim ratio, ≤100 iters, tol 1e-6) against an 8-point
   template T, swept over 12 initial rotations (every 30°)
7. Best-error fit yields rotation `R`; angle = `atan2(R[1,0], R[0,0])` in
   **degrees**, plus `init_angle` offset

Returns `(center, angle_deg, error)` or `(None, None, None)` if no contour
clears the area threshold.

### 2.3 Output semantics

- **Coordinate frame**: pixel coordinates of the input frame, NOT image-normalised.
  The supervisor processes WM frames at 512×512 (resizing 128 → 512 first); REAL
  frames at 480×480. We will run at our native 128×128 with `t_scale = 128/512 = 0.25`
  applied to the template — same convention they use, just at a smaller resolution.
- **Angle**: degrees. Sign: standard image-coordinate atan2 (x right, y down),
  so positive `angle` is clockwise visually. Range: principle output is in
  `(-180°, 180°]` due to atan2; the +30° init sweep can push out of range, so
  consumers should `((angle + 180) % 360) - 180` to normalise.
- **Periodicity**: the T-block has no rotational symmetry (it's a T, not an X),
  so `θ` and `θ + 180` are visually distinct. The ICP cost can still flip
  between them on poorly-resolved masks; the supervisor's downstream code
  (`analyze.py:402-411`) does temporal smoothing with a wrap-aware EMA. We get
  no temporal context at label time, so the probe will need to learn the
  bimodality from data; representing the target as `(sin θ, cos θ)` removes the
  2π discontinuity from the loss.
- **Differentiable**: **no**. `cv2.findContours`, `argsort`, ICP iteration,
  and the SVD-based rotation update are all non-differentiable. This is what
  motivates the offline-label-then-probe strategy (§3 below).

### 2.4 Outputs that DO NOT exist

The original brief asked for arm end-effector positions and gripper open/close
states for both arms. **There is no source of these in the supervisor's repo.**
The closest visual cue is the `goal_mask.npy` and `grey_card.png` referenced
by the WM evaluation pipeline, but those are static pose targets / colour
calibration cards, not estimators.

This is the core finding that motivated the scope narrowing in §1 of the v5
brief.

### 2.5 HSV preset values

From `analyze.py:57-64`:

| Preset | HSV lower | HSV upper | Tuned for |
|---|---|---|---|
| REAL | (160, 50, 100) | (179, 200, 244) | Real RealSense images of a real T-block (RGB ≈ (172, 88, 99)) |
| WM   | (140, 50, 100) | (179, 255, 255) | Pink/magenta T-block in **their** WM renders |

Neither was tuned against our IWS decoder. Whether one (or both) work on our
outputs is the open question the sanity check is built to answer (§5 below).

### 2.6 Performance (from code, not measured yet)

ICP sweep is 12 init angles × ≤100 iters × cKDTree query of 200 points on
each iter. The supervisor's `analyze.py` runs at ~0.5 s per episode wall
clock with 16-way multiprocessing (so ~8 s/episode single-threaded;
amortised across many frames per episode, this is well below real-time per
frame). For our offline labelling pipeline, even a 50-ms-per-frame budget
gives us ~20 fps on a single thread — fine.

---

## 3. Resolved decisions (from §1 of the v5 brief)

| Decision | Resolution |
|---|---|
| Scope | T-block pose only (cx, cy, θ). No arms, no grippers. |
| Approach | Offline label → train MLP probe `latent → (cx, cy, sin θ, cos θ)`. |
| Decode-in-loop POC | Skipped. Non-diff CV → no analytic-gradient path. |
| Angle representation | `(sin θ, cos θ)` to avoid 2π discontinuity in loss. Reward uses `1 − (sin·sin_g + cos·cos_g)`. |
| Acceptance | Position ≤ 3 px, angle ≤ 5°, plus visual gate (validation grid). |
| Re-decode vs. raw HDF5 | Default re-decode (Option A); fallback to raw HDF5 only if HSV drop rate > 30 % after calibration. |
| Train/val split | By episode, not by frame. |
| Branches | A: probe training (this branch). B: reward integration (separate, after A merges). |

---

## 4. Code-level integration plan (Branch A only)

Files this branch will add (Phase 3-A):

- `rl/labeling/cv_labeler.py` — wraps `estimate_current_pose` with HSV calibration
  hooks and quality filters; uses `importlib` to load `analyze.py` directly until
  we're ready to vendor the relevant functions in tree.
- `scripts/label_replay_buffer.py` — one-off: decode latents → CV → labels.
- `rl/models/state_probe.py` — MLP probe.
- `scripts/train_state_probe.py` — train + checkpoint + val metrics.
- `scripts/compute_state_goal.py` — `decode(z_goal) → CV → state_goal.pt`.
- `rl/visualization/state_viz.py` — drawing primitives. **Created in Phase 1.**
- `tests/state_estimator/{sanity_check, test_labeling, test_probe}.py`.

Files this branch will **not** touch (per §2.1 of the brief): anything under
`rl/training/`, `rl/utils/`, `rl/models/{actor,critic,world_model}.py`, `main.py`.

---

## 5. The sanity check (Phase 1 deliverable)

`tests/state_estimator/sanity_check.py` does:

1. Load `DifferentiableDynamics` from `outputs/pusht_cam1/checkpoints/best.ckpt`.
2. Load `tests/goal_selection/z_goal.pt` (shape `(1, 4, 32, 32)`).
3. `wm.decode(z_goal, resolution=128)` → `(1, 3, 128, 128)` in [0, 1].
4. Convert to BGR uint8 (CV pipeline expects BGR — `cvtColor(BGR→HSV)` is the first step).
5. Run `estimate_current_pose` once with `HSV_REAL`, once with `HSV_WM`. Time each.
6. Save: `00_decoded_rgb.png`, `01_mask_real.png`, `01_mask_wm.png`,
   `02_annotated_real.png`, `02_annotated_wm.png`, `03_annotated_both.png`,
   and a `results.json` with per-preset success flag, mask pixel count,
   largest contour area, ICP residual, center, angle.

Annotation uses the `render_state_on_image` primitive from
`rl/visualization/state_viz.py` (Section 6.1 of brief).

Loading the supervisor module: via `importlib.util.spec_from_file_location`
to bypass `aloha/__init__.py`'s ROS-only imports. Documented in
`migration_log.md`.

---

## 6. GPU availability and run-time observations

### 6.1 Snapshot at Phase 1 start (deferred run)
```
RTX 5070 Ti Laptop, 12227 MiB total, 10059 MiB used, 1724 MiB free
pid 54706: python train.py --seed 1 --env-id coinrun --encoder multiscale_token
           --no-use-aug --total-timesteps 25000000 --logdir runs/coinrun
           --exp-name coinrun_multiscale_token_s1
```
Below the §8.1 4-GiB threshold; sanity check held off and user notified.

### 6.2 Snapshot when sanity check ran (2026-04-20)
```
12227 MiB total, 6117 MiB used, 5666 MiB free
```
- WM ckpt load: 4.3 s, ~560 MiB resident
- `decode(z_goal)` (resolution=128): 0.74 s, ~330 MiB peak additional
- WM total footprint after decode: ~890 MiB

Implication for Phase 2: under sustained contention with a ~10-GiB
co-tenant, the probe training budget is ~3 GiB peak. Start batch 16 per §8.2.

---

## 7. HSV mask observations (sanity check on z_goal)

Both presets succeeded with **identical numbers** because the WM HSV range
(H∈[140,179], S∈[50,255], V∈[100,255]) is a strict superset of the REAL
range (H∈[160,179], S∈[50,200], V∈[100,244]) and our pink T-block falls
entirely inside the REAL subset.

| Metric | Value |
|---|---|
| Mask pixels (post morphology) | 476 |
| Largest contour area | 415 px (single connected component) |
| Detected center | (55.51, 62.70) px |
| Detected angle | +1.53° |
| ICP residual | 0.289 |
| ICP wall time | 0.06–0.08 s per call |

Visual (`02_annotated_real.png`, `02_annotated_wm.png`, `03_annotated_both.png`):
center marker and orientation arrow land on the pink T in the decoded image.
Mask is clean — no spurious pickup from the orange gripper tips (they sit
below the H≥140 cutoff). Decoded RGB shows a cleanly-rendered pink T flanked
by both ALOHA gripper assemblies — the WM render quality is good for this
goal frame.

**Coverage concern, flagged for Phase 2**: at scale=128/512, `T_BLOCK_SHAPE`
total area = ~471 px. We get 415 px detected (~88%). The conservative V
upper bound (REAL=244) is shaving edge pixels. For the goal frame ICP
residual is low (0.289) so pose accuracy is unaffected, but during bulk
labelling on out-of-distribution decoder frames a tighter HSV could push
some frames below the 100-px contour-area reject threshold and inflate the
drop rate. **Action**: HSV calibration in Phase 2 will spot-check ~20
decoded frames and either (a) keep REAL preset as-is, (b) raise V upper to
255 to recover edge pixels, or (c) introduce an `iws_wm_render` preset.

**Note on preset equivalence**: because both presets returned identical
numbers, this single sanity-check run does not distinguish them. The
calibration step in Phase 2 will exercise both on a diverse mini-batch of
decoded frames to confirm the equivalence holds in distribution, not just
on the goal.

---

## 8. Phase 1 deliverables status

| Deliverable | Status |
|---|---|
| `state_estimator_research.md` | ✅ this file |
| `tests/state_estimator/sanity_check.py` | ✅ written and executed |
| `tests/state_estimator/sanity_outputs/` | ✅ 6 PNGs + `results.json` |
| `migration_log.md` | ✅ four entries: preset choice, importlib loader, sanity-check observations, GPU contention |
| `rl/visualization/state_viz.py` (§6.1 primitive) | ✅ `render_state_on_image` |
| `nvidia-smi` recorded | ✅ §6.1 (deferred) and §6.2 (run time) |
| HSV mask observations | ✅ §7 |

Phase 1 closes here. Phase 2 (`state_estimator_design.md`) starts only after
the user signs off on the §7 observations.
