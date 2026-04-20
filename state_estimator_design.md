# State Estimator Design — Phase 2

Branch: `state-probe-training` (Branch A)
Date: 2026-04-20
Phase: Design only. No code changes in this commit set.

Section numbering mirrors v5 §1 for line-by-line review.

---

## 1.1 HSV preset equivalence — corrected framing

Phase 1's framing ("WM is a superset of REAL, our pixels fall in the REAL
subset") was geometrically correct for the single `z_goal` frame but
overreached as a generalisation. Corrected statement:

> For a given frame, the two presets produce identical masks **iff every
> pixel that either preset selects falls inside the pixel-wise
> intersection of the two masks** — i.e., inside the REAL subset. One
> frame exhibiting this does not imply the property holds across the
> decoder's full output distribution. A single frame containing even one
> pixel in the WM-only region (H∈[140, 160), or S>200, or V>244) breaks
> the equivalence for that frame.

### Calibration questions (explicit)

The Phase 2 calibration answers both of these with quantitative evidence,
not hand-waving:

1. **Absolute performance**: which preset (or a tuned `iws_wm_render`)
   minimises `drop_rate` and maximises `contour_area` on our decoder
   output distribution, given fixed ICP hyperparameters?
2. **Agreement**: on what fraction of our decoded frames do REAL and WM
   presets produce **pixel-identical** masks? On disagreement, which
   pixels does each preset include that the other does not, and does the
   disagreement move the downstream `(cx, cy, θ)` estimate by more than
   the probe's acceptance tolerance (3 px / 5°)?

### Agreement reporting

For every calibration frame the report will record:
- `n_pixels_real_only` (in REAL mask, not WM mask) → always 0 since REAL⊂WM
- `n_pixels_wm_only` (in WM mask, not REAL mask) → the telling number
- `pose_delta_px`, `pose_delta_deg` = |pose_REAL − pose_WM| where both succeed
- `agreement_flag`: `identical` / `mask_differs_but_pose_agrees` /
  `mask_and_pose_differ` / `one_failed`

If agreement flag is `identical` on all 20 calibration frames, we report
"equivalence holds on the calibration set" and move forward with REAL;
otherwise we pick the preset that wins on drop rate and flag any
disagreement in `migration_log.md`.

---

## 1.2 Area-shortfall root cause — diagnosis plan

Phase 1 showed goal-frame detected area 415 px vs theoretical ~471 px
(88% coverage). Multiple plausible causes; we isolate rather than
assume.

### Per-channel relaxation test

For each of the 20 calibration frames and both presets, run six variants
and report raw-mask pixel count:

| variant         | H lower | H upper | S lower | S upper | V lower | V upper |
|-----------------|---------|---------|---------|---------|---------|---------|
| baseline        | preset  | preset  | preset  | preset  | preset  | preset  |
| H widened ±5°   | −5      | +5      | preset  | preset  | preset  | preset  |
| S_lower → 0     | preset  | preset  | 0       | preset  | preset  | preset  |
| S_upper → 255   | preset  | preset  | preset  | 255     | preset  | preset  |
| V_lower → 0     | preset  | preset  | preset  | preset  | 0       | preset  |
| V_upper → 255   | preset  | preset  | preset  | preset  | preset  | 255     |

Report mean area recovery in **percentage points** (of theoretical 471px
at 128-res).

### Raw vs post-morphology sizes

Every calibration row in §1.3 reports **both** `raw_mask_px` (output of
`cv2.inRange` alone) and `post_morph_px` (after the 3×3 open+close). Gap
between them quantifies morphology erosion independent of threshold
tightness.

### Decision criterion

The chosen preset must satisfy all three:
1. `drop_rate` ≤ 10% across calibration frames (drop = no contour ≥ 100 px
   found or ICP residual above an upper bound determined empirically).
2. `post_morph_px` median ≥ 90% of theoretical area.
3. **No new false positives**: no added contour area attributable to
   gripper tips, shadows, or other non-T-block regions. Visual eyeball
   check on the calibration grid (me + doc reviewer).

If a single-channel relaxation clears the bar, use it. Only introduce a
new `iws_wm_render` preset if multi-channel widening is required.

---

## 1.3 Calibration report format

`outputs/state_probe/hsv_calibration/` holds:

1. `calibration_table.csv` with one row per `(frame_id, preset_variant)`:
   ```
   frame_id, episode, t_idx, preset, raw_mask_px, post_morph_px,
   contour_count, contour_area, icp_residual, cv_success,
   n_pixels_wm_only_vs_real, pose_delta_px, pose_delta_deg,
   agreement_flag, notes
   ```
2. `calibration_grid.png` — 5×4 grid of 20 frames, each showing:
   decoded RGB, post-morph mask for chosen preset, annotated overlay.
3. `per_channel_relaxation.csv` — 6 variants × 20 frames area recovery
   table.
4. `calibration_summary.md` — short prose summarising:
   - Drop rate per preset (REAL / WM / any tuned variant)
   - Mean/median area recovery per relaxation
   - Per-frame REAL-vs-WM disagreement rate
   - **Decision + the numbers that justify it**.

The decision paragraph in `calibration_summary.md` must quote three
quantities: drop rate, area recovery, disagreement rate. If any of the
three is not a number, the doc fails review.

---

## 1.4 Frame sampling for bulk labeling

### Inventory (measured, not guessed)

| split | episodes | frames per episode | total frames |
|-------|----------|--------------------|--------------|
| train | 5 (episode_{0..4}.hdf5) | 200, 200, 200, 200, 199 | **999** |
| val   | 5 (episode_{0..4}.hdf5) | 200, 200, 200, 200, 200 | **1000** |
| **total** | **10** | — | **1999** |

Image source: `data/mini/pusht/{train,val}/episode_*.hdf5`, key
`obs/images/camera_1_color`, 480×640 uint8 RGB. Preprocessing: centre-crop
480×480 → resize 128×128 → divide by 255 → normalizer.

### Sampling policy

Label **every frame**, not subsampled. Rationale:
- Frames within an episode are correlated (10 Hz ALOHA); adjacent frames
  differ by a single small action. But the probe isn't learning dynamics —
  it's learning a per-frame latent→pose map. Correlated labels don't bias
  it; more data helps.
- 1999 frames total is small. Subsampling trades data for a tiny speed
  gain that doesn't matter given §1.6's ~6-min serial estimate.
- A subset of frames will be dropped by CV quality filters (§3.1); the
  "label every frame" policy is the upstream max — drop rate applies
  post-filter.

### Train/val split

Use the **existing dataset split**. Train episodes (IDs 0–4 in
`data/mini/pusht/train/`) for probe training; val episodes (IDs 0–4 in
`data/mini/pusht/val/`) for probe acceptance. Split is by episode — the
v5 §3.2 requirement — because train and val live in different
directories and contain distinct trajectories. No random frame-level
shuffle across the boundary.

### Coverage stratification

Skipped. Justification: with 5 val episodes × 200 frames each, the val
set covers the PushT pose distribution by sheer volume. Manually
stratifying 1000 frames by (cx, cy, θ) bins adds complexity for no
evident gain. If the validation grid (§1.11) shows systematic holes —
e.g., all val frames clustered in one quadrant of the image — we
reconsider. This is a "measure first" decision, not an open question.

### Angle normalisation at label time

Every label goes through a one-time normalisation **before** storage:
```
theta_deg_raw       ← supervisor's estimate_current_pose output
theta_deg_normalised = ((theta_deg_raw + 180.0) % 360.0) - 180.0   # (-180, 180]
theta_rad           = radians(theta_deg_normalised)
label               = (cx, cy, sin(theta_rad), cos(theta_rad))
```
This addresses the v5 §3.2 requirement "do the angle conversion at
label time" and eliminates the cross-frame `+240°` vs `+60°`
inconsistencies we saw in the Phase 1 multi-frame viz.

---

## 1.5 GPU memory budget — arithmetic

### Device and co-tenant accounting

| quantity | value | source |
|----------|-------|--------|
| Device total | 12.0 GiB (12227 MiB) | `nvidia-smi`, 2026-04-20 |
| CoinRun worst-case co-tenant | 9.8 GiB (10059 MiB) | observed peak during Phase 1 blocker |
| WM decoder resident | 0.87 GiB (890 MiB) | Phase 1 measurement (load + one decode) |
| Safety headroom (30%) | 0.4 GiB | ≥30% of post-co-tenant free (per §8.2) |
| **Probe training budget B** | **≈ 0.9 GiB** | 12.0 − 9.8 − 0.87 − 0.4 |

Current reality (2026-04-20): CoinRun not running → free ≈ 11.8 GiB, so
in practice we have far more headroom. The 0.9 GiB budget is the
worst-case design point to make the pipeline robust under co-tenant
return.

### Probe footprint at batch 16 (architecture per §1.7)

Architecture: MLP `[4096 → 512 → 256 → 128 → 4]`, GELU, LayerNorm on
hidden layers.

| item | size |
|------|------|
| Params | `4096·512 + 512·256 + 256·128 + 128·4 + biases` ≈ 2.23M params |
| Params @ fp32 | 2.23M × 4 B = **8.9 MiB** |
| Adam state (m + v) | 2 × 8.9 = **17.8 MiB** |
| Grad buffer | 8.9 MiB |
| Latent batch (16, 4, 32, 32) fp32 | 16 × 4 × 32 × 32 × 4 = **256 KiB** |
| Forward activations (16, 512) + (16, 256) + (16, 128) | ~60 KiB |
| Backward activations | ~2× forward ≈ 120 KiB |
| **Training-side total (peak)** | **≈ 36 MiB** |

36 MiB fits inside B ≈ 0.9 GiB with 25× headroom. We will not be batch-
size-limited by memory at any credible architecture size.

### Batch size decision

Keep v5 §8.2 default of **batch 16** for the initial training run;
measure peak after 10 steps; only scale up if probe val loss plateaus
high (§1.12 risk). Given the budget, scaling to 256 or 1000 (full train
set per batch) is feasible if useful. First-pass default stays at 16 on
general-purpose grounds — stochasticity in mini-batch SGD is known to
help generalisation on small datasets.

---

## 1.6 Bulk labeling — parallelism plan

### Serial time estimate (measured components)

| component | per-frame | total (1999 frames) |
|-----------|-----------|----------------------|
| HDF5 read + preprocess | ~1 ms | 2 s |
| Encoder `encode()` | batched @ 32: ~5 ms/frame | 10 s |
| Decoder `decode()` | 0.74 s → batched @ 8 ≈ 0.1 s/frame | ~200 s |
| CV pipeline (HSV + contour + ICP) | 0.07 s | ~140 s |
| **Total serial** | — | **≈ 6 minutes** |

Dominant cost is the decoder (~50% of wall), CV second (~35%).

### Decision: **no parallelisation**

6 minutes is under the 20-minute threshold v5 §1.6 sets for the
parallelisation bar. Adding a decode-GPU / CV-CPU pipeline with a
multiprocessing pool would shave maybe 2–3 minutes at the cost of
nontrivial bug surface (inter-process HDF5 file handles, tqdm
synchronisation, signal handling on worker OOM). Not worth it.

The labeling script runs in a single process with:
- Decoder batch 8 on GPU (bounded by our ~3 GiB probe-training budget
  note in §1.5 — decoder activations are the bigger cost at batch 8, but
  well under WM's 0.87 GiB steady-state).
- CV on the main CPU thread after each decoded batch is copied to host.

### Failure-mode handling

CV quality filters (per v5 §3.1):
- `contour_count == 0` → drop, reason=`no_contour`
- `largest_contour_area < 100` → drop, reason=`contour_too_small`
- `icp_residual > RESIDUAL_MAX` → drop, reason=`icp_diverged`
- `None` returned → drop, reason=`cv_failed`

`RESIDUAL_MAX` is set empirically from calibration (§1.3): take p95 of
residuals on successful calibration frames, add 50% margin. Dropped
frames are NOT silently discarded — the labeling script writes a
`drops.csv` (`episode`, `t_idx`, `reason`, `raw_mask_px`,
`contour_area`, `icp_residual`) and prints a summary at end. If
post-filter drop rate > 10% on the train split, we stop and calibrate
HSV again before probe training.

Labels file format: `outputs/state_probe/labels/labels_{split}.pt`,
a dict containing:
```python
{
    "episodes": List[int],          # episode id per frame
    "t_idx":    LongTensor (N,),
    "latents":  Tensor (N, 4, 32, 32),  # optional cached z (§1.7)
    "labels":   Tensor (N, 4),      # (cx, cy, sin, cos) — cx, cy in pixels
    "meta":     {preset, hsv, ckpt_hash, drop_rate, ...}
}
```
The labels tensor stores `cx, cy` in raw pixels (0–128) not normalised;
normalisation happens inside the training loader so the raw numbers stay
human-inspectable.

---

## 1.7 Probe architecture — choice and justification

### MLP vs CNN

**Decision: MLP.**

Reasoning:
1. The latent `(4, 32, 32)` is already spatially structured by the WM
   encoder. The encoder is trained for reconstruction (and dynamics),
   which preserves spatial layout — a pixel at `(row, col)` in latent
   space corresponds to a consistent image region. An MLP over the
   flattened 4096-dim vector does not need translation equivariance
   because the downstream task is a global pose regression, not a
   per-patch classification.
2. A small CNN on `(4, 32, 32)` would be roughly the same parameter
   budget as the MLP, and would force us to learn features the encoder
   already encoded. The marginal accuracy gain is unlikely to clear the
   3-px / 5° acceptance bar difference.
3. MLP code is trivially simpler (no padding, no conv shape bookkeeping)
   and trivially fast at this scale.

We keep CNN as a §1.12 risk-mitigation escape hatch if the MLP plateaus
above acceptance.

### Hidden dims and depth

```
Input: (B, 4096)  [flattened latent]
LinearGELU 4096 → 512
LayerNorm 512
LinearGELU 512 → 256
LayerNorm 256
LinearGELU 256 → 128
LayerNorm 128
Linear 128 → 4
```

Parameter count ≈ 2.23M. Training set size ≈ 999 frames → ratio
parameters/frames ≈ 2200. This is high in absolute terms but the task
is extremely constrained (4 scalar outputs per frame, deterministic
labels from CV). LayerNorm + moderate depth regularises well enough at
this scale; a 2-layer `[4096 → 256 → 4]` shallower variant is the
fallback (§1.8 dropout also discussed).

### Output head normalisation

Single linear head, no activation, 4 raw outputs. Convention:
- Position outputs: trained to match `cx/128, cy/128` (normalised to
  [0, 1]). `MSE` in normalised space keeps gradient scale unit-free;
  at inference time multiply by 128 to get pixels. No sigmoid — we want
  out-of-range values to be visible errors, not silently clamped.
- `(sin, cos)` outputs: raw linear. No tanh, no `F.normalize`. MSE
  against unit-norm targets pushes the network toward the unit circle
  naturally; forcing exact unit norm via `F.normalize` in the forward
  pass creates a singularity at origin (gradient blows up) that we don't
  need. At inference time for reward, we do `(s, c) / sqrt(s² + c² + eps)`
  before computing `1 − (s·s_g + c·c_g)` — one line, no training
  complication.

### Loss

```
L = λ_pos · MSE(pos_pred, pos_true_norm)   # pos in [0,1]²
  + λ_ang · MSE(sincos_pred, sincos_true)  # sin, cos in [-1, 1]
```

Scale analysis at the acceptance threshold (3 px, 5°):
- Pos MSE target ≈ (3/128)² = 5.5e-4
- sin/cos MSE target ≈ sin(2.5°)² × 2 ≈ 3.8e-3

Raw sin/cos MSE is ~7× the pos MSE at acceptance. To make both terms
contribute comparably to gradient updates, we up-weight position:
**λ_pos = 7, λ_ang = 1**. Both terms logged separately every step so we
can see if one stalls while the other converges.

If §1.9's "angle pass, pos fail" or vice-versa outcome appears, we
re-tune these. Starting point is simple and grounded.

---

## 1.8 Probe training config

Concrete choices (all targetting a stable first run, not optimised):

| knob | value | note |
|------|-------|------|
| Optimizer | AdamW | standard |
| LR | 3e-4 | MLP-regression default |
| LR schedule | cosine with linear warmup 500 steps, min LR 1e-6 | |
| Weight decay | 1e-4 | LayerNorm/bias excluded |
| Batch size | 16 (§1.5) | |
| Epochs | 100 | 999 frames × 100 / 16 ≈ 6.2k steps — fast |
| Early stop | val p95 pos ≤ 2 px AND val p95 ang ≤ 3° for 5 consecutive evals | evals every 100 steps |
| Gradient clip | 1.0 | L2 norm |
| Dropout | none in first pass | add p=0.2 after LayerNorm only if overfitting |
| Mixed precision | off | model is tiny, fp32 is fine |
| Seed | 0 for probe run; sweep 0/1/2 if acceptance is marginal | |

Checkpoints: `outputs/state_probe/<run_name>/{best.pt, last.pt,
train_log.csv, val_log.csv, config.yaml}`. `best.pt` is the checkpoint
with lowest combined val MSE. The path `tests/goal_selection/state_goal.pt`
is produced separately by `scripts/compute_state_goal.py` — decodes
`z_goal.pt`, runs CV with the chosen preset, stores
`(cx, cy, sin θ, cos θ)` plus metadata.

---

## 1.9 Acceptance criteria — exact measurement

Restated per v5 §1.3 with precise statistics:

| criterion | metric | threshold | failure mode handling |
|-----------|--------|-----------|------------------------|
| Position | **p95** Euclidean error on val set, in pixels | ≤ 3 px | see below |
| Angle | **p95** `min(|Δθ|, 360 − |Δθ|)` on val set, in degrees | ≤ 5° | see below |
| Visual | validation grid + worst-K inspection | passes user sign-off | me |

Mean is reported for context but does not drive the decision. p95 forces
the tails to be reasonable; a probe with mean 1 px but worst case 30 px
on a small val set is useless for RL reward because RL rollouts will hit
those tail cases routinely.

Angle error via `min(|Δθ|, 360 − |Δθ|)` handles the wrap-around: θ and
θ+360 are the same orientation. We do NOT treat θ and θ+180 as equal
here — the T-block is not rotationally symmetric, and the probe is
expected to resolve that, even though the CV labels are themselves
sometimes bimodal. If the probe converges to the mean of the two modes
instead of picking one, `min(|Δθ|, 360 − |Δθ|)` correctly penalises
that.

### Visual sign-off

**User (task owner) signs off.** The doc commits to this so there's no
ambiguity at merge time. Branch A does not merge until the user OKs the
validation grid and worst-K images.

### Split failure outcomes

- **Position passes (p95 ≤ 3 px), angle fails (p95 > 5°)**: not shippable.
  Actions in order: (a) raise `λ_ang` to 2 or 3; (b) inspect worst-K
  angle frames — are they all near θ=±π wraps? If yes, the network is
  finding a local minimum that predicts sin=cos=0 for ambiguous frames.
  Add a unit-norm regulariser `(s² + c² − 1)²` × 0.1 during training.
  (c) If still failing, bump probe capacity by one layer.
- **Angle passes, position fails**: not shippable. Suggests the MLP is
  learning the angle (which depends on local shape) but missing position
  (which depends on global spatial layout, precisely what a CNN would
  handle better). Action: drop to `[4096 → 256 → 4]` MLP (simpler,
  less overfitting) OR switch to a small CNN `(4, 32, 32) → (32, 16, 16)
  → (64, 8, 8) → Flatten → 4`. Decide based on whether the worst-position
  cases have any spatial pattern (all frames where T is near image edge?
  those near arm occlusion?).
- **Both fail**: labels are the prime suspect. Re-audit HSV calibration
  and ICP residual distribution before re-training.

---

## 1.10 Reward function (Branch B preview)

```
r_state(z_pred, z_goal_state) =
    − α · ||pos_pred − pos_goal||₂ / image_diagonal
    − β · (1 − (sin_pred·sin_goal + cos_pred·cos_goal))
```
where `pos_pred = (cx_pred, cy_pred)` in pixels, `image_diagonal = √(128² + 128²) ≈ 181.02`.

### Proposed weights

**α = 1.0, β = 1.0**.

Reasoning:
- The position term is in `[0, 1]` (normalised by diagonal).
- The angle term is in `[0, 2]` (`1 − cosΔ` ranges over that).
- Unweighted → angle dominates by 2×.
- The cosine reward's documented failure mode is "arm-sensitive, T-block-
  blind." The T-block's **orientation** is the specific signal the
  cosine reward ignores most (position of the T does show up in the
  latent's high-level structure; orientation is suppressed by the
  L2-normalised `32`-norm constraint). So biasing the reward toward
  angle matches the failure we're fixing.
- A 2× angle-dominance is mild — at the acceptance threshold
  (3 px / 5°), position contributes `-0.0166` and angle contributes
  `-0.0038`, so actually position dominates at small errors while angle
  dominates at large errors. This cross-over is fine; it pushes the
  policy to first fix orientation roughly, then polish position.

### Shaping: dense every step

Dense shaped reward, computed at every imagination step. Sparse is a
non-starter — Dreamer's policy gradient needs non-zero signal
throughout the rollout horizon H=15 or actor updates are noise.

### Reward floor / exponentiation

**No clamping, no exp.** Expected range of `r_state` given the reward
formula is `[-3, 0]` (≤ 1 from position + ≤ 2 from angle). Actor network
handles that range natively. An exp is a common "make reward positive"
hack but introduces a new non-linearity; let's not add unless we see an
actor-learning issue in Branch B.

### Goal state source

`tests/goal_selection/state_goal.pt` is produced by
`scripts/compute_state_goal.py`: loads `z_goal.pt`, decodes, runs CV with
the chosen HSV preset, stores `(cx, cy, sin θ, cos θ)` plus the preset
metadata for reproducibility. Branch A ships this file. Branch B
consumes it.

---

## 1.11 Visualisation plan

### Output paths

```
outputs/state_probe/
├── hsv_calibration/
│   ├── calibration_table.csv
│   ├── per_channel_relaxation.csv
│   ├── calibration_grid.png      # 5×4 tiled calibration frames
│   └── calibration_summary.md
├── <run_name>/
│   ├── probe_validation_grid.png  # 4×4, 16 val frames
│   ├── probe_worst_cases.png      # 3×4, 12 worst-error frames
│   ├── anchor_progression/        # optional (§1.11 below)
│   │   └── step_XXXXXX.png
│   ├── tb/                        # TensorBoard event files
│   ├── best.pt, last.pt
│   ├── train_log.csv, val_log.csv
│   └── config.yaml

tests/state_estimator/sanity_outputs/           # Phase 1 (shipped)
outputs/rl_runs/<run_name>/probe_viz/           # Branch B, hook only
```

### Validation grid (v5 §6.2, required)

16 tiles from the val set. Selection: **stratified by error quartile on
the first val eval** — 4 tiles from best-25%, 4 from 25–50%, 4 from
50–75%, 4 from worst-25%. Same 16 frames reused at every subsequent
eval (fixed seed across runs = the quartile boundaries from epoch 1).
This shows the probe improving on already-hard cases, not just a
cherry-picked easy subset.

Each tile: decoded RGB at ×4 upscale, CV label (green arrow),
probe prediction (red arrow). Caption: `ep{E} t={T}  pos_err={X}px
ang_err={Y}°`.

### Worst-K (v5 §6.3, required)

Top-12 val frames by combined error
(`pos_err_norm + (1 − sincos_dot)/2`, both in [0, 1]). Same tile format.
If a pattern jumps out (all near-occlusion, all near θ=±π wraps, all
edge-of-frame) it's flagged to the user in the Branch A merge request.

### Training progression (v5 §6.4) — **in scope**, lightweight

4 anchor frames picked from val (one per quartile of error at epoch 1).
At each val eval (every 100 steps), render overlay for each anchor; save
as single composite PNG per eval step; append all to TensorBoard as an
image sequence. Cost: 4 forward passes × 60 evals = trivial.

### RL rollout hook (v5 §6.5, Branch B, build-but-don't-exercise)

Config flag: `rl.visualisation.probe_overlay: bool = False`. When on,
the imagination loop dumps one MP4 / GIF per rollout at interval
`log_every_rollout_viz: int = 1000` training steps. Default off. Files
land in `outputs/rl_runs/<run_name>/probe_viz/step_XXXXX/`. The code path
is built in Branch B; per v5 §8.3, we do NOT exercise it.

### TensorBoard

Scalar logs: `train/loss_pos`, `train/loss_ang`, `val/pos_mean`,
`val/pos_p95`, `val/ang_mean`, `val/ang_p95`, LR, grad norm.
Image logs: the 4 anchor-progression composite per eval. Not the full
validation grid every eval — that's what `probe_validation_grid.png` is
for at end-of-training.

---

## 1.12 Risks and triggers

### Risk 1 — CV labels are noisy / bimodal in angle

- **Symptom**: `calibration_summary.md` shows ICP residual distribution
  bimodal, or drop rate > 10% even after relaxation, or Phase 1's
  observed θ vs θ+180 bimodality replicated across >10% of calibration
  frames.
- **Mitigation (first)**: tighten the ICP contour-area reject threshold
  from 100 → 200 px; introduce a lower bound on ICP residual ratio
  (`best_error / second_best_error` < 0.9) to reject ambiguous fits.
  Report new drop rate.
- **Mitigation (second)**: re-seed ICP initial rotation count from 12 to
  36 (every 10° instead of 30°) — kills ambiguous-fit failure modes at
  the cost of 3× CV time, still trivial at our scale.
- **Abandon trigger**: drop rate stays > 20% after both mitigations, OR
  visual calibration grid still shows obvious incorrect mask/pose on
  > 5 of 20 frames. At that point, the CV pipeline isn't sufficient
  evidence to build a probe against — flag back to user, may need to
  swap CV for a hand-labelled seed set.

### Risk 2 — Probe val accuracy plateaus above acceptance

- **Symptom**: after 100 epochs, val p95 pos > 3 px or val p95 ang > 5°,
  not improving over last 20 epochs.
- **Mitigation (first)**: review `train` vs `val` curves — if train loss
  is much lower than val, add dropout 0.2 after each LayerNorm.
- **Mitigation (second)**: if train and val both plateau, architecture
  is underfit — switch to small CNN (spec in §1.9 split-failure
  outcomes) OR add a `[4096 → 1024 → 256 → 4]` variant.
- **Abandon trigger**: val p95 pos > 10 px OR val p95 ang > 15° after
  three architecture revisions. At this point the problem is not
  architecture; check labels or consider that the latent distribution is
  not informative enough about T-block pose (would be a surprising
  negative result worth reporting).

### Risk 3 — `(sin, cos)` head collapses toward (0, 0) or unit-norm drifts

- **Symptom**: val sin/cos MSE converges but the predicted
  `sqrt(sin² + cos²)` drifts away from 1 (expected 1.0, see e.g.
  mean in logs). Or the predicted norm is consistently ≪ 1 on ambiguous
  frames (network learning "I don't know → predict zero" as a loss
  minimiser on bimodal labels).
- **Mitigation (first)**: add unit-norm regulariser `γ · (s² + c² − 1)²`,
  γ = 0.1.
- **Mitigation (second)**: inspect whether collapse is correlated with
  specific CV residual ranges (worst ICP fits → worst probe predictions)
  and if so tighten the CV residual reject bound upstream.
- **Abandon trigger**: regulariser doesn't restore unit norm (mean
  predicted norm stays < 0.8 on >20% of val) — indicates labels are too
  noisy; back to Risk 1 mitigations.

Secondary concerns flagged but not top-3: decoder-distribution drift vs
training latents (cannot be validated in Branch A, surface in Branch B);
train/val episode distribution shift (mitigated by §1.4 "measure first"
decision).

---

## 1.13 Branch split recap

All items in this design doc with concrete file impact map to v5 §2 as
follows:

**Branch A (`state-probe-training`, this branch):**
- §1.1–1.6: HSV calibration + bulk labeling
  → `rl/labeling/cv_labeler.py`, `scripts/label_replay_buffer.py`.
- §1.7–1.9: probe architecture + training + acceptance
  → `rl/models/state_probe.py`, `scripts/train_state_probe.py`,
  `tests/state_estimator/test_probe.py`, `tests/state_estimator/test_labeling.py`.
- §1.10 goal state artefact production → `scripts/compute_state_goal.py`
  (produces `tests/goal_selection/state_goal.pt`).
- §1.11 visualisation primitives + validation/worst-K grids + anchor
  progression → reuse `rl/visualization/state_viz.py` (already in
  Branch A), plus grid-composition helpers inside probe training
  script.
- All §1.12 mitigations.

**Branch B (`state-reward-integration`, after A merges):**
- §1.10 reward function implementation + config wiring
  → `rl/training/reward.py` (new), modify `rl/training/imagination.py`
  (reward call + probe load), modify `rl/utils/config.py` (`reward_mode`).
- §1.11 RL rollout hook **build only**, not exercised.
- New tests: `tests/state_estimator/test_reward.py`,
  `tests/state_estimator/test_probe_gradient.py`.

The Branch A files do **not** touch `rl/training/`, `rl/utils/`,
`rl/models/{actor, critic, world_model}.py`, or `main.py`. The Branch B
files do **not** touch `rl/models/state_probe.py`, `rl/labeling/`,
`rl/visualization/`, or any Branch A script. Cross-checked against v5
§2.1 and §2.2 file lists.

---

## Summary of judgment calls made without user input (flagged per §2 process rule)

These I decided and wrote down. Flagging here as a batch so you can push
back if you want:

1. Label every frame (no subsampling). §1.4.
2. No coverage stratification for val set; re-visit only if the
   validation grid shows holes. §1.4.
3. MLP, not CNN, at 2.23M params. §1.7.
4. Normalise position targets to `[0, 1]` for training; unit-norm (sin,
   cos) via post-hoc division at inference, not training. §1.7.
5. Loss weights `λ_pos = 7, λ_ang = 1`. §1.7.
6. Reward weights `α = β = 1.0` (letting angle dominate 2×). §1.10.
7. Training progression viz (v5 §6.4 optional) is **in scope** for Phase
   3-A, at trivial cost. §1.11.
8. Serial labeling pipeline, no multiprocessing, given ~6-min estimate.
   §1.6.
9. p95 (not mean) drives acceptance. §1.9.
10. Visual sign-off = user (task owner). §1.9.

If any of these is wrong, flag it in review and I'll revise. Otherwise
Phase 3-A starts from this as written.
