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

Architecture: MLP `[4096 → 256 → 128 → 4]`, GELU, LayerNorm on hidden
layers (revised down from 2.23M to 1.08M params per Phase 2 review).

| item | size |
|------|------|
| Params | `4096·256 + 256·128 + 128·4 + biases` ≈ 1.08M params |
| Params @ fp32 | 1.08M × 4 B = **4.3 MiB** |
| Adam state (m + v) | 2 × 4.3 = **8.6 MiB** |
| Grad buffer | 4.3 MiB |
| Latent batch (16, 4, 32, 32) fp32 | 16 × 4 × 32 × 32 × 4 = **256 KiB** |
| Forward activations (16, 256) + (16, 128) | ~25 KiB |
| Backward activations | ~2× forward ≈ 50 KiB |
| Label/optim scratch + fragmentation slack | ~1 MiB |
| **Training-side total (peak)** | **≈ 18 MiB** |

18 MiB fits inside B ≈ 0.9 GiB with ~50× headroom. The large margin is
**intentional** — it lets us scale up to the 2.23M fallback architecture
or the small-CNN escape hatch (§1.12) without re-planning memory, and
absorbs any transient allocator fragmentation under a CoinRun co-tenant
without triggering CUDA OOM. We are not batch-size-limited by memory.

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

**Starting architecture** (revised per Phase 2 review):

```
Input: (B, 4096)  [flattened latent]
LinearGELU 4096 → 256
LayerNorm 256
LinearGELU 256 → 128
LayerNorm 128
Linear 128 → 4
```

Parameter count ≈ `4096·256 + 256·128 + 128·4 + biases` ≈ **1.08M params**.
Training set size ≈ 999 frames × ~1 effective-sample-per-step after
accounting for intra-episode correlation (10 Hz ALOHA). 2.23M params
overfits this effective sample count easily, so we start at 1.08M and
escalate only on evidence.

**Escalation rule**: scale up (back to `[4096 → 512 → 256 → 128 → 4]` at
2.23M, then to a small CNN if still stuck) **only if** the validation
metrics fail acceptance AND the train/val gap is small — i.e., the
model is underfitting, not overfitting. Concretely: escalate only when
`val_loss / train_loss < 1.5` AND `val_pos_p95_pooled > 3 px` (or angle
equivalent) after full training. If the gap is large (>2.5), we are
already overfitting — adding capacity worsens it; instead try
regularisation, more labels, or dropout per §1.8.

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

### Loss (revised — adds soft unit-norm term)

```
L = λ_pos  · MSE(pos_pred, pos_true_norm)                     # pos in [0,1]²
  + λ_ang  · MSE((sin_pred, cos_pred), (sin_true, cos_true))  # components in [-1, 1]
  + λ_norm · ((sin_pred² + cos_pred²) - 1)²                   # soft unit-norm on (sin, cos)
```

Scale analysis at the acceptance threshold (3 px, 5°):
- Pos MSE target ≈ (3/128)² = 5.5e-4
- sin/cos MSE target ≈ sin(2.5°)² × 2 ≈ 3.8e-3

Raw sin/cos MSE is ~7× the pos MSE at acceptance. To make the two
supervised terms contribute comparably to gradient updates, we up-weight
position: **λ_pos = 7, λ_ang = 1**. The unit-norm regulariser is cheap
and starts small: **λ_norm = 0.01**. Rationale for the regulariser:
without any constraint during training, MSE alone lets the network
output `(sin, cos)` with arbitrary magnitude (especially on CV-ambiguous
frames where the bimodal θ vs θ+180 label drives the net toward small-
magnitude outputs as a loss minimiser). Post-hoc normalisation at
inference recovers direction but training gradient signal is weaker
than it needs to be. The soft constraint fixes this at near-zero
compute cost. We still apply post-hoc `(s, c) / sqrt(s² + c² + eps)` at
RL-reward time for numerical safety — the soft regulariser doesn't
guarantee exact unit norm.

All three terms logged separately every training step so we can see if
one stalls while the others converge. Plus a val-side diagnostic:
**`val/mean_pred_norm = mean(sqrt(sin_pred² + cos_pred²))` every epoch**.
If this drifts below ~0.9 or above ~1.1 during training, `λ_norm` is
wrong — first bump to 0.05, then to 0.1.

If §1.9's "angle pass, pos fail" or vice-versa outcome appears, we
re-tune λ_pos / λ_ang. Starting point is simple and grounded.

### Overfitting monitoring (required)

Logged every epoch to TensorBoard and `train_log.csv`:

- `train_loss`, `val_loss` — scalars
- `val_loss / train_loss` — as its own scalar, the overfit signal
- `val_pos_p95_pooled`, `val_angle_p95_pooled` — trajectory, not just
  final (needed to see when overfitting begins and where best-val lives)
- `val_pos_p95_worst_episode`, `val_angle_p95_worst_episode` — per §1.9
- `val/mean_pred_norm` — unit-norm drift signal above

**Overfit flag**: if at any point `val_loss / train_loss > 2.5`, emit a
loud warning in the training log (`WARN: overfitting detected at epoch
E, val/train = X`) and save a diagnostic dump to
`outputs/state_probe/<run_name>/overfit_dumps/epoch_E.pt` containing
a batch of val predictions and targets. Training **continues** after
the warning — the signal is recorded but does not halt.

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
| Epochs | 100 max | 999 frames × 100 / 16 ≈ 6.2k steps — fast |
| Minimum epochs | 20 | MLPs on small data can show delayed generalisation — don't stop early even if val plateaus before epoch 20 |
| Early stop (patience) | 10 epochs on `λ_pos · val_pos_p95_pooled + λ_ang · val_angle_p95_pooled` (weights match the loss) | if best-val hasn't improved in 10 consecutive epochs, stop |
| Gradient clip | 1.0 | L2 norm |
| Dropout | none in first pass | add `p=0.2` after each LayerNorm only if the overfit flag (§1.7) fires or train/val gap > 2.5 at end of training |
| Mixed precision | off | model is tiny, fp32 is fine |
| Seed | 0 for probe run; sweep 0/1/2 if acceptance is marginal | |

Val evaluations every epoch (not "every 100 steps" — at batch 16 and
999 frames, one epoch is 62 steps, so per-epoch eval is natural and
cheap). The per-epoch cadence is consistent with the monitoring
requirements in §1.7.

Checkpoints: `outputs/state_probe/<run_name>/{best.pt, last.pt,
train_log.csv, val_log.csv, config.yaml}`. `best.pt` is the checkpoint
with lowest combined val MSE. The path `tests/goal_selection/state_goal.pt`
is produced separately by `scripts/compute_state_goal.py` — decodes
`z_goal.pt`, runs CV with the chosen preset, stores
`(cx, cy, sin θ, cos θ)` plus metadata.

---

## 1.9 Acceptance criteria — exact measurement

Restated per v5 §1.3 with precise statistics. Revision: report **both
pooled and per-episode worst p95**; pooled is the gate, per-episode is
diagnostic only.

| criterion | metric | threshold | gate? |
|-----------|--------|-----------|-------|
| Position (pooled) | p95 of per-frame Euclidean error, pooled across all val frames, in pixels | ≤ 3 px | **YES — blocks merge** |
| Angle (pooled) | p95 of `min(|Δθ|, 360 − |Δθ|)` across all val frames, in degrees | ≤ 5° | **YES — blocks merge** |
| Position (worst episode) | max over val episodes of the within-episode p95 position error | — | NO, diagnostic |
| Angle (worst episode) | max over val episodes of the within-episode p95 angle error | — | NO, diagnostic |
| Visual | validation grid + worst-K inspection | user sign-off | **YES — blocks merge** |

**Why both p95s**: pooled p95 is simple, interpretable, and what
downstream RL cares about (actor gradient quality over a batch of val-
distribution frames). Per-episode worst catches the "one val episode is
systematically broken and the pooled metric averages it out" failure
mode. We report per-episode worst but **don't gate on it**, because val
episode count (5) is too small for an episode-level gate to be
statistically meaningful — e.g., one bad episode raises the gate to an
artificially strict level.

Mean is reported for context but does not drive the decision. p95 forces
the tails to be reasonable; a probe with mean 1 px but worst case 30 px
is useless for RL reward because RL rollouts will hit those tail cases
routinely.

Angle error via `min(|Δθ|, 360 − |Δθ|)` handles the wrap-around: θ and
θ+360 are the same orientation. We do NOT treat θ and θ+180 as equal
here — the T-block is not rotationally symmetric, and the probe is
expected to resolve that, even though the CV labels are themselves
sometimes bimodal. If the probe converges to the mean of the two modes
instead of picking one, `min(|Δθ|, 360 − |Δθ|)` correctly penalises
that.

**Report format** (mandatory in `outputs/state_probe/<run_name>/
acceptance_report.md`):

```
Position error: p95 pooled = X.XX px,  p95 worst-episode = Y.YY px
                mean = Z.ZZ px
Angle error:    p95 pooled = X.XX°,    p95 worst-episode = Y.YY°
                mean = Z.ZZ°
Unit-norm drift: mean(sqrt(sin² + cos²)) = V.VV (target ≈ 1.0)
Merge gate (pooled p95 ≤ 3 px AND pooled p95 ≤ 5°): PASS / FAIL
Visual sign-off (user review of validation grid + worst-K): PENDING / SIGNED
```

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
    − β · (1 − (sin_pred·sin_goal + cos_pred·cos_goal)) / 2
```
where `pos_pred = (cx_pred, cy_pred)` in pixels,
`image_diagonal = √(128² + 128²) ≈ 181.02`.

### Proposed weights

**α = 1.0, β = 1.0** with the **angle term divided by 2** so both terms
are intrinsically in `[0, 1]`.

Reasoning:
- Position term is `||Δpos||₂ / diag ∈ [0, 1]`.
- Angle term raw is `(1 − cos Δθ) ∈ [0, 2]`; dividing by 2 puts it
  in `[0, 1]`.
- With both in `[0, 1]` and unit weights, position and angle contribute
  equally when errors are proportionally large. This aligns the reward
  design with the probe loss design (§1.7 weights `λ_pos = 7, λ_ang = 1`
  already balance position and angle at the MSE level for acceptance-
  scale errors — now the reward side is also balanced).
- The earlier design had an unintended 2× angle dominance from the raw
  `1 − cos Δθ` range. That was justified post-hoc as "matches the
  cosine-reward failure mode," but the more honest version is: treat
  position and angle symmetrically, and leave any future bias to an
  explicit `α ≠ β` tuning decision after we see how RL runs behave.
- At the acceptance threshold (3 px / 5°): position contributes
  `-0.0166`, angle contributes `-0.0019` (half of the old `-0.0038`).
  Both small; sanity-check that neither term saturates at the
  acceptance boundary.

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

Scalar logs (per-epoch unless noted):
- Training (per-step cadence ok): `train/loss_total`, `train/loss_pos`,
  `train/loss_ang`, `train/loss_norm`, `train/lr`, `train/grad_norm`
- Validation (per-epoch): `val/loss_total`, `val/loss_pos`,
  `val/loss_ang`, `val/loss_norm`, `val/pos_mean`, `val/pos_p95_pooled`,
  `val/pos_p95_worst_episode`, `val/ang_mean`, `val/ang_p95_pooled`,
  `val/ang_p95_worst_episode`, `val/mean_pred_norm`,
  `val/val_over_train_ratio`

Image logs: the 4 anchor-progression composite per epoch eval. Not the
full validation grid every eval — that's what
`probe_validation_grid.png` is for at end-of-training.

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

- **Symptom**: after 100 epochs (or earlier early-stop), `val_pos_p95_pooled > 3 px`
  or `val_ang_p95_pooled > 5°`, not improving over the last 20 epochs.
- **Mitigation (first)** — suspected overfit: if `val_loss / train_loss > 2.5`
  or the overfit flag (§1.7) fired, add dropout 0.2 after each LayerNorm
  and retrain.
- **Mitigation (second)** — suspected underfit: if `val_loss / train_loss < 1.5`
  at the plateau, scale up. First to `[4096 → 512 → 256 → 128 → 4]`
  (2.23M params), then to the small CNN fallback
  `(4, 32, 32) → (32, 16, 16) → (64, 8, 8) → flatten → 4`.
- **Abandon trigger**: `val_pos_p95_pooled > 10 px` OR `val_ang_p95_pooled > 15°`
  after **three architecture revisions, at least one of which is the CNN
  variant**. At that point the problem is not architecture; check labels
  (Risk 1) or consider that the latent distribution is not informative
  enough about T-block pose (would be a surprising negative result
  worth reporting).

### Risk 3 — `(sin, cos)` head collapses toward (0, 0) or unit-norm drifts

- **Symptom**: `val/mean_pred_norm` drifts below 0.9 or above 1.1
  during training. Or the predicted norm is consistently ≪ 1 on
  ambiguous frames (network learning "I don't know → predict zero" as a
  loss minimiser on bimodal labels, so MSE is reduced but direction is
  lost).
- **Mitigation (first)**: baseline loss already includes the soft
  unit-norm term `λ_norm · ((s² + c² − 1)²)` at `λ_norm = 0.01`
  (§1.7). First escalation: bump `λ_norm → 0.05`. If still drifting,
  bump to `λ_norm = 0.1`.
- **Mitigation (second)**: inspect whether collapse is correlated with
  specific CV residual ranges (worst ICP fits → worst probe predictions)
  and if so tighten the CV residual reject bound upstream (Risk 1
  mitigations).
- **Abandon trigger**: `λ_norm = 0.1` still leaves `mean_pred_norm < 0.8`
  on > 20% of val frames — indicates the CV labels are systematically
  bimodal / inconsistent and no amount of regularisation fixes it.
  Return to Risk 1 mitigations (HSV re-calibration, ICP residual
  tightening) and consider a ranking-based loss instead of MSE.

Secondary concerns flagged but not top-3: decoder-distribution drift vs
training latents (cannot be validated in Branch A, surface in Branch B);
train/val episode distribution shift (mitigated by §1.4 "measure first"
decision).

---

## 1.13 Self-check — confirming the four coverage items

Phase 2 review asked me to verify each of (a)–(d) is addressed in the
doc. Here's the pointer table:

| Item | Addressed in | Specifics |
|------|--------------|-----------|
| (a) Per-frame REAL-vs-WM mask agreement reported across 20 calibration frames | §1.1 (agreement reporting) + §1.3 (`agreement_flag` column in `calibration_table.csv`, + agreement-rate required in `calibration_summary.md`'s three mandatory numbers) | Every calibration row logs `n_pixels_wm_only`, `pose_delta_px`, `pose_delta_deg`, and `agreement_flag ∈ {identical, mask_differs_but_pose_agrees, mask_and_pose_differ, one_failed}` |
| (b) Per-channel ablation: H band widening, S lower relaxation, V upper lifting tested independently | §1.2 (Per-channel relaxation test) | Six variants in the table: baseline, H ±5°, S_lower → 0, S_upper → 255, V_lower → 0, V_upper → 255. Each channel relaxed one at a time, area recovery reported per variant in `per_channel_relaxation.csv`. |
| (c) VRAM arithmetic: concrete table with params × 4 B + Adam 2× + activation estimate | §1.5 (revised table) | Params 1.08M → fp32 4.3 MiB, Adam 2× → 8.6 MiB, grad 4.3 MiB, latent batch 256 KiB, activations 25 + 50 KiB, slack 1 MiB → peak ≈ 18 MiB. 50× headroom explicitly flagged as intentional margin, not slop. |
| (d) Abandon triggers as specific numeric thresholds | §1.12 all three risks | Risk 1 abandon: drop rate > 20% after both mitigations OR >5/20 calibration frames visually broken. Risk 2 abandon: `val_pos_p95_pooled > 10 px` OR `val_ang_p95_pooled > 15°` after three architecture revisions including at least one CNN variant. Risk 3 abandon: `mean_pred_norm < 0.8 on >20% of val` after regulariser bumped to `λ_norm = 0.1`. |

All four items were already addressed before this revision (except
Risk 2's "at least one CNN variant" clarification and Risk 3's "after
regulariser bumped" clarification — both added now to make the
thresholds sharper).

---

## 1.14 Branch split recap

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

These I decided and wrote down. Items marked ✱ were revised per the
Phase 2 review round:

1. Label every frame (no subsampling). §1.4.
2. No coverage stratification for val set; re-visit only if the
   validation grid shows holes. §1.4.
3. ✱ MLP, not CNN, **at 1.08M params** `[4096 → 256 → 128 → 4]` (shrunk
   from 2.23M). Escalation to 2.23M and then CNN gated on underfit
   evidence (`val/train < 1.5` at plateau). §1.7, §1.12.
4. Normalise position targets to `[0, 1]` for training; `(sin, cos)`
   supervised by MSE against unit-norm targets + soft unit-norm
   regulariser during training; post-hoc `F.normalize` at inference.
   §1.7.
5. ✱ Loss weights `λ_pos = 7, λ_ang = 1, λ_norm = 0.01`. λ_norm added
   this round. §1.7.
6. ✱ Reward weights `α = β = 1.0` with the angle term divided by 2 so
   both terms live in `[0, 1]` and contribute equally. Previous 2×
   angle dominance was unintentional scale mismatch, now removed.
   §1.10.
7. Training progression viz (v5 §6.4 optional) is **in scope** for Phase
   3-A, at trivial cost. §1.11.
8. Serial labeling pipeline, no multiprocessing, given ~6-min estimate.
   §1.6.
9. ✱ p95 drives acceptance. **Both pooled and per-episode worst** p95
   reported; only pooled gates merge. §1.9.
10. Visual sign-off = user (task owner). §1.9.
11. (new) Early stop: patience 10 epochs on the combined metric
    `λ_pos · val_pos_p95_pooled + λ_ang · val_angle_p95_pooled`, minimum
    20 epochs. §1.8.
12. (new) Overfit flag at `val_loss / train_loss > 2.5` — warn + dump,
    training continues. §1.7.

If any of these is wrong, flag it in review and I'll revise. Otherwise
Phase 3-A starts from this as written.

---

## Revision Notes (Phase 2 review round)

Revisions from the Phase 2 review round had no second-order consequences
that required working around. (Review-round items labelled "Review 1.x"
below; they land in this doc at different §1.x numbers, noted in
parentheses.) For completeness:

- **Review 1.1 — Shrunk architecture** (lands in this doc's §1.7 + §1.12):
  VRAM peak dropped from 36 MiB to 18 MiB (this doc's §1.5 recomputed).
  Conclusion unchanged — not memory-limited. Fallback architectures
  (2.23M MLP, small CNN) still fit inside the worst-case probe-training
  budget with >30× headroom.
- **Review 1.2 — λ_norm added** (lands in this doc's §1.7 + §1.11):
  adds one extra loss-term-per-step gradient computation; cost
  negligible. `val/mean_pred_norm` added to the TensorBoard scalar list.
  Risk 3 mitigation ladder (§1.12) updated from "add regulariser" to
  "bump regulariser" since the regulariser is now in the baseline loss.
- **Review 1.3 — Reward angle /2** (lands in this doc's §1.10): pure
  scalar change in `rl/training/reward.py` (Branch B). No impact on
  Branch A scope or Branch A artefacts. The saved `state_goal.pt`
  format is unchanged.
- **Review 1.4 — Dual p95 reporting** (lands in this doc's §1.9 +
  §1.11): adds 2 scalar logs per eval and one more line in
  `acceptance_report.md`. No impact on training time.
- **Review 1.5 — Self-check** (this doc's §1.13, new): existing Branch
  split recap moved to §1.14. Sanity-check against v5 §2.1 / §2.2 file
  lists still holds.
