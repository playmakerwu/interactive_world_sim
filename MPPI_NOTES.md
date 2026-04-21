# MPPI on IWS — Design Notes and Results

A minimal MPPI (Model Predictive Path Integral) controller layered on top of the
pretrained IWS latent world model, using the classical CV T-block pose estimator
(vendored from the supervisor's repo at `rl/labeling/cv_labeler.py`) as reward.

**Purpose:** diagnose whether the IWS world model + CV state estimator, as a
pair, carry enough signal to close a control loop on PushT before we commit
more effort to the learned-probe / Dreamer-RL path that has stalled on sin-cos
collapse.

This is a diagnostic, not a production controller. No warm start, no
refinement, no CEM/CMA-ES extensions in v0.

## Algorithm (v0)

At each control step t:
1. Sample N action sequences `a_k[0..H-1] ~ N(0, σ² I)`.
2. Batched rollout through IWS dynamics: `z_k[t'+1] = F_ψ(z_k[t'], a_k[t'])`.
3. Decode last-step latents: `rgb_k = D_θ(z_k[H])`.
4. Reward per trajectory:
   `R_k = −||pos_k − pos_goal||₂ / image_diagonal − |1 − (sin_k·sin_goal + cos_k·cos_goal)|`
   with CV-failure penalty `−10`.
5. Softmax weighting: `w_k = softmax((R_k − max R) / τ)`.
6. Execute `a* = Σ_k w_k · a_k[0]` through WM dynamics; advance real latent.

## Hyperparameters (v0)

| Parameter               | Cloud (L40S) | Local (this machine) | Rationale |
|-------------------------|--------------|----------------------|-----------|
| N (sample count)        | 128          | 32                   | Cloud: production; local: diagnostic fits 11.5 GiB |
| H (planning horizon)    | 10           | 10                   | Matches WM `n_frames=10` — no sliding-window needed |
| σ (action noise)        | 0.1          | 0.1                  | Matches `rl/compare_policies.py` "moderate random" scale |
| τ (softmax temp)        | 1.0          | 1.0                  | Neutral |
| image_diagonal          | √(128²+128²)≈181 | same             | Used to normalize position reward to ~[-1, 0] |
| control_steps           | 50           | 50                   | Enough to show convergence or failure |
| decode_strategy         | last-step    | last-step            | Cheapest v0; dense-reward is a v1 consideration |
| CV-fail penalty         | -10          | -10                  | Well below any typical reward, discourages off-distribution samples |
| action_dim              | 4            | 4                    | **Not** the 10 from default config — this checkpoint uses 4 (see `outputs/pusht_cam1/.hydra/config.yaml:105`) |

## Step 1 — Batched rollout

### What shipped

- `rl/mppi/utils.py::batched_rollout(z0, actions, wm, hist_context=10)` —
  thin wrapper over `DifferentiableDynamics.rollout`. Takes `z0` of shape
  `(B, C, H_lat, W_lat)` (no time dim) and `actions` of shape `(B, H, A)`,
  returns `(B, H+1, C, H_lat, W_lat)`. Validates dim/batch-size mismatches
  loudly so downstream MPPI code can't silently misroute shapes.
- `tests/mppi/test_rollout.py` — 4 tests:
  - `test_zero_action_rollout_stays_close` (B=4, H=5)
  - `test_random_action_rollout_diverges_more` (B=4, H=5)
  - `test_rollout_small` (N=32, H=10) — always runs locally
  - `test_rollout_scaling` (N=128, H=10) — gated behind GPU free ≥ 20 GiB
    (cloud L40S only; skips locally).

### Local test results (GPU 0, ~11.5 GiB total)

```
test_zero_action_rollout_stays_close    PASSED   mean ||z_H − z_0|| = 2.07
test_random_action_rollout_diverges_more PASSED  zero=2.06, rand=4.31  (σ=0.3, H=5)
test_rollout_small                       PASSED   peak VRAM 5488 MiB
test_rollout_scaling                     SKIPPED  (local GPU < 20 GiB)
```

Cloud verification of `test_rollout_scaling` is pending — should run on L40S
before we ship Step 4.

### Viz run (`scripts/viz_step1_rollout.py`)

Configuration: `N=32, H=10, σ=0.1, seed=0`, starting latent = `z_goal`.

| Metric              | Value |
|---------------------|-------|
| Wall time           | 7.38 s |
| Peak VRAM           | 5488 MiB |
| Mean ||z_H − z_0||  | 3.91 (min 3.31, max 4.66) |

Panel output: `outputs/mppi/step1_rollout_check/rollout_panel.png`.

### Observations

1. **Decoder output is plausible.** All 5 decoded frames (z_0 plus 4 z_H) look
   like realistic PushT scenes — T-block visible, arms visible, no drift into
   noise or off-manifold garbage after H=10 steps.

2. **Different action sequences produce visibly different final frames.** The
   T-block pose and gripper positions vary across the 4 samples as expected —
   the action channel genuinely affects the rollout, which is the precondition
   for MPPI to have any chance of working.

3. **Zero-action drift is small, random-action drift is ~2× larger.** On H=5,
   σ=0.3: `d_zero = 2.06`, `d_rand = 4.31`. Note: per-pixel unit-norm means
   the maximum possible L2 difference between two latents of shape `(4, 32, 32)`
   is `2 · √(32·32) ≈ 64`, so these drifts are 3%–7% of that upper bound —
   modest, not pathological.
   **Follow-up:** in cosine-sim units on the 32² unit-sphere-per-pixel, the
   zero-action drift of 2.07 corresponds to `cos ≈ 1 − 2.07²/(2·32²) ≈ 0.998`,
   i.e. the zero-action rollout stays on the same latent direction to ~3σ.
   This is consistent with the known stochasticity of the consistency-model
   decoder — each denoising step draws fresh noise, so identity-under-zero-
   action holds only approximately.

4. **Local VRAM budget:** projection was ~3 GiB, actual ~5.5 GiB for N=32.
   The difference is in transformer attention activations inside the dynamics
   model (not the latent tensor itself, which is tiny). N=128 OOMs at ~12 GiB
   needed vs 11.5 GiB available on the local GPU — which is why
   `test_rollout_scaling` is cloud-only. Production cloud run (L40S, 45 GiB)
   will need to verify the N=128 VRAM claim empirically.

5. **Spec mismatch caught:** the task spec stated `actions (B, H, 4)` which
   I initially misread as a typo — the default
   `configurations/algorithm/latent_world_model.yaml:10` has `action_dim: 10`.
   But the PushT checkpoint (`outputs/pusht_cam1/.hydra/config.yaml:105`)
   overrides to `action_dim: 4`. So the spec was right; `action_dim = 4` is
   pinned throughout the MPPI code. `batched_rollout` doesn't hard-code it
   anywhere — it reads from `actions.shape[-1]`.

## Step 2 — CV reward

### What shipped

- `rl/mppi/reward.py::state_reward(rgb, state_goal, *, labeler=None,
  image_diagonal, large_penalty) -> (float, CVLabelResult)` —
  `R = −||pos − pos_goal||/diag − |1 − cos(Δθ)|`, CV-fail → `−10`.
  Angle term computed via sin·sin + cos·cos directly (no atan2).
- `tests/mppi/test_reward.py` — 5 tests: self-reward, far-from-goal,
  180°-flip synthetic, CV-fail, IMAGE_DIAGONAL constant lock.
- `scripts/viz_step2_reward.py` — 10 frames × encode→decode→CV→reward,
  writes `reward_curve.png`, `frame_grid.png`, `summary.json`.

### Test results (all pass)

```
test_reward_goal_frame_against_self     PASSED   reward = -0.0002
test_reward_far_from_goal_is_negative   PASSED   reward = -0.3014
test_reward_synthetic_180_flip          PASSED   reward = -2.009 (predicted -2)
test_reward_cv_fail_returns_penalty     PASSED   reward = -10.0000
test_image_diagonal_constant            PASSED   181.019...
```

The 180-flip test landed at reward = -2.009 vs the theoretical -2 predicted
by `|1 − cos(180°)| = 2`. The 0.009 residual is sub-pixel position jitter
from the `fillPoly` rasterizer at the PINK_RGB threshold.

### Viz observations (data/mini/pusht/train/episode_3, 10 frames)

`outputs/mppi/step2_reward_check/reward_curve.png` shows a clear dynamic
range on this episode: rewards fluctuate between -1.71 (worst) and -0.20
(best), peaking at t=154 when the T transiently lines up with the goal
orientation before the CV estimator re-acquires a 180°-flipped solution.

| Metric                      | Value |
|-----------------------------|-------|
| Min reward over 10 frames   | -1.71 |
| Max reward over 10 frames   | -0.20 |
| WM decode wall (N=10)       | 0.87 s (87 ms/frame) |
| CV label wall (N=10)        | 0.20 s (20 ms/frame) |
| Implied per-step cost @ N=128 | ~14 s (11 s decode + 2.6 s CV) |

Key findings for Step 3+:

1. **Reward function is working.** Visually-similar-to-goal frames score
   near zero; far frames score strongly negative. The 180-flip synthetic
   test nails the exact theoretical angle penalty (2.009 vs predicted 2).

2. **Local data caveat.** `data/full/pusht/` is not on this machine, so
   local viz falls back to `data/mini/pusht/`. Mini's episode_0 happens
   to be static (T barely moves) — I switched to episode_3, which shows
   a genuine push trajectory from cx=48 → cx=88. Cloud run will use
   `data/full/` per spec.

3. **Decoder is stochastic.** Rerunning `viz_step2_reward.py` with the
   same inputs produces different rewards — the consistency-model decoder
   samples fresh noise per call. Across two back-to-back runs, reward
   at t=154 changed from -0.20 to -0.48 and reward at t=176 from -1.71
   to -1.44. **Implication for MPPI:** reward estimates carry variance
   of order 0.1–0.3. With τ=1.0 softmax, this is tolerable (dominant
   terms are still separable), but single-trajectory rankings are
   noisy. Per-trajectory reward averaging across multiple decodes may
   be needed if Step 5 shows high-frequency oscillation in a*.

4. **CV angle can flip 180° between adjacent frames.** At t=154 the CV
   reports θ = 4.95°; at t=176 it reports θ = -119.48°. The T-block is
   approximately 180°-symmetric when the bar-vs-stem orientation is
   hard to distinguish, and HSV-mask noise can push the ICP into the
   flipped local minimum. **Implication for MPPI:** reward curve has
   discontinuities of up to Δ=2 at CV-symmetry flips even with no
   change in the actual T pose. This is an irreducible reward-signal
   noise source.

5. **Wall-time projection is reasonable but not fast.** N=128 per plan
   step ≈ 14 s, implying 50 control steps ≈ 12 minutes per run. Under
   the "v0 diagnostic" budget this is fine. If Step 5 shows per-step
   cost exceeding ~20 s, need to revisit.

6. **WM decode introduces pose drift.** Raw mini episode_0 t=0 CVs as
   pose (39, 83, -86°); after encode→decode through the WM the same
   frame CVs as (39.8, 83.3, -103°) — position matches, angle drifts
   by ~17°. Not a reward-module bug (reward is consistent in either
   direction), but it means "reward near zero on a decoded near-goal
   frame" is not strictly the same as "zero reward on the raw goal
   frame". The `state_goal` in `tests/goal_selection/state_goal.pt`
   was computed on a decoded frame, so the closed loop is internally
   consistent.


## Step 3 — Batched reward
(pending)

## Step 4 — Single plan step
(pending)

## Step 5 — Full execution
(pending)

## Final Results
(pending)

## Failure analysis (if applicable)
(pending)

## What this tells us about IWS + CV control feasibility
(pending)
