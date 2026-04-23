# MPPI reference algorithm — diffusion-forcing planner_v0_0

Source: `~/Documents/diffusion-forcing/algorithms/latent_dynamics/planner_v0_0.py`
(class `Planner`, `planner_type = "MPPI"`).
Default config: `~/Documents/diffusion-forcing/configurations/planner/planner_v0_0.yaml`.

This document is the authoritative spec for `rl/mppi/mppi_planner.py`. Every
algorithmic choice in our implementation is intentionally faithful to this
reference (the only deviation is the reward function — we use our CV-based
state reward via `env.compute_reward`, where the reference uses a generic
`evaluate_traj_fn` callback).

## Three exact formulas

### 1. Sampler — `sample_action_sequences_default` (planner_v0_0.py:196-256)

Given an `act_seq` of shape `(H, A)` and `n_sample = N`, return
`act_seqs` of shape `(N, H, A)` with **smooth correlated noise across the
horizon**:

```python
act_seqs = stack([act_seq.clone()] * N)              # (N, H, A)
act_residual = zeros(N, A)
for i in range(H):
    noise_sample = randn(N, A) * noise_level         # fresh per step
    act_residual = beta_filter * noise_sample + (1 - beta_filter) * act_residual
    act_seqs[:, i] += act_residual
    act_seqs[:, i] = clamp(act_seqs[:, i], action_lower_lim, action_upper_lim)
```

**`beta_filter` is intra-horizon noise smoothing**, not a cross-plan-step
warm-start. A higher `beta_filter` (closer to 1) means each horizon step's
noise is closer to fresh-sampled; a lower `beta_filter` (closer to 0) means
the noise residual carries over from the previous horizon step almost
unchanged, producing slowly-varying sampled action curves. With the
reference default `beta_filter = 0.7`, the noise applied to step `i+1` is
70% fresh + 30% the residual that was applied to step `i`. This produces
sampled trajectories that are noisy-but-not-shock-discontinuous along the
horizon — much more in-distribution for a WM that was trained on smooth
demonstration actions.

### 2. Softmax aggregator — `optimize_action_mppi` (planner_v0_0.py:390-400)

```python
softmax_weight = F.softmax(reward_seqs * reward_weight, dim=0)  # (N,)
act_seq = sum(act_seqs * softmax_weight[..., None, None], dim=0)  # (H, A)
return act_seq
```

Notes:
- `reward_weight` directly multiplies the rewards; there is **no
  separate temperature τ** and **no max-subtraction** before softmax.
- For our refactor we'll add the standard `R - R.max()` shift inside
  the softmax for numerical stability — this is mathematically a no-op
  but prevents overflow when `reward_weight=200` and rewards are
  modestly negative. The reference is fragile to this in our setting
  (rewards in `[-2, 0]` × 200 = `[-400, 0]` is fine; CV-fail
  rewards `-10 × 200 = -2000` underflow to zero contribution, which
  is exactly what we want anyway).

### 3. Iterative refinement — `trajectory_optimization_mppi` (planner_v0_0.py:428-483)

```python
for _ in range(n_update_iter):
    act_seqs = sample_action_sequences(act_seq)            # (N, H, A) noisy variants
    state_seqs = model_rollout(state_cur, act_seqs)         # (N, H, state_dim)
    reward_seqs = evaluate_traj(state_seqs, act_seqs)       # (N,)
    act_seq = optimize_action_mppi(act_seqs, reward_seqs)   # (H, A) weighted mean
if rollout_best:
    best_state_seqs = model_rollout(state_cur, act_seq.unsqueeze(0))
return TrajOptOutput(act_seq=act_seq, ...)
```

The function takes `act_seq` as INPUT and returns the refined `act_seq` as
OUTPUT. **There is no cross-call state inside the planner.** Planner is
entirely stateless across calls.

## What the reference does NOT do

Worth listing because two of these were misinterpreted in the prior
project notes:

- **No cross-plan-step warm-start.** `beta_filter` is intra-horizon, not
  cross-plan. We confirmed by reading the caller
  (`experiments/exp_sim_control.py:122`) which re-initializes
  `act_seq = curr_pos.repeat(H, 1)` every iteration — the previous
  plan's converged `act_seq` is not propagated forward.
- **No max-subtraction in softmax.** Pure `softmax(R * w)`.
- **No CEM-style covariance update.** The sampling distribution stays
  centered on the running `act_seq`; only the mean updates between
  iterations, not the spread.

## Default hyperparameters (from planner_v0_0.yaml)

```yaml
action_dim:        ${algorithm.action_dim}
state_dim:         2
n_sample:          100
n_look_ahead:      40
n_update_iter:     50
reward_weight:     200.0
action_lower_lim:  [-1.0, -1.0]
action_upper_lim:  [1.0, 1.0]
planner_type:      MPPI
device:            cuda
verbose:           True
noise_level:       0.05
rollout_best:      True
beta_filter:       0.7
```

The reference's `state_dim: 2` and `action_dim: 2` are for their 2-D
PushT environment; we use `action_dim: 4` for the IWS bimanual ALOHA
WM. We also reduce `n_update_iter` from 50 to 5 because the IWS decoder
is compute-bound and 50 iterations × N=100 rollouts × decode-and-CV
per trajectory would put a single plan_step at minutes (vs. the
reference's pure-2D simulator that runs in milliseconds per rollout).

## Pure-deviation list (intentional differences in our implementation)

| Aspect | Reference | Ours | Reason |
|---|---|---|---|
| Reward source | callback (`evaluate_traj`) | `env.compute_reward` (CV-based) | reuse our `PushTWMEnv` |
| Softmax stability | none | subtract `max` first | numerical safety with our `reward_weight=200` × moderately-negative rewards |
| `n_update_iter` default | 50 | 5 | IWS decoder cost — see comment in `configs/mppi/default.yaml` |
| `n_look_ahead` default | 40 | 10 | matches IWS WM `n_frames=10`; sliding-window not implemented yet |
| `action_dim` default | 2 | 4 | bimanual ALOHA |

Everything else — sampling formula, optimizer formula, refinement loop
structure, action clipping, lack of cross-call state — matches exactly.
