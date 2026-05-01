# Action normalization & scaling audit

Pre-flight: branch `mppi-baseline` at `eafa33d` (sliding-window commit). No
code changes proposed in this report — diagnostic only.

## Headline finding

**The `[-1, 1]` action bounds in our config are correct** — they match the
training-time normalized action range. But two related issues compress
MPPI's exploration into a region of action space the demos rarely visit:

1. **MPPI starts each plan_step from `act_seq = 0`** (no warm-start). In
   the normalized space the WM was trained on, the demo distribution has
   non-zero per-dim means (notably dim 1 = +0.60, dim 3 = −0.70). MPPI
   starts ~12-14 σ_MPPI away from where the demo distribution lives.

2. **MPPI's per-step noise σ = 0.05** is **6.5-9.8× smaller than the
   per-dim std of demo actions in the same normalized space** (demo std
   per dim: 0.32-0.49). After 30 iterations the running mean can drift,
   but it never reaches dim 1 ≈ +0.6 or dim 3 ≈ −0.7 — looking at the
   `iter30` cloud run's executed actions, all four dims stay in [-0.2,
   +0.3] across all 100 control steps.

This isn't an OOD problem (small actions are within training distribution
— the normalizer maps them to a small region of the trained space, not
outside it). It IS a **systematic under-exploration**: MPPI never proposes
the kind of actions that real demos use to push the T-block decisively.

Likely contribution to Phase 3a's stalled trajectories.

---

## Section 1 — Layer-by-layer action accounting

| Layer | Action shape | Range | Scaling applied | Clipping |
|---|---|---|---|---|
| **A. Demo hdf5** (`data/mini/pusht/val/episode_*.hdf5` `action`) | `(T, 4)` float32 | dim 0: [−0.05, +0.36], dim 1: [−0.21, +0.17], dim 2: [−0.06, +0.36], dim 3: [−0.39, −0.12] | none — raw ALOHA action | none |
| **B. Training dataloader** (in `interactive_world_sim`) | `(B, T, 4)` | normalized | `normalizer["action"].normalize(raw)` → maps to `[-1, +1]` per-dim (min-max) | implicit by min-max range |
| **C. WM dynamics module** (training input) | `(t, B, 4)` | `[-1, +1]` per-dim | none (already normalized by step B) | none |
| **D. WM dynamics module** (inference, called from `DifferentiableDynamics.step`) | `(t, B, 4)` | whatever caller passes | **NONE — caller's responsibility to pass normalized actions** | none |
| **E. `env.PushTWMEnv.dynamics_step` / `rollout`** | `(B, H, 4)` | passes through | **none** — directly forwards to WM | none |
| **F. MPPI sampler** (`rl/mppi/mppi_planner.py`) | `(N, H, 4)` | samples around running `act_seq` (init 0) with σ=0.05, β=0.7 inter-step correlation | none | `torch.clamp` to `cfg.action_lower_lim`/`upper_lim` (= `[-1, 1]`) per step |
| **G. Reward computation** | n/a | reads CV pose (cx, cy, θ) from decoded RGB; no action involvement | n/a | n/a |

**The chain works because by convention everything between layer B and
layer F speaks "normalized [-1, 1] action space".** MPPI samples in this
space and feeds them straight to the WM, which was trained to ingest
exactly this space. The convention is **implicit, not enforced** —
`env.PushTWMEnv.rollout` does not call `normalizer["action"].normalize()`;
it relies on the caller to pre-normalize. With MPPI as the caller this is
fine because MPPI's bounds happen to match the trained space.

A risk worth noting: anyone in the future who calls `env.dynamics_step`
or `env.rollout` with **raw** physical actions (e.g. by reading actions
directly from an hdf5) will silently get junk output, because the WM will
interpret raw action `+0.36` as if it were normalized `+0.36` (~ 95th
percentile of normalized space). Worth a future docstring or assertion.

---

## Section 2 — Reference's action conventions

Diffusion-forcing's `experiments/exp_sim_control.py` is **explicit** about
the boundary that ours is implicit about:

```python
# line 123, 127 — INTO planner: normalize first
act_seq    = self.algo.normalizer["action"].normalize(act_seq).to(device)
self.curr_pos = self.algo.normalizer["action"].normalize(self.curr_pos)

# line 153, 174 — OUT of planner: unnormalize before sending to env
next_action     = self.algo.normalizer["action"].unnormalize(next_action)
unnorm_act_seq  = self.algo.normalizer["action"].unnormalize(res.act_seq)
```

So the reference's MPPI also operates in normalized [-1, 1] action space.
The difference is purely architectural: reference talks to a real-physics
environment so it has to convert at the boundary; we talk to a WM that
was trained with normalized inputs, so no conversion needed.

| Aspect | Reference | Ours |
|---|---|---|
| MPPI internal action space | normalized to `[-1, 1]` | normalized to `[-1, 1]` ✓ |
| Training-time action normalization | min-max to `[-1, 1]` | min-max to `[-1, 1]` ✓ |
| `action_lower_lim` / `action_upper_lim` | `[-1, +1]` (in normalized space) | `[-1, +1]` ✓ |
| `noise_level` (σ) | 0.05 | 0.05 ✓ |
| `beta_filter` | 0.7 | 0.7 ✓ |
| `n_update_iter` default | 50 | 5 (we run 30 in cloud experiments) |
| `n_look_ahead` default | 40 | 10 (constrained by IWS WM `n_frames=10`) |
| MPPI `act_seq` initial value | zeros | zeros ✓ |
| Where normalize/unnormalize is called | env-boundary (`exp_sim_control.py`) | nowhere (implicit — MPPI's [-1,1] outputs go straight to WM) |

**Our [-1, 1] is not a copy-paste mistake — it genuinely matches the
training-time normalized space.** Verified by inspecting the WM
checkpoint's `normalizer["action"]` parameters and confirming that
`normalize(raw_min) = -1` and `normalize(raw_max) = +1` exactly.

---

## Section 3 — Resolving the 0.02 vs [-1, 1] mystery

The keyboard-MPPI era used `KeyboardAtomSampler(atom_magnitude=0.02)`
producing 9 atoms like `(0, 0.02, 0, 0)`. From `rl/mppi/action_sampling.py`:

> *"Sample one of 9 keyboard atoms per step: noop + 8 axis-aligned deltas
> of magnitude 0.02, exactly matching the PushT keyboard teleop"*

> *"`jitter_sigma` should be much smaller than the demo per-step delta L2
> (≈ 0.012) to avoid dominating the demo signal — pick something like
> 0.005-0.02 for safe perturbation."*

**The 0.02 was a *delta size*, not an absolute action value.** It was
chosen to match the demo's typical per-step *change* in action (which
empirically is ≈ 0.012 in raw units, p95 ≈ 0.025). Computed from the val
hdf5 directly:

```
Per-step delta L2 (raw):   mean=0.0133  std=0.015  p50=0.011  p95=0.025
```

So 0.02 ≈ p95 of demo per-step delta. The keyboard era's planner used an
**accumulator** mode: each step's *atom* (delta) was added to a running
sum (`curr_action`), and the WM saw the *running sum* — equivalent to
moving the absolute action by 0.02 per atom. This mirrored the human
keyboard teleop where each keypress produces an atomic delta.

Current MPPI is **not** delta-mode. It samples the **absolute action per
step** directly from `Normal(running_mean, σ=0.05)` in normalized space.
There is no accumulation. So the `0.02` from the keyboard era and the
`σ=0.05` from current MPPI are NOT the same kind of quantity:

- `0.02` raw delta ≈ `0.08` normalized delta (multiply by avg scale ≈ 4) →
  larger than current σ=0.05 of normalized **absolute action**.
- The keyboard-era atom of 0.02 was a "small reasonable nudge in raw
  units"; current σ=0.05 is "small noise around the mean of the
  normalized space".

These don't conflict; they live in different spaces with different
semantics. The 0.02 doesn't help us decide whether σ=0.05 is appropriate.

---

## Section 4 — Demo action statistics

Source: `data/mini/pusht/val/episode_{0..4}.hdf5`, 5 episodes × 200 frames
= 1000 raw action samples (4-dim per sample).

### Raw demo action stats (in physical units, as stored in hdf5)

| dim | min | max | mean | std | p5 | p50 | p95 |
|----:|----:|----:|----:|----:|----:|----:|----:|
| 0 | −0.054 | +0.360 | +0.148 | 0.123 | −0.020 | +0.131 | +0.360 |
| 1 | −0.208 | +0.170 | +0.090 | 0.078 | −0.093 | +0.109 | +0.170 |
| 2 | −0.058 | +0.360 | +0.165 | 0.100 | +0.002 | +0.165 | +0.312 |
| 3 | −0.390 | −0.122 | −0.339 | 0.067 | −0.390 | −0.367 | −0.180 |

### Same actions in NORMALIZED [-1, 1] space (what the WM actually sees)

(Computed by passing raw values through `wm.normalizer["action"].normalize`)

| dim | mean | std | min | max |
|----:|----:|----:|----:|----:|
| 0 | +0.137 | 0.489 | −0.664 | +0.981 |
| 1 | **+0.598** | 0.355 | −0.754 | +0.959 |
| 2 | +0.209 | 0.405 | −0.687 | +0.997 |
| 3 | **−0.698** | 0.324 | −0.943 | +0.342 |

Per-dim demo means are **not** centered on zero in normalized space —
dim 1 sits at +0.60, dim 3 at −0.70. The min-max normalization preserves
the asymmetric distribution.

### Per-step delta stats (raw units)

| dim | mean Δ | std Δ | \|Δ\|_p50 | \|Δ\|_p95 |
|----:|----:|----:|----:|----:|
| 0 | -0.00003 | 0.0123 | 0.0038 | 0.0191 |
| 1 | -0.00007 | 0.0069 | 0.0018 | 0.0159 |
| 2 | -0.00006 | 0.0103 | 0.0039 | 0.0153 |
| 3 |  0.00000 | 0.0106 | 0.0008 | 0.0129 |

**Demo actions change very slowly between consecutive timesteps**
(per-step delta mostly < 2% of the action range). The action sequence is
nearly piecewise-constant. This explains the keyboard-era 0.02 atom size.

### Cumulative L2 over a 10-step window

| Source | Mean cumulative L2 over 10 steps (normalized) |
|---|---|
| Demo trajectories | **0.57** |
| MPPI sampling at σ=0.05, β=0.7 | **0.73** |

These are surprisingly close in *path length per 10 steps*. MPPI's
within-trajectory variation is comparable to demo's. The problem isn't
the trajectory variation; it's the **starting point** (next section).

Histogram visualization: `outputs/demo_action_hist.png` — overlays demo
action distribution (blue) with MPPI sampling distribution (orange) per
dim. Demo is broad, off-center for dims 1 and 3; MPPI is a narrow spike
at zero.

---

## Section 5 — Diagnosis: is MPPI sampling distribution OOD vs training?

**Not OOD.** MPPI samples in [−1, +1] are within the training distribution.

But the sampling is **systematically under-exploring** the demo
distribution.

### Numerical comparison (all in normalized space)

| Quantity | Value |
|---|---|
| Demo per-dim std (normalized) | dim 0: 0.49, dim 1: 0.36, dim 2: 0.41, dim 3: 0.32 |
| MPPI per-step σ | 0.05 (all dims) |
| **Ratio (demo std / MPPI σ)** | **6.5–9.8×** |
| Demo per-dim mean (normalized) | dim 0: +0.14, dim 1: +0.60, dim 2: +0.21, dim 3: −0.70 |
| MPPI initial running_mean (act_seq) | zeros (all dims) |
| Distance from MPPI start to demo mean (normalized) | dim 0: 0.14, dim 1: 0.60, dim 2: 0.21, **dim 3: 0.70** |
| In units of σ_MPPI (=0.05) | dim 0: 2.7σ, dim 1: 12σ, **dim 2: 4.2σ, dim 3: 14σ** |

### What this looks like in actual MPPI runs

Executed actions from the `iter30` cloud run (N=100, n_iter=30, 100
control steps) sampled every 10 steps, in **normalized space**:

```
t=10  action = [-0.00, -0.11, +0.01, +0.09]
t=20  action = [-0.06, -0.01, +0.32, -0.04]
t=30  action = [-0.00, +0.15, +0.27, -0.19]
t=40  action = [-0.10, -0.07, -0.03, +0.16]
t=50  action = [+0.15, -0.17, -0.17, +0.01]
```

All four dims stay in `[-0.2, +0.3]` across all 100 plan_steps. Compare
to demo means `(+0.14, +0.60, +0.21, -0.70)` — MPPI **never even
approaches dim 1 ≈ +0.6 or dim 3 ≈ −0.7**, even after 30 iterations of
softmax-weighted refinement per plan_step.

**Verdict**: MPPI sampling is **sub-distribution**:
- not OOD (the WM still produces sensible predictions for these small
  actions; we see clean rollouts and mostly-monotone iter-reward curves)
- but exploring a much narrower volume than the demo distribution covers.

The reward landscape MPPI sees is the WM's prediction of "what happens if
I apply small action a near zero". It never tests "what happens if I
apply a demo-magnitude action". So even if there's a great
T-block-pushing action at e.g. norm=(0.0, +0.6, 0.0, -0.7), MPPI can't
find it.

### Why this connects to the iter30 trajectory weirdness

In the prior debugging session ("trajectory looks discontinuous"), the
explanation was CV pose ambiguity on stochastic decoder outputs. That
remains the *proximate* cause of the spiky reward curve. But this audit
suggests a *contributing* factor: because MPPI never proposes
demo-magnitude actions, the executed trajectory consists of small noisy
nudges that don't produce coherent T motion. The CV pose has nothing
clean to lock onto frame-to-frame, amplifying the symmetry-flip issue.

---

## Section 6 — Recommendations

Ranked by expected impact for solving "MPPI can't move T-block 50 px":

### 1. **Increase σ to match demo std** (highest impact, lowest risk)

Change `noise_level` from `0.05` → `0.30` (roughly the average of demo
std per-dim in normalized space). Or set per-dim σ matching each demo
std.

- **Why**: lets MPPI propose actions across the full demo distribution
  in fewer iterations.
- **Risk**: with broader sampling and reward_weight=200 (sharp argmax),
  the planner may pick noisier "best samples" that don't generalize.
  Probably want to also reduce reward_weight (e.g., 50–100) or increase
  N. Reference uses σ=0.05 but for a 2D toy environment whose action
  scale is different.
- **Effort**: 1-line config change. Re-run an existing pair to check.
- **Validation cost**: ~6 hours per pair on cloud at the current per-step
  budget. Cheaper than another 16-hour iter=30 run.

### 2. **Initialize MPPI's `act_seq` to demo mean instead of zeros** (medium impact, low risk)

In `rl/mppi/mppi_planner.py::trajectory_optimization`, change
`act_seq = torch.zeros(...)` to `act_seq = demo_mean.unsqueeze(0).expand(H, -1)`.

- **Why**: MPPI no longer wastes the first ~5-10 iterations drifting
  toward the demo distribution. Concentrates iterations on selecting
  *which* demo-like action best advances the goal, rather than first
  *finding* demo-like actions.
- **Risk**: small. Demo mean is precomputable from val hdf5.
- **Effort**: 1 hour to implement + 1 unit test. Could even make the
  init action a warm-start from the previous plan_step's converged
  `act_seq` (matches what some MPPI variants do, though reference doesn't).
- **Note**: this changes the algorithm vs reference. Currently we match
  reference exactly; this would be an intentional documented deviation
  for IWS specifically.

### 3. **Add an action-space normalization assertion in env.rollout** (low impact, low risk, future-proofing)

Document that `env.PushTWMEnv.rollout` expects actions in normalized [-1,
+1] space. Add an assertion `assert (-1.5 <= actions <= 1.5).all()` (loose
bound) that catches future bugs where someone passes raw actions.

- **Why**: prevents the silent-misuse failure mode I flagged in Section 1.
- **Risk**: zero — assertion is loose enough not to fire on legitimate
  near-bound samples.
- **Effort**: 30 minutes.

### 4. **Reduce reward_weight (loosen softmax)** (medium impact, medium risk)

Drop `reward_weight: 200` → `reward_weight: 50` (or even 20).

- **Why**: with a broader σ, sharp argmax picks one outlier sample. A
  softer softmax averages across more samples and is more robust to
  CV-noise-induced reward spikes (the symmetry-flip issue from the prior
  debugging session).
- **Risk**: this is intentional algorithm-level deviation from reference.
  Reference's 200 was tuned for their 2D PushT; our task may want
  different.
- **Effort**: 1-line config change, but should A/B test against the
  current σ=0.05 + reward_weight=200 to disentangle effects.

### 5. **(Out of scope for this audit but related) symmetry-aware reward**

Use `−|sin(Δθ)|` instead of `−(1 − cos(Δθ))` to sidestep CV's 180°
symmetry confusion. Discussed in the prior "trajectory looks
discontinuous" debugging session. Independent of action-scaling issues
but compounds with them.

---

## Section 7 — What I didn't check / unknowns

- **Original IWS WM training repo**: I read the saved hydra config in
  `outputs/pusht_cam1/.hydra/config.yaml` but not the actual training
  scripts (`interactive_world_sim` package internals). I confirmed the
  normalizer is loaded with the checkpoint and contains action stats
  matching the val hdf5 (within 1-2%, the diff is just min-max stats
  computed on train vs val splits). The schema is correct.
- **Whether the ALOHA action semantics are joint angles, EE positions,
  or something else**: the action is `(4,)` and the values ([+0.14,
  +0.60, +0.21, -0.70] for the demo mean in normalized space, with
  per-dim ranges differing) suggest 2 dims per arm (left/right end-
  effector x, y maybe — but I'm not sure). Doesn't affect the audit
  conclusion. If you want the semantics nailed down, check the original
  ALOHA teleop script.
- **I did NOT run an experiment to validate Recommendation 1** (increase
  σ). This is a paper-only audit. Empirically validating "σ=0.30 gives
  better convergence" needs a control run.
- **Diffusion-forcing's training-time action stats**: I didn't compute
  them. So I don't know whether reference's σ=0.05 vs their demo std
  ratio is similar to ours (6.5-9.8×) or much smaller (in which case
  reference's σ=0.05 is well-tuned for their data and ours is wrong
  for ours). Worth comparing if you can find their training data.

---

## Files referenced

- `outputs/pusht_cam1/.hydra/config.yaml` — training-time hydra config
  (action_dim=4)
- `outputs/pusht_cam1/checkpoints/best.ckpt` — WM checkpoint with
  normalizer attached
- `data/mini/pusht/val/episode_{0..4}.hdf5` — demo action source
- `rl/mppi/mppi_planner.py` — MPPI planner
- `rl/mppi/action_sampling.py:93-123` — legacy KeyboardAtomSampler
- `rl/models/world_model.py:111-191` — `DifferentiableDynamics.step` (no
  action normalization in forward path)
- `~/Documents/diffusion-forcing/experiments/exp_sim_control.py:123-174` —
  reference's explicit normalize/unnormalize at env boundary
- `~/Documents/diffusion-forcing/configurations/planner/planner_v0_0.yaml` —
  reference planner defaults (σ=0.05, β=0.7, action_lim=[-1,1])
- `outputs/demo_action_hist.png` — visualization (demo blue vs MPPI orange)
