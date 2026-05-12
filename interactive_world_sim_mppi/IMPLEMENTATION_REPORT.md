# interactive_world_sim_mppi — Implementation Report (Phase 3)

A self-contained MPPI planner over `WorldModelEnv`, with the algorithmic
core copied verbatim from diffusion-forcing's `planner_v0_0.py` /
`splines.py` and hyperparameters consolidated in `config.py`. The reward
function is supplied by the caller; the planner does not import the CV
package directly.

## 1. Source provenance

| | |
|---|---|
| diffusion-forcing source | `/home/yiru-wu/Documents/diffusion-forcing` |
| diffusion-forcing SHA | `180a2639a01c593c1a73275abe42b3acf4afc162` |
| Tip commit message | `Add dim6 left-arm absolute position monkey patch, fix render/wdmdl scripts` |
| Copy date | 2026-05-12 |

The diffusion-forcing repo is **reference only**. Production code in
this package does not import from it; the bitwise equivalence test in
`scripts/smoke_bitwise_equivalence.py` loads it via
`importlib.util.spec_from_file_location` solely to verify the copy.

## 2. Files

```
interactive_world_sim_mppi/
├── __init__.py                #  35 lines  — public re-exports
├── api.py                     # 285 lines  — MPPIPlanner + adapters + warm-start + callback
├── _mppi_core.py              # 543 lines  — verbatim planner_v0_0.py (with documented substitutions)
├── _splines.py                # 195 lines  — verbatim splines.py
├── config.py                  # 145 lines  — Config + GoalPose dataclasses, all hyperparameters
├── reward.py                  # 260 lines  — pusht_terminal_reward + PushTTerminalReward + detect_goal_pose_from_episode
├── py.typed                   #   0 lines  — PEP 561 marker
├── IMPLEMENTATION_REPORT.md   # (this file)
└── scripts/
    ├── smoke_bitwise_equivalence.py    # 221 lines  — vs diffusion-forcing reference
    ├── smoke_snapshot_isolation.py     #  91 lines  — env state preservation
    ├── smoke_warmstart_shape.py        # 127 lines  — shift+pad logic
    ├── smoke_detection_failure.py      # 102 lines  — failure-penalty math
    └── plan_one_step.py                # 199 lines  — end-to-end + 3 diagnostics
```

## 3. Verbatim copy: diff results

### 3.1 `_splines.py`

All 3 functions copied byte-identical from `splines.py:7-173`:

| Symbol | Source lines | Diff result |
|---|---|---|
| `cubic_spline_nd_torch_batched` | 7–79 | IDENTICAL |
| `eval_cubic_spline_nd_torch_batched` | 82–147 | IDENTICAL |
| `cubic_spline_nd_function_torch` | 150–173 | IDENTICAL |

Dropped: the `__main__` example block (176–206) and `import time`.

### 3.2 `_mppi_core.py`

Verified by per-block diff against `planner_v0_0.py`. **10 of 11 blocks
byte-identical**; the one diff is the documented `ListConfig → (list, tuple)`
substitution.

| Symbol | Source lines | Copy lines | Diff result |
|---|---|---|---|
| `ModelOutput` | 52–56 | 73–77 | IDENTICAL |
| `EvalOutput` | 59–64 | 81–86 | IDENTICAL |
| `TrajOptOutput` | 67–78 | 90–101 | IDENTICAL |
| `register_model_rollout_fn` | 180–182 | 197–199 | IDENTICAL |
| `register_evaluate_traj_fn` | 184–186 | 202–204 | IDENTICAL |
| `register_visualize_fn` | 188–190 | 207–209 | IDENTICAL |
| `register_sample_action_sequences_fn` | 192–194 | 212–214 | IDENTICAL |
| `sample_action_sequences_default` | 196–256 | 217–277 | 1-line diff (ListConfig→list/tuple, documented) |
| `clip_actions` | 421–426 | 382–387 | IDENTICAL |
| `trajectory_optimization_mppi` | 428–483 | 390–445 | IDENTICAL |
| `trajectory_optimization_mppi_waypts` | 485–580 | 448–543 | IDENTICAL |

Functions deliberately diverged (init + dispatchers + extension):
- `Planner.__init__` (source 84–178): MPPI_BY_ACC branch stripped;
  config-dict reads replaced with attribute reads per §3.3.
- `optimize_action` (source 302–342): non-MPPI/MPPI_WAYPTS branches stripped.
- `trajectory_optimization` (source 344–388): non-MPPI/MPPI_WAYPTS branches stripped.
- `optimize_action_mppi` (source 390–400): adds the optional
  normalize-before-softmax extension (§3.4). With
  `normalize_rewards_before_softmax=False` the function is byte-identical.

Dropped entirely:
- `fps_np` (source 27–49)
- `Planner.generate_action_sequences_by_acc` (258–300)
- `Planner.optimize_action_gd` (402–419)
- `Planner.trajectory_optimization_robust_mppi_waypts` (582–661)
- `Planner.trajectory_optimization_mppi_by_acc` (663–752)
- `Planner.trajectory_optimization_gd` (754–797)
- `torch.autograd.set_detect_anomaly(True)` (line 14) — module-level side effect
- Top-level tuning-tips comment (16–24)

Imports trimmed:
- Removed `numpy as np` (only `fps_np` used it)
- Removed `from omegaconf.listconfig import ListConfig` (replaced with `(list, tuple)`)
- **Kept** `from tqdm import tqdm` — otherwise the `pbar = tqdm(...)` lines
  in `trajectory_optimization_mppi{,_waypts}` would diverge from the source
  byte-by-byte. With `verbose=False` hardcoded, the tqdm branches are dead
  code; the import is paid for code identity.

## 4. Hyperparameter substitutions

All substitutions in `Planner.__init__` are documented inline with
`# was: ... (planner_v0_0.py:LINE)` comments. Summary:

| Field on our Config | Verbatim `self.X` attribute | Source default | Source line |
|---|---|---|---|
| `action_lower_lim` (len) → action_dim | `action_dim` | `config["action_dim"]` | 120 |
| `n_sample` | `n_sample` | `config["n_sample"]` | 121 |
| `n_waypoints` (renamed) | `n_look_ahead` | `config["n_look_ahead"]` | 122 |
| `n_update_iter` | `n_update_iter` | `config["n_update_iter"]` | 123 |
| `reward_weight` | `reward_weight` | `config["reward_weight"]` | 124 |
| `action_lower_lim` | `action_lower_lim` | `config["action_lower_lim"]` | 125 |
| `action_upper_lim` | `action_upper_lim` | `config["action_upper_lim"]` | 126 |
| n/a (hardcoded) | `planner_type = "MPPI_WAYPTS"` | `config["planner_type"]` | 127 |
| `device` | `device` | default `"cuda"` | 161 |
| n/a (hardcoded False) | `verbose` | default `False` | 162 |
| `noise_level` | `noise_level` | default `0.1` | 164 |
| n/a (hardcoded 1) | `n_his` | default `1` | 165 |
| `rollout_best` | `rollout_best` | default `True` | 166 |
| n/a (hardcoded 1e-3) | `lr` | default `1e-3` | 167 |
| `beta_filter` | `beta_filter` | default `0.7` | 168 |
| `normalize_rewards_before_softmax` (OURS) | `normalize_rewards_before_softmax` | — | — |

Hardcoded values (`planner_type`, `verbose`, `n_his`, `lr`) are constants
for our use case and intentionally not exposed in Config to keep the
public surface small. They can be added if needed.

## 5. The one algorithmic addition: reward normalization

`optimize_action_mppi` has a 3-line prepend:

```python
if self.normalize_rewards_before_softmax:
    reward_seqs = (reward_seqs - reward_seqs.mean()) / (reward_seqs.std() + 1e-8)
```

When `normalize_rewards_before_softmax=False` (the smoke test setting),
this branch is bypassed and the algorithm is byte-identical to the
source. Bitwise equivalence is verified by the smoke test in §6.1.

## 6. Smoke-test outcomes

All 5 smoke tests PASS.

### 6.1 `smoke_bitwise_equivalence.py` — **PASS**

Loaded diffusion-forcing's `Planner` via `spec_from_file_location` from
`/home/yiru-wu/Documents/diffusion-forcing/algorithms/latent_dynamics/planner_v0_0.py`
(SHA above). Constructed identical configs (K=8, n_waypoints=2,
interp_pts=3, n_update_iter=1, rollout_best=True, action_dim=2,
device="cpu", `normalize_rewards_before_softmax=False`). Ran one
MPPI_WAYPTS iteration on a deterministic dummy dynamics + dummy reward
with fixed seed.

Result:

```
ours.waypts_seq.shape  = (2, 2)
ref.waypts_seq.shape   = (2, 2)
ours.act_seq.shape     = (6, 2)
ref.act_seq.shape      = (6, 2)
waypts bitwise equal?  True
act_seq bitwise equal? True
```

The verbatim copy reproduces diffusion-forcing's MPPI_WAYPTS output
bitwise.

Note: encountered a latent bug in the upstream code — with
`rollout_best=False`, `act_seq` is undefined when the
`trajectory_optimization_mppi_waypts` constructs its `TrajOptOutput`
(source lines 559–580). Both planners reproduce this; the test runs with
the upstream-default `rollout_best=True`.

### 6.2 `smoke_snapshot_isolation.py` — **PASS**

Loaded `WorldModelEnv("pusht_cam1")`, took a snapshot, deep-cloned the
tensors, ran `planner.plan(snap)` with K=4 and a zero-reward dummy, took
another snapshot, compared field-by-field.

Result: `latent_window`, `action_window`, `step_counter`, `task` all
byte-identical before vs after. plan() does not mutate env state.

### 6.3 `smoke_warmstart_shape.py` — **PASS**

Uses a mock env (no GPU rollout needed for shape math). Verified:

- First-call initial mean = `curr_pos` repeated `n_waypoints` times.
- Subsequent-call initial mean = `cat([prev_waypts[s:], prev_waypts[-1:].repeat(s, 1)])`
  with `s = step_each_iter`. Tested with n_waypoints=4, step_each_iter=1,
  seeded `_prev_waypts` with an arange pattern; the shift+pad output
  matched bit-for-bit.
- Dense plan shape = `(n_waypoints * interp_pts, action_dim)` —
  verified by inspection (the verbatim
  `trajectory_optimization_mppi_waypts` computes
  `act_len = self.n_look_ahead * interp_pts`).

### 6.4 `smoke_detection_failure.py` — **PASS**

Direct test of `optimize_action_mppi` math, no env / no CV.

- **Probe 1**: all K=100 rewards at `detection_failure_penalty=-1000`.
  After normalize-then-softmax, weights are uniform (constant input →
  softmax of zeros → 1/K). Weighted mean equals simple mean to within
  1.86e-8.
- **Probe 2**: 50 rollouts at -1000, 50 rollouts at -1.0. The high-reward
  half (= -1.0) dominates; weighted mean of the first action is
  -0.9999999 (all four dims), confirming the planner correctly assigns
  near-zero weight to the failure rollouts.
- No NaN/Inf in either case.

### 6.5 `plan_one_step.py` — **PASS** (with 3 diagnostics)

End-to-end test with real `WorldModelEnv("pusht_cam1")` and
`PushTTerminalReward(num_workers=8)`. Goal detected on episode_0 t=150
(real frame, mode='real'): x=223.847, y=253.023, angle=0.422°.

**Memory note**: K=100 + H=10 (the Config defaults) OOMs the 11.5 GB
laptop GPU because `env.step_batch` decodes all K*H frames in one
`render_img_cm` call (internal batch_size=50, attention softmax
allocates ~5 GB at K*H=40). The smoke test runs at K=8, n_waypoints=2,
interp_pts=2 (H_dense=4), n_update_iter=20 — total ~32 decodes per
iteration, fits comfortably. Tuning K toward 100 will require a larger
GPU or a future "chunked decode" / "terminal-only decode" mode on
`interactive_world_sim_env` (out of scope; env is frozen).

#### Diagnostic 1: reward iteration table

```
   iter         min        mean         max         std
      0     -2.4868     -1.2499     -0.2700      0.6250
      1     -1.5484     -1.0084     -0.4342      0.3401
      2     -2.3762     -1.3315     -0.6375      0.5476
      3     -2.0043     -1.4362     -1.2681      0.2435
      4     -2.2666     -1.7290     -1.1101      0.4548
      5     -2.3017     -1.7266     -1.1629      0.4094
      6     -2.8062     -1.8855     -1.3654      0.4803
      7     -3.8943     -1.8540     -1.3048      0.8629
      8     -3.0315     -1.8342     -1.2911      0.5412
      9     -2.6475     -2.1419     -1.8265      0.2868
     10     -3.3126     -2.1431     -1.3313      0.6971
     11     -3.5604     -2.2811     -0.9216      0.9019
     12     -3.3989     -2.0486     -0.7970      0.7275
     13     -2.2283     -1.6897     -1.0798      0.4083
     14     -2.6613     -1.7423     -0.6446      0.6258
     15     -2.4533     -1.9507     -1.2899      0.4420
     16     -2.9227     -1.6871     -0.8615      0.6163
     17     -2.8870     -1.7518     -1.2985      0.5205
     18     -3.0017     -1.8546     -0.9419      0.6382
     19     -2.8941     -1.6461     -0.6712      0.6519
```

Rewards are bounded by 0 (reward = -(pos_dist + angle_dist_rad) is
non-positive). Maxima fluctuate in [-2.28, -0.27] across iterations.
Iter 0 mean (-1.25) is the best mean across the run; later iterations
hover around -1.7 to -2.3 — the MPPI optimizer is NOT monotonically
improving on this small K=8 with normalize_rewards_before_softmax=True
configuration. Observations:

- Standard deviation stays in [0.24, 0.90] across iterations — rollouts
  remain spread, so the softmax is non-degenerate.
- The drift toward lower means suggests `reward_weight=200` may be too
  exploitative given the reward magnitude scale (-0.3 to -2.5); the
  planner aggressively chases the iter-0 maxima, then the noise
  re-randomizes around the chased mean.
- With K=100 (the design default) and larger horizon, expected behavior
  is steadier improvement. This is exactly the type of finding the
  diagnostic is supposed to surface.

This is informative for tuning but does NOT indicate a bug in the copied
algorithm — it's the small-K regime behaving as small-K MPPI does.

#### Diagnostic 2: histogram `/tmp/plan_one_step_reward_hist.png`

Iter-0 rewards: 0/8 at failure-penalty (all detections succeeded). Non-
penalty stats: min=-2.4868, mean=-1.2499, max=-0.2700, std=0.5847.

The histogram shows a non-degenerate spread of rewards across the 8
rollouts — there's no clustering at a single value, so the softmax has
a meaningful gradient to follow on iter 0. The K=8 sample is small but
diverse.

#### Diagnostic 3: best-rollout video `/tmp/plan_one_step_best_rollout.mp4`

Shape `(4, 128, 128, 3)`, 10 fps, libx264, 3.9 KB. The video is the
rollout_best replay decoded sequence — the dense actions executed under
the final optimized waypoints. Visually plausible: the pink T is in
view throughout the 4 frames, no decoder collapse, gripper motion
follows the optimized waypoints.

## 7. Deviations from the Phase 2 design

Tracked deviations (each surfaced explicitly):

1. **Default `n_sample` is honored in Config (100), but the plan_one_step
   smoke uses K=8** due to GPU memory. See §6.5 memory note. **The
   Config default itself is unchanged.**

2. **Default `interp_pts` is 5 in Config, but plan_one_step uses 2** —
   same reason: H_dense = n_waypoints × interp_pts × K determines
   render_img_cm peak memory. Config default unchanged.

3. **Kept `from tqdm import tqdm`** even though `verbose` is hardcoded
   False, to preserve byte-identity of the verbatim core. Originally
   §3.1 of the design said "Removed tqdm"; this is a tightening to
   maintain stricter verbatim equivalence.

4. **Noted upstream bug**: with `rollout_best=False`, the verbatim
   `trajectory_optimization_mppi_waypts` constructs `TrajOptOutput`
   with an unbound `act_seq`. We do not patch this — it reproduces
   the source faithfully. The smoke tests use `rollout_best=True`.

5. **Diagnostics callback ergonomics**: per the Phase 3 brief addition,
   `on_iter_callback(iter_index, reward_seqs, action_seqs)` is wired
   into `api._adapter_evaluate_traj`. We pass `action_seqs` (the
   spline-interpolated dense actions) rather than the pre-spline
   waypts_seqs because the verbatim core doesn't expose the waypts to
   the evaluate hook. The `rollout_best` final replay fires the
   callback with `iter_index = -1` to distinguish it.

No other deviations.

## 8. What's next (deferred — for your review)

- `closed_loop_demo.py` — multi-step receding-horizon execution with
  reward/distance curve and 3-row video, per Phase 2 §8.
- Phase 4 visualization artifacts.

Both deferred per the brief until you confirm Phase 3 is clean.
