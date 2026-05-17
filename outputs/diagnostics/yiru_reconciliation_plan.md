# Yiru Reconciliation Plan — Semantic Match to production

Date: 2026-05-16
production HEAD: `5e2d48e902db5accb6c3f5b7beca8916bde36084`
yiru HEAD: `6ede1577b30efda434c3b3d28d9ff3e20660e658`
backup tag: `pre-reconcile-yiru-2026-05-16`

This plan reconciles the **algorithmic** content of yiru's MPPI to match
production. yiru's organizational style (package split, dataclass Config,
typed dataclasses for Observation/EnvState, dedicated CV package) is
preserved; only the math is rewritten.

## Decision: Restructure within yiru

The verbatim-from-diffusion-forcing `_mppi_core.py` is **incompatible with
production semantics** without invasive edits to the verbatim code, which
would defeat the purpose of preserving it. The clean approach is to:

- **Add** a new module `_planner.py` under `interactive_world_sim_mppi/`
  that implements production's MPPI semantics (per-step AR(1), sample
  delta clip, cube clamp, dedicated RNG generator, etc.).
- **Replace** `api.py`'s `MPPIPlanner` to delegate to `_planner.py`'s
  `Planner` class (keeping the public class name and the snapshot-based
  contract).
- **Keep** `_mppi_core.py` and `_splines.py` intact — they remain the
  documented diffusion-forcing reference but become inactive. The
  bitwise-equivalence smoke test continues to work against them.

## Subsystem reconciliation table

| # | Subsystem | production source | yiru destination | gap | plan |
|---|-----------|-------------------|------------------|-----|------|
| 1 | Vanilla AR(1) sampler | `rl/mppi/mppi_planner.py:289-367` | `interactive_world_sim_mppi/_planner.py::Planner.sample_action_sequences` (new) | yiru hardwires MPPI_WAYPTS; vanilla path absent | Implement production's vanilla sampler: `randn(generator=self._gen) * sigma → AR(1) → +mean → cube clamp → sample_delta_clip → cube re-clamp` |
| 2 | WAYPTS sampler + Catmull-Rom interp | `rl/mppi/mppi_planner.py:386-503` (`_sample_waypoints_then_interp`, `_interp_along_dim`) | `interactive_world_sim_mppi/_planner.py::Planner._sample_waypoints_then_interp` + `_interp_along_dim` (new) | yiru has natural cubic spline through 3 knots, hardwired; production Catmull-Rom variant, optional | Port production's K-waypoint sampler + linear/Catmull-Rom interp verbatim |
| 3 | `sample_delta_clip` | `rl/mppi/mppi_planner.py:241-285` | `interactive_world_sim_mppi/_planner.py::Planner._apply_sample_delta_clip` (new) | absent in yiru | Port production's clip + cube re-clamp loop |
| 4 | Cube clamp position | `mppi_planner.py:337-343` (clamp inside per-step loop) + `:241-285` (re-clamp inside delta clip) | new `_planner.py` | yiru's spline path doesn't cube-clamp until after sample; order differs | Match production: clamp inside the loop then re-clamp in delta clip |
| 5 | RNG generator | `mppi_planner.py:80-82`: `torch.Generator(device).manual_seed(cfg.seed)` | new `_planner.py::__init__` | yiru: no seed, global CUDA RNG | Build dedicated `self._gen`, pass `generator=self._gen` to every `torch.randn` |
| 6 | `evaluate_trajectories` | `mppi_planner.py:529-569` | new `_planner.py::Planner.evaluate_trajectories` | yiru does this via env.step_batch + user reward callable; production uses env.rollout + env.compute_reward directly | New `evaluate_trajectories` calls a `_decode_and_score` helper that decodes final latent at 128×128, runs CV at 128 with preset "REAL", applies production reward |
| 7 | `optimize_action_mppi` | `mppi_planner.py:505-525` | new `_planner.py::Planner.optimize_action_mppi` | yiru has optional `(R-μ)/σ` standardization (ON by default); production does max-subtract only | Match production exactly: `softmax((R*rw) - max(R*rw))` — no standardization |
| 8 | Reward formula | `env/pusht_wm_env.py:328-385` (`compute_reward`) | `interactive_world_sim_mppi/reward.py::pusht_terminal_reward_production` (new) | yiru uses `-(pos+acos)`, 512px, -1000 fail; production uses `-(pos/diag + 1-cos)`, 128px, -10 fail | New reward function matches production: position term normalized by image diagonal, angle as `1-cos`, fail penalty = `cv_fail_penalty` |
| 9 | CV labeler | `rl/labeling/cv_labeler.py` preset "REAL" at 128 res | `interactive_world_sim_cv` mode "real" at 128 px (via `processing_resolution` argument) | yiru defaults to 512 + "wm" for rollouts | New reward calls `detect(rgb, mode='real', processing_resolution=128)`. Goal pose detection unchanged (uses "real" already). Production preset "REAL" matches `_detection.HSV_LOWER_REAL` (same numerical bounds). |
| 10 | WM rollout call | `env.rollout(z0_batch, act_seqs)` returns `(N, H+1, C, h, w)` with z0 at index 0 | `WorldModelEnv.step_batch(act_seqs)` returns `BatchedObservation(latents=(K,H,C,h,w))` no z0 | Different shapes; yiru needs z_final separately | Use `step_batch` and grab `batched.latents[:, -1]` for z_final; decode via `env.latent_to_rgb` |
| 11 | Audit log | `mppi_planner.py:124-125, 700-707, 759-782` + `run_mppi_v2.py:956-966` | new yiru driver writes `audit_log.json` per same schema | absent in yiru | Driver emits same JSON schema (`plan_call_idx`, `niter`, `mean`, `best_sample_reward`, optional `samples`/`sample_rewards`) |
| 12 | Config defaults | `configs/mppi/default.yaml` | `interactive_world_sim_mppi/config.py::Config` (dataclass) | many fields differ in value/presence | Update Config to expose every production yaml field with matching defaults |
| 13 | Argparse defaults | `scripts/run_mppi_v2.py:1011-1099` | new yiru driver | absent | New driver replicates production argparse signature |
| 14 | SEI / control loop | `run_mppi_v2.py:633-766` | new yiru driver | absent | Port the while-loop with `step_each_iter` semantics |
| 15 | Output dump | `run_mppi_v2.py:776-967` (summary.json, trajectory.mp4, audit_log.json, reward_curve.png) | new yiru driver | absent | Port the output writers |

## File-level edit plan

### NEW files

1. **`interactive_world_sim_mppi/_planner.py`** — Production-faithful
   MPPI core. Wraps WorldModelEnv. Methods:
   - `__init__(env, config, image_diagonal=181.02)`
   - `set_anchor(anchor)`
   - `_apply_sample_delta_clip(samples, anchor)`
   - `sample_action_sequences(act_seq)`
   - `_sample_waypoints_then_interp(act_seq, K)` + `_interp_along_dim`
   - `_sampler_sigma_and_bounds(A)`
   - `optimize_action_mppi(act_seqs, rewards)` — max-subtract only
   - `evaluate_trajectories(z_current_unused, act_seqs, goal_state)`
     — calls env.step_batch then production-style reward via CV at 128
   - `trajectory_optimization(z_current_unused, goal_state, init_act_seq)`
   - `plan_step(z_current_unused, goal_state, init_act_seq, anchor=)`
   - `_validate_config()`
   - `get_audit_log()`

2. **`interactive_world_sim_mppi/scripts/run_mppi.py`** — Closed-loop
   driver. Mirrors `scripts/run_mppi_v2.py` but writes to yiru
   conventions (uses WorldModelEnv from `interactive_world_sim_env`,
   uses `_planner.py::Planner`). Argparse defaults match production
   (file paths point at yiru-loaded ckpt via task name).

3. **`tests/mppi/test_yiru_smoke.py`** — Smoke test asserting the new
   planner imports, constructs, runs 1 niter, and returns the expected
   shape. (Repo root `tests/` directory exists — confirmed.)

### MODIFIED files

1. **`interactive_world_sim_mppi/config.py`** — Add every production yaml
   field with production default values:
   - `n_sample: 100` (already correct)
   - `n_look_ahead: 10` (replaces `n_waypoints + interp_pts`)
   - `n_update_iter: 5` (yiru: 50 → 5)
   - `noise_level: 0.05` (already correct)
   - `reward_weight: 200.0` (already correct)
   - `beta_filter: 0.7` (already correct)
   - `cv_fail_penalty: -10.0` (yiru: `detection_failure_penalty=-1000`)
   - `waypoints_n: None` (new)
   - `waypoints_interp: "linear"` (new)
   - `action_dim: 4` (already implicit via action_lower_lim)
   - `action_lower_lim`, `action_upper_lim` (already correct)
   - `delta_mode: False`, `delta_action_lim: 0.0872`,
     `noise_level_delta: 0.02`, `cumulative_drift_log_only: True`
   - `sample_delta_clip: True`,
     `per_step_delta_lim: [0.0975, 0.0919, 0.0760, 0.0909]`
   - `step_each_iter: 1`
   - `control_steps: 50`
   - `seed: 0`
   - `cv_n_workers: 16`
   - `cv_processing_resolution: 128` (yiru: 512)
   - `image_diagonal: float = sqrt(128² + 128²)` (new)
   - `rollout_best`: REMOVE (production has no analog)
   - `normalize_rewards_before_softmax`: REMOVE (this was THE divergence)
   - `pos_weight`, `angle_weight`: REMOVE (production formula has no
     pos_weight; it's `-pos/diag`)
   - `detector_num_workers`: keep but rename to `cv_n_workers`
     (production name)
   - `GoalPose`: keep (used by other surface area, but the new reward
     path uses dict goal_state)

2. **`interactive_world_sim_mppi/reward.py`** — Add a new
   `production_terminal_reward(latents, rgbs, actions, config)` and a
   `PushTTerminalRewardProduction` class for the pool-backed version.
   The new functions:
   - Take `rgbs[:, -1]` (terminal frame, no change here)
   - Resize to 128×128 with INTER_AREA (NOT 512+CUBIC)
   - Call `detect(rgb, mode='real', processing_resolution=128)`
   - Use production reward formula: `-pos_dist/image_diagonal -
     (1 - cos(Δθ))`
   - Fail penalty: `cv_fail_penalty` (default -10.0) from config
   - Goal: dict-based `{'cx', 'cy', 'sin_theta', 'cos_theta',
     'theta_deg'}` to match production's goal_state
   - Add `detect_goal_pose_dict_from_episode` helper that returns dict
     instead of GoalPose, also detects in mode='real' at 128 px
   - Keep the old `pusht_terminal_reward` for backwards compat with
     existing smoke scripts, BUT mark as deprecated.

3. **`interactive_world_sim_mppi/api.py`** — Replace `MPPIPlanner` with
   a thin façade that delegates to `_planner.py::Planner`. Maintain the
   public class name `MPPIPlanner` and the `plan(snapshot)` method, but
   the implementation uses the new production-faithful Planner.

4. **`interactive_world_sim_mppi/__init__.py`** — Export changes (drop
   `GoalPose` from the active path or keep both surfaces).

## Static byte-equivalence preview (Phase 10 checks)

| # | Item | How to check |
|---|------|--------------|
| 1 | Yaml defaults match | Diff `Config` defaults vs `default.yaml` |
| 2 | RNG generator setup identical | Inspect `__init__` for `torch.Generator(device).manual_seed(cfg.seed)` |
| 3 | Sample order: noise→AR(1)→mean→clamp→sample_delta_clip→re-clamp | Read the new `sample_action_sequences` and compare line-by-line with production |
| 4 | AR(1) coefficient + formula | `act_residual = beta * noise + (1.0-beta) * act_residual` with `beta = config.beta_filter` |
| 5 | Cube clamp position | `new_step = clamp(act_seqs[:, i] + act_residual, lo, hi)` matching production |
| 6 | sample_delta_clip per-dim p99 default + formula | Verify `per_step_delta_lim` default and the in-place clip+cube-re-clamp loop |
| 7 | WAYPTS K→H interp | Verify `_interp_along_dim` linear branch math |
| 8 | Cubic Catmull-Rom coefficients | Verify the four-point Hermite expansion |
| 9 | optimize_action softmax (max-subtract) | `weights = F.softmax(R*rw - (R*rw).max(), dim=0)` — no standardization |
| 10 | evaluate_trajectories chain | env.step_batch → rgbs[:,-1] → CV(128, 'real') → compute_reward |
| 11 | Reward formula | `-pos_dist/image_diagonal - (1-cos_delta)` with `cv_fail_penalty=-10` |
| 12 | CV labeler (preset, resolution, preprocessing) | mode='real', resolution=128, no further resize after env decode (env already produces 128x128) |
| 13 | WM rollout call signature + inference_mode | `torch.no_grad()` + `step_batch`; env already wraps `dynamics_forward` in no_grad |
| 14 | SEI loop + control_step semantics | `while n_actions_done < control_steps: n_this = min(SEI, control_steps - done); ...` matching production |

## Notes / risks

- The yiru env decodes at 128×128 (per registry `resolution=128`) so the
  decoded RGB is naturally at the CV labeler's expected size. No resize
  is needed in the reward path. Production's `_preprocess_rgb_uint8`
  applied INTER_AREA only at encode time (HDF5 → encoder input); the
  decoder output is already at 128 px in both branches.
- The yiru env's `step_batch` returns latents WITHOUT a leading z0 entry
  (production has `H+1`). We use `latents[:, -1]` as `z_final`, which is
  semantically equivalent to production's `traj[:, -1]`.
- The verbatim diffusion-forcing core (`_mppi_core.py`) and `_splines.py`
  remain in the tree but become unused by the public API. The
  `smoke_bitwise_equivalence.py` test continues to work against them.
- We keep yiru's `_extract_curr_pos` snapshot handling because the env's
  snapshot is the only way to "restore between rollouts" in yiru's
  step_batch model. Production's planner takes a single latent and
  doesn't need restore semantics; yiru's planner will internally
  snapshot/restore exactly as it does today, but the algorithmic
  primitives (sampling, optimizing, scoring) match production.
- `image_diagonal` is computed as `sqrt(128² + 128²) ≈ 181.019` in
  production. We store this constant in the new Planner and pass it to
  the reward at every call.
