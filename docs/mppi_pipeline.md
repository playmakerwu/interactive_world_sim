# MPPI Pipeline Architecture (post-cleanup)

**Cleanup date**: 2026-05-13
**Pre-cleanup tag**: `pre-cleanup-2026-05-10` → `bac9ab7c99b6819ee740e7802a5611b4eaa3a00b`
**Revert**: `git reset --hard pre-cleanup-2026-05-10`

## 1. Entry point

`scripts/run_mppi_v2.py` — given a `(initial_hdf5, initial_frame, goal.pt, wm_ckpt)`, runs an MPPI control episode and writes artifacts (`summary.json`, `trajectory.mp4`, `trajectory_combined.mp4`, `reward_curve.png`, `iter_reward_heatmap.png`, `action_history.pt`, `iteration_log.pt`, `action_dist/*.png`) to `--output_dir`.

## 2. Pipeline overview

```
                   ┌───────────────────────────────────────────────────┐
                   │            scripts/run_mppi_v2.py                 │
                   │  (env bootstrapping, multi-GPU launch, artifact   │
                   │   writing, per-step viz)                          │
                   └──────────┬────────────────────────────────────────┘
                              │
                              ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  PushTWMEnv (env/pusht_wm_env.py)                            │
   │  • encode_initial_state(hdf5, frame) → z0                     │
   │  • encode_goal(goal.pt) → z_goal (plus pose state via CV)     │
   │  • step(action, z) → z_next, decoded RGB, CV pose, reward     │
   │  • Cam-1 ONLY; horizon ≤ MAX_HORIZON=50 (drift-validated)     │
   └──────┬─────────────────────┬─────────────────────────────────┘
          │                     │
          ▼                     ▼
   ┌──────────────┐       ┌──────────────────────────────────────┐
   │ Differentiable│       │ CVLabeler (rl/labeling/cv_labeler.py) │
   │ Dynamics      │       │ HSV + contour + ICP T-pose extractor  │
   │ (frozen WM)   │       │ (vendored from supervisor's analyze.py)│
   │ ↓ loads ckpt  │       └──────────────────────────────────────┘
   │ LatentWorld   │
   │ Model         │
   └──────────────┘
                              │
                              ▼
   ┌───────────────────────────────────────────────────────────────┐
   │  MPPIPlanner (rl/mppi/mppi_planner.py)  — for each plan_step: │
   │   for niter = 0..n_update_iter-1:                              │
   │     1. sample_action_sequences(mean, sigma, beta_filter, K, …) │
   │        → (N, H, A) candidate trajectories                      │
   │     2. _interp_along_dim(samples, mode)  (if waypoint mode)    │
   │        → up-sampled (N, H, A)                                  │
   │     3. WM rollout: env.rollout(z0, samples) → (N, H+1, latents)│
   │     4. decode → (N, H, RGB) ; CV → (N, H, pose)                │
   │     5. reward(pose, goal_pose) → (N, H)                        │
   │     6. weights = softmax(reward_weight · sum(rewards, dim=H))  │
   │     7. mean ← weights · samples (sum over N)                   │
   │   return mean[0]  (execute first action)                       │
   └───────────────────────────────────────────────────────────────┘
                              │
                              ▼
                  ┌────────────────────────────┐
                  │  Distributed shard helpers  │
                  │  (rl/mppi/distributed.py)   │
                  │  Multi-GPU sample sharding  │
                  │  via torch.distributed      │
                  │  (single-process no-ops on  │
                  │  laptop)                    │
                  └────────────────────────────┘
```

The MPPI planner is **stateless across plan_step calls** — no warm-start.
`beta_filter` smooths noise *within* a horizon, not across plan steps
(see [MPPI_REFERENCE_NOTES.md](../MPPI_REFERENCE_NOTES.md)).

## 3. Module inventory (post-cleanup)

### Entry & config
| Path | Purpose |
|------|---------|
| `scripts/run_mppi_v2.py` | Runner: bootstraps env, runs MPPI, writes artifacts |
| `scripts/run_config.py` | Run-time toggles (`USE_WARM_START`, etc.) |
| `scripts/viz/action_sampling_viz.py` | Per-niter sample-iteration visualization |
| `scripts/viz/closed_loop_goal_viz.py` | Closed-loop WM rollout with goal overlay |
| `configs/mppi/default.yaml` | All hyperparameters + algorithmic deviations |
| `configs/mppi/local_hard_K16.yaml` | Laptop-friendly K=16 variant |

### MPPI core
| Path | Purpose |
|------|---------|
| `rl/mppi/mppi_planner.py` | `MPPIPlanner` — `plan_step`, `sample_action_sequences`, `_interp_along_dim`, `trajectory_optimization` |
| `rl/mppi/distributed.py` | Multi-GPU sample-shard helpers (rank/broadcast/all_gather) |
| `rl/mppi/__init__.py` | Package init |

### World model + state estimator (shared across pipelines)
| Path | Purpose |
|------|---------|
| `env/pusht_wm_env.py` | `PushTWMEnv` atomic ops (encode/rollout/decode/CV/reward) |
| `env/expert_action.py` | Expert action extraction from episode HDF5 |
| `rl/models/world_model.py` | `DifferentiableDynamics` — frozen-WM wrapper |
| `rl/labeling/cv_labeler.py` | HSV + contour + ICP T-pose labeler (vendored) |
| `interactive_world_sim/algorithms/latent_dynamics/latent_world_model.py` | `LatentWorldModel` PyTorch Lightning module |
| `interactive_world_sim/algorithms/latent_dynamics/models/cm_latent_dynamics.py` | `CMLatentDynamics` (Hydra-instantiated from ckpt) |
| `interactive_world_sim/algorithms/common/base_pytorch_algo.py` | Base Lightning algorithm |
| `interactive_world_sim/algorithms/common/diffusion_helper.py` | `render_img_cm` |
| `interactive_world_sim/algorithms/common/metrics/{__init__,fid,fvd,lpips}.py` | Training-time eval metrics (imported at WM load) |
| `interactive_world_sim/algorithms/models/{attention,cm_decoder,cm_controlnet,diffae_unet,embeddings,utils}.py` | WM neural-network building blocks |
| `interactive_world_sim/utils/{cm_utils,dict_of_tensor_mixin,draw_utils,logging_utils,normalizer,pose_utils,pytorch_util}.py` | WM utility helpers |

### Visualization helpers (used by run_mppi_v2)
| Path | Purpose |
|------|---------|
| `rl/visualization/combined_video.py` | `render_combined_video` — per-frame trajectory + reward-iteration video |
| `rl/visualization/action_distribution.py` | `render_action_distributions` — per-step action histograms |
| `rl/visualization/demo_action_stats.py` | Cached demo action stats + WM normalizer params |
| `rl/visualization/state_viz.py` | `render_state_on_image` — overlay state estimates on RGB |

### Tests retained (3 core)
| Path | What it verifies |
|------|------------------|
| `tests/mppi/test_waypoints.py` | Waypoint sampling: linear/cubic interp, action-bound clamp, count invariants |
| `tests/mppi/test_sample_delta_clip.py` | Per-step delta clipping in vanilla mode (anchor-relative for first step) |
| `tests/mppi/test_audit_log.py` | yaml-gated audit log (per-niter `mean` + `samples` dumps) |

Total: **52 .py files** retained.

## 4. Key YAML fields (configs/mppi/default.yaml)

| Field | Default | Meaning |
|-------|---------|---------|
| `n_sample` | 100 | N: trajectory samples per refinement iteration |
| `n_look_ahead` | 10 | H: planning horizon (reference uses 40; IWS reduced because n_frames=10) |
| `n_update_iter` | 5 | Inner refinement iters per plan_step (reference uses 50; reduced because decoder is compute-bound) |
| `noise_level` | 0.05 | σ for Gaussian action noise |
| `beta_filter` | 0.7 | Intra-horizon AR(1) noise smoothing (NOT cross-plan warm-start) |
| `reward_weight` | 200.0 | Softmax temperature multiplier |
| `cv_fail_penalty` | -10.0 | Reward when CV labeler can't detect T pose |
| `waypoints_n` | null | K: sparse waypoint count (null = per-step sampling) |
| `waypoints_interp` | linear | "linear" or "cubic" |
| `delta_mode` | false | Sample per-step deltas + cumsum onto anchor (off by default) |
| `delta_action_lim` | 0.0872 | Symmetric per-dim p99 from demos |
| `sample_delta_clip` | true | Vanilla-mode per-step Δ cap (default ON since 2026-05-12) |
| `per_step_delta_lim` | `[0.0975, 0.0919, 0.0760, 0.0909]` | Per-dim normalized bound |
| `audit_log_enabled` | false | yaml-gated per-niter trace dump |
| `debug_mode` | false | Paper-grade artifact bundle |
| `step_each_iter` | 1 | Actions executed per MPPI plan_step call |
| `control_steps` | 50 | plan_step calls per episode |
| `cv_n_workers` | 16 | CV labeler worker-pool size |
| `action_dim` | 4 | ALOHA bimanual end-effector deltas |

See [MPPI_REFERENCE_NOTES.md](../MPPI_REFERENCE_NOTES.md) for the algorithm spec.

## 5. Files deleted (summary)

Cleanup deleted **157 .py files + 11 .yaml + 1 .md + 3 stale empty dirs** under
the pre-cleanup tag `pre-cleanup-2026-05-10`. By category:

| Category | Files | Notes |
|----------|------:|-------|
| WM training (`main.py` + `interactive_world_sim/{datasets,environments,experiments,real_world,utils,…}/`) | 74 | WM is frozen at MPPI inference; training code unused |
| Other MPPI runners/tests (not 3 core) | 23 | Old `planner.py`, `run_mppi.py`, `run_mppi_pairs.py`, `run_mppi_warmup_phase_c.py`, 15 non-core MPPI tests |
| Non-MPPI tests | 20 | `tests/{env,goal_selection,gradient_flow,latent_*,scripts,state_estimator,visualization}/` |
| RL training (Dreamer) | 16 | `rl/training/`, `rl/utils/`, `rl/models/{actor,actor_discrete,critic}`, `rl/{train,evaluate*,compare_policies,analyze_initial_states}.py` |
| Demo / data collection | 18 | `deploy/`, `scripts/{calibrate_hsv,compute_state_goal*,label_replay_buffer,download_*,upload_*}.py`, `scripts/data_collection/`, `scripts/inference/` |
| Notebooks / experiments | 6 | `scripts/{viz_step1..4,wm_interactive_replay,make_rollout_showcase}.py` |
| Ad-hoc scripts (`_` prefix) | 2 | `scripts/_profile_mppi.py`, `scripts/_validate_batch_scaling.py` |
| Stale empty top-level dirs | 3 | `interactive_world_sim_{cv,env,mppi}/` |
| Hydra WM-training configs | 11 yaml | `configurations/` (used only by deleted `main.py`) |
| Pre-refactor archive note | 1 md | `MPPI_NOTES_archive_pre_refactor.md` (superseded) |

**8 files** initially deleted then **restored** because they're loaded
either as relative-import siblings of kept modules or via Hydra string
resolution at WM-checkpoint load time (which AST-based BFS doesn't see):
`interactive_world_sim/algorithms/common/metrics/{fid,fvd,lpips}.py`,
`interactive_world_sim/algorithms/models/{cm_controlnet,embeddings,diffae_unet}.py`,
`interactive_world_sim/utils/{pose_utils,pytorch_util}.py`,
`interactive_world_sim/algorithms/latent_dynamics/models/cm_latent_dynamics.py`.

## 6. Sanity check (post-cleanup)

- **3 core tests**: 18/18 PASS (`tests/mppi/{test_waypoints,test_sample_delta_clip,test_audit_log}.py`).
- **Smoke MPPI run**: `outputs/smoke/post_cleanup_20260513_142725/`
  - `final_pos_distance_px`: 54.81
  - `final_angle_error_deg`: -57.16
  - `n_cv_failures`: 0
  - `wall_time_s`: 9.10 (N=4, H=5, niter=3, T=2; tiny smoke config)
  - Artifacts: `summary.json`, `trajectory.mp4`, `trajectory_combined.mp4`,
    `reward_curve.png`, `iter_reward_heatmap.png`, `action_history.pt`,
    `iteration_log.pt`, `initial_latent.pt`, `trajectory_latents.pt`,
    `action_dist/` (2 PNGs), `iter_viz/`.

## 7. What changed in code

- **Module-level docstrings** added to 27 files that lacked them (mostly
  `interactive_world_sim/` internals + `__init__.py` files). One short
  line each per the project's "no multi-line comment blocks" rule.
- **No algorithm changes.** `sample_action_sequences`,
  `_interp_along_dim`, `trajectory_optimization`, and every public method
  on `MPPIPlanner` were untouched.

## 8. Revert instructions

If anything regresses:

```bash
git reset --hard pre-cleanup-2026-05-10
```

(The tag is pushed to `origin/pre-cleanup-2026-05-10`.)
