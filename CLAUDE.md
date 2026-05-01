# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

The repo runs inside the `iws` conda env (Python 3.11, CUDA 12.6, PyTorch 2.7.1). All commands assume `conda activate iws` first. The package is installed editable (`pip install -e .`).

`pyproject.toml` pins `pythonpath = ["."]` for pytest because top-level packages (`env/`, `rl/`, `interactive_world_sim/`) sit alongside each other rather than under one src/. Running `pytest` from anywhere other than the repo root will import-fail without that setting.

## Common commands

```bash
# Tests (always run from repo root)
pytest tests/                               # full suite
pytest tests/mppi/test_mppi_planner.py      # one file
pytest tests/mppi -k consistency            # by keyword

# Lint / format (pre-commit drives ruff + black + mypy)
pre-commit run --all-files
ruff check . --config pyproject.toml --fix
```

### World-model training (Hydra, three stages)

```bash
# Stage 1: autoencoder. Stage 2: dynamics. Stage 3: AE finetune.
# All three use main.py with algorithm=latent_world_model, experiment=exp_latent_dyn,
# differing in algorithm.training_stage and dataset horizons. See README.md for the full
# command per stage and which stage's checkpoint becomes the next stage's algorithm.load_ae.
python main.py +name=<run_name> algorithm=latent_world_model experiment=exp_latent_dyn \
    dataset=real_aloha_dataset dataset.dataset_dir=data/mini/pusht ... \
    algorithm.training_stage=<1|2|3>
```

A WandB entity is required (set in `configurations/config.yaml` or via `wandb.entity=...`). Without it, `main.py` raises before training starts.

### Dreamer-style RL training

```bash
# Single GPU
bash rl/train_cloud.sh 1
# Or direct, overriding any DreamerConfig field (see rl/utils/config.py):
python rl/train.py --total_steps 200000 --batch_size 64 --imagination_horizon 15
# Small-VRAM fallback
python rl/train.py --batch_size 8 --imagination_horizon 10 --use_gradient_checkpointing true
```

Defaults target a single 80 GB A100. Eval: `python rl/evaluate.py --checkpoint rl/outputs/checkpoints/final.pt`.

### MPPI control runs

```bash
python scripts/run_mppi_v2.py \
    --config configs/mppi/default.yaml \
    --initial_hdf5 data/mini/pusht/val/episode_0.hdf5 --initial_frame 0 \
    --goal tests/goal_selection/state_goal.pt \
    --output_dir outputs/mppi/<run_name> \
    --wm_ckpt outputs/pusht_cam1/checkpoints/best.ckpt
```

Algorithm-changing flags (`--n_sample`, `--n_update_iter`) require `--override_reason`; `--control_steps` and `--seed` are per-run knobs that don't.

The 4-GPU multi-pair sweep launcher is `scripts/run_multipair_experiment.sh` (writes to `outputs/mppi/multipair_cloud/`, refuses to clobber an existing dir).

### One-time setup steps

```bash
bash scripts/download_checkpoints.sh        # 7 WM ckpts → outputs/<task>_<cam>/
bash scripts/download_mini_data.sh          # mini ALOHA dataset → data/mini/
git submodule update --init --recursive     # gym-aloha for sim data collection
uv pip install -e external/gym-aloha/
# MPPI/Dreamer goal latent (one-time)
python tests/goal_selection/extract_final_frames.py
python tests/goal_selection/encode_goal.py --image_path tests/goal_selection/frames/episode_0_final.png
```

## Architecture

The repo is **three loosely-coupled pipelines that share one frozen world model**. Knowing which pipeline a file belongs to is the single most useful orientation:

1. **World-model training** — `main.py` + `interactive_world_sim/` (Hydra + PyTorch Lightning, configured under `configurations/`). Trains the latent dynamics model in three stages and produces `outputs/<task>_<cam>/checkpoints/best.ckpt`. This is the only pipeline that *trains the WM*.

2. **MPPI control** — `scripts/run_mppi_v2.py` + `rl/mppi/` + `env/pusht_wm_env.py` (configured under `configs/mppi/`, plain OmegaConf, no Hydra). Loads a frozen WM checkpoint and plans actions by sampling action sequences, rolling them out through the WM, and softmax-weighting them by a CV-based reward. Stateless across plan steps.

3. **Dreamer RL** — `rl/train.py` + `rl/{models,training,utils}/` (plain `argparse` over a `DreamerConfig` dataclass, no Hydra). Loads the same frozen WM and trains an actor + critic via imagination rollouts. `DiscreteActor` (Gumbel-Softmax ST) is the default; continuous `Actor` exists too.

Pipelines 2 and 3 both depend on:
- `env/pusht_wm_env.py::PushTWMEnv` — the shared atomic-ops wrapper around the WM (`rl/models/world_model.py::DifferentiableDynamics`) and the CV state estimator (`rl/labeling/cv_labeler.py::CVLabeler`). All public methods support both batched and unbatched input. Anything that wants to encode an obs, roll the WM forward, decode a latent, or estimate a state goes through here.
- The frozen WM checkpoint at `outputs/pusht_cam1/checkpoints/best.ckpt`. The Dreamer config and MPPI config both point at this path by default.

### Hard-won invariants (do not regress)

- **Camera is hardcoded.** `PushTWMEnv.load_initial_from_hdf5` uses `obs_key='camera_1_color'` because the IWS PushT WM was trained on cam1, not cam0. This was a real bug (commit `fd875ef`) where inference scripts silently used cam0; the fix was to remove camera as a parameter. **Do not re-expose it.** See `env/pusht_wm_env.py:36-38` and `MPPI_NOTES.md`.
- **Rollout horizon cap.** `PushTWMEnv` enforces `MAX_HORIZON = 50` because drift quantification (`tests/env/test_rollout_drift.py`, `outputs/drift_report.json`) has only been validated up to 50 steps. The WM internally does sliding-window rollout, so longer horizons would *run* but aren't trustworthy.
- **MPPI is a faithful port of diffusion-forcing's `planner_v0_0`.** `MPPI_REFERENCE_NOTES.md` is the authoritative spec; deviations are explicitly enumerated there. In particular: `beta_filter` is **intra-horizon noise smoothing**, not a cross-plan-step warm-start (this was misinterpreted in older notes). The planner is stateless across `plan_step` calls.
- **MPPI defaults trade reference fidelity for compute.** `n_update_iter=5` (vs reference 50) and `n_look_ahead=10` (vs reference 40) are intentional in `configs/mppi/default.yaml` — the IWS decoder is compute-bound. The flag-gated `config_deviation` machinery in `scripts/run_mppi_v2.py` exists so that an algorithm-changing override is recorded in every run's `summary.json`.

### Notes worth reading before non-trivial work

- `MPPI_NOTES.md` — current MPPI status, Phase 2 sanity-run results, and open questions for cloud reruns.
- `MPPI_REFERENCE_NOTES.md` — the three exact MPPI formulas + the deviation list.
- `MPPI_BOTTLENECK_ANALYSIS.md` — why the decoder dominates wall time.
- `ACTION_NORMALIZATION_AUDIT.md` — action-space conventions across pipelines (the `[-1, 1]` unit cube is enforced inside MPPI but the WM was trained on raw demonstration actions; mismatch matters).
- `migration_log.md` — every deviation from the upstream supervisor's CV pose estimator (`~/Documents/aloha/.../analyze.py`), including why the supervisor module is loaded via `importlib` rather than imported normally (avoids ROS deps).
- `state_estimator_research.md` — CV labeler design notes.
- `CLOUD_SETUP.md` — end-to-end cloud GPU box bring-up.

## Conventions

- Lint config in `pyproject.toml` enables ruff with pydocstyle (`D101/D102/D2/D3`), bugbear, return-style, and PIE. Pre-commit also runs black, mypy (`--disallow-untyped-defs`), and rejects binary files. The `external/` submodule and a couple of real-world hardware files are excluded — see `.pre-commit-config.yaml`.
- Output artifacts go to `outputs/`; per-run subdirs are expected to contain `summary.json`, `trajectory.mp4`, `reward_curve.png`, and `reproduce_on_cloud.sh`.
- Episode data is HDF5 on disk, cached as zarr for fast loading during WM training.
