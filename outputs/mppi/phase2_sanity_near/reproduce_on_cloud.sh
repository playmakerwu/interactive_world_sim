#!/usr/bin/env bash
# Cloud reproduction of this run at the configured-default hyperparameters
# (no --n_sample override). Requires GPU with >=20 GiB free for decoder
# attention scratch at the default n_sample.
#
# This local run used: n_sample: 100 -> 16
# Reason: Local GPU has 11.7 GiB free. Phase 1 5-iteration refinement loop accumulates peak memory across iterations beyond the single-iteration measurement; N=32 OOMs even with empty_cache between iterations. N=16 fits within budget. All other config values (n_update_iter=5, noise_level=0.05, reward_weight=200, beta_filter=0.7) match configs/mppi/default.yaml unchanged.
set -e
cd "$(dirname "$0")/../../.."
python scripts/run_mppi_v2.py \
    --config configs/mppi/default.yaml \
    --wm_ckpt outputs/pusht_cam1/checkpoints/best.ckpt \
    --initial_hdf5 data/mini/pusht/val/episode_0.hdf5 \
    --initial_frame 0 \
    --goal tests/goal_selection/state_goal.pt \
    --output_dir outputs/mppi/phase2_sanity_near_cloud_repro
