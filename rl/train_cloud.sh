#!/bin/bash
# Cloud training launch script.
# Usage: bash rl/train_cloud.sh [num_gpus]
#
# Run from repo root. Assumes the `iws` conda env is set up.
set -e

NUM_GPUS=${1:-1}

# pick up conda
if [ -z "$CONDA_PREFIX" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate iws

# ── prerequisites ────────────────────────────────────────────────────
if [ ! -d "outputs/pusht_cam1" ]; then
    echo ">> Downloading world-model checkpoints …"
    bash scripts/download_checkpoints.sh
fi
if [ ! -d "data/mini/pusht" ]; then
    echo ">> Downloading mini dataset …"
    bash scripts/download_mini_data.sh
fi
if [ ! -f "tests/goal_selection/z_goal.pt" ]; then
    echo ">> Generating goal latent …"
    python tests/goal_selection/extract_final_frames.py
    python tests/goal_selection/encode_goal.py \
        --image_path tests/goal_selection/frames/episode_0_final.png
fi

# ── multi-GPU toggle ─────────────────────────────────────────────────
MULTI_GPU_FLAG=""
if [ "$NUM_GPUS" -gt 1 ]; then
    MULTI_GPU_FLAG="--use_multi_gpu true"
fi

# ── train ────────────────────────────────────────────────────────────
# Defaults come from DreamerConfig (imagination_horizon=15, batch_size=64,
# total_steps=200000, use_gradient_checkpointing=False). Override any knob
# here or from the command line.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python rl/train.py \
    --num_gpus "$NUM_GPUS" \
    $MULTI_GPU_FLAG \
    "${@:2}"
