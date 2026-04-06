#!/usr/bin/env bash
# Launch ensemble of 4 independent Stage 2 Latent Dynamics models.
#
# Each model uses:
#   - The SAME frozen Autoencoder from Checkpoint 1
#   - A DIFFERENT dynamics_init_seed (weight initialization)
#   - A DIFFERENT bootstrap_seed (data resampling with replacement)
#   - A DIFFERENT GPU
#
# Usage:
#   # Production (8x L40 cluster):
#   bash scripts/launch_ensemble.sh
#
#   # Dry-run (prints commands without executing):
#   bash scripts/launch_ensemble.sh --dry-run
#
#   # Local test (single GPU, 1 model, 10 steps):
#   bash scripts/launch_ensemble.sh --local-test

set -e

# ============================================================
# Python — set CONDA_ENV to your env name, or override PYTHON
# ============================================================
CONDA_ENV="${CONDA_ENV:-iws}"
if [ -d "${CONDA_PREFIX%/*}/envs/${CONDA_ENV}" ]; then
    PYTHON="${CONDA_PREFIX%/*}/envs/${CONDA_ENV}/bin/python"
elif [ -d "${HOME}/miniconda3/envs/${CONDA_ENV}" ]; then
    PYTHON="${HOME}/miniconda3/envs/${CONDA_ENV}/bin/python"
else
    PYTHON="${PYTHON:-python}"
fi
echo "Using Python: ${PYTHON}"

# ============================================================
# Configuration — edit these for your setup
# ============================================================
CKPT_PATH="outputs/pusht_cam1/checkpoints/best.ckpt"
LATENT_DIR="data/mini/pusht_latent"  # Change to data/full/pusht_latent for production

# Task-specific params (must match the checkpoint's training config)
ACTION_DIM=4
ACTION_MODE="single_ee"
OBS_KEYS="[camera_1_color]"

# Wandb
WANDB_PROJECT="iws-ensemble"
WANDB_ENTITY=""  # Set your W&B entity
WANDB_MODE="online"  # "online", "offline", or "disabled"

# Model IDs and their seed pairs: (dynamics_init_seed, bootstrap_seed)
MODEL_IDS=(2 3 4 5)
DYN_SEEDS=(100 200 300 400)
BOOT_SEEDS=(1000 2000 3000 4000)
GPUS=(0 1 2 3)  # GPU assignments

# Training params
MAX_STEPS=200005
BATCH_SIZE=4
HORIZON=16
# ============================================================

DRY_RUN=false
LOCAL_TEST=false

for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        --local-test) LOCAL_TEST=true ;;
    esac
done

if [ "$LOCAL_TEST" = true ]; then
    MODEL_IDS=(2)
    DYN_SEEDS=(100)
    BOOT_SEEDS=(1000)
    GPUS=(0)
    MAX_STEPS=10
    BATCH_SIZE=2
    WANDB_MODE="disabled"
    echo "=== LOCAL TEST MODE (1 model, ${MAX_STEPS} steps, GPU 0) ==="
fi

echo "=== Ensemble Launch Plan ==="
echo "Checkpoint:  ${CKPT_PATH}"
echo "Latent data: ${LATENT_DIR}"
echo "Action dim:  ${ACTION_DIM}, mode: ${ACTION_MODE}"
echo "Models:      ${MODEL_IDS[*]}"
echo ""

PIDS=()

for i in "${!MODEL_IDS[@]}"; do
    MID=${MODEL_IDS[$i]}
    DYN_SEED=${DYN_SEEDS[$i]}
    BOOT_SEED=${BOOT_SEEDS[$i]}
    GPU=${GPUS[$i]}

    CMD="CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} main.py \
  +name=ensemble_dyn_model_${MID} \
  experiment=exp_latent_dyn \
  dataset=latent_dataset \
  algorithm=latent_world_model \
  algorithm.training_stage=2 \
  algorithm.load_ae=${CKPT_PATH} \
  algorithm.dynamics_init_seed=${DYN_SEED} \
  algorithm.use_prebaked_latent=true \
  algorithm.action_dim=${ACTION_DIM} \
  \"algorithm.obs_keys=${OBS_KEYS}\" \
  dataset.dataset_dir=${LATENT_DIR} \
  dataset.bootstrap_seed=${BOOT_SEED} \
  dataset.action_mode=${ACTION_MODE} \
  dataset.horizon=${HORIZON} \
  dataset.val_horizon=${HORIZON} \
  experiment.training.batch_size=${BATCH_SIZE} \
  experiment.training.max_steps=${MAX_STEPS} \
  wandb.project=${WANDB_PROJECT} \
  wandb.mode=${WANDB_MODE}"

    if [ -n "$WANDB_ENTITY" ]; then
        CMD="${CMD} wandb.entity=${WANDB_ENTITY}"
    else
        CMD="${CMD} wandb.entity=local"
    fi

    echo "--- Model ${MID}: GPU=${GPU}, dyn_seed=${DYN_SEED}, boot_seed=${BOOT_SEED} ---"

    if [ "$DRY_RUN" = true ]; then
        echo "  ${CMD}"
        echo ""
    else
        echo "  Launching on GPU ${GPU}..."
        eval "${CMD}" &
        PIDS+=($!)
        echo "  PID: ${PIDS[-1]}"
        echo ""
    fi
done

if [ "$DRY_RUN" = false ] && [ ${#PIDS[@]} -gt 0 ]; then
    echo "=== All ${#PIDS[@]} models launched. PIDs: ${PIDS[*]} ==="
    echo "Waiting for all to complete..."
    for pid in "${PIDS[@]}"; do
        wait "$pid"
        echo "  PID $pid finished with exit code $?"
    done
    echo "=== All ensemble models completed ==="
fi
