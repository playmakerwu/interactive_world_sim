# Cloud Training Setup

End-to-end instructions for running the Dreamer-style RL pipeline on a cloud
GPU box (single 80 GB A100 recommended; multi-GPU optional).

## 1. Clone and checkout the RL branch

```bash
git clone git@github.com:playmakerwu/interactive_world_sim.git
cd interactive_world_sim
git checkout dreamer-rl
git submodule update --init --recursive
```

## 2. Environment setup

```bash
mamba env create -f conda_env.yaml
conda activate iws
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126/
pip install -e .

# tensorboard for training curves
pip install tensorboard
```

## 3. Download data and checkpoints

```bash
bash scripts/download_checkpoints.sh
bash scripts/download_mini_data.sh
```

## 4. Generate goal latent (one-time)

```bash
python tests/goal_selection/extract_final_frames.py
python tests/goal_selection/encode_goal.py \
    --image_path tests/goal_selection/frames/episode_0_final.png
```

## 5. Verify GPU setup

```bash
python -c "import torch; n=torch.cuda.device_count(); print(f'GPUs: {n}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory/1e9:.1f} GB)') for i in range(n)]"
```

## 6. Run training

Defaults in `rl/utils/config.py` are set for a single 80 GB A100:
`H=15`, `B=64`, `total_steps=200000`, gradient checkpointing OFF.

```bash
# Option A — launch script (recommended)
bash rl/train_cloud.sh 1              # single GPU
bash rl/train_cloud.sh 4              # four GPUs (wraps actor/critic in DataParallel)

# Option B — direct (override any DreamerConfig field as --<field>)
python rl/train.py \
    --total_steps 200000 \
    --batch_size 64 \
    --imagination_horizon 15

# Option C — small-VRAM fallback (e.g. 16 GB card)
python rl/train.py \
    --batch_size 8 \
    --imagination_horizon 10 \
    --use_gradient_checkpointing true
```

## 7. Monitor training

```bash
tensorboard --logdir rl/outputs/logs/ --port 6006
```

## 8. Evaluate after training

```bash
python rl/evaluate.py \
    --checkpoint rl/outputs/checkpoints/final.pt \
    --horizon 100 --episodes 10
```

Writes videos + `rl/outputs/eval_metrics.json`.

## Troubleshooting

- **OOM at B=64, H=15**: drop `--batch_size 32` or flip `--use_gradient_checkpointing true`.
- **`No available kernel` during attention**: your GPU's compute capability hit the flash-attention path that requires fp16/bf16. Already patched for non-A100 GPUs via `rl/models/world_model.py::_patch_attention_backends`; if it still fires, run with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Multi-GPU slower than single**: actor/critic are tiny MLPs, so DataParallel helps very little. Stick with `--num_gpus 1` on an 80 GB A100.
- **Lightning checkpoint load fails**: confirm `outputs/pusht_cam1/checkpoints/best.ckpt` and `outputs/pusht_cam1/.hydra/config.yaml` both exist.
