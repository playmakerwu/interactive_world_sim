# Reproduce the Baseline MPPI Episodes (easy / medium / hard)

All commands run from the repo root. Goals are committed under `repro_assets/goals/`.

## 0. Setup

```bash
git clone https://github.com/playmakerwu/interactive_world_sim.git
cd interactive_world_sim

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda env create -f conda_env.yaml
conda activate iws
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126/
pip install -e .
```

Blackwell / RTX 5090 (sm_120) only:

```bash
uv pip install --reinstall torch==2.7.1+cu128 torchvision==0.22.1+cu128 --index-url https://download.pytorch.org/whl/cu128/
uv pip install --reinstall numpy==1.26.4
```

## 1. Preflight

```bash
python -u scripts/run_mppi_v2.py --help
python -c "import torch; assert torch.cuda.is_available(); x=torch.randn(64,64,device='cuda'); (x@x).sum().item()"
```

## 2. Download checkpoint + data

```bash
python scripts/download_checkpoints_hf.py --repo yixuan1999/interactive-world-sim-checkpoints --subdir pusht_cam1
bash scripts/download_mini_data.sh
```

## 3. Run

```bash
# easy
TS=$(date +%Y%m%d_%H%M%S); OUTDIR=outputs/mppi/easy_${TS}; mkdir -p "$OUTDIR"
python -u scripts/run_mppi_v2.py \
    --initial_hdf5 data/mini/pusht/train/episode_4.hdf5 --initial_frame 32 \
    --goal repro_assets/goals/easy_goal.pt --output_dir "$OUTDIR" --cv_n_workers 4 \
    2>&1 | tee "$OUTDIR/run.log"

# medium
TS=$(date +%Y%m%d_%H%M%S); OUTDIR=outputs/mppi/medium_${TS}; mkdir -p "$OUTDIR"
python -u scripts/run_mppi_v2.py \
    --initial_hdf5 data/mini/pusht/val/episode_3.hdf5 --initial_frame 71 \
    --goal repro_assets/goals/medium_goal.pt --output_dir "$OUTDIR" --cv_n_workers 4 \
    2>&1 | tee "$OUTDIR/run.log"

# hard
TS=$(date +%Y%m%d_%H%M%S); OUTDIR=outputs/mppi/hard_${TS}; mkdir -p "$OUTDIR"
python -u scripts/run_mppi_v2.py \
    --initial_hdf5 data/mini/pusht/val/episode_1.hdf5 --initial_frame 59 \
    --goal repro_assets/goals/hard_goal.pt --output_dir "$OUTDIR" --cv_n_workers 4 \
    2>&1 | tee "$OUTDIR/run.log"
```

## 4. Inspect

```bash
python -m json.tool <OUTDIR>/summary.json | head -40
```

## Or run all at once

```bash
bash scripts/reproduce_mppi_baselines.sh [easy|medium|hard|all]
```
