# Cloud Execution Tutorial — Branch A on Full Data

Execute the full state-probe pipeline on a cloud GPU instance, end-to-end,
from data inventory through PR. You are the operator; this document is
your checklist. Pause at every **CHECKPOINT** marker and verify the
stated outputs before proceeding. If a **STOP — contact me** appears,
do not improvise — surface the state and wait.

This tutorial fixes the choices already locked in earlier:

| Choice | Value | Source |
|---|---|---|
| CV preset | `REAL` | Phase 3-A Checkpoint 3 |
| Labeling path | Option A (decode-then-label) | Design doc §1.4 |
| Split kind | random 80/20 **by episode**, seed-selected | Design doc §3.2 + Phase 3-A retrospective |
| Probe arch | MLP `[4096 → 256 → 128 → 4]`, ~1.08M params | Design doc §1.7 revised |
| Loss | `λ_pos = 7`, `λ_ang = 1`, `λ_norm = 0.01` | Design doc §1.7 revised |
| Acceptance | pooled `val_pos_p95 ≤ 3 px` AND pooled `val_ang_p95 ≤ 5°` AND visual sign-off | Design doc §1.9 revised |

**DO NOT** re-decide any of these during the run. If one of them becomes
unviable, stop and surface it.

**Known unfixed issue**: the CV estimator is bimodal between `θ` and
`θ + 180°` on some frames (Phase 1 observation). Angle p95 may be
somewhat worse than position p95 as a result. If angle p95 misses
acceptance while position passes, **do not try to fix**: surface and
stop. The label-noise fix is a separate future task.

Total expected wall time (A100 class, ~100 episodes, ~20 k frames):
~60–90 minutes of user-facing work, plus ~20–40 minutes of instance
compute.

---

## Assumed starting state

Pre-conditions. Verify each before Stage 0.

1. You have SSH access to a cloud GPU instance. Provider, region, disk
   size, spot vs on-demand — all your call.
2. Instance has **one GPU of A100 or H100 class** (40 GB+ VRAM). Lower
   specs will work but the batch-size recommendations in Stage 7 need
   manual adjustment.
3. Instance has **`nvidia-smi`** and a working CUDA driver.
4. Instance has **`git`, `conda` (miniconda/miniforge), `tmux` or
   `screen`**. If `conda` isn't present, install miniforge:
   `curl -L -o mf.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && bash mf.sh -b && ~/miniforge3/bin/conda init bash && exec bash`.
5. The full dataset is on the instance at `~/data/full/`. Structure will
   be verified in Stage 1; tutorial tolerates a few common layouts.
6. The repo is cloned to the instance and on branch
   **`state-probe-training`**.
7. The WM checkpoint is available: either in the repo at
   `outputs/pusht_cam1/checkpoints/best.ckpt` (note: this file is
   gitignored, 366 MiB, you must transfer it separately — see Stage 0.4)
   or at some path you will symlink to the expected location.

Branch hygiene: do all cloud-only edits on a new branch branched off
`state-probe-training`. The first stage covers this.

---

## Stage 0 — Instance setup and environment verification

**Purpose**: confirm the instance is ready to execute the pipeline. No
edits, no training, just verification. If any check fails, fix that
first before moving on.

**Preconditions**: the seven items above.

### 0.1 GPU availability

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv
```

**What to check**: you see one GPU listed, `memory.total >= 40000 MiB`,
and `memory.free >= 35000 MiB` (i.e., no co-tenant). If you see < 20 GB
free, something else is using the card — investigate (`nvidia-smi
--query-compute-apps=pid,process_name,used_memory --format=csv`) before
continuing.

### 0.2 Create a working branch

```bash
cd ~/interactive_world_sim  # or wherever you cloned
git fetch origin
git checkout state-probe-training
git pull origin state-probe-training
git checkout -b state-probe-training-cloud
```

All cloud-specific edits go on `state-probe-training-cloud`. When the
run is done you'll push this branch and open a PR back to `dreamer-rl`.

### 0.3 Python environment

Check if the `iws` env exists:

```bash
conda env list | grep -E "^iws\b"
```

If present, activate:

```bash
conda activate iws
```

If not present, create from the repo's env spec (check for
`environment.yml` / `setup.sh` / `pyproject.toml`):

```bash
ls environment.yml setup.sh pyproject.toml 2>/dev/null
```

Follow whichever is there. If none, bootstrap manually (minimal —
enough to run labeling, training, and tests):

```bash
conda create -n iws python=3.11 -y
conda activate iws
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python scipy h5py numpy shapely scikit-learn matplotlib tqdm \
            einops lightning hydra-core omegaconf torchmetrics tensorboard pytest
# plus whatever the WM loader needs (`interactive_world_sim` package);
# install the repo in editable mode:
pip install -e .
```

Verify:

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import cv2, scipy, shapely, h5py, numpy; print('cv2', cv2.__version__)"
```

Both lines must succeed.

### 0.4 WM checkpoint

```bash
ls -lh outputs/pusht_cam1/checkpoints/best.ckpt 2>/dev/null || echo "MISSING"
ls -lh tests/goal_selection/z_goal.pt 2>/dev/null || echo "MISSING"
```

`best.ckpt` is gitignored (366 MiB). If missing, transfer it from your
local machine or a bucket. `z_goal.pt` is committed (18 KiB) so should
already be present.

### 0.5 Vendored CV pipeline

Verify the CV labeler imports cleanly **without** needing the
supervisor's repo on disk:

```bash
python -c "
from rl.labeling.cv_labeler import CVLabeler, HSV_PRESETS, normalize_angle_deg
print('presets:', list(HSV_PRESETS.keys()))
print('cv_labeler OK')
"
```

The CV code is vendored into `rl/labeling/cv_labeler.py` (see
`migration_log.md`). The supervisor's repo at `~/Documents/aloha` is
**not** a runtime dependency. Only `tests/state_estimator/sanity_check.py`
and an older Phase 1 code path reference it; you don't need to run
either on cloud.

### 0.6 Smoke test

```bash
pytest tests/state_estimator/test_state_viz.py tests/state_estimator/test_labeling.py tests/state_estimator/test_probe.py -v
```

All ~38 tests must pass. If any fail, stop — your env is broken.

### CHECKPOINT 0 (Go / No-go)

Go if all six sub-steps passed. Free VRAM ≥ 35 GiB, all tests green,
CV labeler importable, WM ckpt and z_goal.pt present, on branch
`state-probe-training-cloud`. Otherwise fix first.

**Expected wall time**: 5–20 minutes, depending on whether `iws` had to
be installed from scratch.

---

## Stage 1 — Data inventory on `data/full`

**Purpose**: discover what the full dataset looks like. The existing
pipeline was built against `data/mini/pusht/{train,val}/`; we need to
know whether the full dataset matches or diverges, because the scripts
need to be adapted accordingly in Stage 2.

**Preconditions**: Stage 0 complete.

### 1.1 Directory structure

```bash
ls ~/data/full/ | head
find ~/data/full -maxdepth 2 -type d | head -20
find ~/data/full -maxdepth 3 -name "episode_*.hdf5" | wc -l
```

You are looking for whether episodes live in `~/data/full/` directly or
under nested subdirs like `~/data/full/train/`, `~/data/full/pusht/`,
etc. Note the actual layout — you'll need it.

### 1.2 Schema inspection

Create a throwaway script and run it:

```bash
cat > /tmp/inspect_full.py <<'PY'
import glob, sys
import h5py, numpy as np
from pathlib import Path

ROOT = Path.home() / "data" / "full"
candidates = sorted(Path(p) for p in glob.glob(str(ROOT / "**/episode_*.hdf5"),
                                               recursive=True))
if not candidates:
    print("FAIL: no episode_*.hdf5 found under", ROOT)
    sys.exit(1)

print(f"found {len(candidates)} episode files")
# inventory: total frames, schema from episode 0
total_frames = 0
frame_counts = []
shapes_ok = True
keys_ok = True

with h5py.File(candidates[0], "r") as f:
    def walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  {name} {obj.shape} {obj.dtype}")
    f.visititems(walk)

    required = ["action", "obs/images/camera_1_color"]
    for k in required:
        if k not in f:
            print(f"FAIL: missing required key {k}")
            keys_ok = False
    if "obs/images/camera_1_color" in f:
        arr = f["obs/images/camera_1_color"]
        if arr.ndim != 4 or arr.shape[-1] != 3 or arr.dtype != np.uint8:
            print(f"FAIL: camera_1_color wrong shape/dtype: {arr.shape} {arr.dtype}")
            shapes_ok = False
        h, w = arr.shape[1], arr.shape[2]
        # Value-range sanity: one middle frame
        sample = arr[min(50, arr.shape[0] - 1)]
        if sample.min() < 0 or sample.max() > 255:
            print("FAIL: pixel values out of [0, 255]")
            shapes_ok = False
        # Channel-order sanity: pink T-block has elevated red + magenta.
        # If image is BGR, red will be in channel 2 not 0.
        # heuristic — not authoritative, just a flag
        print(f"  sample frame shape={sample.shape} dtype={sample.dtype} min={sample.min()} max={sample.max()}")
        r, g, b = sample.mean(axis=(0, 1))
        print(f"  mean channels (assumed RGB): ({r:.1f}, {g:.1f}, {b:.1f})")

# Count frames per episode
for p in candidates:
    with h5py.File(p, "r") as f:
        n = f["action"].shape[0] if "action" in f else 0
    frame_counts.append(n)
    total_frames += n

print()
print(f"total episodes: {len(candidates)}")
print(f"total frames:   {total_frames}")
print(f"per-episode frame count: min {min(frame_counts)}  "
      f"mean {np.mean(frame_counts):.1f}  max {max(frame_counts)}")
print()
print("PASS" if (keys_ok and shapes_ok) else "FAIL")
PY
python /tmp/inspect_full.py
```

**What to check**:
- `total episodes` ≥ ~20 (anything less and the by-episode split in
  Stage 6 is likely to hit the mini-dataset failure mode again — stop
  and contact me).
- `total frames` ≥ ~4000 (similar reason).
- `obs/images/camera_1_color` shape `(N, H, W, 3)` with `dtype=uint8`.
- The final line reads `PASS`.
- Mean RGB channel order looks plausible (pink scene has elevated red
  and some blue; mean should roughly be R > G, R ≈ B on a scene with a
  pink T-block against a white/grey backdrop — close enough).

### 1.3 Compare to `data/mini/pusht/` expectations

The existing scripts expect, per `data/mini/pusht/train/episode_0.hdf5`:

```
action                         (200, 4) float32
obs/images/camera_1_color      (200, 480, 640, 3) uint8
obs/ee_pos                     (200, 14) float32
timestamp                      (200,) float64
```

Minimum required for the pipeline: `action` and
`obs/images/camera_1_color`. Other keys are ignored by this pipeline
(they're used by the WM training code, not by the probe pipeline).

**Failure recovery**: *"If `data/full` schema differs from mini"* —
`obs/images/camera_1_color` must exist with uint8 RGB shape
`(T, H, W, 3)`, `action` must exist. If either is missing, or the image
key has a different name (e.g., `obs/images/top_color`), stop and
contact me — patching the pipeline for a new key name is doable but
requires editing `scripts/label_replay_buffer.py` (search/replace
`camera_1_color` globally).

### CHECKPOINT 1 (Go / No-go, user review)

Go if inspection reports `PASS`, episodes ≥ 20, frames ≥ 4000. Post the
output of `inspect_full.py` if there is anything odd (weird channel
means, unexpected keys, fewer episodes than mini). Otherwise proceed.

**Expected wall time**: 2–5 minutes.

---

## Stage 2 — Pipeline parameterization

**Purpose**: remove `data/mini/pusht` hardcodes from the two scripts
that read raw episodes (`calibrate_hsv.py`, `label_replay_buffer.py`)
and verify the edits by labelling a tiny subset.

**Preconditions**: Stage 1 confirmed the full-data format matches mini.

### 2.1 Find the hardcodes

```bash
grep -n "data/mini/pusht\|DATA_ROOT\|TRAIN_DIR" scripts/*.py rl/labeling/*.py
```

Expected hits:

- `scripts/calibrate_hsv.py:63: TRAIN_DIR = REPO_ROOT / "data" / "mini" / "pusht" / "train"`
- `scripts/label_replay_buffer.py:48: DATA_ROOT = REPO_ROOT / "data" / "mini" / "pusht"`
- `scripts/label_replay_buffer.py:185: split_dir = DATA_ROOT / split`  *(this expects `train/` and `val/` subdirs — needs to be changed because full data is flat)*

The training scripts (`train_state_probe.py`,
`train_state_probe_diagnostic.py`) only reference
`outputs/state_probe/labels/labels_*.pt` — **no raw-data hardcode**, so
no patch needed there.

### 2.2 Patch `scripts/calibrate_hsv.py`

Replace the block around line 58–63:

```python
# BEFORE (lines ~58-63)
CKPT_PATH = REPO_ROOT / "outputs" / "pusht_cam1" / "checkpoints" / "best.ckpt"
TRAIN_DIR = REPO_ROOT / "data" / "mini" / "pusht" / "train"
OUT_DIR = REPO_ROOT / "tests" / "state_estimator" / "calibration_outputs"

# AFTER
import argparse as _argparse
_cli = _argparse.ArgumentParser()
_cli.add_argument("--data_dir", type=str,
                  default="data/mini/pusht/train",
                  help="directory containing episode_*.hdf5 files")
_cli.add_argument("--ckpt", type=str,
                  default="outputs/pusht_cam1/checkpoints/best.ckpt")
_cli.add_argument("--out_dir", type=str,
                  default="tests/state_estimator/calibration_outputs")
_args, _ = _cli.parse_known_args()
CKPT_PATH = REPO_ROOT / _args.ckpt
TRAIN_DIR = REPO_ROOT / _args.data_dir
OUT_DIR = REPO_ROOT / _args.out_dir
```

Keep the rest of the file as is. `parse_known_args` avoids conflict
with `main()` which doesn't use argparse directly.

### 2.3 Patch `scripts/label_replay_buffer.py`

Two changes. First the top of the file (around lines 45–52):

```python
# BEFORE
CKPT_PATH = REPO_ROOT / "outputs" / "pusht_cam1" / "checkpoints" / "best.ckpt"
DATA_ROOT = REPO_ROOT / "data" / "mini" / "pusht"
OUT_DIR = REPO_ROOT / "outputs" / "state_probe" / "labels"

# AFTER
import argparse as _argparse
_cli = _argparse.ArgumentParser()
_cli.add_argument("--data_root", type=str, default="data/mini/pusht")
_cli.add_argument("--ckpt", type=str,
                  default="outputs/pusht_cam1/checkpoints/best.ckpt")
_cli.add_argument("--out_dir", type=str,
                  default="outputs/state_probe/labels")
_cli.add_argument("--splits", type=str, default="train,val",
                  help="comma-separated subdir names under --data_root, "
                       "or 'flat' if episodes live directly under data_root")
_cli.add_argument("--decode_batch", type=int, default=8)
_cli.add_argument("--encode_batch", type=int, default=32)
_args, _ = _cli.parse_known_args()
CKPT_PATH = REPO_ROOT / _args.ckpt
DATA_ROOT = REPO_ROOT / _args.data_root
OUT_DIR = REPO_ROOT / _args.out_dir
SPLITS = _args.splits.split(",") if _args.splits != "flat" else ["flat"]
DECODE_BATCH = _args.decode_batch
ENCODE_BATCH = _args.encode_batch
```

Second, replace the original `ENCODE_BATCH = 32` / `DECODE_BATCH = 8`
constants (now shadowed by the CLI-provided values — delete those lines
or leave them, they'll be overwritten).

Third, fix the split-dir handling. Find the `label_split()` function
(around line 180) and modify its first few lines:

```python
# BEFORE
def label_split(
    split: str,
    wm: DifferentiableDynamics,
    labeler: CVLabeler,
    ckpt_mtime: float,
) -> dict:
    split_dir = DATA_ROOT / split
    episodes = sorted(split_dir.glob("episode_*.hdf5"))
    ...

# AFTER
def label_split(
    split: str,
    wm: DifferentiableDynamics,
    labeler: CVLabeler,
    ckpt_mtime: float,
) -> dict:
    split_dir = DATA_ROOT if split == "flat" else DATA_ROOT / split
    episodes = sorted(split_dir.glob("episode_*.hdf5"))
    ...
```

And at the bottom of `main()`, replace the hardcoded `for split in
("train", "val"):` loop with `for split in SPLITS:`.

### 2.4 Commit the patches

```bash
git add scripts/calibrate_hsv.py scripts/label_replay_buffer.py
git commit -m "Cloud: parameterize data paths + flat-layout support"
```

### 2.5 Smoke test with 2 episodes

```bash
mkdir -p /tmp/data_smoke
# Copy 2 episodes to a scratch dir for the smoke test
# If your data is flat:
cp ~/data/full/episode_0.hdf5 ~/data/full/episode_1.hdf5 /tmp/data_smoke/
# If your data is nested:
# cp ~/data/full/<subdir>/episode_0.hdf5 ~/data/full/<subdir>/episode_1.hdf5 /tmp/data_smoke/

python scripts/label_replay_buffer.py \
    --data_root /tmp/data_smoke \
    --splits flat \
    --out_dir /tmp/smoke_labels
```

**What to check**:
- No crash, no KeyError on HDF5 keys.
- Per-episode drop rate printed for each of the 2 episodes.
- Output files written to `/tmp/smoke_labels/labels_flat.pt`,
  `/tmp/smoke_labels/meta.json`, `/tmp/smoke_labels/drops.csv`.
- Overall drop rate ideally 0%, tolerably ≤ 10%.

### CHECKPOINT 2 (Go / No-go)

Go if the 2-episode smoke finishes cleanly with low drop rate. Otherwise
inspect the drops: if it's a schema mismatch (exceptions on HDF5 keys),
stop and contact me. If it's high drop rate, skip to Stage 3 which is
exactly what checks that on a larger sample.

**Expected wall time**: 15–30 minutes for patching + commit + smoke test.

---

## Stage 3 — HSV calibration sanity-check on full data

**Purpose**: the `REAL` HSV preset was validated on mini. Same WM
checkpoint produces the same decoder output distribution, so `REAL`
should still be fine — but verify empirically on 20 full-data frames
before bulk-labeling thousands.

**Preconditions**: Stages 0–2 complete. `data/full` accessible; patched
scripts committed on `state-probe-training-cloud`.

### 3.1 Run calibration

If your data is flat at `~/data/full`:

```bash
python scripts/calibrate_hsv.py --data_dir ~/data/full
```

If data is under a subdir like `~/data/full/train/`, point at that.
(The calibration script uses the first 5 episodes, stratified-sampled.
If the full dataset has way more than 5 episodes, 5 is still enough to
sanity-check.)

### 3.2 What to check

The script produces
`tests/state_estimator/calibration_outputs/`:

- `calibration_summary.md` — read the "three mandatory numbers" table
- `calibration_grid_REAL.png` — 20-tile overlay, eyeball
- `per_channel_relaxation.csv`, `agreement_table.csv` — detail

**Go criteria**:

| metric | threshold | mini baseline (for comparison) |
|---|---|---|
| REAL drop rate | < 10% | 0% |
| REAL median post-morph area | ≥ 90% of theoretical 471.8 px = 424.6 px | 512.5 px (108.9%) |
| REAL median post-morph area within ±5 pp of mini's 108.9% (so ≥ 103.9% or ≤ 113.9%) | preferred | — |
| REAL-vs-WM agreement | ≥ 95% identical | 100% identical |
| False positives in `calibration_grid_REAL.png` | none visible on gripper tips, shadows, bundles | none |

### 3.3 Failure recovery

*"If REAL drop rate is > 10% on full data"*: stop and contact me. This
means the full-data decoder distribution differs from mini's —
unexpected since the WM checkpoint is the same, but possible if the
full dataset includes very different lighting/backgrounds. Do not
retrofit thresholds on your own.

*"If the grid shows false positives on gripper tips"*: same — stop and
contact. The REAL preset's H ∈ [160, 179] excluded gripper-orange on
mini; if that fails on full data, the decoder is producing subtly
different output distributions.

### CHECKPOINT 3 (Go / No-go, user review)

Go if REAL meets all four thresholds above. Post the
`calibration_summary.md` contents (short) and `calibration_grid_REAL.png`
URL/attachment if anything looks off. Otherwise proceed.

**Expected wall time**: 3–5 minutes for the calibration run; 2–5 minutes
for review.

---

## Stage 4 — Bulk labeling on full data

**Purpose**: label every frame in `data/full` into `(latent, label)`
pairs using Option A (decode-then-label). This is the labeled dataset
the probe trains on.

**Preconditions**: Stage 3 confirmed `REAL` transfers. Stages 0–3
complete.

### 4.1 Run labeling

Under `tmux` or `screen` so a dropped SSH session doesn't abort the
job:

```bash
tmux new -s label
# inside tmux:
conda activate iws
cd ~/interactive_world_sim

# Flat layout:
python scripts/label_replay_buffer.py \
    --data_root ~/data/full \
    --splits flat \
    --out_dir outputs/state_probe/labels_full \
    --decode_batch 32 \
    --encode_batch 64 \
    2>&1 | tee outputs/state_probe/labels_full.log

# Detach: Ctrl-b d
# Reattach: tmux attach -t label
```

If the dataset has `train/` and `val/` subdirs, use `--splits train,val`
and you'll get two output `.pt` files (labels_train.pt,
labels_val.pt). The Stage 6 split-selection code handles either case.

**Batch sizes**: the defaults are dev-box conservative (decode 8,
encode 32). On A100/H100 you can push decode to 32–64 and encode to
64–128, cutting wall time roughly 3–4× without OOM risk. If you see
OOM, halve and restart.

### 4.2 Expected wall time

Rough extrapolation from the local run (2 k frames, 3.2 min on an
RTX 5070 Ti):

| frames | dev-box (RTX 5070 Ti, batch 8/32) | A100 (batch 32/64) | H100 (batch 64/128) |
|---|---|---|---|
| 2 k   | ~3 min | ~1 min | ~30 s |
| 10 k  | ~15 min | ~4 min | ~2 min |
| 50 k  | ~75 min | ~20 min | ~10 min |

If your extrapolated wall time exceeds 2 hours, pause and reconsider —
either the batch size is wrong for your GPU, or the dataset is
unexpectedly large.

### 4.3 Progress monitoring

```bash
# In a second shell, while labeling runs:
tail -f outputs/state_probe/labels_full.log
```

You should see `ep{id}: kept N/N (drop K, X%)` lines, one per episode.

### 4.4 Verify output

```bash
ls -lh outputs/state_probe/labels_full/
python -c "
import torch, json
d = torch.load('outputs/state_probe/labels_full/labels_flat.pt', weights_only=False)
print('latents:', tuple(d['latents'].shape), d['latents'].dtype)
print('labels:',  tuple(d['labels'].shape),  d['labels'].dtype)
print('episodes:', sorted(d['episode'].unique().tolist())[:20], '...')
print('cx range:', float(d['labels'][:,0].min()), '-', float(d['labels'][:,0].max()))
print('cy range:', float(d['labels'][:,1].min()), '-', float(d['labels'][:,1].max()))
norm = (d['labels'][:,2]**2 + d['labels'][:,3]**2).sqrt()
print(f'unit-norm (sin, cos): mean {norm.mean():.4f}  min {norm.min():.4f}  max {norm.max():.4f}')
with open('outputs/state_probe/labels_full/meta.json') as f:
    m = json.load(f)
print('overall drop rate:', m['splits']['flat']['drop_rate'] if 'flat' in m['splits'] else '(see meta.json)')
"
```

**What to check**:
- Latents `(N, 4, 32, 32) float32`, labels `(N, 4) float32`.
- `cx`, `cy` ranges plausible (mostly within `[0, 128]`; frames where
  the T-block partially leaves the frame can extrapolate slightly).
- Unit-norm of `(sin, cos)` = 1.0000 on every frame.
- Overall drop rate < 10%.

### 4.5 Failure recovery

*"If bulk labeling OOMs"*: halve `--decode_batch` and restart. If it
still OOMs at decode_batch=4, halve again. Labeling is idempotent — you
can delete the partial output and re-run.

*"If drop rate > 10% overall or any episode > 30%"*: stop and contact
me. Do not tune thresholds. The calibration at Stage 3 should have
caught this; if it didn't, we have a distribution-shift issue to
diagnose together.

*"If the instance disconnects mid-labeling"*: re-attach tmux
(`tmux attach -t label`) to see where it got. The script processes
episodes in order and writes output only at the end per split, so a
mid-run disconnect loses work — start over.

### CHECKPOINT 4 (Go / No-go, user review)

Go if overall drop < 10% and no single episode > 30% drop. Post the
`meta.json` stats. Otherwise stop as above.

**Expected wall time**: 10 minutes to ~2 hours depending on dataset
size and GPU (see 4.2).

---

## Stage 5 — Goal state extraction

**Purpose**: produce `tests/goal_selection/state_goal.pt` from `z_goal`
so Branch B can load a concrete `(cx, cy, sin θ, cos θ)` goal tensor.

**Preconditions**: Stage 4 done. `z_goal.pt` and WM ckpt present.

### 5.1 Run

```bash
python scripts/compute_state_goal.py
```

### 5.2 What to check

The script prints the extracted goal state and writes two files:

- `tests/goal_selection/state_goal.pt` — payload tensor + metadata
- `tests/goal_selection/state_goal_overlay.png` — 4× upscaled visual

Confirm:

- `cv_success: True`
- `contour area` > 300 px
- `icp_residual` < 0.5
- Overlay PNG: green marker is on the pink T-block, arrow direction is
  sensible

### 5.3 Failure recovery

*"If CV fails on z_goal"*: the script raises `RuntimeError`. This is a
hard blocker. The goal frame must label; if it doesn't, `z_goal.pt` is
incompatible with the current WM checkpoint. Stop and contact me.

### CHECKPOINT 5

No user review — auto-advance to Stage 6 once the file is written.

**Expected wall time**: 1–2 minutes.

---

## Stage 6 — Train/val split decision

**Purpose**: choose a by-episode 80/20 split whose train and val label
distributions substantially overlap. On mini, the existing `train/` vs
`val/` split was disjoint in `cx` and `θ`; the probe couldn't
generalise. On full data with more episodes, a random split should
work — but we need to verify rather than trust.

**Preconditions**: Stage 4 labels exist at
`outputs/state_probe/labels_full/labels_{flat,train,val}.pt`.

### 6.1 Split-selection script

Create `scripts/select_split.py`:

```bash
cat > scripts/select_split.py <<'PY'
"""Pick an 80/20 by-episode split whose train/val label distributions
overlap the full set. Tries N seeds, scores by coverage, saves the best.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]

def load_all_labels(labels_dir: Path):
    """Concatenate labels_flat / labels_train+labels_val into one pool.
    Return (labels_px, episodes, t_idx, latents).  Episodes get made
    unique by offsetting the second file's ids by max+1."""
    files = sorted(labels_dir.glob("labels_*.pt"))
    assert files, f"no labels_*.pt in {labels_dir}"
    latents, labels, eps, ts = [], [], [], []
    offset = 0
    for f in files:
        d = torch.load(f, weights_only=False)
        e = d["episode"].clone() + offset
        offset = int(e.max()) + 1
        latents.append(d["latents"]); labels.append(d["labels"])
        eps.append(e); ts.append(d["t_idx"])
    return (torch.cat(latents, 0), torch.cat(labels, 0),
            torch.cat(eps, 0), torch.cat(ts, 0))

def score_split(labels_full, mask_train, mask_val, nbins=18):
    """Lower is better. Combines cx, cy, theta histogram KL-like
    distance."""
    def hist(vals, lo, hi):
        h, _ = np.histogram(vals, bins=nbins, range=(lo, hi))
        h = h.astype(np.float64) + 1e-6
        return h / h.sum()
    tr = labels_full[mask_train]; va = labels_full[mask_val]
    theta_full = np.degrees(np.arctan2(labels_full[:,2].numpy(),
                                        labels_full[:,3].numpy()))
    theta_tr = theta_full[mask_train.numpy()]
    theta_va = theta_full[mask_val.numpy()]

    score = 0.0
    for vals_tr, vals_va, lo, hi in (
        (tr[:,0].numpy(), va[:,0].numpy(), 0, 128),
        (tr[:,1].numpy(), va[:,1].numpy(), 0, 128),
        (theta_tr,         theta_va,        -180, 180),
    ):
        p, q = hist(vals_tr, lo, hi), hist(vals_va, lo, hi)
        # symmetric KL; low = similar distributions
        score += float(0.5 * ((p * np.log(p/q)).sum() + (q * np.log(q/p)).sum()))
    return score

def coverage(labels_full, mask):
    """Fraction of the full theta range and cx range covered by mask."""
    theta_full = np.degrees(np.arctan2(labels_full[:,2].numpy(),
                                        labels_full[:,3].numpy()))
    t_rng_full = theta_full.max() - theta_full.min()
    cx_rng_full = float(labels_full[:,0].max() - labels_full[:,0].min())
    sub = labels_full[mask]
    t_sub = theta_full[mask.numpy()]
    t_rng_sub = t_sub.max() - t_sub.min()
    cx_rng_sub = float(sub[:,0].max() - sub[:,0].min())
    return (t_rng_sub / max(t_rng_full, 1e-6),
            cx_rng_sub / max(cx_rng_full, 1e-6))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_dir", type=str,
                    default="outputs/state_probe/labels_full")
    ap.add_argument("--out", type=str,
                    default="outputs/state_probe/split_spec.json")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--n_seeds", type=int, default=20)
    ap.add_argument("--min_coverage", type=float, default=0.80)
    args = ap.parse_args()

    latents, labels, episodes, ts = load_all_labels(
        Path(args.labels_dir))
    uniq_eps = sorted(torch.unique(episodes).tolist())
    n_val = max(1, int(round(args.val_frac * len(uniq_eps))))
    n_train = len(uniq_eps) - n_val
    print(f"{len(uniq_eps)} episodes -> train {n_train}, val {n_val}")

    candidates = []
    for seed in range(args.n_seeds):
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(uniq_eps)
        val_eps = set(int(x) for x in shuffled[:n_val].tolist())
        train_eps = set(int(x) for x in shuffled[n_val:].tolist())
        mask_val = torch.tensor([int(e) in val_eps for e in episodes.tolist()])
        mask_train = ~mask_val
        tr_t_cov, tr_cx_cov = coverage(labels, mask_train)
        va_t_cov, va_cx_cov = coverage(labels, mask_val)
        s = score_split(labels, mask_train, mask_val)
        passes = (min(tr_t_cov, va_t_cov) >= args.min_coverage
                  and min(tr_cx_cov, va_cx_cov) >= args.min_coverage)
        candidates.append({
            "seed": seed,
            "train_eps": sorted(train_eps),
            "val_eps": sorted(val_eps),
            "n_train_frames": int(mask_train.sum()),
            "n_val_frames": int(mask_val.sum()),
            "train_theta_coverage": tr_t_cov,
            "val_theta_coverage": va_t_cov,
            "train_cx_coverage": tr_cx_cov,
            "val_cx_coverage": va_cx_cov,
            "kl_score": s,
            "passes": passes,
        })

    passing = [c for c in candidates if c["passes"]]
    print(f"{len(passing)} / {args.n_seeds} seeds pass "
          f"the {args.min_coverage:.0%} coverage floor")
    if not passing:
        print("FAIL: no seed passes the coverage floor. Stop here.")
        with open(args.out, "w") as f:
            json.dump({"status": "FAIL", "candidates": candidates}, f, indent=2)
        sys.exit(2)

    best = min(passing, key=lambda c: c["kl_score"])
    print(f"selected seed {best['seed']}: "
          f"kl={best['kl_score']:.4f}  "
          f"train_theta_cov={best['train_theta_coverage']:.2%}  "
          f"val_theta_cov={best['val_theta_coverage']:.2%}")

    # save spec
    with open(args.out, "w") as f:
        json.dump({
            "status": "PASS",
            "labels_dir": args.labels_dir,
            "val_frac": args.val_frac,
            "min_coverage": args.min_coverage,
            "selected": best,
            "all_candidates": candidates,
        }, f, indent=2)
    print(f"wrote {args.out}")

    # produce overlap plots for the selected seed
    val_eps = set(best["val_eps"])
    mask_val = torch.tensor([int(e) in val_eps for e in episodes.tolist()])
    mask_train = ~mask_val
    tr, va = labels[mask_train], labels[mask_val]
    theta_tr = np.degrees(np.arctan2(tr[:,2].numpy(), tr[:,3].numpy()))
    theta_va = np.degrees(np.arctan2(va[:,2].numpy(), va[:,3].numpy()))

    out_prefix = Path(args.out).parent / "split_distribution"
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(tr[:,0].numpy(), tr[:,1].numpy(), s=6, alpha=0.4,
               c="#2874A6", label=f"train ({len(tr)})")
    ax.scatter(va[:,0].numpy(), va[:,1].numpy(), s=6, alpha=0.4,
               c="#E67E22", label=f"val ({len(va)})")
    ax.set_xlim(0, 128); ax.set_ylim(128, 0)
    ax.set_xlabel("cx (px)"); ax.set_ylabel("cy (px)")
    ax.set_title(f"Split seed {best['seed']} — position overlap")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect("equal")
    fig.savefig(f"{out_prefix}_position.png", dpi=120, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(-180, 180, 37)
    ax.hist(theta_tr, bins=bins, alpha=0.5, color="#2874A6",
            label=f"train ({len(theta_tr)})")
    ax.hist(theta_va, bins=bins, alpha=0.5, color="#E67E22",
            label=f"val ({len(theta_va)})")
    ax.set_xlabel("theta (deg)"); ax.set_ylabel("count")
    ax.set_title(f"Split seed {best['seed']} — angle overlap")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(f"{out_prefix}_theta.png", dpi=120, bbox_inches="tight")
    plt.close()

    print(f"plots: {out_prefix}_position.png, {out_prefix}_theta.png")


if __name__ == "__main__":
    main()
PY
git add scripts/select_split.py
git commit -m "Cloud: split-selection script with KL-based seed scoring"
```

### 6.2 Run it

```bash
python scripts/select_split.py \
    --labels_dir outputs/state_probe/labels_full \
    --out outputs/state_probe/split_spec.json \
    --n_seeds 20
```

### 6.3 What to check

- Console printed `selected seed <N>` and wrote `split_spec.json`.
- Both overlap plots look like the `probe_v2_odd_even` baseline
  (`outputs/state_probe/probe_v2_odd_even/label_distribution_{position_scatter,angle_hist}.png`)
  — broadly similar distributions across train and val.
- `train_theta_coverage` and `val_theta_coverage` both ≥ 80%.

### 6.4 Failure recovery

*"If no seed passes the coverage floor"*: try `--n_seeds 100`. If still
failing, the dataset doesn't have enough episode diversity — stop and
contact me. This would be a dataset problem (a handful of episodes
contain all the diversity, and any random 80/20 excludes one of them
from train).

*"If first selected seed looks bad visually even though it passed the
KL score"*: re-run with `--n_seeds 100` to pick a better candidate. The
KL score is a proxy; your eyeball is the ground truth.

### CHECKPOINT 6 (Go / No-go, user review)

Go if coverage floors are met and plots look like the `probe_v2`
baseline. Post both plots if anything seems off.

**Expected wall time**: 2–5 minutes.

---

## Stage 7 — Probe training

**Purpose**: train the probe on the Stage-6-selected split and hit the
acceptance thresholds: pooled `val_pos_p95 ≤ 3 px`, pooled
`val_ang_p95 ≤ 5°`.

**Preconditions**: Stages 0–6 complete. `split_spec.json` written.

### 7.1 Make the training script split-spec-aware

The existing `scripts/train_state_probe.py` loads
`LABELS_DIR/labels_train.pt` and `LABELS_DIR/labels_val.pt` by
hardcoded name. On cloud we want it to consume `split_spec.json` instead.
Create a thin wrapper:

```bash
cat > scripts/train_state_probe_cloud.py <<'PY'
"""Cloud probe training: consumes a split_spec.json to select train/val
frame indices from a pooled labels_*.pt dataset.

All hyperparameters are unchanged from scripts/train_state_probe.py
(arch, loss, optimizer, schedule, acceptance). Only the data loading
path differs.
"""
import argparse, csv, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from rl.models.state_probe import StateProbe, split_output  # noqa

RESOLUTION = 128
DEVICE = "cuda:0"
LAMBDA_POS, LAMBDA_ANG, LAMBDA_NORM = 7.0, 1.0, 0.01
LR, WD, WARMUP_STEPS = 3e-4, 1e-4, 500
MIN_EPOCHS, PATIENCE, GRAD_CLIP = 20, 10, 1.0
OVERFIT_THRESHOLD = 2.5

def load_pooled(labels_dir: Path):
    files = sorted(labels_dir.glob("labels_*.pt"))
    latents, labels, eps, ts = [], [], [], []
    offset = 0
    for f in files:
        d = torch.load(f, weights_only=False)
        e = d["episode"] + offset
        offset = int(e.max()) + 1
        latents.append(d["latents"]); labels.append(d["labels"])
        eps.append(e); ts.append(d["t_idx"])
    return (torch.cat(latents, 0), torch.cat(labels, 0),
            torch.cat(eps, 0), torch.cat(ts, 0))

class Subset(Dataset):
    def __init__(self, lat, lab_px, eps, ts, indices):
        idx = torch.as_tensor(indices, dtype=torch.long)
        self.latents = lat[idx]
        self.labels_raw = lab_px[idx]
        self.episodes = eps[idx]
        self.t_idx = ts[idx]
        lbls = self.labels_raw.clone()
        lbls[:, 0] /= RESOLUTION; lbls[:, 1] /= RESOLUTION
        self.labels_norm = lbls
    def __len__(self): return self.latents.shape[0]
    def __getitem__(self, i):
        return (self.latents[i], self.labels_norm[i],
                int(self.episodes[i]), int(self.t_idx[i]))

def loss_fn(pred, tgt):
    pp, sp = split_output(pred); pt, st = split_output(tgt)
    lp = F.mse_loss(pp, pt); la = F.mse_loss(sp, st)
    ln = (((sp**2).sum(-1) - 1.0)**2).mean()
    return LAMBDA_POS*lp + LAMBDA_ANG*la + LAMBDA_NORM*ln, lp, la, ln

def per_frame_err(pred, tgt, res=RESOLUTION):
    pp, sp = split_output(pred); pt, st = split_output(tgt)
    pos = ((pp - pt)*res).norm(dim=-1)
    sp_u = sp / sp.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cos_d = (sp_u * st).sum(-1).clamp(-1.0, 1.0)
    return pos, torch.rad2deg(torch.acos(cos_d))

def eval_probe(probe, dl, dev):
    probe.eval()
    pp, tt, ee = [], [], []
    s = {"loss":0, "p":0, "a":0, "n":0}; nb = 0
    with torch.no_grad():
        for z, y, ep, _ in dl:
            z, y = z.to(dev), y.to(dev)
            p = probe(z); t, lp, la, ln = loss_fn(p, y)
            for k, v in zip(s.keys(), (t, lp, la, ln)):
                s[k] += float(v)
            nb += 1
            pp.append(p.cpu()); tt.append(y.cpu())
            ee.append(ep.clone() if isinstance(ep, torch.Tensor) else torch.tensor(ep))
    preds, tgts, eps = torch.cat(pp,0), torch.cat(tt,0), torch.cat(ee,0)
    p_err, a_err = per_frame_err(preds, tgts)
    p95p = float(np.percentile(p_err.numpy(), 95))
    a95p = float(np.percentile(a_err.numpy(), 95))
    wp, wa = 0.0, 0.0
    for e in torch.unique(eps).tolist():
        m = eps == e
        wp = max(wp, float(np.percentile(p_err[m].numpy(), 95)))
        wa = max(wa, float(np.percentile(a_err[m].numpy(), 95)))
    _, scp = split_output(preds)
    probe.train()
    for k in s: s[k] /= max(1, nb)
    return {
        "val_loss": s["loss"], "val_loss_pos": s["p"],
        "val_loss_ang": s["a"], "val_loss_norm": s["n"],
        "val_pos_mean": float(p_err.mean()),
        "val_pos_p95_pooled": p95p, "val_pos_p95_worst_episode": wp,
        "val_ang_mean": float(a_err.mean()),
        "val_ang_p95_pooled": a95p, "val_ang_p95_worst_episode": wa,
        "mean_pred_norm": float(scp.norm(dim=-1).mean()),
    }

def lr_sched(step, total, base):
    if step < WARMUP_STEPS: return base * (step+1)/WARMUP_STEPS
    p = min(1.0, max(0.0, (step-WARMUP_STEPS)/max(1,total-WARMUP_STEPS)))
    return 1e-6 + 0.5*(base-1e-6)*(1.0 + math.cos(math.pi*p))

def plot_curves(val_log, out):
    ep = [r["epoch"] for r in val_log]
    tl, vl = [r["train_loss"] for r in val_log], [r["val_loss"] for r in val_log]
    ratio = [v/max(t,1e-9) for v,t in zip(vl,tl)]
    pp, pw = [r["val_pos_p95_pooled"] for r in val_log], [r["val_pos_p95_worst_episode"] for r in val_log]
    ap, aw = [r["val_ang_p95_pooled"] for r in val_log], [r["val_ang_p95_worst_episode"] for r in val_log]
    bi = int(np.argmin([r["combined_gate"] for r in val_log]))
    mi = int(np.argmax(ratio))
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    a = ax[0,0]
    a.plot(ep, tl, label="train"); a.plot(ep, vl, label="val")
    a.axhline(vl[bi], ls="--", color="gray", label=f"best @ ep {ep[bi]}")
    a.set_yscale("log"); a.set_title("Loss"); a.legend(); a.grid(alpha=0.3)
    a = ax[0,1]
    a.plot(ep, ratio); a.axhline(1.0, ls="--", color="gray")
    a.axhline(OVERFIT_THRESHOLD, ls="--", color="red")
    a.annotate(f"max {ratio[mi]:.2f} @ ep {ep[mi]}", xy=(ep[mi], ratio[mi]),
               xytext=(5,-20), textcoords="offset points")
    a.set_title("val/train ratio"); a.grid(alpha=0.3)
    a = ax[1,0]
    a.plot(ep, pp, label="pooled"); a.plot(ep, pw, label="worst-ep", ls="--")
    a.axhline(3.0, ls="--", color="red", label="acceptance 3 px")
    a.set_title("pos p95 (px)"); a.legend(); a.grid(alpha=0.3)
    a = ax[1,1]
    a.plot(ep, ap, label="pooled"); a.plot(ep, aw, label="worst-ep", ls="--")
    a.axhline(5.0, ls="--", color="red", label="acceptance 5°")
    a.set_title("ang p95 (deg)"); a.legend(); a.grid(alpha=0.3)
    fig.suptitle("cloud probe run", fontsize=14)
    plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()

def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--labels_dir", type=str, required=True)
    cli.add_argument("--split_spec", type=str, required=True)
    cli.add_argument("--run_name", type=str,
                     default=f"run_{time.strftime('%Y%m%d_%H%M%S')}")
    cli.add_argument("--batch_size", type=int, default=64)
    cli.add_argument("--max_epochs", type=int, default=200)
    cli.add_argument("--seed", type=int, default=0)
    args = cli.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    run_dir = REPO_ROOT / "outputs" / "state_probe" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "overfit_dumps").mkdir(exist_ok=True)
    tb = SummaryWriter(log_dir=str(run_dir / "tb"))

    free, tot = torch.cuda.mem_get_info(0)
    print(f"[gpu] free {free/1024**2:.0f} / {tot/1024**2:.0f} MiB")

    with open(args.split_spec) as f:
        spec = json.load(f)
    assert spec.get("status") == "PASS", "split_spec did not pass"
    sel = spec["selected"]
    train_eps = set(sel["train_eps"]); val_eps = set(sel["val_eps"])
    print(f"[split] seed {sel['seed']}  train eps {len(train_eps)}  val eps {len(val_eps)}")

    lat, lab, eps, ts = load_pooled(Path(args.labels_dir))
    train_mask = torch.tensor([int(e) in train_eps for e in eps.tolist()])
    val_mask = torch.tensor([int(e) in val_eps for e in eps.tolist()])
    train_idx = torch.nonzero(train_mask, as_tuple=True)[0]
    val_idx = torch.nonzero(val_mask, as_tuple=True)[0]
    print(f"[data] train frames {len(train_idx)}  val frames {len(val_idx)}")

    tr = Subset(lat, lab, eps, ts, train_idx)
    va = Subset(lat, lab, eps, ts, val_idx)
    tdl = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                     num_workers=0, drop_last=False)
    vdl = DataLoader(va, batch_size=256, shuffle=False, num_workers=0)

    probe = StateProbe().to(DEVICE)
    print(f"[model] params {probe.num_params:,}")
    decay, nodecay = [], []
    for n, p in probe.named_parameters():
        (nodecay if ("norm" in n.lower() or "bias" in n.lower()) else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WD},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=LR, betas=(0.9, 0.999))
    steps_per_ep = max(1, (len(tr) + args.batch_size - 1) // args.batch_size)
    total_steps = steps_per_ep * args.max_epochs

    val_log = []
    best_gate = float("inf"); best_ep = -1; patience_ctr = 0
    overfit_eps = []; peak_mib = None
    wall_t0 = time.time(); step = 0

    for epoch in range(args.max_epochs):
        probe.train()
        s = {"l":0,"p":0,"a":0,"n":0}; nb = 0
        for z, y, _, _ in tdl:
            for pg in opt.param_groups: pg["lr"] = lr_sched(step, total_steps, LR)
            z, y = z.to(DEVICE), y.to(DEVICE)
            pred = probe(z); loss, lp, la, ln = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(probe.parameters(), GRAD_CLIP)
            opt.step()
            for k, v in zip(s.keys(), (loss, lp, la, ln)): s[k] += float(v)
            nb += 1; step += 1
            if step == 10 and peak_mib is None:
                torch.cuda.synchronize()
                peak_mib = torch.cuda.max_memory_allocated(0)/1024**2
                print(f"[mem] peak after 10 steps: {peak_mib:.1f} MiB")

        tr_row = {
            "train_loss": s["l"]/nb, "train_loss_pos": s["p"]/nb,
            "train_loss_ang": s["a"]/nb, "train_loss_norm": s["n"]/nb,
        }
        vm = eval_probe(probe, vdl, DEVICE)
        gate = LAMBDA_POS * vm["val_pos_p95_pooled"] + LAMBDA_ANG * vm["val_ang_p95_pooled"]
        row = {"epoch": epoch, "step": step, **tr_row, **vm,
               "combined_gate": gate,
               "val_over_train": vm["val_loss"]/max(tr_row["train_loss"],1e-9)}
        val_log.append(row)
        for k in ("train_loss","val_loss","val_pos_p95_pooled","val_pos_p95_worst_episode",
                  "val_ang_p95_pooled","val_ang_p95_worst_episode","mean_pred_norm"):
            tb.add_scalar(f"epoch/{k}", row[k], epoch)
        tb.add_scalar("epoch/val_over_train", row["val_over_train"], epoch)
        tb.add_scalar("epoch/combined_gate", gate, epoch)
        print(f"ep{epoch:03d} tr={row['train_loss']:.5f} vl={row['val_loss']:.5f} "
              f"v/t={row['val_over_train']:.2f}  "
              f"pos_p95={vm['val_pos_p95_pooled']:5.2f}px "
              f"(worst_ep {vm['val_pos_p95_worst_episode']:5.2f})  "
              f"ang_p95={vm['val_ang_p95_pooled']:5.2f}d "
              f"(worst_ep {vm['val_ang_p95_worst_episode']:5.2f})  "
              f"||sc||={vm['mean_pred_norm']:.3f}")
        if row["val_over_train"] > OVERFIT_THRESHOLD:
            overfit_eps.append(epoch)
            with torch.no_grad():
                z_d, y_d, ep_d, t_d = next(iter(vdl))
                p_d = probe(z_d[:64].to(DEVICE)).cpu()
            torch.save({"epoch": epoch, "pred": p_d, "target": y_d[:64]},
                       run_dir / f"overfit_dumps/epoch_{epoch:03d}.pt")
        if gate < best_gate:
            best_gate, best_ep, patience_ctr = gate, epoch, 0
            torch.save({"state_dict": probe.state_dict(), "epoch": epoch,
                        "row": row, "split_spec": sel},
                       run_dir / "best.pt")
        else:
            patience_ctr += 1
        if epoch+1 >= MIN_EPOCHS and patience_ctr >= PATIENCE:
            print(f"[early-stop] patience {PATIENCE}; best @ ep {best_ep}")
            break

    torch.save({"state_dict": probe.state_dict(), "epoch": epoch,
                "row": val_log[-1], "split_spec": sel},
               run_dir / "last.pt")
    with (run_dir / "val_log.csv").open("w", newline="") as f:
        cols = list(val_log[0].keys()); w = csv.writer(f); w.writerow(cols)
        for r in val_log: w.writerow([r[c] for c in cols])
    plot_curves(val_log, run_dir / "learning_curves.png")
    with (run_dir / "config.json").open("w") as f:
        json.dump({"run_name": args.run_name, "batch_size": args.batch_size,
                   "best_epoch": best_ep, "best_gate": best_gate,
                   "peak_mib_after_10_steps": peak_mib,
                   "wall_s": time.time()-wall_t0,
                   "overfit_epochs": overfit_eps,
                   "n_train_frames": len(tr), "n_val_frames": len(va),
                   "split_seed": sel["seed"]}, f, indent=2)
    tb.close()
    br = val_log[best_ep]
    print("="*60)
    print(f"best ep {best_ep}  pos_p95 {br['val_pos_p95_pooled']:.3f} px  "
          f"ang_p95 {br['val_ang_p95_pooled']:.3f}°  "
          f"||sc||_norm {br['mean_pred_norm']:.4f}")
    ok = br["val_pos_p95_pooled"] <= 3.0 and br["val_ang_p95_pooled"] <= 5.0
    print("ACCEPTANCE:", "PASS" if ok else "FAIL")

if __name__ == "__main__":
    main()
PY
git add scripts/train_state_probe_cloud.py
git commit -m "Cloud: split-spec-aware probe training script"
```

### 7.2 Run training

In a tmux:

```bash
tmux new -s train
conda activate iws
cd ~/interactive_world_sim

python scripts/train_state_probe_cloud.py \
    --labels_dir outputs/state_probe/labels_full \
    --split_spec outputs/state_probe/split_spec.json \
    --run_name probe_cloud_v1 \
    --batch_size 64 \
    --max_epochs 200 \
    2>&1 | tee outputs/state_probe/probe_cloud_v1.log
```

**Batch size guidance**:

| GPU | starting `--batch_size` | if OOM |
|---|---|---|
| RTX 5070 Ti (12 GB, dev) | 16 | 8 |
| A100 40 GB | 64 | 32 |
| A100 80 GB / H100 | 128 | 64 |

After the first `[mem] peak after 10 steps` line prints, you'll see the
actual peak. If peak is well under 20% of total VRAM, doubling the
batch size is safe.

### 7.3 Known issues to recognise, not fix

**Angle p95 worse than position p95**: the CV labels have a θ vs
θ + 180° ambiguity on some frames (documented in Phase 1 and in
`outputs/state_probe/probe_v2_odd_even/report.md`). Typical pattern: if
pos p95 = 2.5 px and ang p95 = 6–8°, that's the label-noise tail
showing up, not an overfitting or capacity issue. **Do not try to fix**
— surface it at Checkpoint 7 and stop.

**`val_loss / train_loss > 2.5` briefly**: the overfit flag fires
occasionally during warmup and settles. Diagnostic dumps are saved to
`overfit_dumps/`. This is normal and expected. Harmful overfitting
shows up as a sustained upward trend — watch the `learning_curves.png`
top-right panel.

### 7.4 Expected wall time

| GPU | epochs | wall time |
|---|---|---|
| A100 40 GB, 20 k frames, batch 64 | 100 | 8–15 min |
| A100 40 GB, 20 k frames, batch 64 | 200 | 15–30 min |
| H100, 20 k frames, batch 128 | 200 | 10–20 min |

### 7.5 What to check on completion

```bash
ls -lh outputs/state_probe/probe_cloud_v1/
cat outputs/state_probe/probe_cloud_v1/config.json | python -m json.tool
tail -20 outputs/state_probe/probe_cloud_v1.log
```

Look for:
- The terminal prints `ACCEPTANCE: PASS` or `FAIL` as the last line.
- `val_pos_p95_pooled` at the best epoch ≤ 3 px.
- `val_ang_p95_pooled` at the best epoch ≤ 5°. *(Allow up to ~6–8° and
  surface as known label-noise if pos passes but angle misses.)*
- `mean_pred_norm` at the best epoch ∈ [0.9, 1.1].
- Total epochs ran: if the run hit the full 200 without early-stop
  firing, the model is still improving — re-run with `--max_epochs 400`
  after reviewing the curves.

### 7.6 Failure recovery

*"If probe val p95 plateaus above acceptance"*: **do not tune**. Surface
at Checkpoint 7.

*"If the run finishes at max_epochs and was still improving"*: re-run
with a higher `--max_epochs`. This is the one allowed re-run. If it
still plateaus after that, stop and contact me.

*"If `mean_pred_norm` stays < 0.9 throughout"*: (sin, cos) collapse.
Do not increase `λ_norm` without asking. Stop and contact me —
Phase 3-A design doc §1.12 Risk 3 covers this; the fix should be a
joint decision.

*"If instance disconnects mid-training"*: re-attach tmux
(`tmux attach -t train`); the run continues. If the process died too,
you lose all progress (checkpointing is only at "best val" — no
time-based checkpoints). Re-run with the same `--run_name` to overwrite
the previous run's artefacts, or pick a new `--run_name`.

### CHECKPOINT 7 (Go / No-go, user review) — HARD STOP

Pause. Review, in order:
1. `outputs/state_probe/probe_cloud_v1/learning_curves.png`
2. `outputs/state_probe/probe_cloud_v1/val_log.csv` (tail)
3. `outputs/state_probe/probe_cloud_v1/config.json`

Go only if:
- pooled `val_pos_p95 ≤ 3 px` at best epoch
- pooled `val_ang_p95 ≤ 5°` at best epoch (or marginal miss with known
  label-noise pattern — see §7.3)
- `mean_pred_norm` ≈ 1.0
- Learning curves show no runaway overfit (val loss stable at best-ep)

Otherwise stop and surface. Do not attempt fixes unprompted.

**Expected wall time**: 10 min for the run + 5 min for review.

---

## Stage 8 — Acceptance evaluation + handover artifacts

**Purpose**: produce the validation grid, worst-K, and `report.md` that
accompany the probe checkpoint into the PR.

**Preconditions**: Checkpoint 7 passed.

### 8.1 Produce the acceptance viz

The diagnostic script already has the viz + report logic. Reuse it
against the best checkpoint:

```bash
# Copy the diagnostic script for the cloud run and point it at the
# cloud probe + split
cat > scripts/probe_acceptance_report.py <<'PY'
"""Produce probe_validation_grid.png, probe_worst_cases.png, report.md
for a completed cloud probe training run."""
import argparse, json, math, sys
from pathlib import Path
import cv2, numpy as np, torch
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
from rl.models.state_probe import StateProbe, split_output, normalize_sincos
from rl.models.world_model import DifferentiableDynamics
from rl.visualization.state_viz import render_state_on_image

RES = 128
DEVICE = "cuda:0"

def load_pooled(labels_dir: Path):
    files = sorted(labels_dir.glob("labels_*.pt"))
    latents, labels, eps, ts = [], [], [], []
    off = 0
    for f in files:
        d = torch.load(f, weights_only=False)
        e = d["episode"] + off; off = int(e.max()) + 1
        latents.append(d["latents"]); labels.append(d["labels"])
        eps.append(e); ts.append(d["t_idx"])
    return (torch.cat(latents,0), torch.cat(labels,0),
            torch.cat(eps,0), torch.cat(ts,0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--split_spec", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--wm_ckpt", default="outputs/pusht_cam1/checkpoints/best.ckpt")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    with open(args.split_spec) as f: spec = json.load(f)["selected"]
    val_eps = set(spec["val_eps"])

    lat, lab, eps, ts = load_pooled(Path(args.labels_dir))
    mask_val = torch.tensor([int(e) in val_eps for e in eps.tolist()])
    idx = torch.nonzero(mask_val, as_tuple=True)[0]

    ckpt = torch.load(run_dir / args.ckpt, weights_only=False)
    probe = StateProbe().to(DEVICE)
    probe.load_state_dict(ckpt["state_dict"]); probe.eval()
    best_row = ckpt["row"]
    print(f"loaded {args.ckpt} @ epoch {ckpt['epoch']}")

    wm = DifferentiableDynamics(args.wm_ckpt, device=DEVICE)

    # compute all val errors
    z_all = lat[idx].to(DEVICE)
    with torch.no_grad():
        preds = probe(z_all).cpu()
    tgts_norm = lab[idx].clone()
    tgts_norm[:, 0] /= RES; tgts_norm[:, 1] /= RES
    pos_p, sc_p = split_output(preds)
    pos_t, sc_t = split_output(tgts_norm)
    pos_err = ((pos_p - pos_t) * RES).norm(dim=-1)
    sc_u = sc_p / sc_p.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    ang_err = torch.rad2deg(torch.acos((sc_u * sc_t).sum(-1).clamp(-1, 1)))

    rng = np.random.default_rng(42)
    grid_idx = rng.choice(len(idx), size=min(16, len(idx)), replace=False)
    combined = pos_err.numpy()/RES + (1.0 - np.cos(np.deg2rad(ang_err.numpy())))/2
    worst_idx = np.argsort(-combined)[:12]

    def render_set(sub_idx, title, out_path):
        tiles = []
        for i in sub_idx:
            i = int(i)
            z = lat[idx[i]].unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                rgb = wm.decode(z, RES)[0].permute(1,2,0).cpu().float().numpy()
            rgb_u8 = np.clip(rgb*255, 0, 255).astype(np.uint8)
            cv_row = lab[idx[i]]
            cv_state = dict(cx=float(cv_row[0]), cy=float(cv_row[1]),
                            sin=float(cv_row[2]), cos=float(cv_row[3]))
            p = preds[i]; pp, sc = split_output(p.unsqueeze(0))
            sc_n = normalize_sincos(sc)[0]
            pr = dict(cx=float(pp[0,0])*RES, cy=float(pp[0,1])*RES,
                      sin=float(sc_n[0]), cos=float(sc_n[1]))
            canvas = rgb_u8.copy()
            canvas = render_state_on_image(canvas, cv_state["cx"], cv_state["cy"],
                                            cv_state["sin"], cv_state["cos"],
                                            color=(0,220,0), label="CV")
            canvas = render_state_on_image(canvas, pr["cx"], pr["cy"],
                                            pr["sin"], pr["cos"],
                                            color=(220,0,0), label="pr")
            big = cv2.resize(canvas, (canvas.shape[1]*4, canvas.shape[0]*4),
                             interpolation=cv2.INTER_NEAREST)
            pad = 28
            out = np.full((big.shape[0]+pad, big.shape[1], 3), 255, dtype=np.uint8)
            out[pad:, :] = big
            dp = math.hypot(pr["cx"]-cv_state["cx"], pr["cy"]-cv_state["cy"])
            cos_d = max(-1, min(1, pr["sin"]*cv_state["sin"] + pr["cos"]*cv_state["cos"]))
            dth = math.degrees(math.acos(cos_d))
            cv2.putText(out, f"ep{int(eps[idx[i]])} t={int(ts[idx[i]])} "
                             f"pos {dp:.2f}px ang {dth:.2f}°",
                        (4,20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)
            tiles.append(out)
        rows = (len(tiles)+3)//4
        h, w = tiles[0].shape[:2]
        pad = np.full((h, w, 3), 255, dtype=np.uint8)
        grid_rows = []
        for r in range(rows):
            rt = [tiles[r*4+c] if r*4+c < len(tiles) else pad for c in range(4)]
            grid_rows.append(np.concatenate(rt, axis=1))
        grid = np.concatenate(grid_rows, axis=0)
        banner = np.full((30, grid.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(banner, title, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_path), cv2.cvtColor(np.concatenate([banner, grid]), cv2.COLOR_RGB2BGR))

    render_set(grid_idx, "Validation grid (cloud run)",
               run_dir / "probe_validation_grid.png")
    render_set(worst_idx, "Worst 12 val frames (cloud run)",
               run_dir / "probe_worst_cases.png")

    pos_p95 = float(np.percentile(pos_err.numpy(), 95))
    ang_p95 = float(np.percentile(ang_err.numpy(), 95))
    gates = pos_p95 <= 3.0 and ang_p95 <= 5.0
    with (run_dir / "report.md").open("w") as f:
        f.write(f"# Cloud probe run — {run_dir.name}\n\n")
        f.write(f"## Final metrics (best ckpt, epoch {ckpt['epoch']})\n\n")
        f.write("| metric | value | gate |\n|---|---|---|\n")
        f.write(f"| val pos p95 pooled | **{pos_p95:.3f} px** | <= 3 |\n")
        f.write(f"| val pos p95 worst-episode | {best_row['val_pos_p95_worst_episode']:.3f} px | - |\n")
        f.write(f"| val pos mean | {float(pos_err.mean()):.3f} px | - |\n")
        f.write(f"| val ang p95 pooled | **{ang_p95:.3f}°** | <= 5 |\n")
        f.write(f"| val ang p95 worst-episode | {best_row['val_ang_p95_worst_episode']:.3f}° | - |\n")
        f.write(f"| val ang mean | {float(ang_err.mean()):.3f}° | - |\n")
        f.write(f"| val mean_pred_norm | {best_row['mean_pred_norm']:.4f} | ~1.0 |\n\n")
        f.write(f"**Acceptance gate (pooled pos p95 <= 3 AND pooled ang p95 <= 5): "
                f"{'PASS' if gates else 'FAIL'}.**\n\n")
        f.write("## Overfitting analysis\n\n")
        f.write("See `learning_curves.png`, `val_log.csv`. "
                "Phase 3-A supplement §2 five-question format:\n\n")
        f.write("1. Train/val loss divergence: [user to fill from curves]\n")
        f.write("2. Max val/train ratio and epoch: [from config.json]\n")
        f.write("3. Did val p95 follow loss: [from curves]\n")
        f.write("4. Final gap at best-val: [from val_log.csv]\n")
        f.write("5. Verdict: [one of the four]\n\n")
        f.write("## Systematic pattern in worst-K\n\n")
        f.write("[user to fill after eyeballing probe_worst_cases.png]\n\n")
        f.write("## Recommendation\n\n")
        f.write(f"{'Ship' if gates else 'Escalate — acceptance gate failed'}.\n")
    print(f"report: {run_dir / 'report.md'}")
    print(f"gate: pos p95 {pos_p95:.3f} (<= 3)  ang p95 {ang_p95:.3f} (<= 5)  "
          f"{'PASS' if gates else 'FAIL'}")

if __name__ == "__main__":
    main()
PY
git add scripts/probe_acceptance_report.py
git commit -m "Cloud: acceptance report + viz for trained probe"
```

Run:

```bash
python scripts/probe_acceptance_report.py \
    --run_dir outputs/state_probe/probe_cloud_v1 \
    --labels_dir outputs/state_probe/labels_full \
    --split_spec outputs/state_probe/split_spec.json
```

### 8.2 Fill in the `report.md` user-authored sections

The script stubs them out; you fill in (short, honest sentences):

1. **Overfitting analysis — the five questions** from the supplement.
   Numbers come from `val_log.csv` and `config.json`.
2. **Systematic pattern in worst-K**. Eyeball
   `probe_worst_cases.png`. Look for: are the worst frames all near-edge
   T-block? Near-π wraps? Near-occlusion by gripper tips? Anything
   spatially consistent? If nothing jumps out, say so.
3. **Recommendation**: `Ship` if gate passes, otherwise state what
   blocks and surface to me.

### 8.3 Copy artefacts for download

On the cloud instance (sizes approximate):

| file | size | where |
|---|---|---|
| probe checkpoint (`best.pt`) | ~4–8 MiB | `outputs/state_probe/probe_cloud_v1/best.pt` |
| `state_goal.pt` | ~18 KiB | `tests/goal_selection/state_goal.pt` |
| learning curves, grids, report | ~1–5 MiB each | `outputs/state_probe/probe_cloud_v1/` |

Either `scp` to your laptop, or push to an S3/GCS bucket:

```bash
# scp example (from your laptop, not the instance)
scp -r user@cloud-instance:~/interactive_world_sim/outputs/state_probe/probe_cloud_v1 ./
scp user@cloud-instance:~/interactive_world_sim/tests/goal_selection/state_goal.pt ./

# Or S3 (on the instance)
aws s3 sync outputs/state_probe/probe_cloud_v1 \
    s3://YOUR_BUCKET/interactive_world_sim/probe_cloud_v1/
```

### 8.4 Commit artefacts and push

```bash
# The .pt files under outputs/state_probe/*/ are gitignored by
# existing rules. Commit the small tracked files only:
git add -f \
    outputs/state_probe/probe_cloud_v1/report.md \
    outputs/state_probe/probe_cloud_v1/config.json \
    outputs/state_probe/probe_cloud_v1/val_log.csv \
    outputs/state_probe/probe_cloud_v1/learning_curves.png \
    outputs/state_probe/probe_cloud_v1/probe_validation_grid.png \
    outputs/state_probe/probe_cloud_v1/probe_worst_cases.png \
    outputs/state_probe/split_spec.json \
    outputs/state_probe/split_distribution_position.png \
    outputs/state_probe/split_distribution_theta.png \
    outputs/state_probe/labels_full/meta.json \
    outputs/state_probe/labels_full/drops.csv

# Add the new tracked scripts (already committed earlier as you went)
git status  # sanity check

git commit -m "Cloud run: probe_cloud_v1 acceptance artefacts"
git push -u origin state-probe-training-cloud
```

### 8.5 Open the PR

In GitHub / your host's UI:

- Source: `state-probe-training-cloud`
- Target: `dreamer-rl`
- PR title: `Branch A: state-probe trained on full data`
- PR body checklist (paste into the PR description):
  - [ ] Acceptance gate pass: pooled pos p95 = X.XX px, pooled ang p95 = Y.YY°
  - [ ] Link: `outputs/state_probe/probe_cloud_v1/report.md`
  - [ ] Link: `outputs/state_probe/probe_cloud_v1/probe_validation_grid.png`
  - [ ] Link: `outputs/state_probe/probe_cloud_v1/probe_worst_cases.png`
  - [ ] Link: `outputs/state_probe/probe_cloud_v1/learning_curves.png`
  - [ ] Link: `migration_log.md` (final summary section)
  - [ ] Label-noise acknowledgement (if angle p95 is weaker than position p95):
        known `θ`-vs-`θ + 180°` CV issue, tracked separately

### CHECKPOINT 8 (Go / No-go, user review) — PR open

Go if the PR shows all artefacts linked, acceptance gate passes, and
you've filled in the user-authored sections of `report.md`. Otherwise
address and push an amendment.

**Expected wall time**: 15–25 minutes.

---

## Stage 9 — Shutdown checklist

**Purpose**: nothing is lost when the instance is destroyed.

### 9.1 Preserve

Before shutdown, confirm these are either committed+pushed or copied
off the instance:

- Probe checkpoint: `outputs/state_probe/probe_cloud_v1/best.pt`
- `tests/goal_selection/state_goal.pt`
- All `outputs/state_probe/probe_cloud_v1/*.png,md,json,csv`
- `outputs/state_probe/split_spec.json`
- (optional) TensorBoard event dirs: `outputs/state_probe/probe_cloud_v1/tb/`
- (optional, bulky) The labeled dataset `outputs/state_probe/labels_full/labels_*.pt`
  — these are reproducible from the raw dataset + `label_replay_buffer.py`,
  so not strictly necessary to preserve, but saves a round of decoding
  on re-run.

### 9.2 Discard freely

- `/tmp/*` (the inspection + smoke-test scratch)
- `outputs/state_probe/probe_cloud_v1/overfit_dumps/` (diagnostic only,
  unless a run had a sustained spike worth reviewing)

### 9.3 Final commit + push

```bash
git status
# If anything is uncommitted:
git add -A
git commit -m "Cloud: final artefacts"
# Push
git push origin state-probe-training-cloud
```

### 9.4 Disk usage at each stage (rough, for capacity planning)

| item | size |
|---|---|
| `~/data/full/` raw HDF5 episodes | depends — count × ~160 MB each |
| WM checkpoint `best.ckpt` | 366 MiB |
| Stage 4 labeled dataset | N_frames × ~16 KiB per frame (latent only) |
| Stage 7 probe checkpoints | ~5–10 MiB each (best + last) |
| TensorBoard event files | ~5–20 MiB per run |

For 20k frames total, Stage 4 output is ~300 MiB. Stage 7 run is
~50 MiB. Budget ~1 GiB for the whole `outputs/state_probe/` tree on
top of the raw data + WM ckpt.

### 9.5 Instance destroy

Your provider's console. No checklist — git push is what matters.

---

## References

Do not duplicate content from these; read them when you need the
reasoning.

- [state_estimator_design.md](state_estimator_design.md) — architecture,
  loss, acceptance rationale (§1.7, §1.9)
- [migration_log.md](migration_log.md) — deviations from the
  supervisor's defaults (mainly the vendoring of the CV pipeline and
  the `iws_wm_render` preset slot)
- [outputs/state_probe/probe_v2_odd_even/report.md](outputs/state_probe/probe_v2_odd_even/report.md)
  — the local architecture-validation baseline. Cloud numbers on a
  properly held-out split should be comparable or better for position,
  and comparable (± label-noise) for angle.
- [tests/state_estimator/calibration_outputs/calibration_summary.md](tests/state_estimator/calibration_outputs/calibration_summary.md)
  — the local HSV calibration result for comparison at Stage 3.

---

## Summary of failure recovery — one-table reference

| symptom | stage | action |
|---|---|---|
| `data/full` schema differs from mini | 1 | stop + contact |
| bulk labeling OOMs | 4 | halve `--decode_batch`, restart |
| REAL drops > 10% on full data | 3 | stop + contact |
| best split seed still has disjoint distributions after 100 tries | 6 | stop + contact |
| probe val p95 plateaus above acceptance | 7 | stop + contact (do not tune) |
| probe val p95 still improving at `max_epochs` | 7 | one re-run with higher `--max_epochs` |
| `mean_pred_norm < 0.9` sustained | 7 | stop + contact (do not increase λ_norm) |
| instance disconnects mid-training | any | tmux re-attach; if process died, restart that stage |

One-line go/no-go per stage:

| stage | go criterion |
|---|---|
| 0 | `nvidia-smi` healthy, `iws` env works, tests green, branch `state-probe-training-cloud` |
| 1 | ≥ 20 episodes, ≥ 4 k frames, `obs/images/camera_1_color` uint8 RGB |
| 2 | 2-episode smoke labels successfully, drop < 10% |
| 3 | REAL drop < 10%, median area ≥ 90% of theoretical, no false positives |
| 4 | overall drop < 10%, no episode > 30% |
| 5 | `cv_success=True` on z_goal, overlay visually correct |
| 6 | both splits cover ≥ 80% of the full θ and cx ranges; plots look like `probe_v2` baseline |
| 7 | pos p95 ≤ 3, ang p95 ≤ 5 (or marginal miss + known label noise), `mean_pred_norm` ≈ 1 |
| 8 | PR open with all artefact links filled in |
| 9 | everything committed + pushed |
