"""Step 1 visualization: batched rollout sanity check.

Runs N=32 rollouts of horizon H=10 from z_goal with sigma=0.1 action noise.
Decodes z_0 plus the final latent z_H of 4 representative rollouts and saves
a 1x5 panel PNG so we can eyeball whether:
  - the decoder outputs plausible RGB
  - different action sequences produce visibly different final frames
  - the WM is not drifting into unrecognizable pixel garbage after H=10 steps

Also writes a small JSON with per-sample L2 drift (||z_H - z_0||) and peak
VRAM for the run.

Writes to outputs/mppi/step1_rollout_check/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.mppi.utils import batched_rollout  # noqa: E402
from rl.models.world_model import DifferentiableDynamics  # noqa: E402

CKPT_PATH = REPO_ROOT / "outputs" / "pusht_cam1" / "checkpoints" / "best.ckpt"
Z_GOAL_PATH = REPO_ROOT / "tests" / "goal_selection" / "z_goal.pt"
OUT_DIR = REPO_ROOT / "outputs" / "mppi" / "step1_rollout_check"

ACTION_DIM = 4
N = 32
H = 10
SIGMA = 0.1
SEED = 0
PANEL_INDICES = (0, 8, 16, 24)  # pick 4 of the 32 for display


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), "This script requires CUDA"
    device = "cuda:0"

    print(f"Loading WM from {CKPT_PATH}")
    wm = DifferentiableDynamics(str(CKPT_PATH), device=device)

    z_goal = torch.load(Z_GOAL_PATH, map_location=device)
    assert z_goal.shape == (1, 4, 32, 32)

    z0 = z_goal.expand(N, -1, -1, -1).contiguous()

    torch.manual_seed(SEED)
    actions = torch.randn(N, H, ACTION_DIM, device=device) * SIGMA

    print(f"Rolling out N={N}, H={H}, sigma={SIGMA}")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        latents = batched_rollout(z0, actions, wm)
    rollout_s = time.time() - t0
    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"  wall time:  {rollout_s:.2f} s")
    print(f"  peak VRAM:  {peak_mb:.1f} MiB")

    l2_drift = (latents[:, -1] - latents[:, 0]).reshape(N, -1).norm(dim=1)
    print(f"  ||z_H - z_0|| mean={l2_drift.mean():.3f}, "
          f"min={l2_drift.min():.3f}, max={l2_drift.max():.3f}")

    # Decode z_0 (same for all) plus the 4 selected z_H's.
    z_to_decode = torch.stack(
        [latents[0, 0]] + [latents[i, -1] for i in PANEL_INDICES], dim=0
    )
    with torch.no_grad():
        rgb = wm.decode(z_to_decode, resolution=128)  # (5, 3, 128, 128) in [0,1]
    rgb = (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    rgb = rgb.transpose(0, 2, 3, 1)  # (5, 128, 128, 3)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
    titles = [
        "z_0 (decoded)",
        *(f"sample #{i} z_H\n||Δ||={l2_drift[i].item():.2f}" for i in PANEL_INDICES),
    ]
    for ax, frame, title in zip(axes, rgb, titles, strict=True):
        ax.imshow(frame)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(
        f"Step 1 rollout check — N={N}, H={H}, sigma={SIGMA}, "
        f"peak VRAM {peak_mb:.0f} MiB",
        fontsize=11,
    )
    fig.tight_layout()
    panel_path = OUT_DIR / "rollout_panel.png"
    fig.savefig(panel_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote: {panel_path}")

    summary = {
        "config": {
            "N": N,
            "H": H,
            "sigma": SIGMA,
            "seed": SEED,
            "action_dim": ACTION_DIM,
            "panel_indices": list(PANEL_INDICES),
            "ckpt": str(CKPT_PATH),
        },
        "rollout_wall_s": round(rollout_s, 3),
        "rollout_peak_vram_MiB": round(peak_mb, 1),
        "l2_drift": {
            "mean": round(l2_drift.mean().item(), 3),
            "min": round(l2_drift.min().item(), 3),
            "max": round(l2_drift.max().item(), 3),
            "per_panel_sample": [round(l2_drift[i].item(), 3) for i in PANEL_INDICES],
        },
    }
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  wrote: {summary_path}")


if __name__ == "__main__":
    main()
