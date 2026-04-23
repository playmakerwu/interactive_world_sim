"""Step 3 visualization: batched reward distribution + wall-time profile.

Runs two N=32 rollouts:
  (A) starting from z_goal      — expected reward distribution near 0
  (B) starting from a far-from-goal latent — expected distribution spread out,
      with a few near-zero if a random action happens to swing T back to goal.

Also measures batched decode wall time (N=32 in a single decoder call) and
compares against the Step 2 single-frame-decode × N projection. The user
specifically asked for this in Step 3 feedback.

Writes:
  outputs/mppi/step3_batch_reward/reward_histograms.png
  outputs/mppi/step3_batch_reward/decode_timing.json
  outputs/mppi/step3_batch_reward/summary.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.labeling.cv_labeler import CVLabeler  # noqa: E402
from rl.models.world_model import DifferentiableDynamics  # noqa: E402
from rl.mppi.reward import batched_state_reward  # noqa: E402
from rl.mppi.utils import batched_rollout  # noqa: E402

CKPT_PATH = REPO_ROOT / "outputs" / "pusht_cam1" / "checkpoints" / "best.ckpt"
Z_GOAL_PATH = REPO_ROOT / "tests" / "goal_selection" / "z_goal.pt"
STATE_GOAL_PATH = REPO_ROOT / "tests" / "goal_selection" / "state_goal.pt"
OUT_DIR = REPO_ROOT / "outputs" / "mppi" / "step3_batch_reward"

# Mini ep2 t=0 has T at (~18, 91, -25°), ~38 px from goal in x, ~28 px in y
# — a real "far from goal" initial state we can encode locally.
FAR_EP_PATH_LOCAL = REPO_ROOT / "data" / "mini" / "pusht" / "train" / "episode_2.hdf5"
FAR_EP_FRAME_IDX = 0
RES = 128
# N=32 rollout fits locally (5.5 GiB) but batched decode of 32 latents
# requires ~4 GiB more for the decoder attention softmax, which pushes
# past 11.5 GiB. Dropping to N=16 is the simplest fix that avoids
# decoder-level chunking (explicitly out of scope per user's Step 1
# feedback). Cloud L40S should re-run this script with N=64 to get the
# full-distribution histogram.
N = 16
H = 10
SIGMA = 0.1
SEED = 0
OBS_KEY = "camera_1_color"


def _preprocess_rgb(raw: np.ndarray) -> np.ndarray:
    h, w = raw.shape[:2]
    s = min(h, w)
    cr = raw[(h - s) // 2 : (h - s) // 2 + s, (w - s) // 2 : (w - s) // 2 + s]
    resized = cv2.resize(cr, (RES, RES), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def _rollout_and_score(
    z0: torch.Tensor,
    wm: DifferentiableDynamics,
    state_goal: dict,
    labeler: CVLabeler,
    *,
    seed: int,
    tag: str,
) -> dict:
    """Rollout N trajectories from z0 and score the final latents."""
    torch.manual_seed(seed)
    device = z0.device
    action_dim = 4  # pinned for this ckpt; see MPPI_NOTES §Step 1
    z0_batch = z0.expand(N, -1, -1, -1).contiguous()
    actions = torch.randn(N, H, action_dim, device=device) * SIGMA

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        latents = batched_rollout(z0_batch, actions, wm)
    rollout_s = time.time() - t0

    # Step-3 headline measurement: batched decode of N latents in one
    # GPU call. Isolated so we can compare to Step-2 serial projection.
    z_last = latents[:, -1].clone()  # (N, C, H, W)
    del latents
    torch.cuda.empty_cache()  # release rollout scratch before attention alloc
    t0 = time.time()
    with torch.no_grad():
        rgb = wm.decode(z_last, resolution=RES)  # (N, 3, H, W) in [0, 1]
    decode_s = time.time() - t0

    t0 = time.time()
    rewards, labels = batched_state_reward(rgb, state_goal, labeler=labeler)
    cv_s = time.time() - t0

    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    n_cv_fail = sum(1 for lbl in labels if not lbl.success)

    print(f"[{tag}]  rollout={rollout_s:.2f}s  decode(N={N})={decode_s:.3f}s  "
          f"cv(N={N})={cv_s:.3f}s  peak={peak_mb:.0f} MiB  cv_fail={n_cv_fail}/{N}")
    print(f"       reward  mean={rewards.mean():+.4f}  std={rewards.std():.4f}  "
          f"min={rewards.min():+.4f}  max={rewards.max():+.4f}")

    return {
        "tag": tag,
        "rewards": rewards.tolist(),
        "cv_success": [lbl.success for lbl in labels],
        "rollout_s": round(rollout_s, 3),
        "decode_s_batched": round(decode_s, 4),
        "cv_s_total": round(cv_s, 3),
        "peak_vram_MiB": round(peak_mb, 1),
        "reward_stats": {
            "mean": float(rewards.mean()),
            "std": float(rewards.std()),
            "min": float(rewards.min()),
            "max": float(rewards.max()),
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda:0"

    print(f"Loading WM from {CKPT_PATH}")
    wm = DifferentiableDynamics(str(CKPT_PATH), device=device)

    g = torch.load(STATE_GOAL_PATH, map_location="cpu")
    state_goal = {
        "cx": g["cx"],
        "cy": g["cy"],
        "sin_theta": g["sin_theta"],
        "cos_theta": g["cos_theta"],
    }
    labeler = CVLabeler(preset="REAL", resolution=RES)

    z_goal = torch.load(Z_GOAL_PATH, map_location=device)

    # -------- (A) start from z_goal --------
    result_a = _rollout_and_score(
        z_goal, wm, state_goal, labeler, seed=SEED, tag="from-z_goal",
    )

    # -------- (B) start from a far latent --------
    assert FAR_EP_PATH_LOCAL.exists(), f"Need {FAR_EP_PATH_LOCAL}"
    with h5py.File(FAR_EP_PATH_LOCAL, "r") as f:
        raw = f[f"obs/images/{OBS_KEY}"][FAR_EP_FRAME_IDX]
    pre = _preprocess_rgb(raw)
    pre_t = torch.from_numpy(pre).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        z_far = wm.encode(pre_t)  # (1, 4, 32, 32)
    print(f"Far-start encoded from data/mini/pusht/train/episode_2.hdf5 t=0")

    result_b = _rollout_and_score(
        z_far, wm, state_goal, labeler, seed=SEED, tag="from-far",
    )

    # -------- histograms --------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for ax, result, color in [
        (axes[0], result_a, "tab:blue"),
        (axes[1], result_b, "tab:red"),
    ]:
        rewards = np.array(result["rewards"])
        ax.hist(rewards, bins=16, color=color, edgecolor="black", alpha=0.8)
        ax.axvline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.set_title(
            f"{result['tag']}  (N={N}, H={H}, σ={SIGMA})\n"
            f"mean={result['reward_stats']['mean']:+.3f} "
            f"std={result['reward_stats']['std']:.3f}",
            fontsize=10,
        )
        ax.set_xlabel("reward")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        "Step 3 — reward distribution over N rollouts at two starting latents",
        fontsize=11,
    )
    fig.tight_layout()
    hist_path = OUT_DIR / "reward_histograms.png"
    fig.savefig(hist_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote: {hist_path}")

    # -------- decode timing: batched vs serial-projection --------
    # Decode one latent 32 times serially to compare to the N=32 batched call.
    print("\nProfiling single-frame decode × N vs batched decode(N):")
    z_single = z_goal
    # warmup
    with torch.no_grad():
        wm.decode(z_single, resolution=RES)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N):
        with torch.no_grad():
            wm.decode(z_single, resolution=RES)
    torch.cuda.synchronize()
    serial_s = time.time() - t0
    # batched
    z_batched = z_goal.expand(N, -1, -1, -1).contiguous()
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        wm.decode(z_batched, resolution=RES)
    torch.cuda.synchronize()
    batched_s = time.time() - t0
    speedup = serial_s / batched_s
    print(
        f"  serial × {N}: {serial_s:.3f} s  ({serial_s/N*1000:.1f} ms/frame)"
    )
    print(f"  batched N={N}: {batched_s:.3f} s  ({batched_s/N*1000:.1f} ms/frame)")
    print(f"  speedup: {speedup:.1f}×")

    timing = {
        "decode_serial_Nx_s": round(serial_s, 4),
        "decode_batched_N_s": round(batched_s, 4),
        "decode_serial_per_frame_ms": round(serial_s / N * 1000, 2),
        "decode_batched_per_frame_ms": round(batched_s / N * 1000, 2),
        "decode_batched_speedup": round(speedup, 2),
        "N": N,
    }
    timing_path = OUT_DIR / "decode_timing.json"
    timing_path.write_text(json.dumps(timing, indent=2))
    print(f"wrote: {timing_path}")

    summary = {
        "config": {"N": N, "H": H, "sigma": SIGMA, "seed": SEED, "res": RES},
        "results": [result_a, result_b],
        "decode_timing": timing,
    }
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote: {summary_path}")


if __name__ == "__main__":
    main()
