"""Step 2 visualization: CV-based reward on 10 frames of a real episode.

Takes 10 evenly-spaced frames from data/mini/pusht/train/episode_3.hdf5
(data/full/ is not available on this machine — cloud will use full),
encodes each through the WM to get the latent, decodes it back through
the WM to get the RGB that MPPI reward will see at runtime, then runs
state_reward against state_goal.

Produces:
  outputs/mppi/step2_reward_check/reward_curve.png  — reward vs frame idx
  outputs/mppi/step2_reward_check/frame_grid.png    — 10-panel of decoded frames
  outputs/mppi/step2_reward_check/summary.json      — numeric results

The expected pattern for a successful push: early frames (T far from goal)
score very negative, later frames (T near goal) score close to zero.
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
from rl.mppi.reward import state_reward  # noqa: E402

CKPT_PATH = REPO_ROOT / "outputs" / "pusht_cam1" / "checkpoints" / "best.ckpt"
STATE_GOAL_PATH = REPO_ROOT / "tests" / "goal_selection" / "state_goal.pt"
EPISODE_IDX = 3  # mini episode_3 is static; ep3 shows a real push trajectory.
EP_PATH_LOCAL = REPO_ROOT / "data" / "mini" / "pusht" / "train" / f"episode_{EPISODE_IDX}.hdf5"
EP_PATH_FULL = REPO_ROOT / "data" / "full" / "pusht" / "train" / f"episode_{EPISODE_IDX}.hdf5"
OUT_DIR = REPO_ROOT / "outputs" / "mppi" / "step2_reward_check"

RES = 128
N_FRAMES = 10
OBS_KEY = "camera_1_color"


def _pick_ep_path() -> Path:
    if EP_PATH_FULL.exists():
        return EP_PATH_FULL
    if EP_PATH_LOCAL.exists():
        return EP_PATH_LOCAL
    raise FileNotFoundError(
        f"Neither {EP_PATH_FULL} nor {EP_PATH_LOCAL} exists; "
        "need at least one PushT episode to run the Step 2 viz."
    )


def _preprocess_rgb(raw: np.ndarray) -> np.ndarray:
    h, w = raw.shape[:2]
    s = min(h, w)
    cropped = raw[(h - s) // 2 : (h - s) // 2 + s, (w - s) // 2 : (w - s) // 2 + s]
    resized = cv2.resize(cropped, (RES, RES), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda:0"

    ep_path = _pick_ep_path()
    print(f"Using episode: {ep_path}")

    with h5py.File(ep_path, "r") as f:
        frames = f[f"obs/images/{OBS_KEY}"][()]  # (T, H, W, 3)
    T = frames.shape[0]
    idxs = np.linspace(0, T - 1, N_FRAMES, dtype=int)
    print(f"  episode length: {T}; selected indices: {idxs.tolist()}")

    # Preprocess to 128x128 float32 [0,1] in (N, 3, 128, 128)
    prep = np.stack([_preprocess_rgb(frames[i]) for i in idxs], axis=0)
    prep_t = torch.from_numpy(prep).permute(0, 3, 1, 2).contiguous().to(device)

    wm = DifferentiableDynamics(str(CKPT_PATH), device=device)

    t0 = time.time()
    with torch.no_grad():
        z = wm.encode(prep_t)                       # (N, 4, 32, 32)
        rgb_decoded = wm.decode(z, resolution=RES)  # (N, 3, 128, 128) in [0,1]
    decode_s = time.time() - t0

    rgb_u8 = (rgb_decoded.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    rgb_u8 = rgb_u8.transpose(0, 2, 3, 1)  # (N, 128, 128, 3)

    goal = torch.load(STATE_GOAL_PATH, map_location="cpu")
    state_goal = {
        "cx": goal["cx"],
        "cy": goal["cy"],
        "sin_theta": goal["sin_theta"],
        "cos_theta": goal["cos_theta"],
    }

    labeler = CVLabeler(preset="REAL", resolution=RES)

    t0 = time.time()
    rewards = []
    per_frame = []
    for k, frame_idx in enumerate(idxs):
        r, label = state_reward(rgb_u8[k], state_goal, labeler=labeler)
        rewards.append(r)
        per_frame.append(
            {
                "frame_idx": int(frame_idx),
                "reward": round(r, 4),
                "cv_success": label.success,
                "cx": round(label.cx, 2) if label.success else None,
                "cy": round(label.cy, 2) if label.success else None,
                "theta_deg": round(label.theta_deg, 2) if label.success else None,
            }
        )
    cv_s = time.time() - t0
    print(f"  decode wall:      {decode_s:.2f} s (N={N_FRAMES})")
    print(f"  CV-label wall:    {cv_s:.3f} s (N={N_FRAMES}), "
          f"per-frame avg {cv_s/N_FRAMES*1000:.1f} ms")
    for row in per_frame:
        print(f"  t={row['frame_idx']:3d}: reward={row['reward']:+.4f}  "
              f"cv={row['cv_success']}  "
              f"pose=({row['cx']}, {row['cy']}, {row['theta_deg']}°)")

    # ------------------ plot reward curve ------------------
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.plot(idxs, rewards, marker="o", color="tab:blue")
    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.set_xlabel("frame index in episode_3")
    ax.set_ylabel("state_reward(decoded frame, state_goal)")
    ax.set_title("Step 2 — CV reward across 10 decoded frames of episode_3")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    curve_path = OUT_DIR / "reward_curve.png"
    fig.savefig(curve_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote: {curve_path}")

    # ------------------ plot frame grid --------------------
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.2))
    for k, ax in enumerate(axes.flat):
        ax.imshow(rgb_u8[k])
        ax.set_title(
            f"t={idxs[k]}  r={rewards[k]:+.3f}", fontsize=9,
        )
        ax.axis("off")
    fig.suptitle(
        "Step 2 — decoded frames, reward annotated", fontsize=11
    )
    fig.tight_layout()
    grid_path = OUT_DIR / "frame_grid.png"
    fig.savefig(grid_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote: {grid_path}")

    summary = {
        "config": {
            "episode": str(ep_path.relative_to(REPO_ROOT)),
            "n_frames": N_FRAMES,
            "frame_indices": idxs.tolist(),
            "resolution": RES,
            "preset": "REAL",
        },
        "timings": {
            "wm_decode_s": round(decode_s, 3),
            "cv_label_total_s": round(cv_s, 3),
            "cv_label_per_frame_ms": round(cv_s / N_FRAMES * 1000, 1),
        },
        "rewards": per_frame,
        "reward_stats": {
            "min": float(np.min(rewards)),
            "max": float(np.max(rewards)),
            "mean": float(np.mean(rewards)),
        },
    }
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  wrote: {summary_path}")


if __name__ == "__main__":
    main()
