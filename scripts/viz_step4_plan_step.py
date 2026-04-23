"""Step 4 visualization: one MPPI plan step, top-8 candidates.

Runs `MPPIPlanner.plan_step` starting from a far-from-goal latent (mini
ep2 t=0, chosen because Step 3 showed its reward distribution is bimodal
and interesting to visualize). Saves:

  outputs/mppi/step4_plan_step/rollout_samples.png
     2x4 grid of the 8 highest-reward candidates' decoded final frames.
     Each annotated with (reward, CV theta, softmax weight) — the CV
     theta is the user's Step 4 ask: if two visually-similar frames
     show reward differing by ~2, that is the 180-flip smoking gun.
  outputs/mppi/step4_plan_step/summary.json
     Numeric a*, naive mean, top-8 indices, and config.

Uses N=16 locally per §Step 3 VRAM finding; cloud re-runs at N=128.
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
from rl.mppi.planner import MPPIPlanner  # noqa: E402

CKPT_PATH = REPO_ROOT / "outputs" / "pusht_cam1" / "checkpoints" / "best.ckpt"
STATE_GOAL_PATH = REPO_ROOT / "tests" / "goal_selection" / "state_goal.pt"
FAR_EP_PATH = REPO_ROOT / "data" / "mini" / "pusht" / "train" / "episode_2.hdf5"
OUT_DIR = REPO_ROOT / "outputs" / "mppi" / "step4_plan_step"

RES = 128
N = 16
H = 10
SIGMA = 0.1
TEMPERATURE = 1.0
SEED = 0
TOP_K = 8
OBS_KEY = "camera_1_color"


def _preprocess_rgb(raw: np.ndarray) -> np.ndarray:
    h, w = raw.shape[:2]
    s = min(h, w)
    cr = raw[(h - s) // 2 : (h - s) // 2 + s, (w - s) // 2 : (w - s) // 2 + s]
    resized = cv2.resize(cr, (RES, RES), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda:0"

    print(f"Loading WM from {CKPT_PATH}")
    wm = DifferentiableDynamics(str(CKPT_PATH), device=device)

    g = torch.load(STATE_GOAL_PATH, map_location="cpu")
    state_goal = {
        "cx": g["cx"], "cy": g["cy"],
        "sin_theta": g["sin_theta"], "cos_theta": g["cos_theta"],
    }

    # Build far-from-goal latent.
    with h5py.File(FAR_EP_PATH, "r") as f:
        raw = f[f"obs/images/{OBS_KEY}"][0]
    pre = _preprocess_rgb(raw)
    pre_t = torch.from_numpy(pre).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        z_current = wm.encode(pre_t)

    planner = MPPIPlanner(
        wm, state_goal,
        N=N, H=H, sigma=SIGMA, temperature=TEMPERATURE,
        action_dim=4, resolution=RES, device=device,
        capture_rgb=True,  # keep decoded RGBs for viz
    )

    t0 = time.time()
    a_star = planner.plan_step(z_current, seed=SEED)
    wall_s = time.time() - t0
    stats = planner.last_stats
    assert stats is not None

    print(f"plan_step wall: {wall_s:.2f} s  (N={N}, H={H})")
    print(f"a_star      = {a_star.tolist()}")
    print(f"a_naive_mean= {stats.a_naive_mean.tolist()}")
    print(f"cv_fail     = {stats.cv_fail_count}/{N}")
    print(
        f"rewards  mean={stats.rewards.mean():+.4f}  std={stats.rewards.std():.4f}  "
        f"min={stats.rewards.min():+.4f}  max={stats.rewards.max():+.4f}"
    )

    # Indices by reward (descending). Show TOP_K at the top for the spec
    # deliverable; show the full sort in a 4x4 grid so the 180-flip
    # smoking gun (similar-T but reward differing by ~2) is visible.
    sort_desc = np.argsort(stats.rewards)[::-1]
    top_idx = sort_desc[:TOP_K]
    print(f"top-{TOP_K} indices (by reward): {top_idx.tolist()}")

    def _annot(idx: int) -> str:
        lbl = stats.labels[int(idx)]
        r = stats.rewards[idx]
        w = stats.weights[idx]
        if lbl.success:
            return (
                f"#{idx}  r={r:+.3f}  w={w:.3f}\n"
                f"({lbl.cx:.0f},{lbl.cy:.0f})  θ={lbl.theta_deg:+.1f}°"
            )
        return f"#{idx}  r={r:+.3f}  w={w:.3f}\nCV-FAIL"

    # ------------------ rollout_samples.png — primary (top-8 per spec) ------
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.2))
    for ax, idx in zip(axes.flat, top_idx, strict=True):
        ax.imshow(stats.decoded_rgb[idx])
        ax.set_title(_annot(int(idx)), fontsize=9.5)
        ax.axis("off")
    fig.suptitle(
        f"Step 4 — top-{TOP_K} candidates of {N} (H={H}, σ={SIGMA})\n"
        f"from far-start (mini ep2 t=0).  cv_fail={stats.cv_fail_count}/{N}",
        fontsize=11,
    )
    fig.tight_layout()
    samples_path = OUT_DIR / "rollout_samples.png"
    fig.savefig(samples_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote: {samples_path}")

    # ------- rollout_all_sorted.png — full N sorted, 4 rows x 4 cols -------
    # Every candidate annotated with CV theta. Two visually similar frames
    # with reward differing by ~2 = 180-flip smoking gun.
    fig, axes = plt.subplots(4, 4, figsize=(13, 13))
    for ax, idx in zip(axes.flat, sort_desc, strict=True):
        ax.imshow(stats.decoded_rgb[idx])
        ax.set_title(_annot(int(idx)), fontsize=9)
        ax.axis("off")
    fig.suptitle(
        f"Step 4 — all {N} candidates sorted by reward "
        f"(top-left = best).\n"
        f"Look for visually similar T poses with Δreward≈2 — "
        f"that's the 180°-flip smoking gun.",
        fontsize=11,
    )
    fig.tight_layout()
    all_path = OUT_DIR / "rollout_all_sorted.png"
    fig.savefig(all_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote: {all_path}")

    # ------------------ summary.json ------------------
    per_candidate = []
    for idx in top_idx:
        lbl = stats.labels[int(idx)]
        per_candidate.append({
            "idx": int(idx),
            "reward": round(float(stats.rewards[idx]), 4),
            "weight": round(float(stats.weights[idx]), 5),
            "cv_success": bool(lbl.success),
            "cx": round(lbl.cx, 2) if lbl.success else None,
            "cy": round(lbl.cy, 2) if lbl.success else None,
            "theta_deg": round(lbl.theta_deg, 2) if lbl.success else None,
            "icp_residual": round(lbl.icp_residual, 4) if lbl.success else None,
            "action_t0": [round(x, 4) for x in stats.actions[idx, 0].tolist()],
        })

    summary = {
        "config": {
            "N": N, "H": H, "sigma": SIGMA, "temperature": TEMPERATURE,
            "seed": SEED, "top_k": TOP_K,
            "start": "mini/pusht/train/episode_2.hdf5 t=0",
        },
        "wall_time_s": round(wall_s, 3),
        "a_star": [round(x, 6) for x in a_star.tolist()],
        "a_naive_mean": [round(x, 6) for x in stats.a_naive_mean.tolist()],
        "a_star_vs_naive_l2": round(
            float(torch.norm(a_star.cpu() - stats.a_naive_mean).item()), 6
        ),
        "reward_stats": {
            "mean": float(stats.rewards.mean()),
            "std": float(stats.rewards.std()),
            "min": float(stats.rewards.min()),
            "max": float(stats.rewards.max()),
        },
        "cv_fail_count": stats.cv_fail_count,
        "top_candidates": per_candidate,
    }
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote: {summary_path}")

    # Sanity check: a* should differ from naive mean when rewards have
    # real variance. On z_goal where all rewards are ~-0.005 this
    # difference will be tiny.
    l2 = summary["a_star_vs_naive_l2"]
    print(f"||a* - a_naive_mean||_2 = {l2:.5f}")


if __name__ == "__main__":
    main()
