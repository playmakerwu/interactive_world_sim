"""Produce tests/goal_selection/state_goal.pt.

Decodes z_goal through the WM, runs the CV labeler with the Phase 3-A
approved REAL preset, and persists the (cx, cy, sin theta, cos theta)
state vector plus metadata. Matches the format produced by
scripts/label_replay_buffer.py so Branch B's reward code can load
`state_goal` and label tensors with the same schema.

Also saves a visualization `tests/goal_selection/state_goal_overlay.png`
using the §6.1 primitive.

HARD precondition: CV must succeed on the goal frame. If it does not,
this script fails loudly — the goal frame IS the reward target, and an
unlabeled goal is a blocker for Branch B.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.labeling.cv_labeler import CVLabeler  # noqa: E402
from rl.models.world_model import DifferentiableDynamics  # noqa: E402
from rl.visualization.state_viz import render_state_on_image  # noqa: E402

CKPT_PATH = REPO_ROOT / "outputs" / "pusht_cam1" / "checkpoints" / "best.ckpt"
Z_GOAL_PATH = REPO_ROOT / "tests" / "goal_selection" / "z_goal.pt"
STATE_GOAL_PATH = REPO_ROOT / "tests" / "goal_selection" / "state_goal.pt"
OVERLAY_PATH = REPO_ROOT / "tests" / "goal_selection" / "state_goal_overlay.png"

RESOLUTION = 128
DEVICE = "cuda:0"
PRESET = "REAL"


def main() -> None:
    assert Z_GOAL_PATH.exists(), f"z_goal missing at {Z_GOAL_PATH}"
    assert CKPT_PATH.exists(), f"WM ckpt missing at {CKPT_PATH}"

    free, _ = torch.cuda.mem_get_info(0)
    print(f"free GPU mem before WM load: {free / 1024**2:.0f} MiB")

    t0 = time.time()
    wm = DifferentiableDynamics(str(CKPT_PATH), device=DEVICE)
    print(f"WM loaded in {time.time() - t0:.1f}s")

    z_goal = torch.load(Z_GOAL_PATH, map_location=DEVICE, weights_only=False)
    print(f"z_goal shape={tuple(z_goal.shape)} norm={z_goal.float().flatten().norm().item():.2f}")

    with torch.no_grad():
        rgb = wm.decode(z_goal.to(DEVICE), RESOLUTION)  # (1, 3, H, W) in [0,1]
    rgb_np = rgb[0].permute(1, 2, 0).detach().cpu().float().numpy()
    rgb_u8 = np.clip(rgb_np * 255.0, 0, 255).astype(np.uint8)

    labeler = CVLabeler(preset=PRESET, resolution=RESOLUTION)
    result = labeler.label(rgb_u8)

    if not result.success:
        # HARD precondition per the kickoff §Step 5
        raise RuntimeError(
            f"CV failed on z_goal — cannot produce state_goal.pt.\n"
            f"contour_count={result.contour_count} "
            f"contour_area={result.contour_area:.1f} "
            f"icp_residual={result.icp_residual}\n"
            f"The goal frame MUST label successfully. Investigate z_goal "
            f"before re-running."
        )

    # Persist as a (4,) tensor matching the column order used in labels_*.pt:
    #   [cx, cy, sin theta, cos theta]
    state_vec = torch.tensor(
        [result.cx, result.cy, result.sin_theta, result.cos_theta],
        dtype=torch.float32,
    )

    payload = {
        "state": state_vec,                      # (4,) float32
        "cx": float(result.cx),
        "cy": float(result.cy),
        "sin_theta": float(result.sin_theta),
        "cos_theta": float(result.cos_theta),
        "theta_rad": float(result.theta_rad),
        "theta_deg": float(result.theta_deg),
        "resolution": RESOLUTION,
        "meta": {
            "preset": PRESET,
            "hsv_lower": labeler.hsv_lower.tolist(),
            "hsv_upper": labeler.hsv_upper.tolist(),
            "ckpt": str(CKPT_PATH),
            "z_goal_source": str(Z_GOAL_PATH),
            "contour_area": float(result.contour_area),
            "icp_residual": float(result.icp_residual),
        },
    }
    STATE_GOAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, STATE_GOAL_PATH)

    # Visualisation
    overlay = render_state_on_image(
        rgb_u8,
        cx=result.cx,
        cy=result.cy,
        sin_theta=result.sin_theta,
        cos_theta=result.cos_theta,
        color=(0, 220, 0),
        label="goal",
    )
    upscaled = cv2.resize(
        overlay, (overlay.shape[1] * 4, overlay.shape[0] * 4),
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.imwrite(str(OVERLAY_PATH), cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR))

    print()
    print("=" * 60)
    print("STATE GOAL EXTRACTED")
    print("=" * 60)
    print(f"preset:        {PRESET}")
    print(f"cx, cy:        ({result.cx:.3f}, {result.cy:.3f}) px")
    print(f"theta_deg:     {result.theta_deg:+.3f}  (sin {result.sin_theta:+.4f}, cos {result.cos_theta:+.4f})")
    print(f"contour area:  {result.contour_area:.1f} px")
    print(f"ICP residual:  {result.icp_residual:.4f}")
    print(f"state tensor:  {STATE_GOAL_PATH}")
    print(f"overlay PNG:   {OVERLAY_PATH}")


if __name__ == "__main__":
    main()
