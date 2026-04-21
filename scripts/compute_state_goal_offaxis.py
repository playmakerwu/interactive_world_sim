"""Produce tests/goal_selection/state_goal_offaxis.pt.

Selects one frame from mini val episodes whose CV-detected T-block θ lies
in [30°, 60°] or [-60°, -30°] — i.e. clearly off-axis so the bar-vs-stem
ICP initialization is unambiguous and the 180° flip degeneracy does not
apply. Selection is by lowest ICP residual among candidates that are
also near the image center (so the MPPI controller has room on all
sides).

Written as the Step 5 follow-up diagnostic companion to
`compute_state_goal.py` (the axis-aligned goal). Same output schema so
downstream MPPI code can load either interchangeably via the
`--goal_path` flag on `run_mppi.py`.

data/full/pusht/val is not available locally; this script falls back to
data/mini/pusht/val. Cloud runs would point the same logic at the full
val split and the picked frame may differ.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.labeling.cv_labeler import CVLabeler  # noqa: E402
from rl.models.world_model import DifferentiableDynamics  # noqa: E402
from rl.visualization.state_viz import render_state_on_image  # noqa: E402

CKPT_PATH = REPO_ROOT / "outputs" / "pusht_cam1" / "checkpoints" / "best.ckpt"
VAL_DIR_LOCAL = REPO_ROOT / "data" / "mini" / "pusht" / "val"
VAL_DIR_FULL = REPO_ROOT / "data" / "full" / "pusht" / "val"
OUT_PT = REPO_ROOT / "tests" / "goal_selection" / "state_goal_offaxis.pt"
OUT_OVERLAY = REPO_ROOT / "tests" / "goal_selection" / "state_goal_offaxis_overlay.png"

RES = 128
OBS_KEY = "camera_0_color"
THETA_ABS_MIN = 30.0
THETA_ABS_MAX = 60.0
SCAN_STRIDE = 5  # evaluate every 5th frame for speed
MAX_DIST_FROM_CENTER = 30.0  # prefer T near image center


def _preprocess_rgb(raw: np.ndarray) -> np.ndarray:
    h, w = raw.shape[:2]
    s = min(h, w)
    cr = raw[(h - s) // 2 : (h - s) // 2 + s, (w - s) // 2 : (w - s) // 2 + s]
    return cv2.resize(cr, (RES, RES), interpolation=cv2.INTER_AREA)


def _scan_val(val_dir: Path, labeler: CVLabeler) -> list:
    """Return list of (icp_residual, dist_from_center, ep, t, cx, cy, theta_deg,
    sin_theta, cos_theta, rgb_u8) for frames meeting the off-axis criteria.
    Sorted by residual ascending."""
    candidates = []
    eps = sorted(val_dir.glob("episode_*.hdf5"))
    print(f"Scanning {len(eps)} episodes in {val_dir}")
    for ep_path in eps:
        ep_idx = int(ep_path.stem.split("_")[1])
        with h5py.File(ep_path, "r") as f:
            frames = f[f"obs/images/{OBS_KEY}"][()]
        for t in range(0, frames.shape[0], SCAN_STRIDE):
            rgb = _preprocess_rgb(frames[t])
            label = labeler.label(rgb)
            if not label.success:
                continue
            if not (THETA_ABS_MIN <= abs(label.theta_deg) <= THETA_ABS_MAX):
                continue
            dist = math.sqrt((label.cx - 64.0) ** 2 + (label.cy - 64.0) ** 2)
            if dist > MAX_DIST_FROM_CENTER:
                continue
            candidates.append((
                label.icp_residual, dist, ep_idx, t,
                label.cx, label.cy, label.theta_deg,
                label.sin_theta, label.cos_theta,
                rgb,
            ))
    candidates.sort(key=lambda x: (x[0], x[1]))  # low residual, then near-center
    return candidates


def main() -> None:
    val_dir = VAL_DIR_FULL if VAL_DIR_FULL.exists() else VAL_DIR_LOCAL
    if not val_dir.exists():
        raise FileNotFoundError(
            f"No val split found at {VAL_DIR_FULL} or {VAL_DIR_LOCAL}"
        )

    labeler = CVLabeler(preset="REAL", resolution=RES)
    candidates = _scan_val(val_dir, labeler)
    if not candidates:
        raise RuntimeError(
            "No off-axis candidate frame found in the val split. Widen "
            "THETA_ABS_MIN/MAX or MAX_DIST_FROM_CENTER."
        )
    print(f"\nTop 5 off-axis candidates:")
    for rec in candidates[:5]:
        res, dc, ep, t, cx, cy, th, _, _, _ = rec
        print(f"  ep={ep} t={t:3d}  ({cx:.1f},{cy:.1f},{th:+.1f}°)  "
              f"resid={res:.3f}  dist_center={dc:.1f}")

    # Pick the best (lowest residual, then nearest center).
    picked = candidates[0]
    res, dist, ep_idx, t_idx, cx, cy, theta_deg, sin_t, cos_t, rgb = picked
    print(f"\nPicked: ep={ep_idx} t={t_idx}  ({cx:.2f}, {cy:.2f}, {theta_deg:+.2f}°)")
    print(f"        icp_residual={res:.3f}, dist_from_center={dist:.2f}")

    # Encode the picked frame to a latent so downstream MPPI can use it as an
    # initial state if desired. Same pattern as compute_state_goal.py.
    print(f"\nLoading WM from {CKPT_PATH}")
    wm = DifferentiableDynamics(str(CKPT_PATH), device="cuda:0")

    rgb_t = torch.from_numpy(rgb.astype(np.float32) / 255.0)
    rgb_t = rgb_t.permute(2, 0, 1).unsqueeze(0).to("cuda:0")
    with torch.no_grad():
        z_goal = wm.encode(rgb_t)
    print(f"  encoded z_goal shape: {tuple(z_goal.shape)}")

    # Build the state_goal dict, same schema as compute_state_goal.py output.
    theta_rad = math.radians(theta_deg)
    state = torch.tensor(
        [cx, cy, sin_t, cos_t], dtype=torch.float32
    )
    out = {
        "state": state,
        "cx": float(cx),
        "cy": float(cy),
        "sin_theta": float(sin_t),
        "cos_theta": float(cos_t),
        "theta_rad": float(theta_rad),
        "theta_deg": float(theta_deg),
        "resolution": RES,
        "meta": {
            "preset": "REAL",
            "hsv_lower": labeler.hsv_lower.tolist(),
            "hsv_upper": labeler.hsv_upper.tolist(),
            "source_val_dir": str(val_dir.relative_to(REPO_ROOT)),
            "source_episode": ep_idx,
            "source_frame_t": t_idx,
            "ckpt": str(CKPT_PATH),
            "icp_residual": float(res),
            "dist_from_image_center_px": float(dist),
        },
    }
    OUT_PT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, OUT_PT)
    # Also save the encoded z for optional use as initial state.
    z_out_path = OUT_PT.with_name("z_goal_offaxis.pt")
    torch.save(z_goal.detach().cpu(), z_out_path)
    print(f"Wrote: {OUT_PT}")
    print(f"Wrote: {z_out_path}")

    # Overlay: draw the CV pose on the source RGB and save for sanity check.
    overlay = render_state_on_image(
        rgb, cx=cx, cy=cy, sin_theta=sin_t, cos_theta=cos_t,
        color=(30, 200, 30), label=f"goal θ={theta_deg:+.0f}°",
    )
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    overlay_4x = cv2.resize(
        overlay_bgr, (RES * 4, RES * 4), interpolation=cv2.INTER_NEAREST
    )
    cv2.imwrite(str(OUT_OVERLAY), overlay_4x)
    print(f"Wrote: {OUT_OVERLAY}")


if __name__ == "__main__":
    main()
