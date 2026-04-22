"""WM-replay diagnostic for an MPPI run's executed action sequence.

Loads `initial_latent.pt` and `action_history.pt` from a Step 5 run dir,
rolls the IWS world model forward one action at a time, decodes each
intermediate latent, runs the classical CV labeler, and dumps per-step
artefacts so the user can step through the trajectory in a file browser.

Outputs (under --output_dir):
  step_NN.png               decoded RGB at step N (upscaled 2x for viewing)
  step_NN_meta.json         action, cv state, latent diagnostics
  replay.mp4                stitched 2-fps video
  action_norms.png          per-step action L2 norm
  latent_drift.png          cos-sim(z_t, z_0) and cos-sim(z_t, z_{t-1}) vs t
  summary.json              first-bad-step thresholds + aggregate stats

Optional --override_from_step + --override_actions {zero,small,<path>} lets
the user replace actions from step N onward to probe counter-factuals.

Note on stochasticity: the WM dynamics inject fresh CUDA noise per
denoising step, so replay traces will not bit-match the original MPPI
execution unless you re-seed exactly. We seed once at startup with
--seed (default 0) so the replay itself is reproducible across runs of
this script. Qualitative trends (when CV breaks, when arms vanish, the
latent-drift slope) carry over regardless.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.labeling.cv_labeler import CVLabeler  # noqa: E402
from rl.models.world_model import DifferentiableDynamics  # noqa: E402
from rl.mppi.utils import batched_rollout  # noqa: E402

CKPT_PATH = REPO_ROOT / "outputs" / "pusht_cam1" / "checkpoints" / "best.ckpt"
RES = 128
DEFAULT_FPS = 2
UPSCALE = 2  # per-step PNG upscale factor for browsing


def _load_z0(path: Path, device: str) -> torch.Tensor:
    z = torch.load(path, weights_only=False, map_location=device)
    if z.dim() == 3:
        z = z.unsqueeze(0)
    if z.dim() != 4 or z.shape[0] != 1:
        raise ValueError(f"z0 must be (1, C, H, W); got {tuple(z.shape)}")
    return z.float()


def _load_actions(path: Path, device: str) -> torch.Tensor:
    a = torch.load(path, weights_only=False, map_location=device)
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if a.dim() != 2:
        raise ValueError(f"actions must be (T, A); got {tuple(a.shape)}")
    return a.float()


def _override_actions(
    actions: torch.Tensor,
    from_step: int,
    mode: str,
) -> torch.Tensor:
    """Replace `actions[from_step:]` according to the override mode."""
    if from_step >= actions.shape[0]:
        return actions
    out = actions.clone()
    tail = out[from_step:]
    if mode == "zero":
        out[from_step:] = torch.zeros_like(tail)
    elif mode == "small":
        out[from_step:] = tail * 0.1
    else:
        # treat as path
        custom = torch.load(Path(mode), weights_only=False, map_location=actions.device)
        custom = custom.float()
        if custom.dim() != 2 or custom.shape[1] != actions.shape[1]:
            raise ValueError(
                f"override .pt must be (T_remaining, A); got {tuple(custom.shape)}"
            )
        n_remaining = actions.shape[0] - from_step
        if custom.shape[0] != n_remaining:
            raise ValueError(
                f"override .pt has {custom.shape[0]} steps but {n_remaining} needed"
            )
        out[from_step:] = custom
    return out


def _decode_one(wm: DifferentiableDynamics, z: torch.Tensor) -> np.ndarray:
    """(1, C, H, W) -> (H, W, 3) uint8 RGB."""
    with torch.no_grad():
        rgb = wm.decode(z, resolution=RES)
    arr = (rgb.clamp(0, 1).cpu().numpy()[0] * 255).astype(np.uint8)
    return arr.transpose(1, 2, 0)


def _save_upscaled_png(rgb: np.ndarray, path: Path) -> None:
    big = cv2.resize(
        rgb, (RES * UPSCALE, RES * UPSCALE), interpolation=cv2.INTER_NEAREST
    )
    cv2.imwrite(str(path), cv2.cvtColor(big, cv2.COLOR_RGB2BGR))


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0))[0])


def _write_mp4(frames: list[np.ndarray], path: Path, fps: int) -> None:
    if not frames:
        return
    H, W = frames[0].shape[:2]
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, fps, (W, H))
    try:
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    finally:
        vw.release()


def run_replay(
    z0: torch.Tensor,
    actions: torch.Tensor,
    wm: DifferentiableDynamics,
    labeler: CVLabeler,
    output_dir: Path,
    fps: int = DEFAULT_FPS,
) -> dict:
    """Drive the WM through `actions` from `z0`, dump per-step artefacts.

    Returns the summary dict (also written to output_dir/summary.json).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device = z0.device
    T = actions.shape[0]
    A = actions.shape[1]
    print(f"Rolling WM: z0 shape {tuple(z0.shape)}, actions {tuple(actions.shape)}")

    # Single forward call. batched_rollout expects (B, H, A); here B=1, H=T.
    with torch.no_grad():
        latents = batched_rollout(z0, actions.unsqueeze(0), wm)
    # latents: (1, T+1, C, H, W) -> drop batch dim
    latents = latents[0].cpu()  # (T+1, C, H, W)
    z0_cpu = z0[0].cpu()

    frames: list[np.ndarray] = []
    rows: list[dict] = []
    prev_latent = None

    for t in range(T + 1):
        z_t = latents[t : t + 1].to(device)  # (1, C, H, W) for decode
        rgb = _decode_one(wm, z_t)
        png_path = output_dir / f"step_{t:02d}.png"
        _save_upscaled_png(rgb, png_path)
        frames.append(rgb)

        label = labeler.label(rgb)
        latent_norm = float(latents[t].norm().item())
        cos_z0 = _cosine_sim(latents[t], z0_cpu)
        cos_prev = (
            _cosine_sim(latents[t], prev_latent) if prev_latent is not None else None
        )
        prev_latent = latents[t].clone()

        if t == 0:
            action_vec = None
            action_norm = None
        else:
            action_vec = actions[t - 1].cpu().tolist()
            action_norm = float(actions[t - 1].norm().item())

        meta = {
            "t": t,
            "action": action_vec,
            "action_norm": action_norm,
            "cv_state": (
                {
                    "cx": label.cx, "cy": label.cy,
                    "sin_theta": label.sin_theta, "cos_theta": label.cos_theta,
                    "theta_deg": label.theta_deg,
                } if label.success else None
            ),
            "cv_success": bool(label.success),
            "contour_area": float(label.contour_area) if label.success else None,
            "icp_residual": (
                float(label.icp_residual) if label.success else None
            ),
            "latent_norm": latent_norm,
            "latent_cosine_sim_to_z0": cos_z0,
            "latent_cosine_sim_to_prev": cos_prev,
        }
        meta_path = output_dir / f"step_{t:02d}_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        rows.append(meta)
        if t % 5 == 0 or not label.success:
            extra = ""
            if label.success:
                extra = (
                    f"  cv=({label.cx:.1f},{label.cy:.1f},{label.theta_deg:+.1f}°)"
                )
            print(
                f"  t={t:02d}  ||a||={action_norm if action_norm is None else f'{action_norm:.4f}'}"
                f"  cos(z0)={cos_z0:+.4f}  norm={latent_norm:.2f}{extra}"
                + ("" if label.success else "  CV-FAIL")
            )

    # Stitched MP4 (use the un-upscaled 128x128 frames so the file stays tiny).
    _write_mp4(frames, output_dir / "replay.mp4", fps)

    # Diagnostic plots.
    ts = np.arange(T + 1)
    action_norms = [r["action_norm"] for r in rows]
    drift_z0 = [r["latent_cosine_sim_to_z0"] for r in rows]
    drift_prev = [r["latent_cosine_sim_to_prev"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(ts[1:], [a for a in action_norms[1:]], marker="o", color="tab:blue", linewidth=1.5)
    ax.set_xlabel("step t")
    ax.set_ylabel("||a_t||  (L2 norm)")
    ax.set_title("Per-step executed action norm")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "action_norms.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(ts, drift_z0, marker="o", color="tab:red", label="cos(z_t, z_0)")
    drift_prev_plot = [v if v is not None else 1.0 for v in drift_prev]
    ax.plot(ts, drift_prev_plot, marker="x", color="tab:gray",
            label="cos(z_t, z_{t-1})", alpha=0.7)
    ax.axhline(0.95, color="orange", linewidth=0.7, linestyle="--", label="0.95")
    ax.axhline(0.90, color="red", linewidth=0.7, linestyle="--", label="0.90")
    ax.set_xlabel("step t")
    ax.set_ylabel("cosine similarity")
    ax.set_title("Latent drift over rollout")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "latent_drift.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Aggregate summary.
    cv_fail_first = next((r["t"] for r in rows if not r["cv_success"]), None)
    drift_below_95 = next(
        (r["t"] for r in rows if r["latent_cosine_sim_to_z0"] < 0.95), None
    )
    drift_below_90 = next(
        (r["t"] for r in rows if r["latent_cosine_sim_to_z0"] < 0.90), None
    )
    summary = {
        "n_steps": T,
        "first_cv_fail_step": cv_fail_first,
        "n_cv_fails": sum(1 for r in rows if not r["cv_success"]),
        "first_drift_below_0_95": drift_below_95,
        "first_drift_below_0_90": drift_below_90,
        "min_cos_z0": min(drift_z0),
        "final_cos_z0": drift_z0[-1],
        "min_latent_norm": min(r["latent_norm"] for r in rows),
        "max_latent_norm": max(r["latent_norm"] for r in rows),
        "min_action_norm": (
            min(a for a in action_norms if a is not None) if T > 0 else None
        ),
        "max_action_norm": (
            max(a for a in action_norms if a is not None) if T > 0 else None
        ),
        "mean_action_norm": (
            float(np.mean([a for a in action_norms if a is not None]))
            if T > 0 else None
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nReplay summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--z0", required=True, help="Path to initial_latent.pt (1, C, H, W)")
    ap.add_argument("--actions", required=True, help="Path to action_history.pt (T, A)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=0,
                    help="Global CUDA seed for reproducible replay")
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--override_from_step", type=int, default=None)
    ap.add_argument("--override_actions", type=str, default=None,
                    help="zero | small | path to .pt (T_remaining, A)")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required for the WM"
    device = "cuda:0"
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"Loading WM from {CKPT_PATH}")
    wm = DifferentiableDynamics(str(CKPT_PATH), device=device)
    labeler = CVLabeler(preset="REAL", resolution=RES)

    z0 = _load_z0(Path(args.z0), device=device)
    actions = _load_actions(Path(args.actions), device=device)

    if args.override_from_step is not None:
        if args.override_actions is None:
            raise ValueError(
                "--override_from_step requires --override_actions {zero|small|<.pt>}"
            )
        actions = _override_actions(
            actions, args.override_from_step, args.override_actions,
        )
        print(
            f"Overrode actions from step {args.override_from_step} "
            f"with mode={args.override_actions}"
        )

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    run_replay(z0, actions, wm, labeler, output_dir, fps=args.fps)


if __name__ == "__main__":
    main()
