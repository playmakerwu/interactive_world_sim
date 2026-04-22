"""Step 5 — full 50-step MPPI execution on PushT.

Default mode: MPPI. Sample N=128 action sequences at each step, rollout
H=10, decode, score against state_goal, softmax, execute weighted-mean
first action. --baseline mode: zero actions every step (no sampling,
no decoding of candidate trajectories), used as a control to see what
the WM does with no policy.

Artifacts (all under outputs/mppi/<run_name>/):
  trajectory.mp4              decoded executed frames
  trajectory_overlay.mp4      same frames + CV pose (green) + goal (red)
  trajectory_latents.pt       (T+1, C, H, W) executed latents
  action_history.pt           (T, action_dim) executed actions
  reward_curve.png            reward(z_t, state_goal) per step
  rollout_samples_step_XX.png  top-8 candidates at steps 0,10,20,30,40 (MPPI only)
  summary.json                config, per-step metrics, success flags
"""

from __future__ import annotations

import argparse
import json
import math
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
from rl.mppi.action_sampling import (  # noqa: E402
    DemoChunkJitterSampler,
    DemoChunkSampler,
    GaussianSampler,
)
from rl.mppi.planner import MPPIPlanner  # noqa: E402
from rl.mppi.reward import (  # noqa: E402
    DEFAULT_LARGE_PENALTY,
    IMAGE_DIAGONAL_128,
    state_reward,
)
from rl.mppi.utils import batched_rollout  # noqa: E402
from rl.visualization.state_viz import render_state_on_image  # noqa: E402

CKPT_PATH = REPO_ROOT / "outputs" / "pusht_cam1" / "checkpoints" / "best.ckpt"
STATE_GOAL_PATH = REPO_ROOT / "tests" / "goal_selection" / "state_goal.pt"
RES = 128
OBS_KEY = "camera_0_color"

SUCCESS_POS_PX = 5.0
SUCCESS_ANGLE_DEG = 10.0
SUCCESS_COS_THRESHOLD = 0.94  # cos(20°) ≈ 0.94
SNAPSHOT_STEPS = (0, 10, 20, 30, 40)


def _preprocess_rgb(raw: np.ndarray) -> np.ndarray:
    h, w = raw.shape[:2]
    s = min(h, w)
    cr = raw[(h - s) // 2 : (h - s) // 2 + s, (w - s) // 2 : (w - s) // 2 + s]
    resized = cv2.resize(cr, (RES, RES), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def _load_initial_latent(source: str, wm: DifferentiableDynamics) -> torch.Tensor:
    """Resolve a `source` spec to (1, C, H, W) latent on the WM device.

    Supported:
      z_goal                      -> tests/goal_selection/z_goal.pt
      mini/val/<ep>/<t>           -> mini val episode frame, encoded
      mini/train/<ep>/<t>         -> mini train episode frame, encoded
      full/val/<ep>/<t>           -> full val episode frame, encoded (cloud only)
      full/train/<ep>/<t>         -> full train episode frame, encoded (cloud only)
      path/to/z.pt                -> a (1, C, H, W) latent .pt
    """
    if source == "z_goal":
        return torch.load(
            REPO_ROOT / "tests" / "goal_selection" / "z_goal.pt",
            map_location=wm._device,
        )
    parts = source.split("/")
    if len(parts) == 4 and parts[0] in {"mini", "full"}:
        dataset, split, ep_str, t_str = parts
        ep_idx = int(ep_str)
        t_idx = int(t_str)
        ep_path = REPO_ROOT / "data" / dataset / "pusht" / split / f"episode_{ep_idx}.hdf5"
        if not ep_path.exists():
            raise FileNotFoundError(
                f"Initial state path {ep_path} not found. For 'full/...' "
                f"sources, the data/full/ tree is only on the cloud."
            )
        with h5py.File(ep_path, "r") as f:
            raw = f[f"obs/images/{OBS_KEY}"][t_idx]
        pre = _preprocess_rgb(raw)
        pre_t = torch.from_numpy(pre).permute(2, 0, 1).unsqueeze(0).to(wm._device)
        with torch.no_grad():
            return wm.encode(pre_t)
    p = Path(source)
    if p.exists() and p.suffix == ".pt":
        z = torch.load(p, map_location=wm._device)
        if z.dim() == 3:
            z = z.unsqueeze(0)
        return z
    raise ValueError(f"unrecognized --initial_state value: {source}")


def _overlay_goal_and_cv(
    rgb: np.ndarray, state_goal: dict, lbl, step: int, reward: float
) -> np.ndarray:
    """Chain-render goal (red) then CV (green) then a small label strip."""
    canvas = render_state_on_image(
        rgb,
        cx=state_goal["cx"],
        cy=state_goal["cy"],
        sin_theta=state_goal["sin_theta"],
        cos_theta=state_goal["cos_theta"],
        color=(220, 30, 30),
        label="goal",
    )
    if lbl.success:
        canvas = render_state_on_image(
            canvas,
            cx=lbl.cx,
            cy=lbl.cy,
            sin_theta=lbl.sin_theta,
            cos_theta=lbl.cos_theta,
            color=(30, 200, 30),
            label=f"cv {lbl.theta_deg:+.0f}°",
        )
    # Bottom-left step + reward text.
    cv2.putText(
        canvas,
        f"t={step:02d} r={reward:+.2f}",
        (2, canvas.shape[0] - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return canvas


def _write_mp4(frames_rgb: list[np.ndarray], out_path: Path, fps: int = 8) -> None:
    if not frames_rgb:
        return
    H, W = frames_rgb[0].shape[:2]
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    try:
        for frame in frames_rgb:
            vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        vw.release()


def _plot_snapshot(
    stats, state_goal: dict, step: int, out_path: Path, top_k: int = 8,
) -> None:
    """4x4 grid of all N candidates at one plan step, sorted by reward."""
    order = np.argsort(stats.rewards)[::-1]
    N = len(order)
    cols = 4
    rows = math.ceil(N / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes_flat = np.atleast_1d(axes).flatten()
    for ax_i, ax in enumerate(axes_flat):
        if ax_i >= N:
            ax.axis("off")
            continue
        idx = int(order[ax_i])
        frame = stats.decoded_rgb[idx]
        lbl = stats.labels[idx]
        if lbl.success:
            title = (
                f"#{idx}  r={stats.rewards[idx]:+.3f}  w={stats.weights[idx]:.3f}\n"
                f"({lbl.cx:.0f},{lbl.cy:.0f})  θ={lbl.theta_deg:+.0f}°"
            )
        else:
            title = f"#{idx}  r={stats.rewards[idx]:+.3f}  CV-FAIL"
        ax.imshow(frame)
        ax.set_title(title, fontsize=8.5)
        ax.axis("off")
    fig.suptitle(
        f"Plan step t={step:02d} — all {N} candidates sorted by reward", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--initial_state", default="mini/val/0/0")
    ap.add_argument(
        "--goal_path",
        default=str(STATE_GOAL_PATH.relative_to(REPO_ROOT)),
        help="Path to a state_goal .pt (same schema as tests/goal_selection/state_goal.pt)",
    )
    ap.add_argument("--baseline", action="store_true",
                    help="Zero-action control (no sampling, no search)")
    ap.add_argument(
        "--symmetry_aware", action="store_true",
        help="Use the symmetry-aware angle penalty (1 - |cos Δθ|). "
             "Treats θ and θ+180° as equivalent. Default: False.",
    )
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--H", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--control_steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--action_dim", type=int, default=4)
    ap.add_argument(
        "--action_source",
        choices=["gaussian", "demo", "demo_jitter"],
        default="gaussian",
        help="Action sampling distribution. 'gaussian' = zero-mean randn*sigma "
             "(v0 default, catastrophically OOD for IWS WM). 'demo' = draw length-H "
             "slices directly from the train action bank. 'demo_jitter' = same as "
             "'demo' plus per-step Gaussian noise at --jitter_sigma.",
    )
    ap.add_argument(
        "--demo_train_dir",
        default="data/mini/pusht/train",
        help="Directory of training episodes to build the demo action bank from.",
    )
    ap.add_argument(
        "--jitter_sigma", type=float, default=0.01,
        help="Per-step Gaussian jitter added on top of demo chunks in "
             "'demo_jitter' mode. Small vs demo step-to-step delta ≈ 0.012.",
    )
    ap.add_argument(
        "--selection_rule", choices=["softmax", "argmax"], default="softmax",
        help="How to pick the executed action from the N sampled trajectories. "
             "'softmax' = weighted mean of first actions. 'argmax' = first "
             "action of the single best trajectory. Argmax preserves the "
             "temporal coherence of individual sampled action sequences.",
    )
    ap.add_argument(
        "--warm_start", action="store_true",
        help="Maintain a running H-step action sequence across plan steps. "
             "Each step samples perturbations around this sequence and then "
             "shifts it. Provides temporal persistence between plan_step calls.",
    )
    args = ap.parse_args()

    device = "cuda:0"
    assert torch.cuda.is_available(), "CUDA required"

    out_dir = REPO_ROOT / "outputs" / "mppi" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading WM from {CKPT_PATH}")
    wm = DifferentiableDynamics(str(CKPT_PATH), device=device)

    goal_path = Path(args.goal_path)
    if not goal_path.is_absolute():
        goal_path = REPO_ROOT / goal_path
    print(f"Loading state_goal from {goal_path}")
    g = torch.load(goal_path, map_location="cpu")
    state_goal = {
        "cx": g["cx"], "cy": g["cy"],
        "sin_theta": g["sin_theta"], "cos_theta": g["cos_theta"],
    }
    state_goal_theta_deg = float(g["theta_deg"])
    labeler = CVLabeler(preset="REAL", resolution=RES)

    print(f"Resolving initial state: {args.initial_state}")
    z_current = _load_initial_latent(args.initial_state, wm)
    if z_current.dim() == 3:
        z_current = z_current.unsqueeze(0)
    assert z_current.shape == (1, 4, 32, 32)
    # Snapshot before the main loop overwrites z_current. Saved for
    # downstream tools (scripts/wm_interactive_replay.py).
    z_initial_snapshot = z_current.detach().cpu().clone()

    # Record the initial (decoded) frame + label. Initial reward uses
    # the SAME reward variant as the controller so the recorded
    # reward_curve is directly comparable across runs.
    with torch.no_grad():
        rgb0 = wm.decode(z_current, resolution=RES)
    rgb0_u8 = (rgb0.clamp(0, 1).cpu().numpy()[0] * 255).astype(np.uint8)
    rgb0_u8 = rgb0_u8.transpose(1, 2, 0)
    r0, lbl0 = state_reward(
        rgb0_u8, state_goal, labeler=labeler,
        symmetry_aware=args.symmetry_aware,
    )

    trajectory_latents: list[torch.Tensor] = [z_current[0].cpu().clone()]
    action_history: list[torch.Tensor] = []
    frames_rgb: list[np.ndarray] = [rgb0_u8]
    overlays: list[np.ndarray] = [_overlay_goal_and_cv(rgb0_u8, state_goal, lbl0, 0, r0)]
    per_step_rows: list[dict] = [{
        "t": 0, "reward": round(float(r0), 4),
        "cv_success": bool(lbl0.success),
        "cx": round(lbl0.cx, 2) if lbl0.success else None,
        "cy": round(lbl0.cy, 2) if lbl0.success else None,
        "theta_deg": round(lbl0.theta_deg, 2) if lbl0.success else None,
        "cv_fail_in_plan": None,
        "plan_wall_s": None,
    }]
    print(f"  t=00  r={r0:+.4f}  cv=({lbl0.cx:.0f},{lbl0.cy:.0f},{lbl0.theta_deg:+.1f}°)")

    # ------------ planner or baseline setup ------------
    planner: MPPIPlanner | None = None
    if not args.baseline:
        demo_dir = REPO_ROOT / args.demo_train_dir
        if args.action_source == "gaussian":
            action_sampler = GaussianSampler(sigma=args.sigma, action_dim=args.action_dim)
        elif args.action_source == "demo":
            action_sampler = DemoChunkSampler(train_dir=demo_dir)
        elif args.action_source == "demo_jitter":
            action_sampler = DemoChunkJitterSampler(
                train_dir=demo_dir, jitter_sigma=args.jitter_sigma,
            )
        else:
            raise AssertionError(args.action_source)
        print(f"Action sampler: {action_sampler}")

        planner = MPPIPlanner(
            wm, state_goal,
            N=args.N, H=args.H, sigma=args.sigma, temperature=args.temperature,
            action_dim=args.action_dim, resolution=RES, device=device,
            symmetry_aware=args.symmetry_aware,
            action_sampler=action_sampler,
            selection_rule=args.selection_rule,
            warm_start=args.warm_start,
            labeler=labeler,
            capture_rgb=True,  # snapshots at specific steps
        )

    # ------------ main loop ------------
    t0_run = time.time()
    for t in range(args.control_steps):
        plan_t0 = time.time()
        if args.baseline:
            a_star = torch.zeros(args.action_dim, device=device)
            cv_fail_in_plan = 0
        else:
            assert planner is not None
            a_star = planner.plan_step(z_current, seed=args.seed + t)
            cv_fail_in_plan = planner.last_stats.cv_fail_count

            # Snapshot at designated steps.
            if t in SNAPSHOT_STEPS:
                _plot_snapshot(
                    planner.last_stats, state_goal, t,
                    out_dir / f"rollout_samples_step_{t:02d}.png",
                    top_k=args.N,
                )

        # Execute a_star through the WM (single step).
        actions_onestep = a_star.view(1, 1, args.action_dim)
        with torch.no_grad():
            z_next_traj = batched_rollout(z_current, actions_onestep, wm)
            # z_next_traj: (1, 2, C, H, W); take z_1
            z_next = z_next_traj[:, 1]
        z_current = z_next

        # Decode & label the executed frame.
        torch.cuda.empty_cache()
        with torch.no_grad():
            rgb_t = wm.decode(z_current, resolution=RES)
        rgb_t_u8 = (rgb_t.clamp(0, 1).cpu().numpy()[0] * 255).astype(np.uint8)
        rgb_t_u8 = rgb_t_u8.transpose(1, 2, 0)
        r_t, lbl_t = state_reward(
            rgb_t_u8, state_goal, labeler=labeler,
            symmetry_aware=args.symmetry_aware,
        )

        plan_s = time.time() - plan_t0

        trajectory_latents.append(z_current[0].cpu().clone())
        action_history.append(a_star.detach().cpu().clone())
        frames_rgb.append(rgb_t_u8)
        overlays.append(_overlay_goal_and_cv(rgb_t_u8, state_goal, lbl_t, t + 1, r_t))
        per_step_rows.append({
            "t": t + 1, "reward": round(float(r_t), 4),
            "cv_success": bool(lbl_t.success),
            "cx": round(lbl_t.cx, 2) if lbl_t.success else None,
            "cy": round(lbl_t.cy, 2) if lbl_t.success else None,
            "theta_deg": round(lbl_t.theta_deg, 2) if lbl_t.success else None,
            "cv_fail_in_plan": int(cv_fail_in_plan),
            "plan_wall_s": round(plan_s, 3),
        })
        print(
            f"  t={t+1:02d}  r={r_t:+.4f}  "
            f"cv=({lbl_t.cx:.0f},{lbl_t.cy:.0f},{lbl_t.theta_deg:+.1f}°)  "
            f"plan={plan_s:.2f}s"
        )

    wall_s = time.time() - t0_run
    print(f"\nRun wall time: {wall_s:.1f} s")

    # ------------ save artifacts ------------
    torch.save(torch.stack(trajectory_latents), out_dir / "trajectory_latents.pt")
    torch.save(torch.stack(action_history), out_dir / "action_history.pt")
    # Persist the initial latent (snapshot taken before the loop) so
    # downstream tools (e.g. scripts/wm_interactive_replay.py) can load
    # it without parsing the full trajectory tensor.
    torch.save(z_initial_snapshot, out_dir / "initial_latent.pt")

    _write_mp4(frames_rgb, out_dir / "trajectory.mp4")
    _write_mp4(overlays, out_dir / "trajectory_overlay.mp4")

    # Reward curve.
    rewards = np.array([row["reward"] for row in per_step_rows])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(range(len(rewards)), rewards, marker="o", color="tab:blue")
    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.set_xlabel("control step t")
    ax.set_ylabel("reward(z_t, state_goal)")
    ax.set_title(
        f"{args.run_name}  —  executed-trajectory reward over {args.control_steps} steps"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "reward_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Success metrics on the final frame.
    final = per_step_rows[-1]
    initial = per_step_rows[0]
    final_pos = None
    final_angle_err = None
    final_cos_sim = None
    success_strict = False
    success_cos = False
    flipped_convergence = False
    if final["cv_success"] and initial["cv_success"]:
        dx = final["cx"] - state_goal["cx"]
        dy = final["cy"] - state_goal["cy"]
        final_pos = math.sqrt(dx * dx + dy * dy)
        # angle error (deg), wrapped to [-180, 180]
        a = final["theta_deg"] - state_goal_theta_deg
        final_angle_err = ((a + 180) % 360) - 180
        final_cos_sim = (
            math.sin(math.radians(final["theta_deg"]))
            * state_goal["sin_theta"]
            + math.cos(math.radians(final["theta_deg"]))
            * state_goal["cos_theta"]
        )
        success_strict = (
            final_pos <= SUCCESS_POS_PX and abs(final_angle_err) <= SUCCESS_ANGLE_DEG
        )
        # success_cos = position close AND cos-sim close.
        # cos(20°) ≈ 0.9397, so threshold 0.94 ≈ 20° tolerance.
        success_cos = final_pos <= SUCCESS_POS_PX and final_cos_sim >= SUCCESS_COS_THRESHOLD
        # Flipped convergence: position ok but angle wrapped 180° away.
        flipped_convergence = (
            final_pos <= SUCCESS_POS_PX
            and not success_strict
            and abs(abs(final_angle_err) - 180) <= SUCCESS_ANGLE_DEG
        )

    initial_pos = None
    if initial["cv_success"]:
        idx_dx = initial["cx"] - state_goal["cx"]
        idx_dy = initial["cy"] - state_goal["cy"]
        initial_pos = math.sqrt(idx_dx * idx_dx + idx_dy * idx_dy)

    summary = {
        "run_name": args.run_name,
        "config": {
            "mode": "baseline" if args.baseline else "mppi",
            "initial_state": args.initial_state,
            "goal_path": str(goal_path.relative_to(REPO_ROOT)) if goal_path.is_relative_to(REPO_ROOT) else str(goal_path),
            "symmetry_aware": bool(args.symmetry_aware),
            "action_source": args.action_source,
            "jitter_sigma": args.jitter_sigma if args.action_source == "demo_jitter" else None,
            "selection_rule": args.selection_rule,
            "warm_start": bool(args.warm_start),
            "N": args.N, "H": args.H, "sigma": args.sigma,
            "temperature": args.temperature, "seed": args.seed,
            "control_steps": args.control_steps,
        },
        "goal_state": {
            "cx": state_goal["cx"], "cy": state_goal["cy"],
            "theta_deg": state_goal_theta_deg,
        },
        "initial_state_measured": {
            "cx": initial["cx"], "cy": initial["cy"],
            "theta_deg": initial["theta_deg"],
            "reward": initial["reward"],
            "distance_to_goal_px": None if initial_pos is None else round(initial_pos, 2),
        },
        "final_state_measured": {
            "cx": final["cx"], "cy": final["cy"],
            "theta_deg": final["theta_deg"],
            "reward": final["reward"],
        },
        "final_pos_distance_px": None if final_pos is None else round(final_pos, 3),
        "final_angle_error_deg": None if final_angle_err is None else round(final_angle_err, 2),
        "final_angle_sim": None if final_cos_sim is None else round(final_cos_sim, 4),
        "success_strict": bool(success_strict),
        "success_cos": bool(success_cos),
        "flipped_convergence": bool(flipped_convergence),
        "reward_trajectory_stats": {
            "min": float(rewards.min()), "max": float(rewards.max()),
            "mean": float(rewards.mean()), "final": float(rewards[-1]),
        },
        "cv_failures_along_trajectory": sum(
            1 for row in per_step_rows if not row["cv_success"]
        ),
        "per_step": per_step_rows,
        "wall_time_s": round(wall_s, 1),
        "wall_per_step_s": round(wall_s / max(1, args.control_steps), 3),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n--- summary ---")
    print(f"success_strict = {success_strict}")
    print(f"success_cos    = {success_cos}")
    print(f"flipped_conv.  = {flipped_convergence}")
    print(f"final_pos_dist = {final_pos}")
    print(f"final_angle_err= {final_angle_err}")
    print(f"final_cos_sim  = {final_cos_sim}")


if __name__ == "__main__":
    main()
