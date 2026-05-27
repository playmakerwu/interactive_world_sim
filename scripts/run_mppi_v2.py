"""Run the refactored MPPI (rl.mppi.mppi_planner) over a control episode.

Thin wrapper: all heavy logic lives in env.PushTWMEnv and rl.mppi.MPPIPlanner.
Only responsibility here is glue + artifact dumping.

Usage:

    python scripts/run_mppi_v2.py \\
        --config configs/mppi/default.yaml \\
        --initial_hdf5 data/mini/pusht/val/episode_0.hdf5 \\
        --initial_frame 0 \\
        --goal tests/goal_selection/state_goal.pt \\
        --output_dir outputs/mppi/refactor_sanity \\
        --wm_ckpt outputs/pusht_cam1/checkpoints/best.ckpt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.pusht_wm_env import PushTWMEnv  # noqa: E402
from rl.mppi import distributed as D  # noqa: E402
from rl.mppi.mppi_planner import MPPIPlanner  # noqa: E402
from scripts.run_config import USE_WARM_START  # noqa: E402


def _apply_cli_overrides(cfg, args) -> dict:
    """Apply CLI overrides to the loaded OmegaConf in place.

    Returns a ``config_deviation`` dict (always with the same shape).
    ``--n_sample`` and ``--n_update_iter`` count as algorithm deviations
    requiring an explicit ``--override_reason``; ``--control_steps`` and ``--seed``
    are legitimate per-run knobs (episode length and which run variant)
    that don't change MPPI's algorithmic behavior.

    Split out so tests can exercise override logic without loading the
    WM or constructing PushTWMEnv.
    """
    config_deviation = {"changed": [], "reason": None, "expected_impact_quantified": None}

    if getattr(args, "control_steps", None) is not None:
        cfg.control_steps = int(args.control_steps)
    if getattr(args, "seed", None) is not None:
        cfg.seed = int(args.seed)
    if getattr(args, "decode_batch_size", None) is not None:
        cfg.decode_batch_size = int(args.decode_batch_size)
    if getattr(args, "cv_n_workers", None) is not None:
        # Infra/perf knob — does not change the CV pipeline numerically
        # (label_batch is bit-exact at any worker count). Mutate cfg
        # silently, no override_reason required.
        cfg.cv_n_workers = int(args.cv_n_workers)

    if getattr(args, "n_sample", None) is not None and int(args.n_sample) != int(cfg.n_sample):
        config_deviation["changed"].append(
            f"n_sample: {int(cfg.n_sample)} -> {int(args.n_sample)}"
        )
        cfg.n_sample = int(args.n_sample)
    if (
        getattr(args, "n_update_iter", None) is not None
        and int(args.n_update_iter) != int(cfg.n_update_iter)
    ):
        config_deviation["changed"].append(
            f"n_update_iter: {int(cfg.n_update_iter)} -> {int(args.n_update_iter)}"
        )
        cfg.n_update_iter = int(args.n_update_iter)
    if getattr(args, "horizon", None) is not None:
        h = int(args.horizon)
        if h < 1 or h > 50:
            raise ValueError(
                f"--horizon must be in [1, 50] (drift has only been "
                f"quantified up to 50; see env.pusht_wm_env.MAX_HORIZON). "
                f"Got {h}."
            )
        if h != int(cfg.n_look_ahead):
            config_deviation["changed"].append(
                f"n_look_ahead: {int(cfg.n_look_ahead)} -> {h}"
            )
            cfg.n_look_ahead = h

    if config_deviation["changed"]:
        if not getattr(args, "override_reason", None):
            raise SystemExit(
                "ERROR: algorithm-level override used "
                "without --override_reason. Every config deviation must be "
                "justified in writing so the audit trail in summary.json "
                "explains why."
            )
        config_deviation["reason"] = args.override_reason
        config_deviation["expected_impact_quantified"] = (
            getattr(args, "expected_impact", None) or
            "This run changed one or more algorithm-level MPPI knobs for "
            "runtime or ablation reasons. Empirical performance may differ "
            "from the configured-default MPPI setting. Replication at the "
            "configured defaults is recommended for paper numbers."
        )
    return config_deviation


def _debug_dump_init(
    debug_dir: Path,
    cfg: Any,
    args: argparse.Namespace,
    env: "PushTWMEnv",
    z0: torch.Tensor,
    goal: dict,
) -> None:
    """Write the run-start debug artifacts: environment / seeds / config /
    init_state / wm_ckpt_meta."""
    import hashlib
    import platform
    import subprocess

    # environment.json
    env_info: dict[str, Any] = {
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        env_info["cuda_version"] = torch.version.cuda
        env_info["gpu_name"] = torch.cuda.get_device_name(0)
        env_info["cudnn_version"] = torch.backends.cudnn.version()
        env_info["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        env_info["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
        env_info["n_visible_gpus"] = torch.cuda.device_count()
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
        ).decode().strip()
        env_info["git_sha"] = sha
    except Exception as e:
        env_info["git_sha"] = f"unavailable: {e}"
    try:
        diff = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(REPO_ROOT),
        ).decode().strip()
        env_info["git_dirty"] = bool(diff)
    except Exception:
        env_info["git_dirty"] = None
    with open(debug_dir / "environment.json", "w") as f:
        json.dump(env_info, f, indent=2)

    # seeds.json — initial RNG states at run start. Hex-encoded raw bytes.
    seeds: dict[str, Any] = {
        "cfg_seed": int(getattr(cfg, "seed", 0)),
        "torch_initial_seed": torch.initial_seed(),
        "torch_rng_state_cpu_hex": torch.get_rng_state().numpy().tobytes().hex(),
        "numpy_rng_state_repr": str(np.random.get_state()),
    }
    if torch.cuda.is_available():
        try:
            seeds["torch_cuda_rng_state_hex"] = (
                torch.cuda.get_rng_state(env.device).numpy().tobytes().hex()
            )
        except Exception as e:
            seeds["torch_cuda_rng_state_hex"] = f"unavailable: {e}"
    with open(debug_dir / "seeds.json", "w") as f:
        json.dump(seeds, f, indent=2)

    # resolved_config.yaml — fully resolved OmegaConf
    with open(debug_dir / "resolved_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    # wm_ckpt_meta.json
    ckpt_path = Path(args.wm_ckpt)
    md5 = hashlib.md5()
    try:
        with open(ckpt_path, "rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                md5.update(chunk)
        ckpt_meta = {
            "path": str(ckpt_path),
            "size_bytes": ckpt_path.stat().st_size,
            "md5": md5.hexdigest(),
        }
    except Exception as e:
        ckpt_meta = {"path": str(ckpt_path), "error": str(e)}
    with open(debug_dir / "wm_ckpt_meta.json", "w") as f:
        json.dump(ckpt_meta, f, indent=2)

    # init_state.npz — z0 + decoded init RGB + goal
    rgb0 = env.decode(z0)
    rgb0_u8 = (rgb0.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
    payload: dict[str, Any] = {
        "z0_latent": z0.detach().cpu().float().numpy(),
        "init_rgb": rgb0_u8,
        "goal_cx": np.float32(goal.get("cx", 0.0)),
        "goal_cy": np.float32(goal.get("cy", 0.0)),
        "goal_theta_deg": np.float32(goal.get("theta_deg", 0.0)),
        "initial_frame": np.int32(args.initial_frame),
        "initial_hdf5": np.array(str(args.initial_hdf5)),
    }
    tmp = debug_dir / ".init_state.partial.npz"
    np.savez(tmp, **payload)
    tmp.replace(debug_dir / "init_state.npz")


def _debug_dump_executed_step(
    plan_dir: Path,
    executed_actions: np.ndarray,
    z_before: np.ndarray | None,
    z_after: np.ndarray,
    rgb_before: np.ndarray | None,
    rgb_after: np.ndarray | None,
    state_before: dict | None,
    state_after: dict | None,
    plan_wall_s: float,
    converged_full: np.ndarray,
) -> None:
    """Dump per-plan-call executed action + frames + CV pose before/after."""
    def _state_to_arr(s: dict | None) -> np.ndarray:
        if s is None or not s.get("cv_success"):
            return np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        return np.array(
            [s.get("cx") or np.nan, s.get("cy") or np.nan,
             s.get("theta_deg") or np.nan], dtype=np.float32,
        )

    payload: dict[str, Any] = {
        "executed_actions": executed_actions.astype(np.float32),
        "converged_full_plan": converged_full.astype(np.float32),
        "z_after": z_after.astype(np.float32),
        "cv_pose_before_cxcy_theta": _state_to_arr(state_before),
        "cv_pose_after_cxcy_theta": _state_to_arr(state_after),
        "plan_wall_s": np.float32(plan_wall_s),
        "reward_after": np.float32(
            state_after.get("reward", float("nan"))
            if state_after else float("nan"),
        ),
    }
    if z_before is not None:
        payload["z_before"] = z_before.astype(np.float32)
    if rgb_before is not None:
        payload["rgb_before"] = np.asarray(rgb_before, dtype=np.uint8)
    if rgb_after is not None:
        payload["rgb_after"] = np.asarray(rgb_after, dtype=np.uint8)
    out = plan_dir / "executed_step.npz"
    tmp = plan_dir / ".executed_step.partial.npz"
    np.savez(tmp, **payload)
    tmp.replace(out)


def _debug_dump_manifest(debug_dir: Path) -> None:
    """Walk the debug dir and write manifest.json listing every file +
    size + total."""
    files = []
    total = 0
    for p in sorted(debug_dir.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            sz = p.stat().st_size
            files.append({
                "path": str(p.relative_to(debug_dir)),
                "size_bytes": sz,
            })
            total += sz
    manifest = {"total_bytes": total, "total_mb": round(total / (1024 ** 2), 2),
                "n_files": len(files), "files": files}
    with open(debug_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def _write_mp4(frames: list[np.ndarray], out_path: Path, fps: int = 8) -> None:
    if not frames:
        return
    H, W = frames[0].shape[:2]
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    try:
        for frame in frames:
            vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        vw.release()


ITER_HEATMAP_STATS = {
    "reward_softmax_weighted": "softmax-weighted mean reward",
    "reward_max": "best reward",
    "reward_mean": "sample mean reward",
    "reward_min": "worst reward",
    "reward_std": "sample reward std",
}


def _distance_to_goal_px(row: dict[str, Any], goal: dict[str, Any]) -> float | None:
    if not row.get("cv_success") or row.get("cx") is None or row.get("cy") is None:
        return None
    return math.sqrt((float(row["cx"]) - float(goal["cx"])) ** 2 +
                     (float(row["cy"]) - float(goal["cy"])) ** 2)


def _iteration_metric_matrix(
    iteration_logs: list[list[dict[str, Any]]],
    metric: str,
) -> np.ndarray:
    if metric not in ITER_HEATMAP_STATS:
        raise ValueError(
            f"unknown iteration heatmap metric {metric!r}; "
            f"choose one of {sorted(ITER_HEATMAP_STATS)}"
        )
    if not iteration_logs:
        return np.empty((0, 0), dtype=np.float32)

    n_steps = len(iteration_logs)
    n_iter = max((len(step_log) for step_log in iteration_logs), default=0)
    mat = np.full((n_iter, n_steps), np.nan, dtype=np.float32)
    for step_idx, step_log in enumerate(iteration_logs):
        for fallback_iter, rec in enumerate(step_log):
            iter_idx = int(rec.get("iter", fallback_iter))
            if 0 <= iter_idx < n_iter:
                mat[iter_idx, step_idx] = float(rec[metric])
    return mat


def _plot_plan_step_iteration_trace(
    step_idx: int,
    step_log: list[dict[str, Any]],
    final_row: dict[str, Any],
    goal: dict[str, Any],
    out_path: Path,
) -> None:
    if not step_log:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    iterations = np.array([int(rec["iter"]) for rec in step_log], dtype=int)
    softmax_reward = np.array(
        [float(rec["reward_softmax_weighted"]) for rec in step_log], dtype=np.float32
    )
    reward_max = np.array([float(rec["reward_max"]) for rec in step_log], dtype=np.float32)
    reward_mean = np.array([float(rec["reward_mean"]) for rec in step_log], dtype=np.float32)
    reward_std = np.array([float(rec["reward_std"]) for rec in step_log], dtype=np.float32)

    final_reward = final_row.get("reward")
    final_reward_str = "nan" if final_reward is None else f"{float(final_reward):+.4f}"
    dist = _distance_to_goal_px(final_row, goal)
    dist_str = "nan" if dist is None else f"{dist:.1f}"

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    fig.suptitle(f"Plan step {step_idx:03d}: iteration reward trace", fontsize=13, y=0.98)
    ax.set_title(
        f"final action reward = {final_reward_str}, distance to goal = {dist_str} px",
        fontsize=9,
        color="dimgray",
        pad=8,
    )

    ax.fill_between(
        iterations,
        reward_mean - reward_std,
        reward_mean + reward_std,
        color="tab:blue",
        alpha=0.14,
        label="sample mean +/- 1 std",
        linewidth=0,
    )
    ax.plot(
        iterations,
        softmax_reward,
        color="tab:blue",
        linewidth=2.4,
        marker="o",
        label="softmax-weighted mean",
    )
    ax.plot(
        iterations,
        reward_max,
        color="tab:cyan",
        linewidth=1.8,
        linestyle="--",
        marker="s",
        label="best sample",
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reward")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if len(iterations) <= 12:
        ax.set_xticks(iterations)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_iteration_heatmap(
    iteration_logs: list[list[dict[str, Any]]],
    cfg,
    out_path: Path,
    metric: str = "reward_softmax_weighted",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    mat = _iteration_metric_matrix(iteration_logs, metric)
    if mat.size == 0:
        return

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    im = ax.imshow(mat, origin="lower", aspect="auto", interpolation="nearest", cmap="viridis")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(ITER_HEATMAP_STATS[metric])

    ax.set_title("Reward refinement across plan_steps and iterations", fontsize=13, pad=18)
    ax.set_xlabel("Control step")
    ax.set_ylabel("Iteration")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if mat.shape[0] <= 12:
        ax.set_yticks(np.arange(mat.shape[0]))
    if mat.shape[1] <= 20:
        ax.set_xticks(np.arange(mat.shape[1]))

    cfg_text = (
        f"N={int(cfg.n_sample)}, n_iter={int(cfg.n_update_iter)}, "
        f"sigma={float(cfg.noise_level):g}, reward_weight={float(cfg.reward_weight):g}, "
        f"beta={float(cfg.beta_filter):g}"
    )
    ax.text(
        0.99, 1.02, cfg_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="dimgray",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _write_iteration_visualizations(
    iteration_logs: list[list[dict[str, Any]]],
    per_step: list[dict[str, Any]],
    goal: dict[str, Any],
    cfg,
    out_dir: Path,
    heatmap_stat: str = "reward_softmax_weighted",
) -> dict[str, str]:
    """Write per-plan-step iteration traces and the aggregate heatmap."""
    if not iteration_logs:
        return {}

    iter_dir = out_dir / "iter_viz"
    iter_dir.mkdir(parents=True, exist_ok=True)
    for step_idx, step_log in enumerate(iteration_logs):
        final_row = per_step[step_idx + 1] if step_idx + 1 < len(per_step) else {}
        _plot_plan_step_iteration_trace(
            step_idx,
            step_log,
            final_row,
            goal,
            iter_dir / f"plan_step_{step_idx:03d}.png",
        )

    heatmap_path = out_dir / "iter_reward_heatmap.png"
    _plot_iteration_heatmap(iteration_logs, cfg, heatmap_path, metric=heatmap_stat)
    return {
        "iteration_log": str(out_dir / "iteration_log.pt"),
        "per_plan_step_dir": str(iter_dir),
        "heatmap": str(heatmap_path),
        "heatmap_stat": heatmap_stat,
    }


def _run_episode(
    args: argparse.Namespace,
    cfg: Any,
    config_deviation: dict,
    out_dir: Path,
) -> None:
    """Run one MPPI control episode. Multi-GPU-safe.

    When ``D.is_dist()`` is False (single-process / ``--n_gpus 1``), this
    is the original single-GPU body. When invoked from a ``_worker``
    process under ``mp.spawn``, ``D.is_dist()`` is True; rank 0 keeps
    doing artifact dumps and progress prints, all other ranks short-
    circuit those side-effects but participate in the planner's
    sample-shard collectives.
    """
    is_rank0 = (D.rank() == 0)
    device = f"cuda:{D.rank()}"

    if is_rank0 and config_deviation["changed"]:
        print(f"\n[!! config deviation] {config_deviation}\n")
    if is_rank0:
        print(f"Effective config:\n{OmegaConf.to_yaml(cfg)}")
        print(f"[step_each_iter] = {int(getattr(cfg, 'step_each_iter', 1))}")
        print(f"Loading WM from {args.wm_ckpt}")
    env = PushTWMEnv(
        args.wm_ckpt,
        device=device,
        cv_n_workers=int(getattr(cfg, "cv_n_workers", 0) or 0),
    )

    if is_rank0:
        print(
            f"Encoding initial state from {args.initial_hdf5} "
            f"frame {args.initial_frame}"
        )
    # Multi-frame warmup: load the 10-frame WM context (frames
    # initial_frame-9 .. initial_frame inclusive) so that env.rollout
    # (planner scoring) and the closed-loop single-step below both run
    # in the WM's trained 10-frame regime instead of the 1..9-frame OOD
    # ramp. Pattern matches scripts/run_mppi_warmup_phase_c.py:376-412.
    window_size = 10  # WM hardcoded n_frames=10 (rl/models/world_model.py:62-63)
    z_history, action_history = env.load_initial_with_warmup(
        args.initial_hdf5,
        end_frame_index=int(args.initial_frame),
        window_size=window_size,
    )
    env.set_warmup(z_history, action_history)
    z = z_history[-1]  # (C, H_lat, W_lat); env.rollout ignores content when warmup is set

    if is_rank0:
        print(f"Loading goal from {args.goal}")
    goal = env.load_goal(args.goal)
    if is_rank0:
        print(
            f"  goal: cx={goal['cx']:.2f} cy={goal['cy']:.2f} "
            f"theta={goal['theta_deg']:.2f}deg"
        )

    planner = MPPIPlanner(env, cfg)

    # ── debug_mode initial dumps (rank 0 only) ──
    # See ``configs/mppi/default.yaml`` docstring for layout.
    debug_mode = bool(getattr(cfg, "debug_mode", False))
    debug_dir: Path | None = None
    if debug_mode and is_rank0:
        debug_path_cfg = getattr(cfg, "debug_dump_dir", None)
        debug_dir = (Path(debug_path_cfg) if debug_path_cfg
                     else out_dir / "debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        planner.set_debug_dir(debug_dir)
        _debug_dump_init(
            debug_dir=debug_dir, cfg=cfg, args=args, env=env,
            z0=z, goal=goal,
        )

    # ── main control loop ──
    trajectory_latents: list[torch.Tensor] = [z.cpu().clone()]
    converged_plans: list[torch.Tensor] = []   # (n_plan_calls, H, A); equals control_steps when step_each_iter=1
    iteration_logs: list[list[dict[str, Any]]] = []
    rgb_frames: list[np.ndarray] = []
    per_step: list[dict] = []

    # initial decode + state for the recording (rank 0 only — other ranks
    # don't write the video and skipping decode/CV saves a few seconds)
    if is_rank0:
        rgb0 = env.decode(z)
        rgb0_u8 = (rgb0.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        state0 = env.estimate_state(rgb0)
        r0 = env.compute_reward(state0, goal, cv_fail_penalty=float(cfg.cv_fail_penalty))
        rgb_frames.append(rgb0_u8)
        per_step.append({
            "t": 0,
            "action": None,
            "reward": float(r0),
            "cv_success": bool(state0["success"]),
            "cx": float(state0["cx"]) if state0["success"] else None,
            "cy": float(state0["cy"]) if state0["success"] else None,
            "theta_deg": float(state0["theta_deg"]) if state0["success"] else None,
            "plan_wall_s": None,
        })

    # Cross-step warm-start state. The converged action sequence from each
    # plan_step is shift-and-padded into ``act_seq_running`` and fed back
    # as ``init_act_seq`` next call (matches upstream exp_sim_control.py
    # control loop). Initialised at zeros — the "stay-put" prior for an
    # action space of bimanual end-effector deltas in [-1, 1]. NOT
    # ``curr_pos.repeat(...)``: upstream uses absolute poses, we use deltas.
    
    anchor_frame = max(0, int(args.initial_frame) - 1)
    import h5py as _h5py
    with _h5py.File(args.initial_hdf5, "r") as _f:
        raw_anchor_np = _f["action"][anchor_frame]
    raw_anchor = torch.as_tensor(raw_anchor_np, dtype=torch.float32)
    anchor_norm = env._wm.normalizer["action"].normalize(
        raw_anchor.unsqueeze(0)
    ).squeeze(0).to(device=env.device, dtype=torch.float32)
        
    H = int(cfg.n_look_ahead)
    A = int(cfg.action_dim)
    act_seq_running = anchor_norm.unsqueeze(0).expand(H, A).clone().to(
            device=env.device, dtype=torch.float32
        )
    # Delta-mode (state-anchored MPPI). The planner internally cumsum-
    # integrates sampled per-step deltas onto an anchor and clamps the
    # resulting absolute trajectory to the cube before WM rollout. The
    # runner just reads the anchor from hdf5 and advances it between
    # plan_step calls — no caller-side integration. See
    # outputs/diagnostics/demo_action_stats_*.md for empirical bounds and
    # outputs/diagnostics/clip_logic_audit_*.md for design rationale.
    sample_delta_clip = bool(getattr(cfg, "sample_delta_clip", False))
    # Anchor is needed when sample_delta_clip is on; the planner consumes
    # it via planner.set_anchor() / plan_step(anchor=...).
    needs_anchor = sample_delta_clip
    if needs_anchor and is_rank0:
        # Read raw action at frame_idx-1 (the action that took us into
        # the initial state; or frame 0 at episode start). Apply the WM's
        # training-time normalizer to land in the same space MPPI samples
        # operate in.
        print(
            f"[sample_delta_clip] anchor from frame {anchor_frame}: "
            f"raw={raw_anchor.tolist()}  normalized={anchor_norm.tolist()}"
        )

    step_each_iter = int(getattr(cfg, "step_each_iter", 1))

    # EMA smoothing on executed mu[0] (Option A). Default OFF: byte-identical
    # to pre-EMA behavior. When enabled, the runner post-processes the
    # action returned by planner.plan_step BEFORE env.rollout — planner
    # internals (anchor advancement, warm-start, sample_delta_clip) are
    # unaffected.
    ema_enabled = bool(OmegaConf.select(cfg, "action_ema.enabled", default=False))
    ema_alpha = float(OmegaConf.select(cfg, "action_ema.alpha", default=0.3))
    ema_init_mode = str(OmegaConf.select(cfg, "action_ema.init_mode", default="first_mu"))
    if ema_enabled and ema_init_mode not in ("first_mu", "anchor"):
        raise ValueError(
            f"unknown action_ema.init_mode: {ema_init_mode!r}; "
            "expected 'first_mu' or 'anchor'"
        )
    a_prev: np.ndarray | None = None

    t0_run = time.time()
    n_actions_done = 0
    while n_actions_done < int(cfg.control_steps):
        # Number of actions executed during this trajectory_optimization
        # call. With the default step_each_iter=1 this is always 1
        # (bit-equivalent to the pre-refactor loop). With step_each_iter=N
        # the planner is called ceil(control_steps/N) times and N actions
        # are executed per call (matches reference exp_sim_control.py
        # lines 108, 151-155, 219-220). The final call may execute fewer
        # than N when control_steps % N != 0.
        n_this = min(step_each_iter, int(cfg.control_steps) - n_actions_done)

        # Snapshot pre-plan state for debug_mode (rank 0 only). The
        # post-plan snapshot happens after the sub-step loop ends.
        debug_z_before = z.detach().cpu().clone() if (debug_mode and is_rank0) else None
        debug_rgb_before = rgb_frames[-1] if (debug_mode and is_rank0 and rgb_frames) else None
        debug_state_before = per_step[-1] if (debug_mode and is_rank0 and per_step) else None

        t0_plan = time.time()
        init_act = act_seq_running if USE_WARM_START else None
        a, iter_log = planner.plan_step(
            z, goal, init_act_seq=init_act, return_iteration_log=True,
            anchor=anchor_norm,  # required when sample_delta_clip=True
        )                                           # (action_dim,), list[n_iter]
        # ``planner.last_stats.act_seq`` lives on CPU as (H, A); every rank
        # has bit-identical bytes (see mppi_planner.optimize_action_mppi
        # determinism note). It is ABSOLUTE; the cube clamp happens inside
        # the planner before WM scoring.
        converged = planner.last_stats.act_seq.detach().cpu().clone()
        plan_wall = time.time() - t0_plan

        # EMA smoothing on executed mu[0]. Only modifies ``a`` (used at
        # s==0 in the substep loop below). ``converged`` is untouched —
        # warm-start, anchor advancement, and substeps for s>0 all see
        # the planner's raw trajectory exactly as before.
        mu_raw_np: np.ndarray | None = None
        ema_l2_val: float = 0.0
        if ema_enabled:
            mu_raw_np = a.detach().cpu().numpy().copy()
            if a_prev is None:
                if ema_init_mode == "first_mu":
                    a_prev = mu_raw_np.copy()
                else:  # "anchor" — validated at config-read time
                    a_prev = anchor_norm.detach().cpu().numpy().copy()
            a_exec_np = ema_alpha * mu_raw_np + (1.0 - ema_alpha) * a_prev
            a_prev = a_exec_np.copy()
            ema_l2_val = float(np.linalg.norm(a_exec_np - mu_raw_np))
            a = torch.from_numpy(a_exec_np).to(a.device).to(a.dtype)

        # Execute n_this absolute actions against the WM. ``converged`` is
        # an absolute trajectory in the cube. a == converged[0]
        # (or, when EMA on, a == EMA-smoothed converged[0]).
        for s in range(n_this):
            a_s = a if s == 0 else converged[s].to(env.device)
            # Warmup-aware single-step: route through env.rollout so the
            # WM sees the full 10-frame context, not just the latest z.
            # Then slide the warmup window and re-install. Mirrors
            # scripts/run_mppi_warmup_phase_c.py:496-513.
            a_dev = a_s.to(env.device)
            actions_one = a_dev.unsqueeze(0).unsqueeze(0)  # (1, 1, A) batched
            with torch.no_grad():
                traj_b = env.rollout(z.unsqueeze(0), actions_one)  # (1, 2, C, H, W)
            z = traj_b[0, 1]  # (C, H, W)
            z_history = torch.cat(
                [z_history[1:], z.unsqueeze(0)], dim=0,
            )
            action_history = torch.cat(
                [action_history[1:], a_dev.unsqueeze(0)], dim=0,
            )
            env.set_warmup(z_history, action_history)
            if is_rank0:
                rgb_t = env.decode(z)
                rgb_t_u8 = (
                    rgb_t.clamp(0, 1).cpu().numpy() * 255
                ).astype(np.uint8).transpose(1, 2, 0)
                state_t = env.estimate_state(rgb_t)
                r_t = env.compute_reward(
                    state_t, goal,
                    cv_fail_penalty=float(cfg.cv_fail_penalty),
                )
                trajectory_latents.append(z.cpu().clone())
                rgb_frames.append(rgb_t_u8)
                # plan_wall_s is recorded on the first sub-step of each
                # plan call; subsequent sub-steps within the same plan
                # report 0.0 so summing per_step still yields total plan
                # wall time.
                _entry = {
                    "t": n_actions_done + s + 1,
                    "action": [float(v) for v in a_s.tolist()],
                    "reward": float(r_t),
                    "cv_success": bool(state_t["success"]),
                    "cx": float(state_t["cx"]) if state_t["success"] else None,
                    "cy": float(state_t["cy"]) if state_t["success"] else None,
                    "theta_deg": float(state_t["theta_deg"]) if state_t["success"] else None,
                    "plan_wall_s": round(plan_wall, 3) if s == 0 else 0.0,
                }
                if ema_enabled:
                    if s == 0 and mu_raw_np is not None:
                        _entry["action_raw"] = [float(v) for v in mu_raw_np.tolist()]
                        _entry["ema_l2"] = ema_l2_val
                    else:
                        # substep (s > 0) — EMA not applied; raw == executed.
                        _entry["action_raw"] = _entry["action"]
                        _entry["ema_l2"] = 0.0
                per_step.append(_entry)
                _ema_token = (
                    f"  ema_l2={ema_l2_val:.4f}" if (ema_enabled and s == 0) else ""
                )
                print(
                    f"  t={n_actions_done + s + 1:02d}  r={float(r_t):+.4f}  "
                    + (
                        f"cv=({float(state_t['cx']):.1f},{float(state_t['cy']):.1f},"
                        f"{float(state_t['theta_deg']):+.1f}deg)"
                        if state_t['success'] else "CV-FAIL"
                    )
                    + (f"  plan={plan_wall:.2f}s" if s == 0 else "  (sub-step)")
                    + _ema_token
                )

        if is_rank0:
            # One converged plan + one iter_log per plan_step call.
            converged_plans.append(converged)
            iteration_logs.append(iter_log)

        # Per-plan executed_step.npz for debug_mode (rank 0 only).
        if debug_mode and is_rank0 and debug_dir is not None:
            plan_idx = int(planner._audit_plan_call_idx)
            plan_dir = debug_dir / f"plan_{plan_idx:03d}"
            plan_dir.mkdir(parents=True, exist_ok=True)
            _debug_dump_executed_step(
                plan_dir=plan_dir,
                executed_actions=converged[:n_this].detach().cpu().numpy(),
                z_before=(debug_z_before.numpy()
                          if debug_z_before is not None else None),
                z_after=z.detach().cpu().numpy(),
                rgb_before=debug_rgb_before,
                rgb_after=rgb_frames[-1] if rgb_frames else None,
                state_before=debug_state_before,
                state_after=per_step[-1] if per_step else None,
                plan_wall_s=plan_wall,
                converged_full=converged.detach().cpu().numpy(),
            )

        if USE_WARM_START:
            # Shift-and-pad by n_this; repeat-last-action is the reference behavior.
            converged_dev = converged.to(env.device)
            new_tail = converged_dev[-1:].repeat(n_this, 1)
            act_seq_running = torch.cat(
                [converged_dev[n_this:], new_tail], dim=0
            )

        if needs_anchor:
            # Advance anchor to the last absolute action we just executed
            # (already in the cube — the planner's clamp ran). Subsequent
            # plans bound their first-step delta against this point.
            anchor_norm = converged[n_this - 1].to(env.device).detach().clone()

        n_actions_done += n_this

    if not is_rank0:
        # Non-rank-0 workers exit here — they do not write artifacts.
        return

    wall_total = time.time() - t0_run
    print(f"\nRun wall time: {wall_total:.1f} s")

    # ── save artifacts ──
    torch.save(torch.stack(trajectory_latents), out_dir / "trajectory_latents.pt")
    # action_history.pt schema: (n_plan_calls, H, A) -- the FULL converged
    # plan from each MPPI call. With step_each_iter=1 this equals
    # control_steps and slice [:, 0, :] recovers the executed sequence.
    # With step_each_iter>1 the leading axis is ceil(control_steps/step_each_iter)
    # and slice [:, :step_each_iter, :] (flattened) recovers the executed sequence
    # (modulo the final partial call). Bumped from (T, A) so downstream
    # debug/replay tools can inspect the entire planner output.
    torch.save(torch.stack(converged_plans), out_dir / "action_history.pt")
    torch.save(trajectory_latents[0], out_dir / "initial_latent.pt")
    torch.save(iteration_logs, out_dir / "iteration_log.pt")
    _write_mp4(rgb_frames, out_dir / "trajectory.mp4")

    # reward curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rewards = np.array([row["reward"] for row in per_step])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(np.arange(len(rewards)), rewards, marker="o", color="tab:blue")
    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.set_xlabel("control step")
    ax.set_ylabel("reward")
    ax.set_title(f"refactored MPPI — {len(rewards)-1} steps")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "reward_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # success summary
    final = per_step[-1]
    initial = per_step[0]
    final_pos = None
    final_ang_err = None
    final_cos_sim = None
    if final["cv_success"] and initial["cv_success"]:
        dx = final["cx"] - goal["cx"]
        dy = final["cy"] - goal["cy"]
        final_pos = math.sqrt(dx * dx + dy * dy)
        a_diff = final["theta_deg"] - goal["theta_deg"]
        final_ang_err = ((a_diff + 180) % 360) - 180
        final_cos_sim = (
            math.sin(math.radians(final["theta_deg"])) * goal["sin_theta"]
            + math.cos(math.radians(final["theta_deg"])) * goal["cos_theta"]
        )

    # Initial measured distance (when CV succeeds on the initial decode).
    initial_distance_to_goal = None
    if initial["cv_success"]:
        idx = initial["cx"] - goal["cx"]
        idy = initial["cy"] - goal["cy"]
        initial_distance_to_goal = math.sqrt(idx * idx + idy * idy)

    # Phase 2 acceptance metrics
    success_strict = (
        final_pos is not None and final_pos <= 5.0
        and final_ang_err is not None and abs(final_ang_err) <= 10.0
    )
    success_cos = (
        final_pos is not None and final_pos <= 5.0
        and final_cos_sim is not None and final_cos_sim >= 0.94
    )
    last_10_rewards = [row["reward"] for row in per_step[-10:]]
    mean_reward_last_10 = float(np.mean(last_10_rewards))

    # Latent drift over the executed trajectory
    latents_t = torch.stack(trajectory_latents).float()  # (T+1, C, H, W)
    z0_flat = latents_t[0].flatten()
    cos_to_z0 = torch.tensor([
        torch.nn.functional.cosine_similarity(
            latents_t[t].flatten().unsqueeze(0), z0_flat.unsqueeze(0)
        ).item() for t in range(latents_t.shape[0])
    ])
    min_cos_to_z0 = float(cos_to_z0.min())

    summary = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "config_deviation": config_deviation,
        "initial_state": {
            "hdf5": args.initial_hdf5,
            "frame": args.initial_frame,
            "cv": initial,
            "distance_to_goal_px": (
                None if initial_distance_to_goal is None
                else round(initial_distance_to_goal, 3)
            ),
        },
        "goal_state": {"cx": goal["cx"], "cy": goal["cy"], "theta_deg": goal["theta_deg"]},
        "final_state": final,
        "final_pos_distance_px": None if final_pos is None else round(final_pos, 3),
        "final_angle_error_deg": None if final_ang_err is None else round(final_ang_err, 2),
        "final_angle_sim": None if final_cos_sim is None else round(final_cos_sim, 4),
        "success_strict": bool(success_strict),
        "success_cos": bool(success_cos),
        "min_latent_cosine_sim_to_z0": round(min_cos_to_z0, 4),
        "n_cv_failures": sum(1 for r in per_step if not r["cv_success"]),
        "mean_reward_last_10_steps": round(mean_reward_last_10, 4),
        "best_reward_in_trajectory": round(
            float(max(row["reward"] for row in per_step)), 4
        ),
        "wall_time_s": round(wall_total, 1),
        "wall_per_step_s": round(wall_total / max(1, int(cfg.control_steps)), 3),
        "n_gpus": int(getattr(args, "n_gpus", 1) or 1),
        "action_ema": OmegaConf.to_container(
            OmegaConf.select(cfg, "action_ema", default=OmegaConf.create({
                "enabled": False, "alpha": 0.3, "init_mode": "first_mu",
            })),
            resolve=True,
        ),
        "per_step": per_step,
    }
    summary["iteration_reward_viz"] = _write_iteration_visualizations(
        iteration_logs,
        per_step,
        goal,
        cfg,
        out_dir,
        heatmap_stat=args.iter_heatmap_stat,
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Combined per-frame trajectory + reward-iteration video. Reads back
    # trajectory.mp4, summary.json, and iteration_log.pt that were just
    # written, so the call must come AFTER the artifact dump above.
    try:
        from rl.visualization.combined_video import render_combined_video
        combined_path = render_combined_video(out_dir, fps=8)
        print(f"  combined viz: {combined_path}")
    except Exception as exc:  # noqa: BLE001 — viz is best-effort, must not nuke a successful run
        print(f"  WARN: combined video render failed: {exc!r}")

    # Per-plan-step action distribution diagnostic plots. Same best-effort
    # convention: viz failure should not nuke a successful run.
    try:
        from rl.visualization.action_distribution import render_action_distributions
        from rl.visualization.demo_action_stats import load_or_compute_demo_action_stats
        demo_stats = load_or_compute_demo_action_stats()
        action_dist_paths = render_action_distributions(
            iteration_logs, per_step, out_dir, demo_stats=demo_stats,
        )
        print(f"  action dist plots: {len(action_dist_paths)} PNGs in "
              f"{out_dir / 'action_dist'}")
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: action distribution render failed: {exc!r}")

    # If the run involved any config deviation, also drop a cloud-ready
    # reproduction script that uses the configured-default values.
    if config_deviation["changed"]:
        cloud_script = out_dir / "reproduce_on_cloud.sh"
        cloud_script.write_text(
            "#!/usr/bin/env bash\n"
            "# Cloud reproduction of this run at the configured-default "
            "hyperparameters\n"
            "# (no --n_sample override). Requires GPU with >=20 GiB free for "
            "decoder\n"
            "# attention scratch at the default n_sample.\n"
            "#\n"
            f"# This local run used: {', '.join(config_deviation['changed'])}\n"
            f"# Reason: {config_deviation['reason']}\n"
            "set -e\n"
            "cd \"$(dirname \"$0\")/../../..\"\n"
            f"python scripts/run_mppi_v2.py \\\n"
            f"    --config {args.config} \\\n"
            f"    --wm_ckpt {args.wm_ckpt} \\\n"
            f"    --initial_hdf5 {args.initial_hdf5} \\\n"
            f"    --initial_frame {args.initial_frame} \\\n"
            f"    --goal {args.goal} \\\n"
            f"    --output_dir {args.output_dir}_cloud_repro\n"
        )
        cloud_script.chmod(0o755)
        print(f"  reproduction script: {cloud_script}")

    if debug_mode and debug_dir is not None:
        _debug_dump_manifest(debug_dir)
        manifest = json.load(open(debug_dir / "manifest.json"))
        print(f"  debug dump: {debug_dir} ({manifest['total_mb']} MB, "
              f"{manifest['n_files']} files)")

    if bool(getattr(cfg, "audit_log_enabled", False)) or debug_mode:
        audit_log = planner.get_audit_log()
        audit_path_cfg = getattr(cfg, "audit_log_path", None)
        audit_out = (
            Path(audit_path_cfg) if audit_path_cfg
            else out_dir / "audit_log.json"
        )
        audit_out.parent.mkdir(parents=True, exist_ok=True)
        with open(audit_out, "w") as af:
            json.dump(audit_log, af)
        print(f"  audit log: {audit_out} ({len(audit_log)} entries)")

    print(f"\nWrote artifacts to {out_dir}")
    print(f"  final_pos_distance_px      = {final_pos}")
    print(f"  final_angle_error_deg      = {final_ang_err}")
    print(f"  min_latent_cosine_sim_to_z0= {min_cos_to_z0:.4f}")
    print(f"  n_cv_failures              = {summary['n_cv_failures']}")
    print(f"  mean_reward_last_10_steps  = {mean_reward_last_10:.4f}")
    print(f"  success_strict / _cos      = {success_strict} / {success_cos}")


def _worker(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    cfg: Any,
    config_deviation: dict,
    out_dir: Path,
    master_port: int,
) -> None:
    """Per-rank entrypoint for ``torch.multiprocessing.spawn``.

    Initializes ``torch.distributed`` (NCCL), pins this process to its
    GPU via ``torch.cuda.set_device(rank)`` (done inside ``init_dist``),
    and runs the same ``_run_episode`` body as the single-GPU path.

    Forward-looking note: WandB integration, when added, must run on
    rank 0 only -- init it here AFTER ``D.init_dist`` and BEFORE
    ``_run_episode``, gated by ``if rank == 0:``. Otherwise every worker
    creates a duplicate run and clobbers metrics.
    """
    D.init_dist(rank, world_size, master_port=master_port)

    # Silence non-rank-0 stdout/stderr so the multi-process console
    # output is readable. Rank 0 keeps both streams.
    if rank != 0:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    try:
        _run_episode(args, cfg, config_deviation, out_dir)
    finally:
        D.destroy_dist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mppi/default.yaml")
    ap.add_argument("--wm_ckpt", default="outputs/pusht_cam1/checkpoints/best.ckpt")
    ap.add_argument("--initial_hdf5", required=True)
    ap.add_argument("--initial_frame", type=int, default=0)
    ap.add_argument("--goal", required=True, help="Path to a state_goal.pt")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument(
        "--n_sample", type=int, default=None,
        help="Override config.n_sample at runtime (for VRAM-constrained "
             "local runs). When used, summary.json records a config_deviation "
             "block listing the override and its rationale (which the caller "
             "must pass via --override_reason).",
    )
    ap.add_argument(
        "--n_update_iter", type=int, default=None,
        help="Override config.n_update_iter at runtime (for visualization "
             "smoke tests or explicit ablations). This is an algorithm-level "
             "override and therefore requires --override_reason.",
    )
    ap.add_argument(
        "--override_reason", type=str, default=None,
        help="Free-text justification for any --n_sample override. Stored in "
             "summary.json under config_deviation.reason.",
    )
    ap.add_argument(
        "--expected_impact", type=str, default=None,
        help="Free-text quantified expected impact of the override. Stored in "
             "summary.json under config_deviation.expected_impact_quantified. "
             "If omitted, a generic placeholder is used.",
    )
    ap.add_argument(
        "--control_steps", type=int, default=None,
        help="Override config.control_steps (episode length). Not an "
             "algorithm deviation, so no --override_reason required.",
    )
    ap.add_argument(
        "--seed", type=int, default=None,
        help="Override config.seed (MPPIPlanner sampler RNG). Useful for "
             "multi-seed variance studies. Not an algorithm deviation, so "
             "no --override_reason required.",
    )
    ap.add_argument(
        "--decode_batch_size", type=int, default=None,
        help="Memory-only override for decoding sampled final latents during "
             "MPPI reward evaluation. Smaller values reduce peak VRAM without "
             "changing N, horizon, or the sampled action set.",
    )
    ap.add_argument(
        "--horizon", type=int, default=None,
        help="Override config.n_look_ahead (planning horizon H). Range "
             "[1, 50]; the IWS WM's internal sliding 10-frame attention "
             "window handles long horizons natively. Algorithm-level "
             "override, so requires --override_reason when set to a value "
             "different from the config default.",
    )
    ap.add_argument(
        "--iter_heatmap_stat", default="reward_softmax_weighted",
        choices=sorted(ITER_HEATMAP_STATS.keys()),
        help="Iteration-log statistic to color in iter_reward_heatmap.png. "
             "Default is the softmax-weighted mean reward.",
    )
    ap.add_argument(
        "--n_gpus", type=int, default=1,
        help="Number of GPUs to shard MPPI samples across. With G>1, this "
             "script spawns G worker processes via torch.multiprocessing.spawn; "
             "each process evaluates n_sample/G action sequences and the "
             "rewards are all-gathered before the softmax. Bit-identical "
             "first-action output to a single-GPU run at the same seed when "
             "n_sample is divisible by G. Not an algorithm deviation.",
    )
    ap.add_argument(
        "--cv_n_workers", type=int, default=None,
        help="Number of CPU worker processes for the batched CV labeler. "
             "0 = sequential (default; bit-exact with the historical loop). "
             ">0 = spawn-pool of this many workers per estimate_state call. "
             "Infra/perf knob, NOT an algorithm change — no --override_reason "
             "required. Pays off only for n_sample sufficiently large that "
             "the CV time amortizes the spawn overhead; see the cv-batching "
             "N-sweep report for the threshold.",
    )
    ap.add_argument(
        "--master_port", type=int, default=29500,
        help="TCP port for the torch.distributed rendezvous when --n_gpus > 1. "
             "Override when running multiple multi-GPU jobs concurrently on "
             "the same machine.",
    )
    ap.add_argument(
        "--cfg_overrides", type=str, nargs="*", default=[],
        help="Generic OmegaConf dotlist overrides applied AFTER --config + "
             "argparse overrides (e.g. --cfg_overrides action_ema.enabled=true "
             "action_ema.alpha=0.3). Lets you toggle yaml-only knobs without "
             "editing default.yaml.",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    # Every runtime override is applied here. Algorithm-level deviations
    # (currently only --n_sample) go into the returned config_deviation
    # block and require --override_reason; per-run knobs (control_steps,
    # seed) mutate cfg silently because they don't change MPPI's algorithm.
    config_deviation = _apply_cli_overrides(cfg, args)
    if args.cfg_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(args.cfg_overrides)))

    # ── multi-GPU dispatch ───────────────────────────────────────────
    n_gpus = int(args.n_gpus)
    if n_gpus < 1:
        raise SystemExit(f"--n_gpus must be >= 1; got {n_gpus}")
    if n_gpus > 1:
        # Validate BEFORE mp.spawn so users get a clean Python error
        # instead of a NCCL initialization timeout.
        n_visible = torch.cuda.device_count()
        if n_visible < n_gpus:
            raise SystemExit(
                f"--n_gpus={n_gpus} but only {n_visible} CUDA device(s) "
                f"visible (torch.cuda.device_count()). Reduce --n_gpus or "
                f"set CUDA_VISIBLE_DEVICES."
            )
        if int(cfg.n_sample) % n_gpus != 0:
            raise SystemExit(
                f"n_sample={int(cfg.n_sample)} must be divisible by "
                f"--n_gpus={n_gpus} for sample-shard parallelism."
            )

    if n_gpus == 1:
        # Single-process path: no spawn, no init_dist. The planner's
        # distributed helpers all short-circuit when dist isn't initialized.
        _run_episode(args, cfg, config_deviation, out_dir)
    else:
        mp.spawn(
            _worker,
            args=(n_gpus, args, cfg, config_deviation, out_dir, int(args.master_port)),
            nprocs=n_gpus,
            join=True,
        )


if __name__ == "__main__":
    main()
