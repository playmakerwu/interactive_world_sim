"""Smoke test 5 + Phase 4 diagnostics: end-to-end one-shot plan.

This is the full plan→reward→optimize→plan loop on a real
WorldModelEnv("pusht_cam1"), against a goal pose detected on the t=150
frame of episode_0.

Three diagnostics (per the Phase 3 brief):

  1. Reward-iteration table: mean/std/min/max of reward_seqs at each of
     the n_update_iter MPPI iterations. Shows whether the optimizer is
     actually moving the rewards higher across iterations.

  2. /tmp/plan_one_step_reward_hist.png — histogram of the K=100 reward
     values at iter 0. Reveals whether rollouts are spread enough to
     drive a non-degenerate softmax.

  3. /tmp/plan_one_step_best_rollout.mp4 — decoded RGB sequence of the
     best rollout (the rollout_best replay's frames), 10 fps.

If any of these fail to produce, the script still tries to print as much
diagnostic info as possible before exiting non-zero.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import imageio.v3 as iio  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # noqa: E402 — must precede pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from interactive_world_sim_env import WorldModelEnv  # noqa: E402
from interactive_world_sim_mppi import (  # noqa: E402
    Config,
    GoalPose,
    MPPIPlanner,
    PushTTerminalReward,
    detect_goal_pose_from_episode,
)


EPISODE = str(_REPO_ROOT / "data" / "mini" / "pusht" / "val" / "episode_0.hdf5")
GOAL_FRAME_T = 150

HIST_PNG = "/tmp/plan_one_step_reward_hist.png"
BEST_MP4 = "/tmp/plan_one_step_best_rollout.mp4"


def main() -> int:
    print("=" * 72)
    print("plan_one_step: end-to-end MPPI plan() with diagnostics")
    print("=" * 72)

    # ---- 1. Resolve goal ----
    print(f"[setup] detecting goal pose at t={GOAL_FRAME_T} on {Path(EPISODE).name}")
    goal = detect_goal_pose_from_episode(EPISODE, t=GOAL_FRAME_T)
    print(f"  goal: x={goal.x:.3f}  y={goal.y:.3f}  angle={goal.angle_deg:.3f}°")

    # ---- 2. Env + warmup ----
    print("[setup] loading WorldModelEnv + 10-frame warmup")
    env = WorldModelEnv("pusht_cam1")
    env.reset(
        init_episode_path=EPISODE,
        init_episode_index=9,
        init_window_size=10,
    )

    # ---- 3. Config ----
    # NOTE on memory: K=100 + H=10 (the Config defaults) OOMs the 11.5 GB
    # laptop GPU. env.step_batch decodes ALL K*H frames in one render_img_cm
    # call which uses an internal batch_size=50; with K=4 and H=10 that
    # exhausts ~5 GB on attention softmax. The env is frozen so we can't
    # expose a smaller batch_size or skip non-terminal decodes from here.
    # We trim interp_pts to 2 (instead of the design default 5) to drop
    # H_dense from 10 to 4 and let K=8 fit. Larger GPUs or a future
    # "terminal-only decode" env API can restore the defaults.
    cfg = Config(
        n_sample=8,
        n_waypoints=2,
        interp_pts=2,
        n_update_iter=20,
        rollout_best=True,
        normalize_rewards_before_softmax=True,
        goal=goal,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"  config: K={cfg.n_sample} n_w={cfg.n_waypoints} ip={cfg.interp_pts} "
          f"n_iter={cfg.n_update_iter} normalize={cfg.normalize_rewards_before_softmax}")

    # ---- 4. Reward + diagnostic callback ----
    # Accumulators captured by the callback.
    reward_iter_log: list[dict[str, float]] = []
    first_iter_rewards: list[float] = []

    def on_iter(iter_index: int, reward_seqs: torch.Tensor, action_seqs: torch.Tensor) -> None:
        r = reward_seqs.detach().cpu().float()
        if iter_index >= 0:
            reward_iter_log.append({
                "iter": iter_index,
                "min": float(r.min().item()),
                "mean": float(r.mean().item()),
                "max": float(r.max().item()),
                "std": float(r.std().item()),
            })
            if iter_index == 0:
                first_iter_rewards.extend(r.tolist())

    reward = PushTTerminalReward(num_workers=cfg.detector_num_workers)
    planner = MPPIPlanner(env, cfg, reward_fn=reward, on_iter_callback=on_iter)

    # ---- 5. Plan ----
    snap = env.snapshot()
    print(f"[plan] running {cfg.n_update_iter} MPPI iterations × K={cfg.n_sample} rollouts...")
    t0 = time.perf_counter()
    plan = planner.plan(snap)
    elapsed = time.perf_counter() - t0
    print(f"  plan complete in {elapsed:.1f}s")
    print(f"  plan.shape = {tuple(plan.shape)}")
    print(f"  plan[0]    = {plan[0].cpu().numpy().tolist()}")

    # ---- DIAGNOSTIC 1: reward-iteration table ----
    print("\n" + "=" * 72)
    print("DIAGNOSTIC 1: reward iteration table")
    print("=" * 72)
    print(f"  {'iter':>5}  {'min':>10}  {'mean':>10}  {'max':>10}  {'std':>10}")
    for row in reward_iter_log:
        print(f"  {row['iter']:>5d}  {row['min']:>10.4f}  {row['mean']:>10.4f}  "
              f"{row['max']:>10.4f}  {row['std']:>10.4f}")

    # ---- DIAGNOSTIC 2: histogram of iter-0 rewards ----
    print("\n" + "=" * 72)
    print("DIAGNOSTIC 2: histogram of iter-0 rewards")
    print("=" * 72)
    if first_iter_rewards:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(first_iter_rewards, bins=20, color="#2874A6", edgecolor="black")
        ax.set_xlabel("reward (higher = closer to goal)")
        ax.set_ylabel("count")
        ax.set_title(f"plan_one_step iter-0 reward distribution  (K={len(first_iter_rewards)})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(HIST_PNG, dpi=100)
        plt.close(fig)
        n_penalty = sum(1 for r in first_iter_rewards if r <= cfg.detection_failure_penalty + 1)
        finite = [r for r in first_iter_rewards if r > cfg.detection_failure_penalty + 1]
        print(f"  wrote {HIST_PNG}")
        print(f"  rollouts at failure-penalty: {n_penalty}/{len(first_iter_rewards)}")
        if finite:
            f = np.asarray(finite)
            print(f"  finite (non-penalty) reward stats:  "
                  f"min={f.min():.4f}  mean={f.mean():.4f}  "
                  f"max={f.max():.4f}  std={f.std():.4f}")
    else:
        print("  WARN — no iter-0 rewards captured")

    # ---- DIAGNOSTIC 3: best-rollout video ----
    print("\n" + "=" * 72)
    print("DIAGNOSTIC 3: best-rollout video")
    print("=" * 72)
    # The rollout_best replay inside the verbatim core already runs one
    # final model_rollout with the optimized waypoints and stashes its
    # rgbs into planner._last_rgbs. Use that directly — it's the
    # dense decoded RGB for the deterministic best rollout.
    if planner._last_rgbs is None:
        print("  WARN — no rgbs stash; cannot write best-rollout video")
    else:
        # Shape: (1, H_dense, 3, H_img, W_img) — single best rollout.
        rgbs = planner._last_rgbs[0]  # (H_dense, 3, H_img, W_img)
        frames = np.ascontiguousarray(rgbs.transpose(0, 2, 3, 1))  # → (H, H_img, W_img, 3)
        try:
            iio.imwrite(BEST_MP4, frames, fps=10, codec="libx264")
            print(f"  wrote {BEST_MP4}  shape={frames.shape}  fps=10")
        except Exception as e:
            # Fallback to gif if libx264 unavailable.
            fallback = BEST_MP4.replace(".mp4", ".gif")
            iio.imwrite(fallback, frames, duration=100, loop=0)
            print(f"  MP4 failed ({e!r}); fell back to {fallback}")

    # ---- Cleanup ----
    planner.close()
    env.close()
    print("\nPASS — plan_one_step ran end-to-end with 3 diagnostics.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
