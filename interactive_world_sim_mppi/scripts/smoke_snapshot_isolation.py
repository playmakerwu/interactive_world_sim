"""Smoke test 2: snapshot isolation.

Confirms that running plan() leaves the env in exactly the same state it
was in when the snapshot was taken. Compares latent_window, action_window,
and step_counter byte-by-byte before and after plan().

Uses a synthetic reward that doesn't need CV (so the test stays fast and
deterministic), constructed inside this script.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import dataclasses  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from interactive_world_sim_env import WorldModelEnv  # noqa: E402
from interactive_world_sim_mppi import Config, GoalPose, MPPIPlanner  # noqa: E402


EPISODE = str(_REPO_ROOT / "data" / "mini" / "pusht" / "val" / "episode_0.hdf5")


def _zero_reward(latents, rgbs, actions, config):
    """Dummy reward — returns zeros. Avoids the CV dependency for this test."""
    return torch.zeros(latents.shape[0], dtype=torch.float32, device=latents.device)


def _snapshot_equal(a, b) -> tuple[bool, list[str]]:
    """Compare two EnvState snapshots field-by-field."""
    diffs = []
    if not torch.equal(a.latent_window, b.latent_window):
        diffs.append("latent_window")
    if not torch.equal(a.action_window, b.action_window):
        diffs.append("action_window")
    if a.step_counter != b.step_counter:
        diffs.append(f"step_counter ({a.step_counter} vs {b.step_counter})")
    if a.task != b.task:
        diffs.append(f"task ({a.task} vs {b.task})")
    return len(diffs) == 0, diffs


def main() -> int:
    print("=" * 72)
    print("smoke_snapshot_isolation: plan() must not mutate env state")
    print("=" * 72)
    env = WorldModelEnv("pusht_cam1")
    env.reset(
        init_episode_path=EPISODE,
        init_episode_index=9,
        init_window_size=10,
    )
    snap = env.snapshot()
    # Deep-clone the snapshot's tensors so we hold reference values
    # that the env can't mutate from under us.
    snap_before = dataclasses.replace(
        snap,
        latent_window=snap.latent_window.clone(),
        action_window=snap.action_window.clone(),
    )

    cfg = Config(
        n_sample=4,
        n_waypoints=2,
        interp_pts=2,
        n_update_iter=2,
        rollout_best=True,
        normalize_rewards_before_softmax=True,
        goal=GoalPose(x=0.0, y=0.0, angle_deg=0.0),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    planner = MPPIPlanner(env, cfg, reward_fn=_zero_reward)
    plan = planner.plan(snap)
    print(f"  plan shape: {tuple(plan.shape)}")

    snap_after = env.snapshot()
    eq, diffs = _snapshot_equal(snap_before, snap_after)
    if eq:
        print("PASS — snapshot before == snapshot after plan().")
        return 0
    print(f"FAIL — env state mutated by plan(): {diffs}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
