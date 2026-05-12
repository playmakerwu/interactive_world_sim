"""Smoke test 4: detection-failure penalty path.

Confirms:
  - With every rollout's terminal reward = config.detection_failure_penalty,
    the optimize_action_mppi softmax is approximately uniform (all 1/K
    after the normalize-and-softmax pass) and the planner does not raise.
  - With mixed rewards (half rollouts at the penalty, half at a small
    near-zero value), the resulting weighted update is dominated by the
    higher-reward rollouts (the penalty samples get ~0 weight).

Tests the math in optimize_action_mppi directly — no env or CV needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from interactive_world_sim_mppi._mppi_core import Planner  # noqa: E402
from interactive_world_sim_mppi.config import Config, GoalPose  # noqa: E402


def main() -> int:
    print("=" * 72)
    print("smoke_detection_failure: failure-penalty path")
    print("=" * 72)

    K = 100
    n_waypoints = 2
    A = 4

    cfg = Config(
        n_sample=K,
        n_waypoints=n_waypoints,
        action_lower_lim=(-1.0,) * A,
        action_upper_lim=(1.0,) * A,
        reward_weight=200.0,
        normalize_rewards_before_softmax=True,
        detection_failure_penalty=-1000.0,
        goal=GoalPose(x=0.0, y=0.0, angle_deg=0.0),
        device="cpu",
    )
    core = Planner(cfg)

    # ---- (1) Uniform-penalty rewards ----
    act_seqs_a = torch.randn(K, n_waypoints, A)
    rewards_a = torch.full((K,), cfg.detection_failure_penalty)
    # We call optimize_action_mppi directly to inspect the weighting.
    # The function returns a weighted mean — for uniform rewards it
    # should equal the mean across samples.
    weighted_a = core.optimize_action_mppi(act_seqs_a.clone(), rewards_a.clone())
    mean_a = act_seqs_a.mean(dim=0)
    diff_a = (weighted_a - mean_a).abs().max().item()
    print(f"  (1) all-penalty: max |weighted_mean - simple_mean| = {diff_a:.3e}")
    # Note: normalize-then-softmax of constants → softmax(0) → uniform.
    # weighted_mean should equal simple_mean to ~ float precision.
    if diff_a > 1e-5:
        print(f"  FAIL — expected uniform softmax under constant rewards")
        return 1
    print(f"      → softmax is uniform under constant rewards ✓")

    # ---- (2) Mixed rewards: half penalty, half near-zero ----
    rewards_b = torch.cat([
        torch.full((K // 2,), cfg.detection_failure_penalty),
        torch.full((K - K // 2,), -1.0),
    ])
    act_seqs_b = torch.zeros(K, n_waypoints, A)
    # First half rollouts at (+1, +1, +1, +1) — the "bad" ones (penalty).
    act_seqs_b[: K // 2] = 1.0
    # Second half rollouts at (-1, -1, -1, -1) — the "good" ones.
    act_seqs_b[K // 2:] = -1.0
    weighted_b = core.optimize_action_mppi(act_seqs_b.clone(), rewards_b.clone())
    # If the high-reward (= -1) half dominates, the result should be
    # close to -1, not 0 or +1.
    print(f"  (2) mixed rewards: half at penalty, half at -1.0")
    print(f"      weighted result first action  = {weighted_b[0].tolist()}")
    print(f"      expected close to -1.0 (high-reward half dominates)")
    if not torch.all(weighted_b < -0.5):
        print(f"  FAIL — high-reward half did not dominate")
        return 1
    print(f"      → high-reward half dominates ✓")

    # ---- (3) Just confirm the optimizer doesn't NaN / Inf ----
    if not torch.isfinite(weighted_a).all() or not torch.isfinite(weighted_b).all():
        print("  FAIL — produced NaN/Inf")
        return 1

    print("\nPASS — failure-penalty path is reachable and well-behaved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
