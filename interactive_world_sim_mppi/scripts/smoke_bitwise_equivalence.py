"""Smoke test 1: bitwise equivalence of MPPI_WAYPTS vs diffusion-forcing.

Loads the diffusion-forcing planner via importlib.util.spec_from_file_location
(the ONLY place in this package's test surface that touches the upstream
repo). Constructs identical configs for ours and the reference, runs ONE
MPPI_WAYPTS iteration on a deterministic dummy dynamics + dummy reward,
and asserts the returned act_seq and waypts_seq match bitwise.

Pre-conditions for the test to actually run (not skip):
  - diffusion-forcing repo present at /home/yiru-wu/Documents/diffusion-forcing
    (or $DIFFUSION_FORCING_REPO_ROOT).
  - The repo's planner_v0_0.py is importable as a standalone module.

If the bitwise check FAILS, this is a STOP condition — the copy diverged
from the reference. Do not proceed to other tests.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import dataclasses  # noqa: E402

import torch  # noqa: E402

from interactive_world_sim_mppi._mppi_core import (  # noqa: E402
    EvalOutput,
    ModelOutput,
    Planner as OursPlanner,
)
from interactive_world_sim_mppi.config import Config, GoalPose  # noqa: E402


# ---- shared dynamics / reward ----


def _dummy_dynamics_factory():
    """Deterministic dynamics: state_seqs = cumsum(actions, dim=1).

    Both planners get the same factory but call it independently — the
    output depends only on actions, no internal state.
    """

    def dynamics(state_cur, actions):
        # actions: (K, H_dense, A). Return (K, H_dense, 1) — state_dim=1.
        # Use sum across action_dim to make a 1D state per step.
        per_step = actions.sum(dim=-1, keepdim=True)
        return ModelOutput(state_seqs=torch.cumsum(per_step, dim=1))

    return dynamics


def _dummy_reward_factory():
    """Deterministic reward: -state_seqs[:, -1, 0] (terminal scalar)."""

    def evaluate(state_seqs, action_seqs):
        return EvalOutput(reward_seqs=-state_seqs[:, -1, 0])

    return evaluate


# ---- reference planner loader ----


def _load_reference_planner_module():
    """Load diffusion-forcing's planner_v0_0.py via spec_from_file_location.

    Returns (module, full_path) or (None, full_path) if unavailable.
    """
    df_root = os.environ.get(
        "DIFFUSION_FORCING_REPO_ROOT", "/home/yiru-wu/Documents/diffusion-forcing"
    )
    planner_path = (
        Path(df_root) / "algorithms" / "latent_dynamics" / "planner_v0_0.py"
    )
    splines_path = Path(df_root) / "algorithms" / "common" / "splines.py"
    if not planner_path.exists() or not splines_path.exists():
        return None, str(planner_path)
    # The planner imports `from algorithms.common.splines import ...`.
    # We pre-load splines under that fully-qualified name so the import
    # inside planner_v0_0 resolves.
    spec_s = importlib.util.spec_from_file_location(
        "algorithms.common.splines", str(splines_path)
    )
    assert spec_s is not None and spec_s.loader is not None
    splines_mod = importlib.util.module_from_spec(spec_s)
    # Stub the parent packages so the absolute import works.
    import types
    if "algorithms" not in sys.modules:
        sys.modules["algorithms"] = types.ModuleType("algorithms")
    if "algorithms.common" not in sys.modules:
        sys.modules["algorithms.common"] = types.ModuleType("algorithms.common")
    sys.modules["algorithms.common.splines"] = splines_mod
    spec_s.loader.exec_module(splines_mod)

    spec_p = importlib.util.spec_from_file_location(
        "_df_reference_planner", str(planner_path)
    )
    assert spec_p is not None and spec_p.loader is not None
    planner_mod = importlib.util.module_from_spec(spec_p)
    sys.modules["_df_reference_planner"] = planner_mod
    spec_p.loader.exec_module(planner_mod)
    return planner_mod, str(planner_path)


# ---- main ----


def main() -> int:
    print("=" * 72)
    print("smoke_bitwise_equivalence: MPPI_WAYPTS one-iter check")
    print("=" * 72)

    ref_mod, ref_path = _load_reference_planner_module()
    if ref_mod is None:
        print(
            f"SKIP — diffusion-forcing not found at {ref_path}.\n"
            "  Set $DIFFUSION_FORCING_REPO_ROOT to override the path.\n"
            "  This test cannot run; the bitwise guard did not fire."
        )
        return 0  # not a failure — explicitly a skip
    print(f"  reference loaded from {ref_path}")

    # ---- Configs ----
    # Small numbers for speed. normalize_rewards_before_softmax=False so
    # our optimize_action_mppi matches the reference math.
    A = 2  # action_dim — use 2D for both (matches DF default config)
    cfg_ours = Config(
        n_sample=8,
        n_waypoints=2,
        interp_pts=3,
        n_update_iter=1,
        reward_weight=200.0,
        noise_level=0.05,
        beta_filter=0.7,
        rollout_best=True,  # default in planner_v0_0.yaml; rollout_best=False
                            # is actually buggy in the source (act_seq is
                            # unbound at TrajOptOutput construction). Both
                            # planners reproduce this so the equivalence
                            # check holds with rollout_best=True.
        action_lower_lim=(-1.0, -1.0),
        action_upper_lim=(1.0, 1.0),
        normalize_rewards_before_softmax=False,
        goal=GoalPose(x=0.0, y=0.0, angle_deg=0.0),  # required by Config; not used
        device="cpu",  # CPU keeps determinism rock-solid
    )

    cfg_ref = dict(
        action_dim=A,
        n_sample=cfg_ours.n_sample,
        n_look_ahead=cfg_ours.n_waypoints,
        n_update_iter=cfg_ours.n_update_iter,
        reward_weight=cfg_ours.reward_weight,
        action_lower_lim=list(cfg_ours.action_lower_lim),
        action_upper_lim=list(cfg_ours.action_upper_lim),
        planner_type="MPPI_WAYPTS",
        device=cfg_ours.device,
        verbose=False,
        noise_level=cfg_ours.noise_level,
        n_his=1,
        rollout_best=cfg_ours.rollout_best,
        lr=1e-3,
        beta_filter=cfg_ours.beta_filter,
    )

    # ---- Construct planners ----
    ours = OursPlanner(cfg_ours)
    ours.register_model_rollout_fn(_dummy_dynamics_factory())
    ours.register_evaluate_traj_fn(_dummy_reward_factory())

    ref = ref_mod.Planner(cfg_ref)
    ref.register_model_rollout_fn(_dummy_dynamics_factory())
    ref.register_evaluate_traj_fn(_dummy_reward_factory())

    # ---- Inputs ----
    waypts_seq = torch.zeros(cfg_ours.n_waypoints, A, dtype=torch.float32)
    curr_pos = torch.zeros(1, A, dtype=torch.float32)
    state_cur = torch.empty(0)

    # ---- Run both with the same seed ----
    torch.manual_seed(12345)
    out_ours = ours.trajectory_optimization(
        state_cur=state_cur, act_seq=waypts_seq.clone(),
        interp_pts=cfg_ours.interp_pts, curr_pos=curr_pos.clone(),
    )

    torch.manual_seed(12345)
    out_ref = ref.trajectory_optimization(
        state_cur=state_cur, act_seq=waypts_seq.clone(),
        interp_pts=cfg_ours.interp_pts, curr_pos=curr_pos.clone(),
    )

    # ---- Compare ----
    same_waypts = torch.equal(out_ours.waypts_seq, out_ref.waypts_seq)
    same_act = torch.equal(out_ours.act_seq, out_ref.act_seq)

    print(f"  ours.waypts_seq.shape  = {tuple(out_ours.waypts_seq.shape)}")
    print(f"  ref.waypts_seq.shape   = {tuple(out_ref.waypts_seq.shape)}")
    print(f"  ours.act_seq.shape     = {tuple(out_ours.act_seq.shape)}")
    print(f"  ref.act_seq.shape      = {tuple(out_ref.act_seq.shape)}")
    print(f"  waypts bitwise equal?  {same_waypts}")
    print(f"  act_seq bitwise equal? {same_act}")

    if not (same_waypts and same_act):
        # Show the magnitude of the diff to aid debugging.
        wd = (out_ours.waypts_seq - out_ref.waypts_seq).abs().max().item()
        ad = (out_ours.act_seq - out_ref.act_seq).abs().max().item()
        print(f"  max |Δwaypts| = {wd:.3e}")
        print(f"  max |Δact_seq| = {ad:.3e}")
        print("FAIL — bitwise equivalence broken. DO NOT proceed.")
        return 1

    print("PASS — MPPI_WAYPTS one-iter is bitwise equivalent to diffusion-forcing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
