"""End-to-end MPPI consistency test (Phase 2 Task E).

Runs both our ``MPPIPlanner`` and a reference build (using the verbatim
helpers from ``_reference_helpers.py``) through a 20-step control loop
with a mock deterministic dynamics + mock distance-based reward. With no
WM stochasticity in the picture, the two implementations should produce
identical action and state trajectories — any divergence reveals an
algorithm-level difference between our outer control loop and the
reference's.

Threshold:
  actions: 1e-4 (Gaussian sampling + 3 inner refinement iters × 20 steps
                  introduces sub-1e-5 FP drift; 1e-4 catches structural
                  divergence)
  states:  1e-3 (states are integrals of actions over 20 steps,
                  accumulated drift is larger)
"""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from rl.mppi.mppi_planner import MPPIPlanner

from tests.mppi._reference_helpers import (  # noqa: E402
    reference_trajectory_optimization,
)

DEVICE = "cpu"  # mock-only; no GPU needed


# ─── Mock Env ───────────────────────────────────────────────────────────

class MockEnv:
    """Mimics the parts of PushTWMEnv that MPPIPlanner depends on, with
    deterministic toy dynamics and a quadratic distance-based reward.

    Uses a fake 3-D latent shape ``(state_dim, 1, 1)`` because
    ``MPPIPlanner.evaluate_trajectories`` does
    ``z_current.unsqueeze(0).expand(N, -1, -1, -1)`` — it expects 3-D
    latents (``(C, H_lat, W_lat)``). The trailing 1×1 spatial dims are
    bookkeeping: dynamics treat the latent as flat ``(state_dim,)``.
    """

    def __init__(self, action_dim: int = 4, dt: float = 0.1):
        self.action_dim = action_dim
        self.device = DEVICE
        self.image_diagonal = 1.0  # arbitrary; reward bypasses normalisation
        self.dt = dt

    def rollout(self, z0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """``(N, state_dim, 1, 1) + (N, H, action_dim) -> (N, H+1, state_dim, 1, 1)``."""
        was_unbatched = z0.dim() == 3
        if was_unbatched:
            z0 = z0.unsqueeze(0)
            actions = actions.unsqueeze(0)
        z0_flat = z0.squeeze(-1).squeeze(-1)  # (N, state_dim)
        zs_flat = [z0_flat]
        z = z0_flat
        for h in range(actions.shape[1]):
            z = z + actions[:, h] * self.dt
            zs_flat.append(z)
        # restore (..., 1, 1)
        traj = torch.stack(zs_flat, dim=1).unsqueeze(-1).unsqueeze(-1)
        return traj if not was_unbatched else traj[0]

    def dynamics_step(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        was_batched = z.dim() == 4
        if not was_batched:
            z = z.unsqueeze(0)
            action = action.unsqueeze(0)
        z_flat = z.squeeze(-1).squeeze(-1)
        z_next_flat = z_flat + action * self.dt
        z_next = z_next_flat.unsqueeze(-1).unsqueeze(-1)
        return z_next if was_batched else z_next[0]

    def estimate_from_latent(self, z: torch.Tensor) -> dict:
        z_flat = z.squeeze(-1).squeeze(-1)  # (state_dim,) or (N, state_dim)
        if z_flat.dim() == 1:
            success = torch.tensor(True)
        else:
            success = torch.ones(z_flat.shape[0], dtype=torch.bool)
        return {"_state": z_flat, "success": success}

    def estimate_state(self, *args, **kwargs):  # not used by planner
        raise NotImplementedError

    def compute_reward(
        self,
        state: dict,
        goal: dict,
        image_diagonal: float | None = None,
        cv_fail_penalty: float = -10.0,
    ) -> torch.Tensor:
        z = state["_state"]
        g = goal["_state"]
        return -(z - g).norm(dim=-1)


# ─── Test ───────────────────────────────────────────────────────────────

def test_e2e_trajectory_matches_reference():
    A = 4
    H = 5
    N = 16
    n_iter = 3
    seed = 7
    n_control_steps = 20

    cfg = OmegaConf.create({
        "n_sample": N,
        "n_look_ahead": H,
        "n_update_iter": n_iter,
        "noise_level": 0.1,
        "reward_weight": 50.0,
        "beta_filter": 0.7,
        "cv_fail_penalty": -10.0,
        "action_dim": A,
        "action_lower_lim": [-1.0] * A,
        "action_upper_lim": [1.0] * A,
        "control_steps": n_control_steps,
        "seed": seed,
    })

    env = MockEnv(action_dim=A)
    z0 = torch.zeros(A, 1, 1, device=DEVICE)  # 3-D fake latent
    goal_vec = torch.tensor([1.0, 1.0, 1.0, 1.0], device=DEVICE)
    goal = {"_state": goal_vec}

    # ── Ours ──
    planner = MPPIPlanner(env, cfg)
    planner._gen.manual_seed(seed)
    z = z0.clone()
    our_actions: list[torch.Tensor] = []
    our_states: list[torch.Tensor] = [z.clone()]
    for _ in range(n_control_steps):
        a = planner.plan_step(z, goal)
        our_actions.append(a.detach().clone())
        z = env.dynamics_step(z, a)
        our_states.append(z.detach().clone())

    # ── Reference (built from helpers) ──
    # Reference uses simple 1-D state (no fake 3-D bookkeeping). Both branches
    # compute the same dynamics internally (z_{t+1} = z_t + a*dt), so trajectories
    # must match.
    ref_gen = torch.Generator(device=DEVICE).manual_seed(seed)
    z_ref_flat = torch.zeros(A, device=DEVICE)
    ref_actions: list[torch.Tensor] = []
    ref_states: list[torch.Tensor] = [z_ref_flat.clone()]
    init_seq = torch.zeros(H, A, device=DEVICE)
    lower = torch.tensor(list(cfg.action_lower_lim), device=DEVICE)
    upper = torch.tensor(list(cfg.action_upper_lim), device=DEVICE)

    def _model_rollout_fn(state_cur, act_seqs):
        # state_cur: (state_dim,) flat; act_seqs: (N, H, action_dim)
        N = act_seqs.shape[0]
        z = state_cur.unsqueeze(0).expand(N, -1).clone()  # (N, state_dim)
        zs = []
        for h in range(act_seqs.shape[1]):
            z = z + act_seqs[:, h] * env.dt
            zs.append(z)
        return torch.stack(zs, dim=1)  # (N, H, state_dim)

    def _evaluate_traj_fn(state_seqs, act_seqs):
        z_final = state_seqs[:, -1]  # (N, state_dim)
        return -(z_final - goal_vec.unsqueeze(0)).norm(dim=-1)

    for _ in range(n_control_steps):
        act_seq = reference_trajectory_optimization(
            z_ref_flat, init_seq,
            n_sample=N,
            n_update_iter=n_iter,
            beta_filter=float(cfg.beta_filter),
            noise_level=float(cfg.noise_level),
            reward_weight=float(cfg.reward_weight),
            action_lower_lim=lower,
            action_upper_lim=upper,
            model_rollout_fn=_model_rollout_fn,
            evaluate_traj_fn=_evaluate_traj_fn,
            generator=ref_gen,
            device=DEVICE,
        )
        a = act_seq[0]
        ref_actions.append(a.detach().clone())
        z_ref_flat = z_ref_flat + a * env.dt  # mock dynamics (1-D version)
        ref_states.append(z_ref_flat.detach().clone())

    # ── Compare ──
    # Flatten our 3-D fake-latent states to 1-D for comparison with reference's
    # 1-D states.
    our_states_flat = [s.squeeze(-1).squeeze(-1) for s in our_states]
    our_actions_t = torch.stack(our_actions)
    ref_actions_t = torch.stack(ref_actions)
    our_states_t = torch.stack(our_states_flat)
    ref_states_t = torch.stack(ref_states)

    max_action_diff = float((our_actions_t - ref_actions_t).abs().max())
    max_state_diff = float((our_states_t - ref_states_t).abs().max())

    print(
        f"[e2e consistency] over {n_control_steps} control steps:\n"
        f"  max |action diff| = {max_action_diff:.3e}\n"
        f"  max |state  diff| = {max_state_diff:.3e}\n"
        f"  final ours: {our_states_flat[-1].tolist()}\n"
        f"  final ref:  {ref_states[-1].tolist()}\n"
        f"  goal:       {goal_vec.tolist()}"
    )

    assert max_action_diff < 1e-4, (
        f"Action trajectories diverge: max diff {max_action_diff:.3e}"
    )
    assert max_state_diff < 1e-3, (
        f"State trajectories diverge: max diff {max_state_diff:.3e}"
    )
