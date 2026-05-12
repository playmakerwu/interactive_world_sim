"""Public API for the MPPI planner.

MPPIPlanner adapts our `WorldModelEnv` and a user-supplied reward
callable to the diffusion-forcing planner core (`_mppi_core.Planner`).
The verbatim core is unchanged; everything env/reward/snapshot-related
lives in this file.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Optional, Protocol

import numpy as np
import torch

from ._mppi_core import EvalOutput, ModelOutput, Planner
from .config import Config, GoalPose


class RewardFn(Protocol):
    """Type for user-supplied reward callables passed to MPPIPlanner.

    The planner invokes reward_fn on every MPPI iteration with the
    current batch of rollouts. The function must return a (K,) tensor of
    rewards (higher = better) on `config.device`.
    """

    def __call__(
        self,
        latents: torch.Tensor,
        rgbs: np.ndarray,
        actions: torch.Tensor,
        config: Config,
    ) -> torch.Tensor: ...


class MPPIPlanner:
    """MPPI planner over WorldModelEnv with a user-supplied reward callable.

    Snapshot semantics: `plan()` snapshots the env once, then forks K
    parallel rollouts from that snapshot for every MPPI iteration. The
    env is restored to the same snapshot before plan() returns; the
    caller decides whether to advance.

    Lifecycle: call `close()` to release CV detection pool resources
    held by the reward_fn (if it has a `.shutdown()` method).

    Diagnostics: pass `on_iter_callback=fn` to receive
        fn(iter_index, reward_seqs, waypts_seqs_before_update)
    once per MPPI iteration. Default None = no overhead.

    Example
    -------
        from interactive_world_sim_env import WorldModelEnv
        from interactive_world_sim_mppi import Config, GoalPose, MPPIPlanner
        from interactive_world_sim_mppi.reward import pusht_terminal_reward

        env = WorldModelEnv("pusht_cam1")
        env.reset(init_episode_path=..., init_window_size=10)

        cfg = Config(goal=GoalPose(x=223.0, y=252.0, angle_deg=-0.5))
        planner = MPPIPlanner(env, cfg, reward_fn=pusht_terminal_reward)

        snap = env.snapshot()
        plan = planner.plan(snap)
        # plan.shape == (n_waypoints * interp_pts, action_dim)
        env.step(plan[0])
    """

    def __init__(
        self,
        env,
        config: Config,
        *,
        reward_fn: RewardFn,
        goal: Optional[GoalPose] = None,
        on_iter_callback: Optional[
            Callable[[int, torch.Tensor, torch.Tensor], None]
        ] = None,
    ) -> None:
        """
        Parameters
        ----------
        env : WorldModelEnv
            Must support snapshot(), restore(), step_batch(), and expose
            a normalized current position via the snapshot's action_window.
        config : Config
            All hyperparameters. If both config.goal and the goal kwarg
            are None, raises ValueError.
        reward_fn : callable
            Signature (latents, rgbs, actions, config) → Tensor[K], on
            config.device. Higher = better.
        goal : GoalPose, optional
            Convenience kwarg. If provided, the planner uses
            dataclasses.replace(config, goal=goal). Mutually exclusive
            with a non-None config.goal.
        on_iter_callback : callable, optional
            Called inside the MPPI inner loop as
                callback(iter_index, reward_seqs, waypts_seqs).
            Default None = no overhead, no behavior change. The verbatim
            core is not modified; this hook fires inside the
            _adapter_evaluate_traj wrapper.
        """
        if goal is not None and config.goal is not None:
            raise ValueError(
                "goal provided both via Config(goal=...) and via the planner "
                "constructor; resolve to a single source."
            )
        if goal is not None:
            config = dataclasses.replace(config, goal=goal)
        if config.goal is None:
            raise ValueError(
                "MPPIPlanner requires a goal: pass either Config(goal=GoalPose(...)) "
                "or MPPIPlanner(env, cfg, reward_fn=..., goal=GoalPose(...))."
            )

        self._env = env
        self._config = config
        self._reward_fn = reward_fn
        self._on_iter_callback = on_iter_callback

        # Construct the verbatim core. Register our adapters as
        # model_rollout and evaluate_traj.
        self._core = Planner(config)
        self._core.register_model_rollout_fn(self._adapter_model_rollout)
        self._core.register_evaluate_traj_fn(self._adapter_evaluate_traj)

        # Warm-start state across plan() calls. Shape (n_waypoints, A).
        self._prev_waypts: Optional[torch.Tensor] = None

        # Per-iteration stash for the rollout→reward handoff.
        self._last_rgbs: Optional[np.ndarray] = None
        self._last_waypts_seqs: Optional[torch.Tensor] = None

        # Snapshot held during plan() so model_rollout can restore.
        self._snapshot = None

        # Diagnostics iteration counter (incremented inside the loop).
        self._iter_index = 0

    # ----- adapters used by the verbatim core -----

    def _adapter_model_rollout(
        self,
        state_cur: torch.Tensor,
        action_seqs: torch.Tensor,
    ) -> ModelOutput:
        """Bridge: verbatim-core model_rollout → env.step_batch.

        state_cur is ignored — the env restores from the snapshot taken
        at the start of plan(). The waypts variant passes a freshly
        spline-interpolated `action_seqs` of shape
        (K, n_waypoints * interp_pts, A) (the rollout_best case at the
        end passes K=1).
        """
        assert self._snapshot is not None, "model_rollout called outside plan()"
        self._env.restore(self._snapshot)
        batched = self._env.step_batch(action_seqs)
        self._last_rgbs = batched.rgbs  # stash for evaluate_traj
        # Also stash the sampled waypts_seqs from the just-finished
        # sample_action_sequences call so the on_iter_callback can see
        # them. The verbatim core calls sample_action_sequences then
        # model_rollout, so the waypts are the input to model_rollout.
        # However, model_rollout receives the spline-interpolated
        # action_seqs, not the waypts directly. We capture the waypts via
        # a different hook — see _adapter_evaluate_traj.
        return ModelOutput(state_seqs=batched.latents)

    def _adapter_evaluate_traj(
        self,
        state_seqs: torch.Tensor,
        action_seqs: torch.Tensor,
    ) -> EvalOutput:
        """Bridge: verbatim-core evaluate_traj → user reward_fn.

        Pulls the rgbs from `_last_rgbs` (set by _adapter_model_rollout
        on the same iteration). Invokes the user reward_fn with
        (latents, rgbs, actions, config). Fires the on_iter_callback if
        present.

        Note: this is also called once at the end of plan() for the
        rollout_best replay, with action_seqs shape (1, H, A). We treat
        that as iter_index "best" for the callback (passed as -1).
        """
        assert self._last_rgbs is not None
        rewards = self._reward_fn(
            latents=state_seqs,
            rgbs=self._last_rgbs,
            actions=action_seqs,
            config=self._config,
        )
        # Diagnostic hook. We pass the (K, H_dense, A) action_seqs as the
        # third arg — these are the post-spline dense actions actually
        # rolled out. The waypts that produced them are not exposed by
        # the verbatim core during the iteration; if a caller needs
        # waypoint-level diagnostics, they can hook
        # sample_action_sequences via register_sample_action_sequences_fn
        # on planner._core.
        if self._on_iter_callback is not None:
            if state_seqs.shape[0] == 1 and self._iter_index > 0:
                # Heuristic: rollout_best replay has K==1 and arrives
                # after the main loop. Use -1 as the iteration index.
                self._on_iter_callback(-1, rewards.detach(), action_seqs.detach())
            else:
                self._on_iter_callback(
                    self._iter_index, rewards.detach(), action_seqs.detach()
                )
                self._iter_index += 1
        return EvalOutput(reward_seqs=rewards)

    # ----- public API -----

    def plan(self, snapshot) -> torch.Tensor:
        """Run MPPI from the given env snapshot and return the dense plan.

        Parameters
        ----------
        snapshot : EnvState
            From env.snapshot(). The env will be restored to this state
            before plan() returns.

        Returns
        -------
        plan : torch.Tensor
            Dense action sequence on config.device, shape
            (n_waypoints * interp_pts, action_dim).
        """
        self._snapshot = snapshot
        self._iter_index = 0
        curr_pos = self._extract_curr_pos(snapshot)  # (action_dim,) on device
        act_seq = self._warm_started_initial_plan(curr_pos)  # (n_waypoints, A)

        # Call the verbatim core. state_cur is unused by our adapter;
        # pass an empty tensor of correct dtype/device for the assert.
        state_cur = torch.empty(0, device=self._config.device)
        out = self._core.trajectory_optimization(
            state_cur=state_cur,
            act_seq=act_seq,
            interp_pts=self._config.interp_pts,
            curr_pos=curr_pos[None],  # (1, A) — n_hist=1
        )

        # Stash new waypoints for next plan()'s warm-start.
        if out.waypts_seq is not None:
            self._prev_waypts = out.waypts_seq.detach().clone()

        # Restore env to the snapshot — caller decides whether to advance.
        self._env.restore(snapshot)
        return out.act_seq.detach()

    def close(self) -> None:
        """Release resources held by the reward_fn (e.g., DetectorPool)."""
        if hasattr(self._reward_fn, "shutdown"):
            try:
                self._reward_fn.shutdown()
            except Exception:
                pass

    # ----- helpers -----

    def _extract_curr_pos(self, snapshot) -> torch.Tensor:
        """Read the normalized current EE position from the snapshot.

        Convention: `snapshot.action_window[-1]` is the action that drove
        INTO the current latent (env convention (b)). This is in
        normalized action space, shape (action_dim,), on env.device.
        """
        a = snapshot.action_window[-1]
        return a.to(self._config.device)

    def _warm_started_initial_plan(self, curr_pos: torch.Tensor) -> torch.Tensor:
        """Initial mean for this plan() call.

        First call: repeat curr_pos n_waypoints times.
        Subsequent calls: shift previous waypoints by step_each_iter,
        pad with copies of the last waypoint.
        """
        n_w = self._config.n_waypoints
        if self._prev_waypts is None:
            return curr_pos[None].repeat(n_w, 1)
        s = self._config.step_each_iter
        if s <= 0 or s > n_w:
            # Defensive: fall back to no-shift / repeat-last.
            s = 1
        kept = self._prev_waypts[s:]
        pad = self._prev_waypts[-1:].repeat(s, 1)
        return torch.cat([kept, pad], dim=0)
