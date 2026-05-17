"""Public API for the MPPI planner.

``MPPIPlanner`` is a thin façade over ``_planner.py::Planner``, the
production-faithful MPPI core. The plan() entry point matches yiru's
original public contract (one-shot plan from a snapshot, env restored
on exit). The closed-loop driver at ``scripts/run_mppi.py`` uses
``Planner.plan_step`` directly for SEI + control-step semantics.

The legacy verbatim diffusion-forcing core lives in ``_mppi_core.py``
and is no longer used by this API; it is retained so the bitwise-
equivalence smoke test (``scripts/smoke_bitwise_equivalence.py``) keeps
working against the diffusion-forcing source.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Optional

import torch

from ._planner import Planner, detect_goal_state_from_episode
from .config import Config, GoalPose


def _goal_pose_to_goal_state(goal: GoalPose) -> dict[str, Any]:
    """Convert legacy ``GoalPose`` to the dict ``goal_state`` the new
    planner expects."""
    rad = math.radians(goal.angle_deg)
    return {
        "cx": float(goal.x),
        "cy": float(goal.y),
        "sin_theta": float(math.sin(rad)),
        "cos_theta": float(math.cos(rad)),
        "theta_deg": float(goal.angle_deg),
    }


class MPPIPlanner:
    """Production-semantics MPPI planner over WorldModelEnv.

    Snapshot semantics: ``plan(snapshot)`` snapshots the env, runs the
    full ``n_update_iter`` loop, restores the env to that snapshot, and
    returns the converged dense action sequence (or the first action,
    depending on the caller's expectation — see Returns).

    Lifecycle: ``close()`` is a no-op now (no CV pool to release; the
    in-module CV calls use a private pool managed by the cv package).

    Example
    -------
        from interactive_world_sim_env import WorldModelEnv
        from interactive_world_sim_mppi import Config, MPPIPlanner
        from interactive_world_sim_mppi._planner import detect_goal_state_from_episode

        env = WorldModelEnv("pusht_cam1")
        env.reset(init_episode_path="data/.../episode_0.hdf5",
                  init_episode_index=9, init_window_size=10)

        goal_state = detect_goal_state_from_episode(
            "data/.../episode_0.hdf5", t=150, processing_resolution=128,
        )
        cfg = Config()
        planner = MPPIPlanner(env, cfg)

        snap = env.snapshot()
        plan = planner.plan(snap, goal_state=goal_state)
        env.step(plan[0])
    """

    def __init__(
        self,
        env,
        config: Config,
        *,
        goal: Optional[GoalPose] = None,
        # Legacy kwargs kept for backwards compat with smoke scripts;
        # the new planner does not use them.
        reward_fn=None,  # noqa: ARG002 — accepted, ignored
        on_iter_callback=None,  # noqa: ARG002 — accepted, ignored
    ) -> None:
        """
        Parameters
        ----------
        env : WorldModelEnv
            yiru's env with snapshot/restore/step_batch.
        config : Config
            Hyperparameters mirroring production's default.yaml.
        goal : GoalPose, optional
            Convenience: if provided, the planner stores
            ``goal_state = _goal_pose_to_goal_state(goal)`` and the
            ``plan(snapshot)`` overload (no goal_state argument) uses it.
        """
        if goal is not None and config.goal is not None:
            raise ValueError(
                "goal provided both via Config(goal=...) and via the planner "
                "constructor; resolve to a single source."
            )
        if goal is not None:
            config = dataclasses.replace(config, goal=goal)

        self._env = env
        self._config = config
        self._core = Planner(env, config)

        # Resolve a default goal_state from config.goal if present, for
        # backwards compatibility with the legacy `plan(snap)` (no
        # goal_state argument) call site.
        self._default_goal_state: dict[str, Any] | None = None
        if config.goal is not None:
            self._default_goal_state = _goal_pose_to_goal_state(config.goal)

        # Warm-start state across plan() calls. Shape (n_look_ahead, A).
        self._prev_act_seq: Optional[torch.Tensor] = None

    # ----- public API -----

    @property
    def core(self) -> Planner:
        """The underlying production-faithful Planner."""
        return self._core

    @property
    def config(self) -> Config:
        return self._config

    def plan(
        self,
        snapshot,
        goal_state: dict[str, Any] | None = None,
        *,
        anchor: torch.Tensor | None = None,
        init_act_seq: torch.Tensor | None = None,
        warm_start: bool = True,
    ) -> torch.Tensor:
        """Run a single MPPI plan from the snapshot. Returns the
        converged full ``(H, A)`` plan on env.device.

        Parameters
        ----------
        snapshot : EnvState
            From ``env.snapshot()``. Env will be restored to this state
            before plan() returns.
        goal_state : dict, optional
            With keys cx, cy, sin_theta, cos_theta, theta_deg. When
            None, the planner falls back to the goal stashed at
            construction time.
        anchor : torch.Tensor, optional
            Required when ``config.delta_mode`` or
            ``config.sample_delta_clip``. Ignored otherwise.
        init_act_seq : torch.Tensor, optional
            (H, A) warm-start. When None and ``warm_start=True``, the
            previous plan's converged tail is shift-and-padded.
        warm_start : bool
            When True, plan k+1 starts from plan k's shift-and-padded
            converged sequence. Matches production's ``USE_WARM_START``.
        """
        gs = goal_state if goal_state is not None else self._default_goal_state
        if gs is None:
            raise ValueError(
                "plan() requires a goal_state (dict with cx, cy, "
                "sin_theta, cos_theta, theta_deg) — either pass it as "
                "goal_state= or construct MPPIPlanner with goal=GoalPose(...)."
            )

        H = int(self._config.n_look_ahead)
        A = int(self._config.action_dim)
        if init_act_seq is None and warm_start and self._prev_act_seq is not None:
            init_act_seq = self._prev_act_seq
        elif init_act_seq is None:
            init_act_seq = torch.zeros(
                H, A, device=self._core.device, dtype=torch.float32,
            )

        result = self._core.plan_step(
            z_current_unused=None,
            goal_state=gs,
            init_act_seq=init_act_seq,
            return_iteration_log=False,
            anchor=anchor,
            snapshot=snapshot,
        )
        # plan_step returns (first action) — the full converged plan is
        # on self._core.last_stats.act_seq.
        full_plan = self._core.last_stats.act_seq.to(self._core.device)
        if warm_start:
            # Shift-and-pad by 1 for the next call's warm-start.
            new_tail = full_plan[-1:].clone()
            self._prev_act_seq = torch.cat(
                [full_plan[1:], new_tail], dim=0,
            ).detach().clone()
        # Silence the unused-variable warning for `result` — we expose
        # the full plan, not the first action.
        del result
        return full_plan

    def close(self) -> None:
        """No-op; retained for backwards compatibility."""
        return
