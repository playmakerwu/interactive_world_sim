"""Run-time toggles for ``scripts/run_mppi_v2.py``.

These are caller-side knobs that don't belong in ``configs/mppi/default.yaml``
(that file is the algorithm spec, kept faithful to the upstream
diffusion-forcing reference). Anything here lives strictly in the wrapper
script.
"""

from __future__ import annotations

USE_WARM_START: bool = True
"""Cross-step warm-start toggle.

When True, the converged action sequence from each ``plan_step`` is
shift-and-padded and fed back as ``init_act_seq`` for the next call. This
matches the upstream ``exp_sim_control.py`` control loop and smooths the
boundary between consecutive plans (the AR(1) ``beta_filter`` only smooths
within a single horizon).

When False, ``init_act_seq`` is left at None for every call (the planner
defaults to ``torch.zeros(H, A)``), producing the pre-fix behaviour for A/B
comparison.
"""

STEP_EACH_ITER: int = 1
"""Number of actions executed between consecutive ``plan_step`` calls.

The MPPI run loop in ``scripts/run_mppi_v2.py`` calls
``planner.plan_step(...)`` once and then executes ``STEP_EACH_ITER``
consecutive actions from the converged plan via ``env.dynamics_step``
before re-planning. The warm-start shift-and-pad uses the same value
(matches reference ``exp_sim_control.py:108, 151-155, 219-220``). The
total number of executed actions in an episode is ``cfg.control_steps``
regardless of this knob; with ``STEP_EACH_ITER=N`` the number of MPPI
calls is ``ceil(control_steps / N)``.
"""
