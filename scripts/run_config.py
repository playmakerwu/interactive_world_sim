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

This is the shift size for the warm-start shift-and-pad. Verified by
reading ``scripts/run_mppi_v2.py``: each loop iteration calls
``env.dynamics_step(z, a)`` exactly once between ``planner.plan_step(...)``
calls. Upstream uses 5 because their loop executes 5 actions per plan; we
execute 1.
"""
