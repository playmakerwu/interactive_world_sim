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

# STEP_EACH_ITER moved to yaml: configs/mppi/default.yaml field
# step_each_iter (default 1). Read in run_mppi_v2.py via
# int(getattr(cfg, "step_each_iter", 1)) for backward compat with
# pre-refactor yamls that lack the field.
