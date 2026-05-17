"""Configuration for the MPPI planner.

Single source of truth for all tunable hyperparameters. Field-for-field
mirror of production's ``configs/mppi/default.yaml`` (production HEAD
``5e2d48e``). The new ``Planner`` in ``_planner.py`` reads these via
attribute access; per-run knobs (control_steps, seed, output paths) are
consumed by the closed-loop driver at ``scripts/run_mppi.py`` rather
than by the planner itself.

production-default yaml fields are reproduced here with the same names
and the same default values. Fields added by this package
(``cv_processing_resolution``, ``image_diagonal``) carry the production
algorithm's 128-px assumptions.

To override a field, use ``dataclasses.replace(cfg, n_sample=200)``.

The dataclass is frozen so an instance can be shared safely across
processes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GoalPose:
    """Target T-block pose. Kept for backwards compat with the legacy
    ``pusht_terminal_reward`` callable. The production-faithful reward
    path consumes a dict-form goal_state instead.

    Attributes
    ----------
    x, y : float
        Pixel coords in the detector's processing-resolution frame
        (default 128 px to match production).
    angle_deg : float
        Aloha-style unwrapped angle in degrees.
    """

    x: float
    y: float
    angle_deg: float


@dataclass(frozen=True)
class Config:
    """All tunable hyperparameters for the MPPI planner.

    Defaults mirror production's ``configs/mppi/default.yaml``.
    """

    # --- algorithm ---
    n_sample: int = 100
    """N: trajectory samples per refinement iteration. Production yaml:6."""

    n_look_ahead: int = 10
    """H: planning horizon. Production yaml:8."""

    n_update_iter: int = 5
    """Inner refinement iterations per plan_step. Production yaml:11."""

    noise_level: float = 0.05
    """Sigma for Gaussian action noise. Production yaml:16."""

    reward_weight: float = 200.0
    """Multiplier inside softmax(R * reward_weight). Production yaml:17."""

    beta_filter: float = 0.7
    """Intra-horizon noise smoothing coefficient. Production yaml:20."""

    cv_fail_penalty: float = -10.0
    """Reward for CV-failed trajectories. Production yaml:24."""

    # --- waypoint sampling (off by default = vanilla MPPI) ---
    waypoints_n: int | None = None
    """K: number of sparse waypoints. None = vanilla per-step sampler.
    Production yaml:27 (null by default)."""

    waypoints_interp: str = "linear"
    """'linear' or 'cubic' (Catmull-Rom). Production yaml:36."""

    # --- action space ---
    action_dim: int = 4
    """ALOHA bimanual end-effector deltas. Production yaml:39."""

    action_lower_lim: tuple[float, ...] = (-1.0, -1.0, -1.0, -1.0)
    """Per-dim cube lower bound. Production yaml:41."""

    action_upper_lim: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    """Per-dim cube upper bound. Production yaml:42."""

    # --- delta-action MPPI mode ---
    delta_mode: bool = False
    """When True, sample per-step DELTAs (cumsum onto anchor at sample
    time). Production yaml:52."""

    delta_action_lim: float = 0.0872
    """Symmetric per-dim p99 from demos. Production yaml:53."""

    noise_level_delta: float = 0.02
    """Sigma in delta_mode. Production yaml:56."""

    cumulative_drift_log_only: bool = True
    """Log only, don't clip cumulative drift. Production yaml:57."""

    # --- vanilla-mode sample-side per-step delta clip ---
    sample_delta_clip: bool = True
    """Cap per-step Δ at demo p99. Production yaml:72 (ON by default)."""

    per_step_delta_lim: tuple[float, ...] = (0.0975, 0.0919, 0.0760, 0.0909)
    """Per-dim p99 from demos (mini dataset, 1989 transitions).
    Production yaml:78."""

    # --- audit logging ---
    audit_log_enabled: bool = False
    """Persistent per-niter audit JSON. Production yaml:96."""

    audit_log_record_samples: bool = False
    """Include samples + sample_rewards in each audit entry.
    Production yaml:97."""

    audit_log_path: str | None = None
    """When None, audit_log.json sits in the run output_dir.
    Production yaml:98."""

    # --- debug mode (stub fields; the driver/planner skip work when
    # debug_mode=False, but the field is exposed for parity with
    # production's yaml so callers can pass identical config objects). ---
    debug_mode: bool = False
    debug_dump_dir: str | None = None
    debug_dump_per_sample_rollout: bool = False
    debug_dump_per_h_cv: bool = False
    debug_dump_decoded_full: bool = False
    debug_storage_warning_gb: int = 100

    # --- run-loop pacing ---
    step_each_iter: int = 1
    """Number of actions executed per plan_step call. Production yaml:150."""

    # --- run control (consumed by the driver, not by the planner) ---
    control_steps: int = 50
    """Plan_step calls per episode. Production yaml:153."""

    seed: int = 0
    """Planner sampler RNG seed (does NOT control WM denoiser noise,
    which uses the global CUDA RNG). Production yaml:154."""

    # --- CV labeler infra ---
    cv_n_workers: int = 16
    """Persistent CV worker pool size for parallel detection.
    Production yaml:158."""

    cv_processing_resolution: int = 128
    """detect() processing resolution. Set to 128 to match production's
    CV-on-decoded-frame pipeline; the WM decodes at 128×128, no further
    resize."""

    image_diagonal: float = field(default_factory=lambda: math.sqrt(128 ** 2 + 128 ** 2))
    """Position-term divisor in the reward. sqrt(128² + 128²) ≈ 181.019
    — matches production's ``PushTWMEnv.image_diagonal``."""

    # --- WorldModelEnv plumbing ---
    device: str = "cuda"
    """Torch device for all planner tensors. Production reads device
    from env; we expose it on the Config for symmetry with diffusion-
    forcing's Config."""

    decode_batch_size: int = 16
    """Chunk size for the WorldModelEnv.step_batch decode pass. Pure
    memory knob — does not change action sampling or scoring."""

    # --- Legacy field kept for backwards compatibility with smoke
    # scripts and tests that take a Config(goal=GoalPose(...)). The
    # production-faithful reward path uses a dict goal_state instead;
    # this field is unused by the new Planner but accepted by Config()
    # construction.
    goal: GoalPose | None = None

    # ----- Legacy backwards-compat fields -----
    # Consumed only by the (now-inactive) verbatim diffusion-forcing
    # core in _mppi_core.py and by the legacy smoke scripts in
    # interactive_world_sim_mppi/scripts/smoke_*.py. The new
    # production-faithful Planner in _planner.py ignores these.
    n_waypoints: int = 2
    """LEGACY: n_look_ahead alias used by _mppi_core.py's Planner.
    Not consumed by the production-faithful _planner.py."""

    interp_pts: int = 5
    """LEGACY: interp factor for MPPI_WAYPTS in _mppi_core.py."""

    rollout_best: bool = True
    """LEGACY: rollout-best replay in _mppi_core.py. Not on
    production's algorithmic path."""

    normalize_rewards_before_softmax: bool = False
    """LEGACY: yiru's prior reward standardization, deliberately
    DISABLED by default in this reconciliation (production's MPPI
    never standardizes; doing so changes the effective softmax
    temperature). Kept as a field so the legacy verbatim core in
    _mppi_core.py still constructs."""

    pos_weight: float = 1.0
    """LEGACY: used by the legacy pusht_terminal_reward (not the
    production-faithful reward path)."""

    angle_weight: float = 1.0
    """LEGACY: used by the legacy pusht_terminal_reward (not the
    production-faithful reward path)."""

    detection_failure_penalty: float = -10.0
    """LEGACY: used by the legacy pusht_terminal_reward. Default is
    -10.0 to match production's cv_fail_penalty (yiru's prior default
    was -1000.0 — that was a divergence)."""

    detector_num_workers: int = 16
    """LEGACY: aliased by the legacy PushTTerminalReward. Default 16
    matches production's cv_n_workers."""
