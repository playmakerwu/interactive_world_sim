"""Configuration for the MPPI planner.

Single source of truth for all tunable hyperparameters. Every field that
the verbatim-copied algorithmic core (in _mppi_core.py) reads is exposed
here. No magic numbers live in _mppi_core.py or _splines.py — the
core reads from a Config instance passed to Planner.__init__.

The Config dataclass is frozen so that an instance can be hashed and
shared safely across processes. To override a field, use
`dataclasses.replace(cfg, n_sample=200)`.

Source citations (where a default value originates) reference:
  - planner_v0_0.yaml in diffusion-forcing (paths relative to that repo)
  - planner_v0_0.py in diffusion-forcing
  - "OURS" for fields we introduced

diffusion-forcing reference SHA: 180a2639a01c593c1a73275abe42b3acf4afc162
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoalPose:
    """Target T-block pose in the detector's processing-resolution pixel space.

    For the default processing_resolution=512, x and y are in [0, 512).
    angle_deg is in degrees (matching aloha's raw unwrapped output).
    Constructed once at startup (e.g., by detecting on a real episode
    frame) and frozen for the duration of the run.
    """

    x: float
    y: float
    angle_deg: float


@dataclass(frozen=True)
class Config:
    """All tunable hyperparameters for the MPPI planner.

    Every field that the algorithmic core uses is exposed here. The
    verbatim-copied planner code reads these via attribute access (set
    in Planner.__init__ from this Config). No magic numbers live in
    _mppi_core.py or _splines.py.
    """

    # ----- MPPI core (from planner_v0_0.yaml) -----
    n_sample: int = 100
    """K: number of action trajectories sampled per MPPI iteration.
    Source: planner_v0_0.yaml:3."""

    n_waypoints: int = 2
    """Number of waypoints per plan; replaces diffusion-forcing's
    n_look_ahead in MPPI_WAYPTS. OURS: small for first runs; spline
    through [curr_pos, w1, w2] = 3 knots produces a 10-step dense
    sequence at interp_pts=5."""

    interp_pts: int = 5
    """Dense actions per waypoint segment; replaces skip_frame.
    OURS: with n_waypoints=2 and interp_pts=5, dense horizon = 10."""

    n_update_iter: int = 50
    """Number of MPPI iterations per plan() call.
    Source: planner_v0_0.yaml:5. n_update_iter=0 is allowed (returns
    the warm-started initial mean unchanged)."""

    reward_weight: float = 200.0
    """Inverse temperature for the softmax over rewards.
    Higher = more peaked weights = exploit; lower = explore.
    Source: planner_v0_0.yaml:6."""

    noise_level: float = 0.05
    """Gaussian std for action noise.
    Source: planner_v0_0.yaml:7."""

    beta_filter: float = 0.7
    """AR(1) low-pass coefficient: residual = beta*new + (1-beta)*old.
    Source: planner_v0_0.yaml:8."""

    rollout_best: bool = True
    """If True, replay the final mean once to capture clean
    best_model_output / best_eval_output. Source: planner_v0_0.yaml:9."""

    action_lower_lim: tuple[float, ...] = (-1.0, -1.0, -1.0, -1.0)
    """Per-dim lower clip on actions. Hardcoded 4D for pusht_cam1.
    Source: planner_v0_0.yaml:10 (which uses 2D for original PushT)."""

    action_upper_lim: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    """Per-dim upper clip on actions. Hardcoded 4D for pusht_cam1.
    Source: planner_v0_0.yaml:11 (2D in original)."""

    # ----- Closed-loop / receding horizon -----
    step_each_iter: int = 1
    """Number of waypoints to advance per plan() call in closed-loop
    operation. The dense actions executed per plan = step_each_iter
    * interp_pts. Source: pattern from exp_sim_control.py:107
    (STEP_EACH_ITER constant); semantics resolved in our Phase 2
    Design §6.2."""

    # ----- OUR extensions to the algorithm -----
    normalize_rewards_before_softmax: bool = True
    """OURS: standardize reward_seqs (subtract mean, divide std)
    before the softmax in optimize_action_mppi. diffusion-forcing
    relies on tuning reward_weight to match reward scale; we
    normalize first for scale-invariance. Set False to recover
    bitwise equivalence with diffusion-forcing's algorithm."""

    # ----- Reward shaping (for pusht_terminal_reward) -----
    pos_weight: float = 1.0
    """OURS: position-distance weight in reward =
    -(pos_weight * pos_dist + angle_weight * angle_dist_rad)."""

    angle_weight: float = 1.0
    """OURS: angle-distance weight (in radians) in the reward."""

    detection_failure_penalty: float = -1000.0
    """OURS: reward assigned when the CV detector returns None
    on a rollout's terminal frame."""

    # ----- Runtime / task -----
    goal: GoalPose | None = None
    """Target pose. Set at startup, frozen thereafter. MPPIPlanner
    requires this to be non-None (either via Config(goal=...) or
    via the planner's goal= kwarg)."""

    device: str = "cuda"
    """Torch device for all planner tensors. Matches
    diffusion-forcing's default. Source: planner_v0_0.yaml:12."""

    detector_num_workers: int = 8
    """Workers for parallel CV detection in PushTTerminalReward.
    0 or 1 → sequential. OURS: not in diffusion-forcing."""

    cv_processing_resolution: int = 512
    """detect()'s processing_resolution. Frozen at 512 (the detector
    default and aloha-pipeline native). Exposed in case future use
    cases change it. OURS: not in diffusion-forcing."""
