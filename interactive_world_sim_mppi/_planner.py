"""Production-faithful MPPI planner over WorldModelEnv.

This module is the semantic equivalent of ``rl/mppi/mppi_planner.py`` on
the production branch (HEAD ``5e2d48e``). The algorithmic primitives —
AR(1) per-step sampler, sample_delta_clip post-pass, cube clamp,
softmax max-subtract, evaluate_trajectories chain — match production
line-for-line. The differences from production are intentional:

  * env: yiru's ``WorldModelEnv`` (snapshot/restore + step_batch) rather
    than production's ``PushTWMEnv`` (rollout). Semantically equivalent
    on the rollout-and-decode path because both forward the same
    underlying ``LatentWorldModel.dynamics_forward`` and
    ``render_img_cm`` calls and return latents/RGBs of matching shape
    modulo the leading z0 entry (production keeps z0 at index 0;
    yiru's step_batch does not — handled inside this module).

  * CV: yiru's ``interactive_world_sim_cv.detect(mode='real',
    processing_resolution=128)`` rather than production's ``CVLabeler``
    at preset "REAL" resolution 128. The detection algorithm is the
    same (both vendored from the supervisor analyze.py); the HSV
    constants for "real" / "REAL" match numerically.

  * Reward: production's ``compute_reward`` formula
    ``R = -‖pos − pos_g‖₂ / image_diagonal − (1 − cos(Δθ))`` with the
    same ``cv_fail_penalty=-10.0`` and ``image_diagonal=sqrt(2)*128``.
    Implemented directly in this module (not delegated to a user
    callable, mirroring production's planner architecture).

  * RNG: dedicated ``torch.Generator(device=self.device)`` seeded from
    ``config.seed``, used for every ``torch.randn`` in this module. The
    global CUDA RNG is left untouched (the WM denoiser consumes it,
    acknowledged in production yaml line 154-156).

  * No reward standardization, no rollout_best replay, no
    ``MPPI_WAYPTS`` hardwiring. Vanilla MPPI is the default;
    ``config.waypoints_n`` may be set to opt into a Catmull-Rom
    waypoint sampler matching production's ``_sample_waypoints_then_interp``.

The legacy ``_mppi_core.py`` + ``_splines.py`` (verbatim diffusion-
forcing source) remain in the package for the bitwise-equivalence
smoke test but are NOT used by ``api.py::MPPIPlanner``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from interactive_world_sim_cv import detect, detect_batch

from .config import Config


def _make_goal_state(
    cx: float,
    cy: float,
    sin_theta: float,
    cos_theta: float,
    theta_deg: float | None = None,
) -> dict[str, Any]:
    """Build a goal_state dict compatible with both production's
    ``env.compute_reward`` and the in-module reward computation here.
    """
    if theta_deg is None:
        theta_deg = math.degrees(math.atan2(sin_theta, cos_theta))
    return {
        "cx": float(cx),
        "cy": float(cy),
        "sin_theta": float(sin_theta),
        "cos_theta": float(cos_theta),
        "theta_deg": float(theta_deg),
    }


def detect_goal_state_from_episode(
    episode_path: str,
    t: int,
    *,
    cam_key: str = "camera_1_color",
    processing_resolution: int = 128,
) -> dict[str, Any]:
    """Detect a production-style goal_state dict from an episode frame.

    Loads the requested frame, center-crops/resizes to
    ``processing_resolution`` (matching production's pre-encode
    pipeline at 128 px), and runs CV detection in mode='real'. Returns
    a dict with keys ``cx, cy, sin_theta, cos_theta, theta_deg`` —
    exactly the shape ``Planner.evaluate_trajectories`` expects as
    ``goal_state``.
    """
    import h5py

    with h5py.File(episode_path, "r") as f:
        raw = np.asarray(f[f"obs/images/{cam_key}"][t])
    h, w = raw.shape[:2]
    s = min(h, w)
    cropped = raw[(h - s) // 2 : (h - s) // 2 + s, (w - s) // 2 : (w - s) // 2 + s]
    resized = cv2.resize(
        cropped, (processing_resolution, processing_resolution),
        interpolation=cv2.INTER_AREA,
    )
    rgb_u8 = resized.astype(np.uint8)
    tpose = detect(rgb_u8, mode="real", processing_resolution=processing_resolution)
    if tpose is None:
        raise RuntimeError(
            f"Detection failed on {episode_path} t={t} cam={cam_key} "
            f"at processing_resolution={processing_resolution}."
        )
    angle_rad = math.radians(tpose.angle_deg)
    return _make_goal_state(
        cx=tpose.x, cy=tpose.y,
        sin_theta=math.sin(angle_rad), cos_theta=math.cos(angle_rad),
        theta_deg=tpose.angle_deg,
    )


def compute_reward_production(
    state: dict[str, Any],
    goal_state: dict[str, Any],
    image_diagonal: float,
    cv_fail_penalty: float,
) -> torch.Tensor:
    """Production reward formula.

    Mirrors ``env/pusht_wm_env.py::compute_reward`` line-for-line:

        R = -‖pos - pos_g‖₂ / image_diagonal - (1 - cos(Δθ))

    with ``cv_fail_penalty`` replacing R wherever ``state['success']``
    is False.
    """
    success = state["success"]
    cx, cy = state["cx"], state["cy"]
    sin_t, cos_t = state["sin_theta"], state["cos_theta"]

    gx = float(goal_state["cx"])
    gy = float(goal_state["cy"])
    gs = float(goal_state["sin_theta"])
    gc = float(goal_state["cos_theta"])

    # nan_to_num so the math is finite; torch.where below overrides the
    # entries where CV failed regardless.
    cx = torch.nan_to_num(cx, nan=gx)
    cy = torch.nan_to_num(cy, nan=gy)
    sin_t = torch.nan_to_num(sin_t, nan=gs)
    cos_t = torch.nan_to_num(cos_t, nan=gc)

    pos_dist = torch.sqrt((cx - gx) ** 2 + (cy - gy) ** 2)
    pos_term = -pos_dist / image_diagonal
    cos_delta = (sin_t * gs + cos_t * gc).clamp(-1.0, 1.0)
    ang_term = -(1.0 - cos_delta)
    reward = pos_term + ang_term
    reward = torch.where(success, reward, torch.full_like(reward, cv_fail_penalty))
    return reward


def _detect_rgbs_to_state_dict(
    rgbs_u8_hwc: np.ndarray,
    processing_resolution: int,
    n_workers: int,
) -> dict[str, torch.Tensor]:
    """Run CV detection on a batch of (N, H, W, 3) uint8 RGB frames.

    mode='real' to match production's CV labeler preset "REAL". The
    HSV constants ``HSV_LOWER_REAL`` / ``HSV_UPPER_REAL`` in yiru's
    ``_detection.py`` are numerically identical to production's
    ``HSV_PRESETS['REAL']``.

    Returns a dict in production's per-sample state-dict format:
    keys cx, cy, sin_theta, cos_theta, theta_deg, success.
    """
    if rgbs_u8_hwc.dtype != np.uint8:
        rgbs_u8_hwc = np.ascontiguousarray(
            np.clip(rgbs_u8_hwc.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
        )
    N = rgbs_u8_hwc.shape[0]
    if n_workers and n_workers > 0:
        poses = detect_batch(
            rgbs_u8_hwc, mode="real",
            processing_resolution=processing_resolution,
            num_workers=n_workers,
        )
    else:
        poses = [
            detect(rgbs_u8_hwc[i], mode="real",
                   processing_resolution=processing_resolution)
            for i in range(N)
        ]

    cx_list = []
    cy_list = []
    sin_list = []
    cos_list = []
    theta_list = []
    success_list = []
    for p in poses:
        if p is None:
            cx_list.append(float("nan"))
            cy_list.append(float("nan"))
            sin_list.append(float("nan"))
            cos_list.append(float("nan"))
            theta_list.append(float("nan"))
            success_list.append(False)
        else:
            cx_list.append(p.x)
            cy_list.append(p.y)
            sin_list.append(p.sin)
            cos_list.append(p.cos)
            theta_list.append(p.angle_deg)
            success_list.append(True)
    return {
        "cx": torch.tensor(cx_list, dtype=torch.float32),
        "cy": torch.tensor(cy_list, dtype=torch.float32),
        "sin_theta": torch.tensor(sin_list, dtype=torch.float32),
        "cos_theta": torch.tensor(cos_list, dtype=torch.float32),
        "theta_deg": torch.tensor(theta_list, dtype=torch.float32),
        "success": torch.tensor(success_list, dtype=torch.bool),
    }


class PlanStats:
    """Per-plan-step debug record. Mirrors production's ``PlanStats``."""

    __slots__ = (
        "act_seq",
        "final_iter_rewards",
        "final_iter_weights",
        "n_cv_failures_final_iter",
        "iter_best_rewards",
        "iteration_log",
    )

    def __init__(
        self,
        act_seq: torch.Tensor,
        final_iter_rewards: torch.Tensor,
        final_iter_weights: torch.Tensor,
        n_cv_failures_final_iter: int,
        iter_best_rewards: list[float] | None = None,
        iteration_log: list[dict[str, Any]] | None = None,
    ) -> None:
        self.act_seq = act_seq
        self.final_iter_rewards = final_iter_rewards
        self.final_iter_weights = final_iter_weights
        self.n_cv_failures_final_iter = n_cv_failures_final_iter
        self.iter_best_rewards = iter_best_rewards or []
        self.iteration_log = iteration_log or []


class Planner:
    """Faithful port of production's ``MPPIPlanner`` over yiru's
    ``WorldModelEnv``.

    Public methods mirror production:

      * ``plan_step(z_current, goal_state, init_act_seq, anchor=)``
        — convenience wrapper returning the first action.
      * ``trajectory_optimization(z_current_unused, goal_state, init_act_seq)``
        — the n_update_iter loop. ``z_current_unused`` is accepted for
        signature parity with production but ignored — yiru's env
        carries its own latent_window/action_window inside the snapshot,
        so the planner restores from snapshot before every rollout.
      * ``sample_action_sequences(act_seq)`` — production's vanilla
        AR(1) sampler.
      * ``optimize_action_mppi(act_seqs, rewards)`` — max-subtract
        softmax-weighted mean (no reward standardization).
      * ``evaluate_trajectories(z_current_unused, act_seqs, goal_state)``
        — env.step_batch → terminal RGB → CV → production reward.
      * ``set_anchor(anchor)`` — for delta_mode / sample_delta_clip.

    Snapshot semantics: the planner uses ``env.snapshot()`` once per
    plan call, then restores before every rollout. The env exits the
    plan in the same state it started.
    """

    def __init__(
        self,
        env,
        config: Config,
        *,
        snapshot=None,
    ) -> None:
        """
        Parameters
        ----------
        env : WorldModelEnv
            yiru's env. Must expose ``snapshot()``, ``restore()``,
            ``step_batch(actions, decode_batch_size=)``,
            ``device`` and the latent/action window contracts.
        config : Config
            Hyperparameters; defaults track production yaml.
        snapshot : EnvState, optional
            Pre-captured snapshot. When None, ``plan_step`` will call
            ``env.snapshot()`` itself.
        """
        self.env = env
        self.cfg = config
        # Resolve device from the env when not pinned in config.
        if hasattr(env, "device") and env.device is not None:
            self.device = env.device
        else:
            self.device = torch.device(config.device)

        self._validate_config()

        self.action_lower_lim = torch.as_tensor(
            list(self.cfg.action_lower_lim),
            dtype=torch.float32, device=self.device,
        )
        self.action_upper_lim = torch.as_tensor(
            list(self.cfg.action_upper_lim),
            dtype=torch.float32, device=self.device,
        )

        self.last_stats: PlanStats | None = None
        self.image_diagonal = float(self.cfg.image_diagonal)

        # Dedicated generator for action noise — production yaml:154-155
        # explicitly documents that this is intentional and does NOT
        # control the WM denoiser noise (global CUDA RNG).
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(int(self.cfg.seed))

        # Anchor for delta_mode AND vanilla sample_delta_clip. Caller
        # sets via plan_step(anchor=...) before each plan_step.
        self._anchor: torch.Tensor | None = None

        # Resolve per_step_delta_lim default if sample_delta_clip is on.
        self._sample_delta_lim: torch.Tensor | None = None
        if bool(self.cfg.sample_delta_clip):
            lim_list = list(self.cfg.per_step_delta_lim)
            if len(lim_list) != int(self.cfg.action_dim):
                raise ValueError(
                    f"per_step_delta_lim length {len(lim_list)} != "
                    f"action_dim {int(self.cfg.action_dim)}"
                )
            self._sample_delta_lim = torch.as_tensor(
                lim_list, dtype=torch.float32, device=self.device,
            )

        # Audit log (config-gated).
        self._audit_log: list[dict[str, Any]] = []
        self._audit_plan_call_idx: int = -1

        # Snapshot captured at plan_step entry.
        self._plan_snapshot = snapshot

    # ── public utility ───────────────────────────────────────────────

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Return the accumulated audit log (empty when feature disabled)."""
        return self._audit_log

    def set_anchor(self, anchor: torch.Tensor) -> None:
        """Set the absolute-action anchor.

        Used by:
          - delta_mode=True: cumsum integration onto this anchor.
          - vanilla + sample_delta_clip=True: first-step delta bounded
            against this anchor.
        """
        self._anchor = anchor.to(
            device=self.device, dtype=torch.float32,
        ).clone().detach()

    # ── algorithmic primitives ───────────────────────────────────────

    def _validate_config(self) -> None:
        if int(self.cfg.action_dim) != len(self.cfg.action_lower_lim):
            raise ValueError(
                f"action_dim={self.cfg.action_dim} disagrees with "
                f"len(action_lower_lim)={len(self.cfg.action_lower_lim)}"
            )
        if len(self.cfg.action_lower_lim) != len(self.cfg.action_upper_lim):
            raise ValueError("action_lower_lim and action_upper_lim differ in length")
        if bool(self.cfg.delta_mode) and bool(self.cfg.sample_delta_clip):
            # Production warns + force-disables sample_delta_clip;
            # we raise instead because both modes consume the anchor in
            # different ways and silently swapping is more surprising.
            raise ValueError(
                "delta_mode=True and sample_delta_clip=True are mutually "
                "exclusive (delta_mode already bounds per-step deltas)."
            )

    def _sampler_sigma_and_bounds(
        self, A: int,
    ) -> tuple[float, torch.Tensor, torch.Tensor]:
        """Pick noise sigma and per-step clip bounds for the active mode.

        Matches production's ``_sampler_sigma_and_bounds`` exactly.
        """
        if bool(self.cfg.delta_mode):
            sigma = float(self.cfg.noise_level_delta)
            lim = float(self.cfg.delta_action_lim)
            lo = torch.full((A,), -lim, dtype=torch.float32, device=self.device)
            hi = torch.full((A,),  lim, dtype=torch.float32, device=self.device)
            return sigma, lo, hi
        return (
            float(self.cfg.noise_level),
            self.action_lower_lim,
            self.action_upper_lim,
        )

    def _apply_sample_delta_clip(
        self,
        samples: torch.Tensor,
        anchor: torch.Tensor | None,
    ) -> torch.Tensor:
        """Sequentially clip per-step deltas to ±per_step_delta_lim.

        Verbatim port of production's ``_apply_sample_delta_clip``.
        Operates in-place on ``samples``.
        """
        assert self._sample_delta_lim is not None
        lim = self._sample_delta_lim
        H = samples.shape[1]

        if anchor is not None:
            anch = anchor.to(samples.device, dtype=samples.dtype)
            d0 = samples[:, 0, :] - anch.unsqueeze(0)
            d0 = torch.clamp(d0, -lim, lim)
            samples[:, 0, :] = anch.unsqueeze(0) + d0
            samples[:, 0, :] = torch.clamp(
                samples[:, 0, :], self.action_lower_lim, self.action_upper_lim,
            )
        else:
            samples[:, 0, :] = torch.clamp(
                samples[:, 0, :], self.action_lower_lim, self.action_upper_lim,
            )

        for t in range(1, H):
            dt = samples[:, t, :] - samples[:, t - 1, :]
            dt = torch.clamp(dt, -lim, lim)
            samples[:, t, :] = samples[:, t - 1, :] + dt
            samples[:, t, :] = torch.clamp(
                samples[:, t, :], self.action_lower_lim, self.action_upper_lim,
            )
        return samples

    def sample_action_sequences(self, act_seq: torch.Tensor) -> torch.Tensor:
        """``(H, A) -> (N, H, A)`` of ABSOLUTE action sequences.

        Verbatim port of production's ``sample_action_sequences``:

          1. noise = randn(generator=self._gen) * sigma
          2. act_residual = beta * noise + (1-beta) * act_residual
          3. new_step = act_seqs[:, i] + act_residual
          4. clamp to [lo, hi]
          5. (delta_mode) cumsum onto anchor + cube clamp
             (vanilla + sample_delta_clip) post-pass clip + cube re-clamp
        """
        K_wp = self.cfg.waypoints_n
        if K_wp is not None and int(K_wp) > 0:
            return self._sample_waypoints_then_interp(act_seq, int(K_wp))

        N = int(self.cfg.n_sample)
        H = int(self.cfg.n_look_ahead)
        A = int(self.cfg.action_dim)
        beta = float(self.cfg.beta_filter)
        delta_mode = bool(self.cfg.delta_mode)
        sigma, lo, hi = self._sampler_sigma_and_bounds(A)

        assert act_seq.shape == (H, A), (
            f"act_seq must be (H={H}, A={A}); got {tuple(act_seq.shape)}"
        )

        act_seqs = act_seq.unsqueeze(0).expand(N, -1, -1).clone().to(self.device)
        act_residual = torch.zeros(N, A, dtype=act_seqs.dtype, device=self.device)

        for i in range(H):
            noise_sample = torch.randn(
                (N, A),
                generator=self._gen, device=self.device,
                dtype=act_seqs.dtype,
            ) * sigma
            act_residual = beta * noise_sample + (1.0 - beta) * act_residual
            new_step = act_seqs[:, i] + act_residual
            new_step = torch.clamp(new_step, lo, hi)
            act_seqs[:, i] = new_step

        if delta_mode:
            if self._anchor is None:
                raise RuntimeError(
                    "sample_action_sequences: delta_mode=True requires "
                    "planner.set_anchor(...) before plan_step. Got "
                    "self._anchor=None."
                )
            abs_seqs = self._anchor.view(1, 1, -1) + act_seqs.cumsum(dim=1)
            abs_seqs = torch.clamp(
                abs_seqs, self.action_lower_lim, self.action_upper_lim,
            )
            return abs_seqs

        if self._sample_delta_lim is not None:
            act_seqs = self._apply_sample_delta_clip(act_seqs, self._anchor)
        return act_seqs

    def _sample_waypoints_then_interp(
        self, act_seq: torch.Tensor, K: int,
    ) -> torch.Tensor:
        """Verbatim port of production's ``_sample_waypoints_then_interp``."""
        N = int(self.cfg.n_sample)
        H = int(self.cfg.n_look_ahead)
        A = int(self.cfg.action_dim)
        beta = float(self.cfg.beta_filter)
        delta_mode = bool(self.cfg.delta_mode)
        sigma, lo, hi = self._sampler_sigma_and_bounds(A)
        mode = str(self.cfg.waypoints_interp)

        assert 2 <= K < H, f"waypoints_n must be in [2, H-1]; got {K} (H={H})"
        assert act_seq.shape == (H, A), (
            f"act_seq must be (H={H}, A={A}); got {tuple(act_seq.shape)}"
        )

        t_wp = torch.linspace(0, H - 1, K).round().long().to(self.device)
        wp_base = act_seq.to(self.device)[t_wp]

        wp_seqs = wp_base.unsqueeze(0).expand(N, -1, -1).clone()
        residual = torch.zeros(N, A, dtype=wp_seqs.dtype, device=self.device)
        for i in range(K):
            noise = torch.randn(
                (N, A),
                generator=self._gen, device=self.device,
                dtype=wp_seqs.dtype,
            ) * sigma
            residual = beta * noise + (1.0 - beta) * residual
            wp_seqs[:, i] = wp_seqs[:, i] + residual

        t_out = torch.arange(H, device=self.device, dtype=torch.float32)
        act_seqs = self._interp_along_dim(wp_seqs, t_wp.float(), t_out, mode=mode)
        act_seqs = torch.clamp(act_seqs, lo, hi)

        if delta_mode:
            if self._anchor is None:
                raise RuntimeError(
                    "_sample_waypoints_then_interp: delta_mode=True requires "
                    "planner.set_anchor(...) before plan_step."
                )
            abs_seqs = self._anchor.view(1, 1, -1) + act_seqs.cumsum(dim=1)
            abs_seqs = torch.clamp(
                abs_seqs, self.action_lower_lim, self.action_upper_lim,
            )
            return abs_seqs

        if self._sample_delta_lim is not None:
            act_seqs = self._apply_sample_delta_clip(act_seqs, self._anchor)
        return act_seqs

    @staticmethod
    def _interp_along_dim(
        wp: torch.Tensor, t_wp: torch.Tensor, t_out: torch.Tensor, mode: str,
    ) -> torch.Tensor:
        """Verbatim port of production's ``_interp_along_dim``.

        mode ∈ {"linear", "cubic"}; cubic is Catmull-Rom.
        """
        N, K, A = wp.shape
        idx_lo = torch.clamp(
            torch.searchsorted(t_wp, t_out, right=True) - 1, 0, K - 2,
        )
        idx_hi = idx_lo + 1
        t_lo = t_wp[idx_lo]
        t_hi = t_wp[idx_hi]
        u = ((t_out - t_lo) / (t_hi - t_lo).clamp_min(1e-6)).clamp(0.0, 1.0)

        if mode == "linear":
            w_lo = wp[:, idx_lo, :]
            w_hi = wp[:, idx_hi, :]
            return w_lo + (w_hi - w_lo) * u.unsqueeze(0).unsqueeze(-1)

        if mode == "cubic":
            idx_prev = torch.clamp(idx_lo - 1, 0, K - 1)
            idx_next = torch.clamp(idx_hi + 1, 0, K - 1)
            w_prev = wp[:, idx_prev, :]
            w_lo = wp[:, idx_lo, :]
            w_hi = wp[:, idx_hi, :]
            w_next = wp[:, idx_next, :]
            u3 = u.unsqueeze(0).unsqueeze(-1)
            u2 = u3 * u3
            u1 = u2 * u3
            return 0.5 * (
                (2.0 * w_lo)
                + (-w_prev + w_hi) * u3
                + (2.0 * w_prev - 5.0 * w_lo + 4.0 * w_hi - w_next) * u2
                + (-w_prev + 3.0 * w_lo - 3.0 * w_hi + w_next) * u1
            )

        raise ValueError(f"unknown waypoints_interp: {mode!r}")

    def optimize_action_mppi(
        self,
        act_seqs: torch.Tensor,
        reward_seqs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Softmax-weighted mean with numerical-stability max-subtract.

        Production: ``softmax(R*w - (R*w).max())`` where w=reward_weight.
        No reward standardization (yiru's old default was a divergence;
        we drop it).
        """
        rw = float(self.cfg.reward_weight)
        reward_seqs = reward_seqs.to(act_seqs.device).float()
        scaled = reward_seqs * rw
        weights = F.softmax(scaled - scaled.max(), dim=0)
        new_mean = (act_seqs * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=0)
        return new_mean, weights

    def evaluate_trajectories(
        self,
        z_current_unused: Any,
        act_seqs: torch.Tensor,
        goal_state: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Roll out N trajectories via env.step_batch, score via CV reward.

        ``z_current_unused`` is accepted for signature parity with
        production but ignored — the env's snapshot carries the
        latent/action history.
        """
        assert self._plan_snapshot is not None, (
            "evaluate_trajectories called outside plan_step; "
            "self._plan_snapshot is None"
        )
        # Restore env state before this rollout. The env will be
        # restored again on the next call.
        self.env.restore(self._plan_snapshot)
        with torch.no_grad():
            batched = self.env.step_batch(
                act_seqs,
                decode_batch_size=int(self.cfg.decode_batch_size),
            )
        # latents: (N, H, C, h_lat, w_lat). z_final equivalent.
        # rgbs: (N, H, 3, H_img, W_img) uint8 channel-first.
        terminal_rgbs_chw = batched.rgbs[:, -1]  # (N, 3, H_img, W_img)
        # CV expects HWC. Transpose now.
        terminal_rgbs_hwc = np.ascontiguousarray(
            terminal_rgbs_chw.transpose(0, 2, 3, 1)
        )
        state = _detect_rgbs_to_state_dict(
            terminal_rgbs_hwc,
            processing_resolution=int(self.cfg.cv_processing_resolution),
            n_workers=int(self.cfg.cv_n_workers),
        )
        rewards = compute_reward_production(
            state, goal_state,
            image_diagonal=self.image_diagonal,
            cv_fail_penalty=float(self.cfg.cv_fail_penalty),
        )
        return rewards, state

    def trajectory_optimization(
        self,
        z_current_unused: Any,
        goal_state: dict[str, Any],
        init_act_seq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, PlanStats]:
        """One full plan_step: returns (act_seq, stats).

        Verbatim port of production's ``trajectory_optimization``
        (modulo the distributed-MPPI machinery which yiru does not have;
        single-process behaviour is bit-identical at the same seed
        modulo WM denoiser noise).
        """
        H = int(self.cfg.n_look_ahead)
        A = int(self.cfg.action_dim)
        if init_act_seq is None:
            act_seq = torch.zeros(H, A, dtype=torch.float32, device=self.device)
        else:
            assert init_act_seq.shape == (H, A), (
                f"init_act_seq must be (H={H}, A={A}); got "
                f"{tuple(init_act_seq.shape)}"
            )
            act_seq = init_act_seq.to(self.device).float()

        iter_best_rewards: list[float] = []
        iteration_log: list[dict[str, Any]] = []
        last_rewards: torch.Tensor | None = None
        last_weights: torch.Tensor | None = None
        last_state: dict[str, Any] | None = None

        audit_enabled = bool(self.cfg.audit_log_enabled) or bool(self.cfg.debug_mode)
        audit_record_samples = (
            bool(self.cfg.audit_log_record_samples) or bool(self.cfg.debug_mode)
        )
        if audit_enabled:
            self._audit_plan_call_idx += 1
            self._audit_log.append({
                "plan_call_idx": int(self._audit_plan_call_idx),
                "niter": -1,
                "mean": act_seq.detach().cpu().float().numpy().tolist(),
                "best_sample_reward": None,
            })

        for iter_idx in range(int(self.cfg.n_update_iter)):
            act_seqs = self.sample_action_sequences(act_seq)
            rewards, state = self.evaluate_trajectories(
                z_current_unused, act_seqs, goal_state,
            )
            act_seq, weights = self.optimize_action_mppi(act_seqs, rewards)

            iter_best_rewards.append(float(rewards.max().item()))
            last_rewards = rewards.detach().cpu()
            last_weights = weights.detach().cpu()
            last_state = state

            iteration_log.append({
                "iter": int(iter_idx),
                "rewards_all": rewards.detach().cpu().float().clone(),
                "weights_all": weights.detach().cpu().float().clone(),
                "reward_max": float(rewards.max().item()),
                "reward_min": float(rewards.min().item()),
                "reward_mean": float(rewards.mean().item()),
                "reward_std": float(rewards.std().item()),
            })
            if audit_enabled:
                entry: dict[str, Any] = {
                    "plan_call_idx": int(self._audit_plan_call_idx),
                    "niter": int(iter_idx),
                    "mean": act_seq.detach().cpu().float().numpy().tolist(),
                    "best_sample_reward": float(rewards.max().item()),
                }
                if audit_record_samples:
                    entry["samples"] = (
                        act_seqs.detach().cpu().float().numpy().tolist()
                    )
                    entry["sample_rewards"] = (
                        rewards.detach().cpu().float().numpy().tolist()
                    )
                self._audit_log.append(entry)

        n_fail = (
            int((~last_state["success"]).sum().item())
            if last_state is not None else 0
        )
        stats = PlanStats(
            act_seq=act_seq.detach().cpu(),
            final_iter_rewards=(
                last_rewards if last_rewards is not None else torch.empty(0)
            ),
            final_iter_weights=(
                last_weights if last_weights is not None else torch.empty(0)
            ),
            n_cv_failures_final_iter=n_fail,
            iter_best_rewards=iter_best_rewards,
            iteration_log=iteration_log,
        )
        self.last_stats = stats
        return act_seq, stats

    def plan_step(
        self,
        z_current_unused: Any,
        goal_state: dict[str, Any],
        init_act_seq: torch.Tensor | None = None,
        return_iteration_log: bool = False,
        *,
        anchor: torch.Tensor | None = None,
        snapshot=None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, Any]]]:
        """Run a full MPPI plan from the current env snapshot.

        Parameters mirror production's ``plan_step`` (the first
        positional arg is the latent in production; here it is unused
        because the env's snapshot carries it).
        """
        delta_mode = bool(self.cfg.delta_mode)
        sample_delta_clip = bool(self.cfg.sample_delta_clip)
        if delta_mode:
            if anchor is None:
                raise RuntimeError(
                    "plan_step: delta_mode=True requires anchor=...; "
                    "got anchor=None."
                )
            self.set_anchor(anchor)
        elif sample_delta_clip:
            if anchor is None:
                raise RuntimeError(
                    "plan_step: sample_delta_clip=True requires anchor=...; "
                    "got anchor=None."
                )
            self.set_anchor(anchor)

        # Capture a snapshot once per plan_step.
        if snapshot is not None:
            self._plan_snapshot = snapshot
        else:
            self._plan_snapshot = self.env.snapshot()

        try:
            act_seq, stats = self.trajectory_optimization(
                z_current_unused, goal_state, init_act_seq,
            )
        finally:
            # Always restore the env to the snapshot — the caller
            # decides whether to advance.
            self.env.restore(self._plan_snapshot)

        action = act_seq[0]
        if return_iteration_log:
            return action, stats.iteration_log
        return action
