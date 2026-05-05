"""MPPI planner — faithful port of diffusion-forcing's planner_v0_0 MPPI.

Algorithm spec: see [MPPI_REFERENCE_NOTES.md](../../MPPI_REFERENCE_NOTES.md)
at the repo root. Three primitives:

  1. ``sample_action_sequences(act_seq) -> (N, H, A)``
     Apply temporally-correlated noise (smoothed across the H horizon by
     ``beta_filter``) onto a running ``act_seq``. Per-step actions clipped
     to ``[action_lower_lim, action_upper_lim]``.

  2. ``optimize_action_mppi(act_seqs, rewards) -> (H, A)``
     Softmax-weighted mean of the N candidate sequences using
     ``softmax(rewards * reward_weight)``. We add a max-subtract for
     numerical stability — see notes file for the (mathematically
     identical, computationally safer) rationale.

  3. ``trajectory_optimization(z_current, goal_state, init_act_seq) ->
     (act_seq, stats)``
     Iterate sampler-rollout-score-aggregator ``n_update_iter`` times.
     Stateless across calls.

The reward function is the only intentional deviation: we use
``env.compute_reward`` (CV-based) where the reference accepts an
``evaluate_traj`` callback. Everything else matches the reference.

Old single-shot ``rl/mppi/planner.py`` is retained for the Phase 2
consistency comparison and will be deprecated after that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from env.pusht_wm_env import PushTWMEnv
from rl.mppi import distributed as D


@dataclass
class PlanStats:
    """Per-plan-step debug record. Captured for offline inspection."""

    act_seq: torch.Tensor                         # (H, A) final converged plan
    final_iter_rewards: torch.Tensor              # (N,) rewards at the LAST iteration
    final_iter_weights: torch.Tensor              # (N,) softmax weights at LAST iter
    n_cv_failures_final_iter: int                 # how many trajectories had CV fail
    iter_best_rewards: list[float] = field(default_factory=list)  # per-iter R.max()
    iteration_log: list[dict[str, Any]] = field(default_factory=list)


class MPPIPlanner:
    """Faithful port of diffusion-forcing's MPPI planner."""

    def __init__(self, env: PushTWMEnv, config: DictConfig | dict[str, Any]) -> None:
        self.env = env
        self.cfg = (
            config if isinstance(config, DictConfig) else DictConfig(config)
        )
        self.device = env.device

        self._validate_config()

        self.action_lower_lim = torch.as_tensor(
            list(self.cfg.action_lower_lim), dtype=torch.float32, device=self.device,
        )
        self.action_upper_lim = torch.as_tensor(
            list(self.cfg.action_upper_lim), dtype=torch.float32, device=self.device,
        )

        # Last plan-step debug record (overwritten each plan_step).
        self.last_stats: PlanStats | None = None

        # Reproducibility: dedicated generator for action noise so we don't
        # disturb the global CUDA RNG that the WM denoiser consumes.
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(int(self.cfg.seed))

    # ── primitives ───────────────────────────────────────────────────

    def sample_action_sequences(self, act_seq: torch.Tensor) -> torch.Tensor:
        """``(H, A) -> (N, H, A)`` faithful port of reference sampler.

        Reference: ``planner_v0_0.py:196-256``. When ``cfg.waypoints_n`` is
        a positive int K < H, we instead sample (N, K, A) waypoint deltas
        and interpolate to (N, H, A); see ``_sample_waypoints_then_interp``.
        That mode mirrors reference's ``trajectory_optimization_mppi_waypts``
        (``planner_v0_0.py:485-580``) adapted to our delta action space:
        waypoints are deltas, interpolation produces smooth deltas, and the
        downstream rollout/reward/MPPI-update operate on the (N, H, A)
        result unchanged.
        """
        K_wp = getattr(self.cfg, "waypoints_n", None)
        if K_wp is not None and int(K_wp) > 0:
            return self._sample_waypoints_then_interp(act_seq, int(K_wp))

        N = int(self.cfg.n_sample)
        H = int(self.cfg.n_look_ahead)
        A = int(self.cfg.action_dim)
        beta = float(self.cfg.beta_filter)
        sigma = float(self.cfg.noise_level)

        assert act_seq.shape == (H, A), (
            f"act_seq must be (H={H}, A={A}); got {tuple(act_seq.shape)}"
        )

        act_seqs = act_seq.unsqueeze(0).expand(N, -1, -1).clone().to(self.device)
        act_residual = torch.zeros(N, A, dtype=act_seqs.dtype, device=self.device)

        for i in range(H):
            noise_sample = torch.randn(
                (N, A), generator=self._gen, device=self.device,
                dtype=act_seqs.dtype,
            ) * sigma
            act_residual = beta * noise_sample + (1.0 - beta) * act_residual
            new_step = act_seqs[:, i] + act_residual
            new_step = torch.clamp(new_step, self.action_lower_lim, self.action_upper_lim)
            act_seqs[:, i] = new_step

        return act_seqs

    def _sample_waypoints_then_interp(
        self, act_seq: torch.Tensor, K: int,
    ) -> torch.Tensor:
        """Sample K waypoint deltas, interpolate to H per-step deltas.

        ``act_seq`` is the warm-started (H, A) running plan; we subsample
        it at K equally-spaced indices to seed the waypoint mean, apply
        the existing AR(1) Gaussian sampler with horizon=K, then up-sample
        to (N, H, A). Interpolation is linear (default) or cubic (Catmull-Rom).
        """
        N = int(self.cfg.n_sample)
        H = int(self.cfg.n_look_ahead)
        A = int(self.cfg.action_dim)
        beta = float(self.cfg.beta_filter)
        sigma = float(self.cfg.noise_level)
        mode = str(getattr(self.cfg, "waypoints_interp", "linear"))

        assert 2 <= K < H, f"waypoints_n must be in [2, H-1]; got {K} (H={H})"
        assert act_seq.shape == (H, A), (
            f"act_seq must be (H={H}, A={A}); got {tuple(act_seq.shape)}"
        )

        # Waypoint timesteps: K integer indices spanning [0, H-1] inclusive.
        t_wp = torch.linspace(0, H - 1, K).round().long().to(self.device)
        # Deduplicate (e.g., if K > H - degenerate case is excluded by assert).
        wp_base = act_seq.to(self.device)[t_wp]                # (K, A)

        # AR(1) Gaussian sampling along the K-axis (mirrors per-step sampler).
        wp_seqs = wp_base.unsqueeze(0).expand(N, -1, -1).clone()
        residual = torch.zeros(N, A, dtype=wp_seqs.dtype, device=self.device)
        for i in range(K):
            noise = torch.randn(
                (N, A), generator=self._gen, device=self.device,
                dtype=wp_seqs.dtype,
            ) * sigma
            residual = beta * noise + (1.0 - beta) * residual
            wp_seqs[:, i] = wp_seqs[:, i] + residual           # no per-waypt clamp

        # Up-sample (N, K, A) -> (N, H, A) at integer timesteps 0..H-1.
        t_out = torch.arange(H, device=self.device, dtype=torch.float32)
        act_seqs = self._interp_along_dim(wp_seqs, t_wp.float(), t_out, mode=mode)
        # Per-step clamp matches reference's per-step clamp on the action seq.
        act_seqs = torch.clamp(act_seqs, self.action_lower_lim, self.action_upper_lim)
        return act_seqs

    @staticmethod
    def _interp_along_dim(
        wp: torch.Tensor, t_wp: torch.Tensor, t_out: torch.Tensor, mode: str,
    ) -> torch.Tensor:
        """Interpolate ``(N, K, A)`` waypoints at ``t_wp`` to ``(N, len(t_out), A)``.

        ``mode`` ∈ {"linear", "cubic"}; "cubic" uses Catmull-Rom (cubic
        Hermite through the four nearest waypoints). Both produce smooth
        sequences in the sense that ``‖Δa‖₂`` between consecutive output
        timesteps is bounded by the waypoint spacing rather than by the
        per-step noise level.
        """
        N, K, A = wp.shape
        # For each t_out, find the segment [t_wp[i], t_wp[i+1]] it falls in.
        # We assume t_wp is sorted ascending integers.
        # idx_lo = largest i such that t_wp[i] <= t_out (clamped to [0, K-2])
        idx_lo = torch.clamp(
            torch.searchsorted(t_wp, t_out, right=True) - 1, 0, K - 2
        )
        idx_hi = idx_lo + 1
        t_lo = t_wp[idx_lo]
        t_hi = t_wp[idx_hi]
        u = ((t_out - t_lo) / (t_hi - t_lo).clamp_min(1e-6)).clamp(0.0, 1.0)  # (T,)

        if mode == "linear":
            # Pick (N, T, A) waypoints at idx_lo and idx_hi.
            w_lo = wp[:, idx_lo, :]
            w_hi = wp[:, idx_hi, :]
            return w_lo + (w_hi - w_lo) * u.unsqueeze(0).unsqueeze(-1)

        if mode == "cubic":
            # Catmull-Rom: needs i-1 and i+2 with edge-clamping.
            idx_prev = torch.clamp(idx_lo - 1, 0, K - 1)
            idx_next = torch.clamp(idx_hi + 1, 0, K - 1)
            w_prev = wp[:, idx_prev, :]                        # (N, T, A)
            w_lo   = wp[:, idx_lo,   :]
            w_hi   = wp[:, idx_hi,   :]
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
    ) -> torch.Tensor:
        """``(N, H, A), (N,) -> (H, A)`` softmax-weighted mean.

        Reference: ``planner_v0_0.py:390-400``. We add a max-subtract for
        numerical stability — the reference omits this, but with
        ``reward_weight=200`` and CV-fail rewards of -10, the unstabilised
        version produces ``exp(-2000)`` which underflows. The shifted form
        is mathematically identical for the softmax output.
        """
        # softmax(R * w) == softmax((R - R.max()) * w) when w > 0
        rw = float(self.cfg.reward_weight)
        # Rewards come from the CPU-side CV pipeline; bring them onto the same
        # device as the action tensors before softmax.
        reward_seqs = reward_seqs.to(act_seqs.device).float()
        scaled = reward_seqs * rw
        weights = F.softmax(scaled - scaled.max(), dim=0)  # (N,)
        return (act_seqs * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=0), weights

    # ── main planning loop ───────────────────────────────────────────

    def evaluate_trajectories(
        self,
        z_current: torch.Tensor,
        act_seqs: torch.Tensor,
        goal_state: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Roll out N trajectories, decode finals, score via CV reward.

        Args:
            z_current: ``(C, H_lat, W_lat)`` single latent.
            act_seqs: ``(N, H, A)``.
            goal_state: dict with scalar ``cx, cy, sin_theta, cos_theta``.

        Returns:
            ``(rewards (N,), state_dict_at_final)``
        """
        N = act_seqs.shape[0]
        z0_batch = z_current.unsqueeze(0).expand(N, -1, -1, -1).contiguous()
        with torch.no_grad():
            traj = self.env.rollout(z0_batch, act_seqs)  # (N, H+1, C, h, w)
        z_final = traj[:, -1]  # (N, C, h, w)
        # Free the rollout-time scratch before decode (Step 3 finding from
        # the prior project — decode attention needs ~4 GiB scratch).
        del traj
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        state = self._estimate_final_states(z_final)  # batched dict
        rewards = self.env.compute_reward(
            state, goal_state,
            image_diagonal=self.env.image_diagonal,
            cv_fail_penalty=float(self.cfg.cv_fail_penalty),
        )
        return rewards, state

    def _estimate_final_states(self, z_final: torch.Tensor) -> dict[str, Any]:
        """Decode/CV-estimate final latents, optionally in memory-safe chunks."""
        decode_batch_size = int(getattr(self.cfg, "decode_batch_size", 0) or 0)
        if decode_batch_size <= 0 or decode_batch_size >= z_final.shape[0]:
            return self.env.estimate_from_latent(z_final)

        chunks: list[dict[str, Any]] = []
        for start in range(0, z_final.shape[0], decode_batch_size):
            end = min(start + decode_batch_size, z_final.shape[0])
            chunks.append(self.env.estimate_from_latent(z_final[start:end]))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return self._cat_state_chunks(chunks)

    @staticmethod
    def _cat_state_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
        if not chunks:
            return {}
        out: dict[str, Any] = {}
        for key in chunks[0].keys():
            vals = [chunk[key] for chunk in chunks]
            if all(isinstance(v, torch.Tensor) for v in vals):
                out[key] = torch.cat([
                    v if v.dim() > 0 else v.unsqueeze(0) for v in vals
                ], dim=0)
            else:
                combined = []
                for v in vals:
                    if isinstance(v, list):
                        combined.extend(v)
                    else:
                        combined.append(v)
                out[key] = combined
        return out

    def _gather_state_dict(self, state_local: dict[str, Any]) -> dict[str, Any]:
        """All-gather a per-sample CV state dict across ranks.

        ``estimate_state`` returns CPU tensors of shape ``(K,)`` with float32
        keys (cx, cy, sin_theta, cos_theta, theta_deg, contour_area,
        icp_residual) and a bool key (success). NCCL only supports a fixed
        set of dtypes, so we promote on the way in (bool → uint8) and demote
        on the way out. The final dict matches the single-process schema
        exactly: same keys, same dtypes, same device (CPU), shape (N,).
        """
        out: dict[str, Any] = {}
        for key, val in state_local.items():
            if not isinstance(val, torch.Tensor):
                # Mock envs may return lists; fall back to plain concatenation
                # via maybe_barrier / gather_object would add complexity for
                # no production benefit, so we just keep the local slice.
                out[key] = val
                continue
            if val.dtype == torch.bool:
                gpu = val.to(self.device, dtype=torch.uint8).contiguous()
                gathered = D.all_gather_concat(gpu).to(torch.bool).cpu()
            else:
                gpu = val.to(self.device, dtype=torch.float32).contiguous()
                gathered = D.all_gather_concat(gpu).cpu()
            out[key] = gathered
        return out

    def trajectory_optimization(
        self,
        z_current: torch.Tensor,
        goal_state: dict[str, Any],
        init_act_seq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, PlanStats]:
        """One full plan: returns ``(act_seq, stats)``.

        Reference: ``planner_v0_0.py:428-483``.

        Args:
            z_current: ``(C, H_lat, W_lat)`` current latent.
            goal_state: dict with scalar ``cx, cy, sin_theta, cos_theta``.
            init_act_seq: optional ``(H, A)`` initial guess. Defaults to
                a zero sequence (matching the reference's stateless behaviour
                — caller may choose to pass the previous plan's result if
                cross-step warm-start is desired, but the reference does not).
        """
        H = int(self.cfg.n_look_ahead)
        A = int(self.cfg.action_dim)
        if init_act_seq is None:
            act_seq = torch.zeros(H, A, dtype=torch.float32, device=self.device)
        else:
            assert init_act_seq.shape == (H, A), (
                f"init_act_seq must be (H={H}, A={A}); got {tuple(init_act_seq.shape)}"
            )
            act_seq = init_act_seq.to(self.device).float()

        z_current = z_current.to(self.device)
        if z_current.dim() == 4:
            assert z_current.shape[0] == 1
            z_current = z_current[0]

        # ── multi-GPU sample-shard setup ─────────────────────────────
        # Single-process path (D.is_dist() is False): rank=0, world=1, no
        # broadcasts/gathers happen — see rl/mppi/distributed.py for the
        # short-circuit semantics. The conditionals below take the
        # else-branches and the math is byte-identical to the pre-fix code.
        N = int(self.cfg.n_sample)
        world = D.world_size()
        r = D.rank()
        if world > 1:
            assert N % world == 0, (
                f"n_sample={N} must be divisible by world_size={world}"
            )
        K = N // world  # per-rank slice size

        iter_best_rewards: list[float] = []
        iteration_log: list[dict[str, Any]] = []
        last_rewards: torch.Tensor | None = None
        last_weights: torch.Tensor | None = None
        last_state: dict[str, Any] | None = None
        last_act_seqs: torch.Tensor | None = None
        last_rewards_full: torch.Tensor | None = None  # GPU copy for re-roll

        for iter_idx in range(int(self.cfg.n_update_iter)):
            # ── 1. Sample (rank 0 only) + broadcast ──────────────────
            # Numerical-equivalence guarantee with single-GPU runs at the
            # same seed: only rank 0 advances self._gen, then broadcasts.
            if r == 0:
                act_seqs = self.sample_action_sequences(act_seq)     # (N, H, A)
            else:
                act_seqs = torch.empty(
                    (N, H, A), dtype=torch.float32, device=self.device,
                )
            D.broadcast_(act_seqs, src=0)

            # ── 2. Evaluate local slice ──────────────────────────────
            local_act = act_seqs[r * K : (r + 1) * K] if world > 1 else act_seqs
            rewards_local, state_local = self.evaluate_trajectories(
                z_current, local_act, goal_state,
            )

            # ── 3. Gather rewards + state ───────────────────────────
            if D.is_dist():
                # rewards_local arrives on CPU from env.compute_reward; move
                # to GPU for NCCL all_gather, then back to CPU to match the
                # single-GPU semantics expected downstream.
                rewards_gpu = rewards_local.to(self.device).contiguous()
                rewards = D.all_gather_concat(rewards_gpu).cpu()
                state = self._gather_state_dict(state_local)
            else:
                rewards = rewards_local
                state = state_local

            # ── 4. Optimize (deterministic on every rank) ───────────
            # Mathematically equivalent to "rank-0 optimize + broadcast"
            # but without the extra collective: every rank holds the full
            # rewards tensor (from all_gather) and runs the same softmax
            # + weighted-mean, producing bit-identical act_seq.
            act_seq, weights = self.optimize_action_mppi(act_seqs, rewards)

            iter_best_rewards.append(float(rewards.max()))
            last_rewards = rewards.detach().cpu()
            last_weights = weights.detach().cpu()
            last_state = state
            last_act_seqs = act_seqs
            last_rewards_full = rewards
            if r == 0:
                iteration_log.append(
                    self._build_iteration_record(iter_idx, rewards, weights, state=state)
                )

        # After the optimization loop, re-roll the top-K trajectories from
        # the last iteration and CV-label every step along the rollout, so
        # downstream visualization can draw full predicted polylines (not
        # just endpoints). Stays on rank 0 in multi-GPU mode (per the
        # design: top-K viz is not worth gathering for shared work).
        if r == 0:
            top_k_record = self._compute_top_k_intermediate_cv(
                z_current, last_act_seqs, last_rewards_full,
                k=int(getattr(self.cfg, "top_k_polylines", 10)),
            )
            if top_k_record is not None and iteration_log:
                iteration_log[-1].update(top_k_record)

            # Persist the LAST iteration's full N-sample action sequences so
            # downstream action-distribution viz can plot the spread MPPI
            # actually explored (not just the top-K). ~16 KB per plan_step at
            # default N=100, H=10 -- negligible.
            if iteration_log and last_act_seqs is not None:
                iteration_log[-1]["last_iter_full_actions"] = (
                    last_act_seqs.detach().cpu().float().clone()
                )

        n_fail = int((~last_state["success"]).sum().item()) if last_state else 0
        stats = PlanStats(
            act_seq=act_seq.detach().cpu(),
            final_iter_rewards=last_rewards if last_rewards is not None else torch.empty(0),
            final_iter_weights=last_weights if last_weights is not None else torch.empty(0),
            n_cv_failures_final_iter=n_fail,
            iter_best_rewards=iter_best_rewards,
            iteration_log=iteration_log,
        )
        self.last_stats = stats
        return act_seq, stats

    def plan_step(
        self,
        z_current: torch.Tensor,
        goal_state: dict[str, Any],
        init_act_seq: torch.Tensor | None = None,
        return_iteration_log: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, Any]]]:
        """Run trajectory optimization and return the first action.

        Args:
            z_current: current latent.
            goal_state: scalar goal-state dict.
            init_act_seq: optional initial action sequence.
            return_iteration_log: when True, also return the per-iteration
                reward/weight records captured during this plan step. Defaults
                to False so older callers that expect only an action keep
                working unchanged.
        """
        act_seq, stats = self.trajectory_optimization(z_current, goal_state, init_act_seq)
        action = act_seq[0]
        if return_iteration_log:
            return action, stats.iteration_log
        return action

    # ── helpers ──────────────────────────────────────────────────────

    def _compute_top_k_intermediate_cv(
        self,
        z_current: torch.Tensor,
        act_seqs: torch.Tensor | None,
        rewards: torch.Tensor | None,
        k: int = 10,
    ) -> dict[str, Any] | None:
        """Re-roll top-K trajectories (by reward) and CV-label every step.

        Returns a dict suitable for ``iteration_log[-1].update(...)`` with
        keys ``top_k_intermediate_cx``, ``top_k_intermediate_cy``,
        ``top_k_intermediate_success`` (each ``(K, H+1)``), plus
        ``top_k_indices`` and ``top_k_rewards`` (each ``(K,)``).

        Returns ``None`` when:
          * inputs are missing (no last iteration captured), or
          * the env's ``estimate_from_latent`` does not provide
            ``cx, cy, success`` (mock envs).
        """
        if act_seqs is None or rewards is None:
            return None
        N = act_seqs.shape[0]
        actual_k = min(int(k), N)
        rewards_cpu = rewards.detach().cpu().float()
        topk_indices = torch.argsort(rewards_cpu, descending=True)[:actual_k]

        top_act_seqs = act_seqs[topk_indices.to(act_seqs.device)]  # (K, H, A)
        z0_batch = z_current.unsqueeze(0).expand(actual_k, -1, -1, -1).contiguous()
        with torch.no_grad():
            traj = self.env.rollout(z0_batch, top_act_seqs)  # (K, H+1, C, h, w)
        K, Hp1 = traj.shape[:2]
        flat = traj.reshape(K * Hp1, *traj.shape[2:])
        del traj
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Reuse the chunked decode+CV path so ``decode_batch_size`` is honoured.
        state = self._estimate_final_states(flat)

        if not all(key in state for key in ("cx", "cy", "success")):
            return None

        cx_flat = state["cx"].detach().cpu().float()
        cy_flat = state["cy"].detach().cpu().float()
        success_flat = state["success"].detach().cpu()
        # Some mock envs return dim-0 tensors when batch-size 1; guard.
        if cx_flat.dim() == 0:
            cx_flat = cx_flat.unsqueeze(0)
            cy_flat = cy_flat.unsqueeze(0)
            success_flat = success_flat.unsqueeze(0)

        return {
            "top_k_intermediate_cx": cx_flat.reshape(K, Hp1),
            "top_k_intermediate_cy": cy_flat.reshape(K, Hp1),
            "top_k_intermediate_success": success_flat.reshape(K, Hp1),
            "top_k_indices": topk_indices.long(),
            "top_k_rewards": rewards_cpu[topk_indices],
        }

    def _build_iteration_record(
        self,
        iter_idx: int,
        rewards: torch.Tensor,
        weights: torch.Tensor,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a CPU-side reward-debug record for one MPPI iteration.

        ``state`` is the per-sample CV pose dict produced by
        ``evaluate_trajectories`` (keys ``cx, cy, sin_theta, cos_theta,
        success``). When provided, those tensors are persisted (CPU clone)
        so downstream visualization can plot per-sample candidate
        endpoints. When the env's ``estimate_from_latent`` returns a
        different schema (e.g. mock envs), the per-sample fields default
        to ``None``.
        """
        rewards_cpu = rewards.detach().cpu().float().clone()
        weights_cpu = weights.detach().cpu().float().clone()
        weighted_reward = (weights_cpu * rewards_cpu).sum()

        sample_cx = sample_cy = sample_sin = sample_cos = sample_success = None
        if state is not None and all(
            k in state for k in ("cx", "cy", "sin_theta", "cos_theta", "success")
        ):
            sample_cx = state["cx"].detach().cpu().float().clone()
            sample_cy = state["cy"].detach().cpu().float().clone()
            sample_sin = state["sin_theta"].detach().cpu().float().clone()
            sample_cos = state["cos_theta"].detach().cpu().float().clone()
            sample_success = state["success"].detach().cpu().clone()

        return {
            "iter": int(iter_idx),
            "rewards_all": rewards_cpu,
            "weights_all": weights_cpu,
            "reward_max": float(rewards_cpu.max().item()),
            "reward_min": float(rewards_cpu.min().item()),
            "reward_mean": float(rewards_cpu.mean().item()),
            "reward_std": float(rewards_cpu.std().item()),
            "reward_softmax_weighted": float(weighted_reward.item()),
            "sample_cx": sample_cx,
            "sample_cy": sample_cy,
            "sample_sin_theta": sample_sin,
            "sample_cos_theta": sample_cos,
            "sample_success": sample_success,
            # Top-K predicted trajectories — populated by
            # _compute_top_k_intermediate_cv on the LAST iteration only,
            # patched in via iteration_log[-1].update(...) post-loop.
            "top_k_intermediate_cx": None,
            "top_k_intermediate_cy": None,
            "top_k_intermediate_success": None,
            "top_k_indices": None,
            "top_k_rewards": None,
            # Full per-sample action set from the LAST iteration. Same
            # patching pattern as top_k_*: only set on the last iteration.
            "last_iter_full_actions": None,
        }

    def _validate_config(self) -> None:
        required = (
            "n_sample", "n_look_ahead", "n_update_iter", "noise_level",
            "reward_weight", "beta_filter",
            "action_lower_lim", "action_upper_lim", "action_dim",
            "cv_fail_penalty", "seed",
        )
        for k in required:
            if k not in self.cfg:
                raise ValueError(f"config missing required key: {k!r}")
        if len(self.cfg.action_lower_lim) != self.cfg.action_dim:
            raise ValueError(
                f"action_lower_lim length {len(self.cfg.action_lower_lim)} "
                f"!= action_dim {self.cfg.action_dim}"
            )
        if len(self.cfg.action_upper_lim) != self.cfg.action_dim:
            raise ValueError(
                f"action_upper_lim length {len(self.cfg.action_upper_lim)} "
                f"!= action_dim {self.cfg.action_dim}"
            )
        if self.cfg.action_dim != self.env.action_dim:
            raise ValueError(
                f"config action_dim {self.cfg.action_dim} != env.action_dim {self.env.action_dim}"
            )
