"""Verbatim-copied MPPI core from diffusion-forcing.

Source: /home/yiru-wu/Documents/diffusion-forcing/algorithms/latent_dynamics/planner_v0_0.py
Reference SHA: 180a2639a01c593c1a73275abe42b3acf4afc162
Copy date: 2026-05-12

DO NOT modify the algorithmic content of this file. If diffusion-forcing
is updated upstream and we want the new behavior, re-copy rather than
editing in place. The bitwise equivalence test in
scripts/smoke_bitwise_equivalence.py (with
normalize_rewards_before_softmax=False) guards against accidental drift.

Symbols copied:
    ModelOutput                                     planner_v0_0.py:52-56
    EvalOutput                                      planner_v0_0.py:59-64
    TrajOptOutput                                   planner_v0_0.py:67-78
    Planner.__init__                                planner_v0_0.py:84-178 (with substitutions)
    Planner.register_model_rollout_fn               planner_v0_0.py:180-182
    Planner.register_evaluate_traj_fn               planner_v0_0.py:184-186
    Planner.register_visualize_fn                   planner_v0_0.py:188-190
    Planner.register_sample_action_sequences_fn     planner_v0_0.py:192-194
    Planner.sample_action_sequences_default         planner_v0_0.py:196-256
    Planner.optimize_action                         planner_v0_0.py:302-342 (MPPI/MPPI_WAYPTS only)
    Planner.optimize_action_mppi                    planner_v0_0.py:390-400 (with OUR normalize extension)
    Planner.clip_actions                            planner_v0_0.py:421-426
    Planner.trajectory_optimization                 planner_v0_0.py:344-388 (MPPI/MPPI_WAYPTS only)
    Planner.trajectory_optimization_mppi            planner_v0_0.py:428-483
    Planner.trajectory_optimization_mppi_waypts     planner_v0_0.py:485-580

Dropped:
    fps_np (planner_v0_0.py:27-49)                              — used only by MPPI_BY_ACC
    Planner.generate_action_sequences_by_acc (258-300)          — MPPI_BY_ACC only
    Planner.optimize_action_gd (402-419)                        — GD variant
    Planner.trajectory_optimization_robust_mppi_waypts (582-661) — research extension
    Planner.trajectory_optimization_mppi_by_acc (663-752)        — MPPI_BY_ACC only
    Planner.trajectory_optimization_gd (754-797)                 — GD variant
    torch.autograd.set_detect_anomaly(True) (line 14)           — module-level side effect
    Top-level tuning-tips comment (16-24)                       — moved to IMPLEMENTATION_REPORT.md

Imports trimmed:
    Removed `numpy as np` (only fps_np used it)
    Removed `from omegaconf.listconfig import ListConfig` (replaced with (list, tuple))
    `from tqdm import tqdm` is KEPT (otherwise the verbose-branch lines
    in trajectory_optimization_mppi{,_waypts} would diverge from the
    source byte-by-byte). With verbose hardcoded to False, the tqdm
    branches are dead code; the import is paid for code identity.

Substitutions (literal → Config attribute), with "# was: ..." comments:
    See Planner.__init__ below — config dict reads become attribute reads on
    our Config dataclass. Every substitution is on a single line with a
    "# was: ... (planner_v0_0.py:LINE)" comment.

The ONE algorithmic extension beyond verbatim is in optimize_action_mppi:
    Optional normalization (subtract mean, divide std) of reward_seqs
    before the softmax, gated by config.normalize_rewards_before_softmax.
    When False, the function is byte-equivalent to the source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ._splines import cubic_spline_nd_function_torch
from .config import Config


# Verbatim from planner_v0_0.py:52-56
@dataclass
class ModelOutput:
    """Model output dataclass for trajectory optimization."""

    state_seqs: torch.Tensor  # (n_sample, n_look_ahead, state_dim)


# Verbatim from planner_v0_0.py:59-64
@dataclass
class EvalOutput:
    """Eval output dataclass for trajectory optimization."""

    reward_seqs: torch.Tensor  # (n_sample)
    latent_reward_seqs: Optional[torch.Tensor] = None  # (n_sample)


# Verbatim from planner_v0_0.py:67-78
@dataclass
class TrajOptOutput:
    """Trajectory optimization output dataclass."""

    act_seq: torch.Tensor  # (n_look_ahead, action_dim)
    waypts_seq: Optional[torch.Tensor] = None  # (n_look_ahead, action_dim)
    model_outputs: Optional[list] = None  # list of ModelOutput
    eval_outputs: Optional[list] = None  # list of EvalOutput
    best_model_output: Optional[ModelOutput] = None
    best_eval_output: Optional[EvalOutput] = None
    acc_seq: Optional[torch.Tensor] = None  # (n_look_ahead, action_dim)
    next_vel: Optional[torch.Tensor] = None  # (action_dim)


class Planner(object):
    """Trajectory optimization planner."""

    # Verbatim from planner_v0_0.py:84-178 (with documented substitutions).
    def __init__(self, config: Config) -> None:
        # config contains following keys:

        # REQUIRED
        # - action_dim: the dimension of the action
        # - model_rollout_fn:
        #   - description: the function to rollout the model
        #   - input:
        #     - state_cur (shape: [n_his, state_dim] torch tensor)
        #     - action_seqs (shape: [n_sample, n_look_ahead, action_dim] torch tensor)
        #   - output: a dict containing the following keys:
        #     - state_seqs: the sequence of the state, shape: [n_sample, n_look_ahead,
        #       state_dim] torch tensor
        #     - any other keys that you want to return
        # - evaluate_traj_fn:
        #   - description: the function to evaluate the trajectory
        #   - input:
        #     - state_seqs (shape: [n_sample, n_look_ahead, state_dim] torch tensor)
        #     - action_seqs (shape: [n_sample, n_look_ahead, action_dim] torch tensor)
        #   - output: a dict containing the following keys:
        #     - reward_seqs (shape: [n_sample] torch tensor)
        #     - any other keys that you want to return
        # - n_sample: the number of action trajectories to sample
        # - n_look_ahead: the number of steps to look ahead
        # - n_update_iter: the number of iterations to update the action sequence
        # - reward_weight: the weight of the reward to aggregate action sequences
        # - action_lower_lim:
        #   - description: the lower limit of the action
        #   - shape: [action_dim]
        #   - type: torch tensor
        # - action_upper_lim: the upper limit of the action
        #   - description: the upper limit of the action
        #   - shape: [action_dim]
        #   - type: torch tensor
        # - planner_type: the type of the planner (options: 'GD', 'MPPI', 'MPPI_GD')
        self.config = config
        self.action_dim = len(config.action_lower_lim)  # was: config["action_dim"] (planner_v0_0.py:120). Inferred from action limits for our 4D Config.
        self.n_sample = config.n_sample  # was: config["n_sample"] (planner_v0_0.py:121)
        self.n_look_ahead = config.n_waypoints  # was: config["n_look_ahead"] (planner_v0_0.py:122). Renamed: our Config field is n_waypoints; algorithm attribute kept as n_look_ahead for verbatim downstream code.
        self.n_update_iter = config.n_update_iter  # was: config["n_update_iter"] (planner_v0_0.py:123)
        self.reward_weight = config.reward_weight  # was: config["reward_weight"] (planner_v0_0.py:124)
        self.action_lower_lim = torch.tensor(config.action_lower_lim)  # was: torch.tensor(config["action_lower_lim"]) (planner_v0_0.py:125)
        self.action_upper_lim = torch.tensor(config.action_upper_lim)  # was: torch.tensor(config["action_upper_lim"]) (planner_v0_0.py:126)
        self.planner_type = "MPPI_WAYPTS"  # was: config["planner_type"] (planner_v0_0.py:127). We only ship MPPI_WAYPTS in the public API; smoke tests can override.
        # Stripped MPPI_BY_ACC branch (planner_v0_0.py:128-133) — not in our scope.
        assert self.planner_type in [
            "GD",
            "MPPI",
            "MPPI_GD",
            "MPPI_WAYPTS",
            "ROBUST_MPPI_WAYPTS",
            "MPPI_BY_ACC",
        ]
        assert self.action_lower_lim.shape == (self.action_dim,)
        assert self.action_upper_lim.shape == (self.action_dim,)
        assert type(self.action_lower_lim) == torch.Tensor
        assert type(self.action_upper_lim) == torch.Tensor

        # OPTIONAL
        # - device: 'cpu' or 'cuda', default: 'cuda'
        # - verbose: True or False, default: False
        # - sampling_action_seq_fn:
        #   - description: the function to sample the action sequence
        #   - input: init_act_seq (shape: [n_look_ahead, action_dim] torch tensor)
        #   - output: act_seqs (shape: [n_sample, n_look_ahead, action_dim]
        #     torch tensor)
        #   - default: sample action sequences from a normal distribution
        # - noise_level: the level of the noise, default: 0.1
        # - n_his: the number of history states to use, default: 1
        # - rollout_best: whether rollout the best act_seq and get model prediction and
        #   reward. True or False, default: True
        # - lr: the learning rate of the optimizer, default: 1e-3
        self.device = config.device  # was: config["device"] if "device" in config else "cuda" (planner_v0_0.py:161)
        self.verbose = False  # was: config["verbose"] if "verbose" in config else False (planner_v0_0.py:162). Hardcoded — we log from outside.
        self.sample_action_sequences: Callable = self.sample_action_sequences_default
        self.noise_level = config.noise_level  # was: config["noise_level"] if "noise_level" in config else 0.1 (planner_v0_0.py:164)
        self.n_his = 1  # was: config["n_his"] if "n_his" in config else 1 (planner_v0_0.py:165). Hardcoded — our env always uses 1-step current-pos history.
        self.rollout_best = config.rollout_best  # was: config["rollout_best"] if "rollout_best" in config else True (planner_v0_0.py:166)
        self.lr = 1e-3  # was: config["lr"] if "lr" in config else 1e-3 (planner_v0_0.py:167). Unused (no GD variant), kept for shape compat.
        self.beta_filter = config.beta_filter  # was: config["beta_filter"] if "beta_filter" in config else 0.7 (planner_v0_0.py:168)
        # OUR extension — see optimize_action_mppi below.
        self.normalize_rewards_before_softmax = config.normalize_rewards_before_softmax

        # convert torch tensor device
        self.action_lower_lim = self.action_lower_lim.to(self.device)
        self.action_upper_lim = self.action_upper_lim.to(self.device)
        # Stripped MPPI_BY_ACC device transfer (planner_v0_0.py:173-178) — not in our scope.

    # Verbatim from planner_v0_0.py:180-182
    def register_model_rollout_fn(self, model_rollout_fn: Callable) -> None:
        """Register the model rollout function."""
        self.model_rollout: Callable = model_rollout_fn

    # Verbatim from planner_v0_0.py:184-186
    def register_evaluate_traj_fn(self, evaluate_traj_fn: Callable) -> None:
        """Register the evaluate trajectory function."""
        self.evaluate_traj: Callable = evaluate_traj_fn

    # Verbatim from planner_v0_0.py:188-190
    def register_visualize_fn(self, visualize_fn: Callable) -> None:
        """Register the visualize function."""
        self.visualize: Callable = visualize_fn

    # Verbatim from planner_v0_0.py:192-194
    def register_sample_action_sequences_fn(self, sample_fn: Callable) -> None:
        """Register the sample action sequences function."""
        self.sample_action_sequences = sample_fn

    # Verbatim from planner_v0_0.py:196-256
    def sample_action_sequences_default(
        self, act_seq: torch.Tensor, n_sample: Optional[int] = None
    ) -> torch.Tensor:
        """Generate a batch of action sequences with added noise.

        Args:
            act_seq (torch.Tensor): Initial action sequence of shape [n_look_ahead,
            action_dim].
            n_sample (int, optional): Number of samples to generate. If None, use the
            default value from the config.

        Returns:
            torch.Tensor: Batch of action sequences of shape [n_sample, n_look_ahead,
            action_dim].

        Raises:
            ValueError: If an unknown noise type is specified.

        Notes:
        - The generated action sequences are clipped to the range specified by
          action_lower_lim and action_upper_lim.
        """
        assert act_seq.shape == (self.n_look_ahead, self.action_dim)

        if n_sample is None:
            n_sample = self.n_sample

        # [n_sample, n_look_ahead, action_dim]
        act_seqs = torch.stack([act_seq.clone()] * n_sample)

        # [n_sample, action_dim]
        act_residual = torch.zeros(
            (n_sample, self.action_dim), dtype=act_seqs.dtype, device=self.device
        )

        # actions that go as input to the dynamics network
        for i in range(self.n_look_ahead):
            if isinstance(self.noise_level, float):
                noise = torch.ones((n_sample, self.action_dim), device=self.device)
                noise = noise * self.noise_level
            elif isinstance(self.noise_level, (list, tuple)):  # was: isinstance(self.noise_level, ListConfig) (planner_v0_0.py:236). Dropped omegaconf dep.
                noise = torch.Tensor(self.noise_level).to(self.device)
                noise = noise[None, :].repeat(n_sample, 1)
            noise_sample = torch.normal(0, noise)
            noise_sample = noise_sample.to(self.device)

            act_residual = self.beta_filter * noise_sample + act_residual * (
                1.0 - self.beta_filter
            )

            # add the perturbation to the action sequence
            act_seqs[:, i] += act_residual

            # clip to range
            act_seqs[:, i] = torch.clamp(
                act_seqs[:, i], self.action_lower_lim, self.action_upper_lim
            )

        assert act_seqs.shape == (n_sample, self.n_look_ahead, self.action_dim)
        assert type(act_seqs) == torch.Tensor
        return act_seqs

    # Verbatim from planner_v0_0.py:302-342, with non-MPPI branches stripped.
    def optimize_action(
        self,
        act_seqs: torch.Tensor,
        reward_seqs: torch.Tensor,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> torch.Tensor:
        """Optimize the action sequences based on the given reward sequences.

        Args:
            act_seqs (torch.Tensor): Action sequences with shape [n_sample, n_look_ahead
            , action_dim].
            reward_seqs (torch.Tensor): Reward sequences with shape [n_sample].
            optimizer (optional): Optimizer for gradient descent, default is None.

        Returns:
            torch.Tensor: Optimized action sequences.

        Raises:
            AssertionError: If the shapes of act_seqs or reward_seqs are incorrect or if
            they are not torch tensors.
            NotImplementedError: If the planner type is "MPPI_GD".
            ValueError: If the planner type is unknown.
        """
        # assert act_seqs.shape == (self.n_sample, self.n_look_ahead, self.action_dim)
        assert reward_seqs.shape == (self.n_sample,)
        assert type(act_seqs) == torch.Tensor
        assert type(reward_seqs) == torch.Tensor

        if self.planner_type in [
            "MPPI",
            "MPPI_WAYPTS",
            "MPPI_BY_ACC",
            "ROBUST_MPPI_WAYPTS",
        ]:
            return self.optimize_action_mppi(act_seqs, reward_seqs)
        # Stripped GD and MPPI_GD branches (planner_v0_0.py:337-340).
        else:
            raise ValueError("unknown planner type: %s" % (self.planner_type))

    # Verbatim from planner_v0_0.py:344-388, with non-MPPI branches stripped.
    def trajectory_optimization(
        self, state_cur: torch.Tensor, act_seq: torch.Tensor, **kwargs: dict
    ) -> TrajOptOutput:
        """Perform trajectory optimization.

        Args:
            state_cur (torch.Tensor): Current state, shape: [n_his, state_dim].
            act_seq (torch.Tensor): Initial action sequence, shape: [n_look_ahead,
            action_dim].
            kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary with the following keys:
            - 'act_seq': Optimized action sequence, shape: [n_look_ahead, action_dim].
            - 'model_outputs': List of model outputs if verbose is True, otherwise None.
            - 'eval_outputs': List of evaluation outputs if verbose is True, otherwise
               None.
            - 'best_model_output': Best model output if rollout_best is True, otherwise
               None.
            - 'best_eval_output': Best evaluation output if rollout_best is True,
               otherwise None.
        """
        assert type(state_cur) == torch.Tensor
        # assert act_seq.shape == (self.n_look_ahead, self.action_dim)
        assert type(act_seq) == torch.Tensor
        if self.planner_type == "MPPI":
            return self.trajectory_optimization_mppi(state_cur, act_seq)
        elif self.planner_type == "MPPI_WAYPTS":
            return self.trajectory_optimization_mppi_waypts(
                state_cur, act_seq, **kwargs  # type: ignore
            )
        # Stripped GD / ROBUST_MPPI_WAYPTS / MPPI_BY_ACC / MPPI_GD branches
        # (planner_v0_0.py:371-372, 377-386).
        else:
            raise ValueError("unknown planner type: %s" % (self.planner_type))

    # Verbatim from planner_v0_0.py:390-400, PLUS one OURS extension at the
    # top: optional reward standardization before softmax. When
    # normalize_rewards_before_softmax=False, this function is byte-identical
    # to the source.
    def optimize_action_mppi(
        self, act_seqs: torch.Tensor, reward_seqs: torch.Tensor
    ) -> torch.Tensor:
        """Optimize the action sequences using MPPI."""
        # OURS: standardize rewards before softmax so reward_weight is
        # scale-invariant. diffusion-forcing tunes reward_weight to match
        # reward magnitudes; we normalize first. When the config flag is
        # False, this branch is bypassed and the math is identical to
        # planner_v0_0.py:390-400.
        if self.normalize_rewards_before_softmax:
            reward_seqs = (reward_seqs - reward_seqs.mean()) / (
                reward_seqs.std() + 1e-8
            )
        softmax_weight = F.softmax(reward_seqs * self.reward_weight, dim=0)
        act_seq = torch.sum(
            act_seqs * softmax_weight.unsqueeze(-1).unsqueeze(-1),
            dim=0,
        )
        # return self.clip_actions(act_seq)
        return act_seq

    # Verbatim from planner_v0_0.py:421-426
    def clip_actions(self, act_seqs: torch.Tensor) -> torch.Tensor:
        """Clip the action sequences."""
        # act_seqs: shape: [**dim, action_dim] torch tensor
        # return: shape: [**dim, action_dim] torch tensor
        act_seqs.data.clamp_(self.action_lower_lim, self.action_upper_lim)
        return act_seqs

    # Verbatim from planner_v0_0.py:428-483
    def trajectory_optimization_mppi(
        self, state_cur: torch.Tensor, act_seq: torch.Tensor
    ) -> TrajOptOutput:
        """Trajectory optimization using MPPI."""
        if self.verbose:
            # model_outputs = []
            eval_outputs = []
        if self.verbose:
            pbar = tqdm(total=self.n_update_iter, desc="MPPI Optimization")
        for _ in range(self.n_update_iter):
            # iter_start_time = time.time()
            with torch.no_grad():
                # sample_start_time = time.time()
                act_seqs = self.sample_action_sequences(act_seq)
                assert act_seqs.shape == (
                    self.n_sample,
                    self.n_look_ahead,
                    self.action_dim,
                )
                assert type(act_seqs) == torch.Tensor
                model_out: ModelOutput = self.model_rollout(state_cur, act_seqs)
                # torch.cuda.synchronize()
                # print("model time:", time.time() - model_start_time)
                state_seqs = model_out.state_seqs
                assert type(state_seqs) == torch.Tensor
                # eval_start_time = time.time()
                eval_out: EvalOutput = self.evaluate_traj(state_seqs, act_seqs)
                # print("eval time:", time.time() - eval_start_time)
                reward_seqs = eval_out.reward_seqs
                # optimizer_start_time = time.time()
                act_seq = self.optimize_action(act_seqs, reward_seqs)
                # print("optimizer time:", time.time() - optimizer_start_time)
                if self.verbose:
                    self.visualize(act_seqs, eval_out)
                    # model_outputs.append(model_out)
                    eval_outputs.append(eval_out)
            if self.verbose:
                pbar.update(1)
            # print("iter time:", time.time() - iter_start_time)

        if self.rollout_best:
            with torch.no_grad():
                act_seq = act_seq.unsqueeze(0)
                best_model_out: ModelOutput = self.model_rollout(state_cur, act_seq)
                best_model_out.state_seqs = best_model_out.state_seqs[0:1]
                best_eval_out = self.evaluate_traj(best_model_out.state_seqs, act_seq)
                act_seq = act_seq.squeeze(0)

        traj_opt_output = TrajOptOutput(act_seq=act_seq)
        if self.verbose:
            # traj_opt_output.model_outputs = model_outputs
            traj_opt_output.eval_outputs = eval_outputs
        if self.rollout_best:
            traj_opt_output.best_model_output = best_model_out
            traj_opt_output.best_eval_output = best_eval_out
        return traj_opt_output

    # Verbatim from planner_v0_0.py:485-580
    def trajectory_optimization_mppi_waypts(
        self,
        state_cur: torch.Tensor,
        waypts_seq: torch.Tensor,
        interp_pts: int,
        curr_pos: torch.Tensor,
    ) -> TrajOptOutput:
        """Trajectory optimization using MPPI with waypoints."""
        if self.verbose:
            eval_outputs = []
        if self.verbose:
            pbar = tqdm(total=self.n_update_iter, desc="MPPI Optimization")
        act_len = self.n_look_ahead * interp_pts
        for _ in range(self.n_update_iter):
            with torch.no_grad():
                n_hist = curr_pos.shape[0]
                waypts_seqs = self.sample_action_sequences(waypts_seq, self.n_sample)
                waypts_seqs = waypts_seqs.to(self.device)
                # valid_waypts_seqs = torch.zeros((0, self.n_look_ahead, \
                #   self.action_dim))
                # valid_waypts_seqs = valid_waypts_seqs.to(self.device)
                # while valid_waypts_seqs.shape[0] < self.n_sample:
                #     waypts_seqs = self.sample_action_sequences(waypts_seq, \
                #       3*self.n_sample)
                #     curr_pos_repeat = curr_pos[None].repeat(3*self.n_sample, 1, 1)
                #     waypts_seqs_cat = torch.cat([curr_pos_repeat, waypts_seqs], dim=1)

                #     # acc test
                #     dt = interp_pts / 30
                #     vel = waypts_seqs_cat[:, n_hist:, :3] - \
                #       waypts_seqs_cat[:, n_hist-1:-1, :3]
                #     vel = vel / dt
                #     acc = vel[:, 1:, :] - vel[:, :-1, :]
                #     acc = acc / dt
                #     acc_threshold = 0.2
                #     valid_mask = ((acc > acc_threshold).sum(-1).sum(-1) == 0)
                #     valid_waypts_seqs = torch.cat([valid_waypts_seqs, \
                #       waypts_seqs[valid_mask]], dim=0)

                #     # cubic test
                #     M = cubic_spline_nd_torch_batched(waypts_seqs_cat)
                #     M_threshold = 0.03
                #     valid_mask = ((M[..., :3] > M_threshold).sum(-1).sum(-1) == 0)
                #     valid_waypts_seqs = torch.cat([valid_waypts_seqs, \
                #       waypts_seqs[valid_mask]], dim=0)
                # waypts_seqs = valid_waypts_seqs[:self.n_sample]
                assert waypts_seqs.shape == (
                    self.n_sample,
                    self.n_look_ahead,
                    self.action_dim,
                )
                assert type(waypts_seqs) == torch.Tensor
                curr_pos_repeat = curr_pos[None].repeat(self.n_sample, 1, 1)
                waypts_seqs_cat = torch.cat([curr_pos_repeat, waypts_seqs], dim=1)
                spline_fn = cubic_spline_nd_function_torch(waypts_seqs_cat)
                all_ts = torch.linspace(
                    n_hist - 1,
                    self.n_look_ahead + n_hist - 1,
                    act_len + 1,
                    device=self.device,
                )
                act_seqs = spline_fn(all_ts[None].repeat(self.n_sample, 1))[:, 1:]
                model_out: ModelOutput = self.model_rollout(state_cur, act_seqs)
                state_seqs = model_out.state_seqs
                assert type(state_seqs) == torch.Tensor
                eval_out: EvalOutput = self.evaluate_traj(state_seqs, act_seqs)
                reward_seqs = eval_out.reward_seqs
                waypts_seq = self.optimize_action(waypts_seqs, reward_seqs)
                if self.verbose:
                    self.visualize(act_seqs, eval_out)
                    eval_outputs.append(eval_out)
            if self.verbose:
                pbar.update(1)

        if self.rollout_best:
            with torch.no_grad():
                curr_pos_repeat = curr_pos[-1:, None]
                all_ts = torch.linspace(
                    0, self.n_look_ahead, act_len + 1, device=self.device
                )
                waypts_seq_cat = torch.cat([curr_pos_repeat, waypts_seq[None]], dim=1)
                spline_fn = cubic_spline_nd_function_torch(waypts_seq_cat)
                act_seq = spline_fn(all_ts[None])[0, 1:]
                act_seq = act_seq.unsqueeze(0)
                best_model_out: ModelOutput = self.model_rollout(state_cur, act_seq)
                best_model_out.state_seqs = best_model_out.state_seqs[0:1]
                best_eval_out = self.evaluate_traj(best_model_out.state_seqs, act_seq)
                act_seq = act_seq.squeeze(0)

        traj_opt_output = TrajOptOutput(act_seq=act_seq, waypts_seq=waypts_seq)
        if self.verbose:
            traj_opt_output.eval_outputs = eval_outputs
        if self.rollout_best:
            traj_opt_output.best_model_output = best_model_out
            traj_opt_output.best_eval_output = best_eval_out
        return traj_opt_output
