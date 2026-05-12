"""Differentiable dynamics wrapper around the pretrained world model.

This is the ONLY interface to the world model for all RL code.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.checkpoint import checkpoint

from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm
from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import (
    LatentWorldModel,
)
from interactive_world_sim.utils.normalizer import LinearNormalizer


def _patch_attention_backends():
    from torch.nn.attention import SDPBackend
    from interactive_world_sim.algorithms.models.attention import Attention

    cap = torch.cuda.get_device_capability()
    if cap[0] >= 8 and cap[0] != 8:
        _orig_init = Attention.__init__

        def _patched_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            self.cuda_backends = [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]

        Attention.__init__ = _patched_init


def _register_resolvers():
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver(
        "torch", lambda x: getattr(torch, x), replace=True
    )


class DifferentiableDynamics(nn.Module):
    """Wraps the pretrained world model for differentiable RL training.

    All world-model parameters are frozen.  The only differentiable path
    is through the dynamics step: actor-produced actions flow into the
    dynamics model, and gradients backpropagate through the consistency-
    model denoising to the actor.
    """

    def __init__(self, ckpt_path: str, device: str = "cuda:0"):
        super().__init__()
        _patch_attention_backends()
        _register_resolvers()

        cfg_path = Path(ckpt_path).parent.parent / ".hydra" / "config.yaml"
        cfg = OmegaConf.load(cfg_path)
        dtype = torch.float32 if "dtype" not in cfg.algorithm else cfg.algorithm.dtype
        cfg.n_frames = 10
        cfg.algorithm.n_frames = 10
        if (
            "diffusion" in cfg.algorithm
            and "sampling_timesteps" in cfg.algorithm.diffusion
        ):
            cfg.algorithm.diffusion.sampling_timesteps = 10
        if (
            "diffusion" in cfg.algorithm.dynamics
            and "sampling_timesteps" in cfg.algorithm.dynamics.diffusion
        ):
            cfg.algorithm.dynamics.diffusion.sampling_timesteps = 10
        cfg.algorithm.load_ae = None

        self.wm: LatentWorldModel = LatentWorldModel.load_from_checkpoint(
            ckpt_path,
            cfg=cfg.algorithm,
            map_location=device,
            dtype=dtype,
            strict=False,
            weights_only=False,
        )
        self.wm.dynamics = self.wm.dynamics.to(dtype)
        self.wm.eval()
        self.wm.dynamics.eval()

        # freeze everything
        for p in self.wm.parameters():
            p.requires_grad_(False)

        self._device = device
        self.normalizer: LinearNormalizer = self.wm.normalizer

    # ── encode ───────────────────────────────────────────────────────
    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Encode preprocessed image(s) to latent.

        Args:
            image: (B, 3, H, W) float32 in [0, 1], NOT yet normalizer'd

        Returns:
            z: (B, C, H_lat, W_lat) float32
        """
        obs_key = self.wm.obs_keys[0]
        x = self.normalizer[obs_key].normalize(image).to(self._device)
        z = self.wm.encoder_forward(x)
        return z.float()

    # ── differentiable single step ───────────────────────────────────
    def step(
        self,
        z_hist: torch.Tensor,       # (B, T_hist, C, H, W)
        action_hist: torch.Tensor,  # (B, T_hist+1, A)
        use_checkpoint: bool = True,
    ) -> torch.Tensor:
        """One differentiable dynamics step.  Returns (B, 1, C, H, W)."""
        z_seq = rearrange(z_hist, "b t c h w -> t b c h w")
        action_seq = rearrange(action_hist, "b t a -> t b a")
        T_hist = z_seq.shape[0]
        B = z_seq.shape[1]
        device = z_seq.device

        # 1. noise for new frame
        noise = torch.randn(
            (1, B, *z_seq.shape[2:]), device=device, dtype=z_seq.dtype
        )
        noise = torch.clamp(noise, -self.wm.clip_noise, self.wm.clip_noise)

        # 2. concat history + noise
        xs = torch.cat([z_seq, noise], dim=0)

        # 3. sliding window
        n_tokens = self.wm.n_tokens
        curr_end = T_hist + 1
        curr_start = max(0, curr_end - n_tokens)
        xs_window = xs[curr_start:]
        action_window = action_seq[curr_start:curr_end]

        # 4. timestep schedule
        window_len = xs_window.shape[0]
        clean_t = (
            torch.ones((window_len - 1,), device=device)
            * self.wm.noise_scheduler.stabilization_level
        )
        timesteps = torch.linspace(
            self.wm.noise_scheduler.timesteps - 1,
            0,
            self.wm.dyn_infer_steps + 1,
            device=device,
        )

        # 5. denoise
        xs_updated = xs_window
        for step_i in range(self.wm.dyn_infer_steps):
            t = timesteps[step_i].unsqueeze(0)
            s = timesteps[step_i + 1].unsqueeze(0)
            t = torch.cat([clean_t, t], 0)[:, None].expand(-1, B).long()
            s = torch.cat([clean_t, s], 0)[:, None].expand(-1, B).long()

            def _denoise(xs_in, t_in, s_in, act_w):
                denoise_fn = lambda x, tt, ss: self.wm.dynamics(
                    x, tt, ss, external_cond=act_w
                )
                return self.wm.noise_scheduler.CTM_calc_out(
                    denoise_fn, xs_in, t_in, s_in
                )

            if use_checkpoint:
                xs_updated = checkpoint(
                    _denoise, xs_updated, t, s, action_window,
                    use_reentrant=False,
                )
            else:
                xs_updated = _denoise(xs_updated, t, s, action_window)

        # 6. extract predicted frame
        z_pred = xs_updated[-1:]  # (1, B, C, H, W)

        # 7. L2-normalize per view
        num_views = len(self.wm.obs_keys)
        c_per_v = z_pred.shape[2] // num_views
        parts = []
        for i in range(num_views):
            chunk = z_pred[:, :, i * c_per_v : (i + 1) * c_per_v]
            chunk = chunk / (torch.norm(chunk, dim=2, keepdim=True) + 1e-8)
            parts.append(chunk)
        z_pred = torch.cat(parts, dim=2)

        return rearrange(z_pred, "t b c h w -> b t c h w")  # (B, 1, C, H, W)

    # ── multi-step rollout ───────────────────────────────────────────
    def rollout(
        self,
        z_init: torch.Tensor,            # (B, 1, C, H, W) or (B, T_warm, C, H, W)
        actions: torch.Tensor,           # (B, H, A)
        hist_context: int = 10,
        action_history: torch.Tensor | None = None,  # (B, T_warm, A) — see notes
    ) -> torch.Tensor:
        """Multi-step dynamics rollout.

        Two modes:

        Single-frame init (legacy, ``action_history=None``):
            ``z_init`` must be ``(B, 1, C, H, W)``. Action history starts
            with a single dummy zero action. Early steps (t=0..8) see
            only 1..9 frames of context — OUT OF DISTRIBUTION relative
            to the model's 10-frame training.

        Multi-frame warmup (NEW, ``action_history`` given):
            ``z_init`` is ``(B, T_warm, C, H, W)`` — pre-encoded latents
            of T_warm consecutive real frames. ``action_history`` is
            ``(B, T_warm, A)`` — the actions that drove those frames
            (with ``action_history[:, 0]`` typically zeros: convention
            "action that drove INTO ``z_init[:, 0]``", which doesn't
            exist for the very first warmup frame). Step 0 of the
            rollout starts with a FULL T_warm-frame context — in
            distribution.

        Returns:
            latents: (B, H+1, C, H, W) — ``z_init[:, -1]`` followed by
                H predicted latents.
        """
        H = actions.shape[1]
        use_ckpt = H > 10

        if action_history is None:
            # Legacy single-frame path. z_init must be (B, 1, C, H, W).
            if z_init.dim() != 5 or z_init.shape[1] != 1:
                raise ValueError(
                    f"rollout(action_history=None) expects z_init of shape "
                    f"(B, 1, C, H, W); got {tuple(z_init.shape)}. Pass "
                    f"action_history to enable multi-frame warmup."
                )
            B = z_init.shape[0]
            z = z_init  # (B, 1, C, H, W)
            latents = [z[:, -1]]
            action_hist = [torch.zeros(B, 1, actions.shape[2], device=z.device)]
        else:
            # Warmup path: prime z and action_hist from the supplied
            # history. The returned latents start at the LAST warmup frame.
            if z_init.dim() != 5 or z_init.shape[1] < 1:
                raise ValueError(
                    f"warmup rollout expects z_init (B, T_warm, C, H, W); "
                    f"got {tuple(z_init.shape)}."
                )
            if action_history.dim() != 3:
                raise ValueError(
                    f"action_history must be (B, T_warm, A); got "
                    f"{tuple(action_history.shape)}."
                )
            if action_history.shape[0] != z_init.shape[0]:
                raise ValueError(
                    f"action_history batch ({action_history.shape[0]}) != "
                    f"z_init batch ({z_init.shape[0]})."
                )
            if action_history.shape[1] != z_init.shape[1]:
                raise ValueError(
                    f"action_history T_warm ({action_history.shape[1]}) != "
                    f"z_init T_warm ({z_init.shape[1]}). Convention: "
                    f"action_history[:, i] drove INTO z_init[:, i]; "
                    f"action_history[:, 0] is typically zeros."
                )
            # Trim warmup to hist_context — model can't see further back.
            z = z_init[:, -hist_context:]
            T_warm = z.shape[1]
            ahist = action_history[:, -T_warm:]
            # action_hist as a list of (B, 1, A) tensors so the existing
            # cat loop below works unchanged.
            action_hist = [ahist[:, i : i + 1] for i in range(T_warm)]
            latents = [z[:, -1]]

        for t in range(H):
            action_hist.append(actions[:, t : t + 1])  # (B, 1, A)
            T_hist = z.shape[1]
            all_act = torch.cat(action_hist, dim=1)
            act_input = all_act[:, -(T_hist + 1) :]

            z_next = self.step(z, act_input, use_checkpoint=use_ckpt)
            z = torch.cat([z, z_next], dim=1)[:, -hist_context:]
            latents.append(z_next[:, 0])

        return torch.stack(latents, dim=1)  # (B, H+1, C, H, W)

    # ── decode (for evaluation / video) ──────────────────────────────
    @torch.no_grad()
    def decode(self, z: torch.Tensor, resolution: int = 128) -> torch.Tensor:
        """Decode latent to image.  z: (B, C, H, W) → (B, 3, res, res) in [0,1]."""
        return render_img_cm(
            self.wm, z, resolution,
            normalizer=self.normalizer, num_views=1,
        ).float()
