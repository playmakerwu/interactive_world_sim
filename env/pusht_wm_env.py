"""PushTWMEnv — atomic operations on the IWS world model + CV state estimator.

Wraps `rl/models/world_model.py::DifferentiableDynamics` (the WM) and
`rl/labeling/cv_labeler.py::CVLabeler` (the CV pose estimator) into a single
reusable interface that both MPPI and future RL code can call.

Design rules:
  * Every public method works on both batched ``(B, ...)`` and unbatched
    ``(...)`` input. If the caller passes unbatched, the result is unbatched.
  * The WM lives on GPU; the CV labeler is CPU-only (numpy/OpenCV). Conversion
    is handled inside `estimate_state` so the caller never has to think about
    the boundary.
  * `load_initial_from_hdf5` *hardcodes* ``obs_key='camera_1_color'`` because
    that is the only camera the IWS PushT checkpoint was trained on (see
    ``outputs/pusht_cam1/.hydra/config.yaml:70-71`` and commit ``fd875ef``).
    Camera is intentionally not exposed as a parameter — leaving it as a
    function arg is what produced the camera bug in the first place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import torch

from rl.labeling.cv_labeler import CVLabeler
from rl.models.world_model import DifferentiableDynamics

DEFAULT_RES = 128
DEFAULT_DIAGONAL = float(np.sqrt(DEFAULT_RES ** 2 + DEFAULT_RES ** 2))  # ≈ 181.02

# The camera the IWS PushT WM was trained on. Hardcoded to prevent a
# regression of the pre-fix camera bug. Don't make this configurable.
PUSHT_CAMERA_KEY = "camera_1_color"


def _is_batched_latent(z: torch.Tensor) -> bool:
    """Latent shape is (C, H, W) unbatched or (B, C, H, W) batched."""
    return z.dim() == 4


def _is_batched_rgb(rgb: torch.Tensor | np.ndarray) -> bool:
    """RGB shape is (3, H, W) unbatched or (B, 3, H, W) batched (torch convention)."""
    return rgb.ndim == 4


def _preprocess_rgb_uint8(raw: np.ndarray, resolution: int = DEFAULT_RES) -> np.ndarray:
    """Center-crop to square, resize to ``resolution``, return float [0, 1] CHW."""
    h, w = raw.shape[:2]
    s = min(h, w)
    cropped = raw[(h - s) // 2 : (h - s) // 2 + s, (w - s) // 2 : (w - s) // 2 + s]
    resized = cv2.resize(cropped, (resolution, resolution), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


class PushTWMEnv:
    """Atomic operations on the IWS PushT WM + CV state estimator.

    Args:
        wm_ckpt_path: path to the IWS Lightning checkpoint (e.g.
            ``outputs/pusht_cam1/checkpoints/best.ckpt``).
        device: torch device for WM tensors. CV labeler stays on CPU.
        resolution: decoded RGB resolution. Defaults to 128 (the only
            resolution the IWS PushT checkpoint was trained at).
        cv_preset: CV labeler HSV preset. ``"REAL"`` matches the pink
            T-block under top-down ALOHA lighting and is the only
            preset validated for this WM.
    """

    def __init__(
        self,
        wm_ckpt_path: str,
        device: str = "cuda:0",
        resolution: int = DEFAULT_RES,
        cv_preset: str = "REAL",
    ) -> None:
        self.wm_ckpt_path = wm_ckpt_path
        self.device = device
        self.resolution = resolution
        self.image_diagonal = float(np.sqrt(resolution ** 2 + resolution ** 2))
        self._wm = DifferentiableDynamics(wm_ckpt_path, device=device)
        self._labeler = CVLabeler(preset=cv_preset, resolution=resolution)
        # Action dim is fixed by the WM checkpoint; cached so callers don't
        # have to dig into hydra config.
        self.action_dim = 4

    # ── encode / decode ───────────────────────────────────────────────

    def encode(self, rgb: torch.Tensor) -> torch.Tensor:
        """Encode RGB to latent.

        Args:
            rgb: ``(3, H, W)`` or ``(B, 3, H, W)``. Float in [0, 1] preferred;
                uint8 also accepted (auto-converted to float / 255).

        Returns:
            Latent ``(C, H_lat, W_lat)`` or ``(B, C, H_lat, W_lat)``,
            L2-normalised to total norm ≈ ``resolution`` (for 128 → 32).
        """
        was_batched = _is_batched_rgb(rgb)
        if not was_batched:
            rgb = rgb.unsqueeze(0)
        if rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255.0
        rgb = rgb.to(self.device)
        z = self._wm.encode(rgb)
        return z if was_batched else z[0]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent back to RGB.

        Args:
            z: ``(C, H_lat, W_lat)`` or ``(B, C, H_lat, W_lat)``.

        Returns:
            RGB ``(3, H, W)`` or ``(B, 3, H, W)``, float in [0, 1].
        """
        was_batched = _is_batched_latent(z)
        if not was_batched:
            z = z.unsqueeze(0)
        z = z.to(self.device)
        rgb = self._wm.decode(z, resolution=self.resolution)
        rgb = rgb.clamp(0.0, 1.0).float()
        return rgb if was_batched else rgb[0]

    # ── dynamics ─────────────────────────────────────────────────────

    def dynamics_step(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Single-step forward through the WM dynamics.

        Args:
            z: ``(C, H_lat, W_lat)`` or ``(B, C, H_lat, W_lat)``.
            action: ``(action_dim,)`` or ``(B, action_dim)``.

        Returns:
            ``z_next`` with the same shape as ``z``.
        """
        was_batched = _is_batched_latent(z)
        if not was_batched:
            z = z.unsqueeze(0)
            action = action.unsqueeze(0)
        z = z.to(self.device)
        action = action.to(self.device)
        # Use rollout with H=1 — this routes through DifferentiableDynamics
        # which handles all the consistency-model denoising machinery.
        actions_one = action.unsqueeze(1)  # (B, 1, A)
        z_init = z.unsqueeze(1)  # (B, 1, C, H, W)
        with torch.no_grad():
            traj = self._wm.rollout(z_init, actions_one)  # (B, 2, C, H, W)
        z_next = traj[:, 1]  # (B, C, H, W) — the post-action latent
        return z_next if was_batched else z_next[0]

    def rollout(
        self,
        z0: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-step batched rollout through the WM dynamics.

        Args:
            z0: ``(C, H_lat, W_lat)`` or ``(B, C, H_lat, W_lat)``.
            actions: ``(H, action_dim)`` or ``(B, H, action_dim)``.

        Returns:
            Latents ``(B, H+1, C, H_lat, W_lat)`` (or ``(H+1, ...)`` if
            input was unbatched). Index 0 is ``z0``; subsequent indices are
            the post-step latents.
        """
        was_batched = _is_batched_latent(z0)
        if not was_batched:
            z0 = z0.unsqueeze(0)
            actions = actions.unsqueeze(0)
        z0 = z0.to(self.device)
        actions = actions.to(self.device)
        z_init = z0.unsqueeze(1)  # (B, 1, C, H_lat, W_lat)
        with torch.no_grad():
            traj = self._wm.rollout(z_init, actions)  # (B, H+1, C, H_lat, W_lat)
        return traj if was_batched else traj[0]

    # ── state estimation ─────────────────────────────────────────────

    def estimate_state(self, rgb: torch.Tensor | np.ndarray) -> dict[str, Any]:
        """Run the CV labeler on each frame.

        Args:
            rgb: ``(3, H, W)`` or ``(B, 3, H, W)`` torch tensor in [0, 1],
                OR ``(H, W, 3)`` / ``(B, H, W, 3)`` uint8 numpy.

        Returns:
            A dict whose every entry is a 1-d tensor of length ``B`` (or a
            scalar tensor if input was unbatched). Keys:

              ``'cx', 'cy'``        — pose centre, pixels
              ``'sin_theta'``       — sin of detected angle
              ``'cos_theta'``       — cos of detected angle
              ``'theta_deg'``       — angle in degrees (-180, 180]
              ``'success'``         — bool tensor
              ``'contour_area'``    — float (NaN where success is False)
              ``'icp_residual'``    — float (NaN where success is False)
        """
        # Normalise to a (B, H, W, 3) uint8 numpy block.
        was_batched: bool
        if isinstance(rgb, torch.Tensor):
            was_batched = _is_batched_rgb(rgb)
            arr = rgb.detach().cpu().float().numpy()
            if not was_batched:
                arr = arr[None]  # (1, 3, H, W)
            arr = arr.transpose(0, 2, 3, 1)  # (B, H, W, 3)
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = np.asarray(rgb)
            if arr.ndim == 3:
                was_batched = False
                arr = arr[None]
            else:
                was_batched = True
            if arr.dtype != np.uint8:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

        N = arr.shape[0]
        results = [self._labeler.label(arr[i]) for i in range(N)]

        keys = ("cx", "cy", "sin_theta", "cos_theta", "theta_deg",
                "contour_area", "icp_residual")
        out: dict[str, Any] = {}
        for k in keys:
            vals = [getattr(r, k) if r.success else float("nan") for r in results]
            out[k] = torch.tensor(vals, dtype=torch.float32)
        out["success"] = torch.tensor([r.success for r in results], dtype=torch.bool)

        if not was_batched:
            out = {k: v[0] for k, v in out.items()}
        return out

    def estimate_from_latent(self, z: torch.Tensor) -> dict[str, Any]:
        """Convenience: decode then estimate. Useful for MPPI reward."""
        rgb = self.decode(z)
        return self.estimate_state(rgb)

    # ── reward ───────────────────────────────────────────────────────

    def compute_reward(
        self,
        state: dict[str, Any],
        goal_state: dict[str, Any],
        image_diagonal: float | None = None,
        cv_fail_penalty: float = -10.0,
    ) -> torch.Tensor:
        """CV-based per-batch reward.

        ``R = -‖pos − pos_goal‖₂ / image_diagonal − (1 − cos(Δθ))``,
        clamped via the CV-fail penalty wherever ``state['success']`` is False.

        Args:
            state: dict as returned by ``estimate_state`` (batched or scalar).
            goal_state: dict with scalar ``cx, cy, sin_theta, cos_theta`` keys
                (e.g. loaded via ``load_goal``).
            image_diagonal: divisor for the position term. Defaults to the
                env's resolution diagonal.
            cv_fail_penalty: per-trajectory reward where CV failed.

        Returns:
            Scalar tensor (unbatched state) or ``(B,)`` tensor (batched state).
        """
        if image_diagonal is None:
            image_diagonal = self.image_diagonal
        success = state["success"]
        was_batched = success.dim() == 1
        if not was_batched:
            success = success.unsqueeze(0)
            cx, cy = state["cx"].unsqueeze(0), state["cy"].unsqueeze(0)
            sin_t, cos_t = state["sin_theta"].unsqueeze(0), state["cos_theta"].unsqueeze(0)
        else:
            cx, cy = state["cx"], state["cy"]
            sin_t, cos_t = state["sin_theta"], state["cos_theta"]

        gx = float(goal_state["cx"])
        gy = float(goal_state["cy"])
        gs = float(goal_state["sin_theta"])
        gc = float(goal_state["cos_theta"])
        # Replace NaN positions in failed entries with goal so the math is
        # finite; we'll overwrite with the penalty below regardless.
        cx = torch.nan_to_num(cx, nan=gx)
        cy = torch.nan_to_num(cy, nan=gy)
        sin_t = torch.nan_to_num(sin_t, nan=gs)
        cos_t = torch.nan_to_num(cos_t, nan=gc)

        pos_dist = torch.sqrt((cx - gx) ** 2 + (cy - gy) ** 2)
        pos_term = -pos_dist / image_diagonal
        cos_delta = (sin_t * gs + cos_t * gc).clamp(-1.0, 1.0)
        ang_term = -(1.0 - cos_delta)
        reward = pos_term + ang_term

        reward = torch.where(
            success, reward, torch.full_like(reward, cv_fail_penalty)
        )
        return reward if was_batched else reward[0]

    # ── utility ──────────────────────────────────────────────────────

    def load_goal(self, goal_path: str | Path) -> dict[str, Any]:
        """Load a saved ``state_goal.pt`` file. Returns a dict with float keys
        ``cx, cy, sin_theta, cos_theta, theta_deg`` and a 4-vector ``state``."""
        g = torch.load(str(goal_path), map_location="cpu", weights_only=False)
        return {
            "cx": float(g["cx"]),
            "cy": float(g["cy"]),
            "sin_theta": float(g["sin_theta"]),
            "cos_theta": float(g["cos_theta"]),
            "theta_deg": float(g["theta_deg"]),
            "state": torch.tensor(
                [g["cx"], g["cy"], g["sin_theta"], g["cos_theta"]],
                dtype=torch.float32,
            ),
        }

    def load_initial_from_hdf5(
        self, hdf5_path: str | Path, frame_idx: int = 0,
    ) -> torch.Tensor:
        """Load one RGB frame from an HDF5 episode and encode it to latent.

        The camera key is hardcoded to ``camera_1_color`` because that is
        the only camera the IWS PushT checkpoint was trained on. See
        commit ``fd875ef``.

        Returns a ``(1, C, H_lat, W_lat)`` latent ready for MPPI / RL.
        """
        with h5py.File(str(hdf5_path), "r") as f:
            raw = f[f"obs/images/{PUSHT_CAMERA_KEY}"][int(frame_idx)]
        pre = _preprocess_rgb_uint8(raw, resolution=self.resolution)
        rgb_t = torch.from_numpy(pre).permute(2, 0, 1).unsqueeze(0)
        return self.encode(rgb_t)
