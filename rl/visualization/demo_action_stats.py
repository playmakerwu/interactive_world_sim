"""Cached demo action statistics + WM action-normalizer params for viz.

These are read by both the gripper-arrow side panels (combined_video.py)
and the action distribution plots (action_distribution.py). Caching to a
single .pt file means viz is fast (no hdf5 scanning, no WM load) and
also works post-hoc on existing run dirs without needing the WM.

Cached fields (computed across all data/mini/pusht/val/episode_*.hdf5):
  raw:
    mean_per_dim (4,)         # raw action mean per dim, in physical units
    std_per_dim  (4,)         # raw action std per dim
    min_per_dim  (4,)
    max_per_dim  (4,)
    max_magnitude_left        # max(sqrt(a[0]^2 + a[1]^2)) across demos
    max_magnitude_right       # max(sqrt(a[2]^2 + a[3]^2)) across demos
    std_magnitude_left        # std-magnitude reference (sqrt(std0^2 + std1^2))
    std_magnitude_right
  normalizer:
    scale  (4,)               # WM action normalizer scale
    offset (4,)               # WM action normalizer offset
                              # forward: x_norm = x_raw * scale + offset
                              # inverse: x_raw  = (x_norm - offset) / scale

Caller-facing helper: ``load_or_compute_demo_action_stats(force_recompute=False)``.
"""

from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
VAL_GLOB = REPO_ROOT / "data" / "mini" / "pusht" / "val"
WM_CKPT = REPO_ROOT / "outputs" / "pusht_cam1" / "checkpoints" / "best.ckpt"
CACHE_PATH = REPO_ROOT / "tests" / "goal_selection" / "demo_action_stats.pt"

ACTION_KEY = "action"


def _compute_demo_action_stats() -> dict:
    """Aggregate raw demo actions across val episodes."""
    eps = sorted(VAL_GLOB.glob("episode_*.hdf5"))
    if not eps:
        raise FileNotFoundError(f"no episode_*.hdf5 found in {VAL_GLOB}")
    chunks = []
    for ep in eps:
        with h5py.File(str(ep), "r") as f:
            chunks.append(f[ACTION_KEY][:])
    A = np.concatenate(chunks, axis=0).astype(np.float32)  # (T_total, 4)
    mean = A.mean(axis=0)
    std = A.std(axis=0)
    mn = A.min(axis=0)
    mx = A.max(axis=0)

    # Per-gripper magnitude statistics
    mag_left = np.sqrt(A[:, 0] ** 2 + A[:, 1] ** 2)
    mag_right = np.sqrt(A[:, 2] ** 2 + A[:, 3] ** 2)
    return {
        "mean_per_dim": torch.tensor(mean, dtype=torch.float32),
        "std_per_dim": torch.tensor(std, dtype=torch.float32),
        "min_per_dim": torch.tensor(mn, dtype=torch.float32),
        "max_per_dim": torch.tensor(mx, dtype=torch.float32),
        "max_magnitude_left": float(mag_left.max()),
        "max_magnitude_right": float(mag_right.max()),
        "p95_magnitude_left": float(np.percentile(mag_left, 95)),
        "p95_magnitude_right": float(np.percentile(mag_right, 95)),
        "std_magnitude_left": float(math.sqrt(std[0] ** 2 + std[1] ** 2)),
        "std_magnitude_right": float(math.sqrt(std[2] ** 2 + std[3] ** 2)),
        "n_samples": int(A.shape[0]),
        "source_episodes": [str(p.relative_to(REPO_ROOT)) for p in eps],
    }


def _load_wm_action_normalizer() -> dict:
    """Pull just the action-normalizer scale + offset out of the WM checkpoint.

    Avoids constructing the full DifferentiableDynamics module so we don't
    pay the WM-load cost during visualization.
    """
    if not WM_CKPT.exists():
        raise FileNotFoundError(f"WM checkpoint missing: {WM_CKPT}")
    ckpt = torch.load(str(WM_CKPT), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict") if isinstance(ckpt, dict) else None
    if state_dict is None:
        raise RuntimeError(
            f"unexpected checkpoint shape at {WM_CKPT}: keys = "
            f"{list(ckpt.keys())[:10]}"
        )
    # Lightning-style key path; verify both possible spellings for robustness
    candidates = [
        ("normalizer.params_dict.action.scale",
         "normalizer.params_dict.action.offset"),
        ("normalizer.action.scale", "normalizer.action.offset"),
    ]
    for scale_key, offset_key in candidates:
        if scale_key in state_dict and offset_key in state_dict:
            return {
                "scale": state_dict[scale_key].detach().cpu().float().clone(),
                "offset": state_dict[offset_key].detach().cpu().float().clone(),
            }
    matches = [k for k in state_dict.keys() if "normalizer" in k and "action" in k]
    raise RuntimeError(
        f"could not find action normalizer in WM checkpoint state_dict. "
        f"Closest matches: {matches[:6]}"
    )


def compute_and_cache(cache_path: Path = CACHE_PATH) -> dict:
    """Build demo stats + load normalizer + persist combined dict."""
    demo = _compute_demo_action_stats()
    norm = _load_wm_action_normalizer()
    payload = {**demo, "normalizer": norm}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def load_or_compute_demo_action_stats(
    force_recompute: bool = False,
    cache_path: Path = CACHE_PATH,
) -> dict:
    """Return the cached payload, computing if missing or forced."""
    if not force_recompute and cache_path.exists():
        return torch.load(str(cache_path), map_location="cpu", weights_only=False)
    return compute_and_cache(cache_path)


# ── Denormalization helper (used in viz) ────────────────────────────────

def denormalize_action(
    norm_action: torch.Tensor | np.ndarray,
    normalizer: dict,
) -> torch.Tensor:
    """Map MPPI-normalized actions in [-1, +1] back to raw physical units.

    forward (training):  x_norm = x_raw * scale + offset
    inverse (this fn):   x_raw  = (x_norm - offset) / scale
    """
    if isinstance(norm_action, np.ndarray):
        norm_action = torch.from_numpy(norm_action)
    scale = normalizer["scale"].to(norm_action.device, dtype=norm_action.dtype)
    offset = normalizer["offset"].to(norm_action.device, dtype=norm_action.dtype)
    return (norm_action - offset) / scale
