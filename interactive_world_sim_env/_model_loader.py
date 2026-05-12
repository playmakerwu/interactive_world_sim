"""The single place in the project allowed to import from
`interactive_world_sim.algorithms.**` and to load .ckpt files.

Anything that needs a LatentWorldModel instance goes through `load_model`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

# All of these imports are localized to this module per the design.
from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import (
    LatentWorldModel,
)

from .registry import RegistryError, TaskSpec, get_task_spec

# OmegaConf resolvers must be registered before loading a config that uses
# ${torch:...} or ${eval:...}. Calling register_new_resolver twice raises;
# guard with replace=True so importing this module repeatedly is safe.
OmegaConf.register_new_resolver(
    "eval", lambda expr: eval(expr, {"np": np}), replace=True  # noqa: S307
)
OmegaConf.register_new_resolver(
    "torch", lambda x: getattr(torch, x), replace=True
)


@dataclass(frozen=True)
class LoadedModel:
    """Bundle of the things env.py needs from a checkpoint."""

    model: LatentWorldModel
    cfg: DictConfig
    dtype: torch.dtype
    task: str
    spec: TaskSpec


def load_model(
    task: str,
    *,
    device: str | torch.device = "cuda",
    repo_root: Path | None = None,
    n_frames: int = 10,
    dyn_infer_steps: Optional[int] = None,
) -> LoadedModel:
    """Load a checkpoint for `task`, cross-check against the registry, and return it.

    Parameters
    ----------
    task: registry key (see :data:`TASKS` in registry.py).
    device: target device for the model and its buffers.
    repo_root: directory the relative ckpt_path is resolved against. Defaults
        to the closest ancestor of this file that contains a ``.git`` directory.
    n_frames: overrides ``cfg.n_frames`` and ``cfg.algorithm.n_frames`` after
        loading. Matches the value used by the keyboard / browser demos so the
        sliding-window dimensions agree.
    dyn_infer_steps: if provided, overrides the dynamics inference-step count
        after loading.

    Raises
    ------
    RegistryError: when the registry hints disagree with the checkpoint config.
    FileNotFoundError: when ``best.ckpt`` or ``.hydra/config.yaml`` is missing.
    """

    spec = get_task_spec(task)
    if repo_root is None:
        repo_root = _find_repo_root()

    ckpt_path = (repo_root / spec.ckpt_path).resolve()
    cfg_path = ckpt_path.parent.parent / ".hydra" / "config.yaml"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found for task {task!r}: {ckpt_path}.\n"
            f"Run scripts/download_checkpoints_hf.py --subdir {task}."
        )
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Sibling config not found for task {task!r}: {cfg_path}.\n"
            f"The checkpoint download may be incomplete."
        )

    cfg = OmegaConf.load(cfg_path)
    _cross_check_registry(task, spec, cfg)

    # Mirror what teleoperate_keyboard.py / deploy/server.py do before
    # load_from_checkpoint: override n_frames and sampling timesteps, and
    # disable load_ae so the loader does not try to chain another ckpt.
    cfg.n_frames = n_frames
    cfg.algorithm.n_frames = n_frames
    if "diffusion" in cfg.algorithm and "sampling_timesteps" in cfg.algorithm.diffusion:
        cfg.algorithm.diffusion.sampling_timesteps = n_frames
    if (
        "diffusion" in cfg.algorithm.dynamics
        and "sampling_timesteps" in cfg.algorithm.dynamics.diffusion
    ):
        cfg.algorithm.dynamics.diffusion.sampling_timesteps = n_frames
    cfg.algorithm.load_ae = None
    if dyn_infer_steps is not None:
        cfg.algorithm.dyn_infer_steps = int(dyn_infer_steps)

    dtype = (
        torch.float32 if "dtype" not in cfg.algorithm else cfg.algorithm.dtype
    )
    device_obj = torch.device(device)
    map_location = str(device_obj)

    model = LatentWorldModel.load_from_checkpoint(
        str(ckpt_path),
        cfg=cfg.algorithm,
        map_location=map_location,
        dtype=dtype,
        strict=False,
        weights_only=False,
    )
    # Match teleop: explicit dtype on the dynamics submodule, then eval mode.
    model.dynamics = model.dynamics.to(dtype)
    model.eval()
    model.dynamics.eval()
    return LoadedModel(model=model, cfg=cfg, dtype=dtype, task=task, spec=spec)


def _cross_check_registry(task: str, spec: TaskSpec, cfg: DictConfig) -> None:
    """Compare the registry hints against the saved checkpoint config.

    action_mode is intentionally NOT checked: the saved value is unreliable for
    some shipped checkpoints (e.g. pusht_cam1 saved "single_ee" which is not in
    interactive_world_sim/utils/action_utils.py). The registry's ctrl_mode is
    the source of truth for the episode-init path.
    """
    cfg_action_dim = int(cfg.algorithm.action_dim)
    if cfg_action_dim != spec.action_dim:
        raise RegistryError(
            f"action_dim mismatch for task {task!r}: "
            f"registry={spec.action_dim} vs ckpt.cfg.algorithm.action_dim={cfg_action_dim}"
        )

    cfg_obs_keys = tuple(cfg.dataset.obs_keys)
    if cfg_obs_keys != spec.obs_keys:
        raise RegistryError(
            f"obs_keys mismatch for task {task!r}: "
            f"registry={spec.obs_keys} vs ckpt.cfg.dataset.obs_keys={cfg_obs_keys}"
        )

    cfg_resolution = int(cfg.dataset.resolution)
    if cfg_resolution != spec.resolution:
        raise RegistryError(
            f"resolution mismatch for task {task!r}: "
            f"registry={spec.resolution} vs ckpt.cfg.dataset.resolution={cfg_resolution}"
        )


def _find_repo_root() -> Path:
    """Walk upward from this file until a directory containing .git is found."""
    p = Path(__file__).resolve().parent
    for parent in (p, *p.parents):
        if (parent / ".git").exists():
            return parent
    raise RuntimeError(
        "Could not locate repo root from "
        f"{Path(__file__).resolve()} (no .git found in any ancestor)."
    )
