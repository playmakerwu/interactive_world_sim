"""WorldModelEnv — a stable, gym-like wrapper around the trained latent
world model. The class is the only public entry point of this package.

This module imports from `interactive_world_sim.algorithms.**` only via
`_model_loader.py`; helpers from yixuan_utilities are used for the same
RGB/HDF5 preprocessing the existing demos use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
import torch
from yixuan_utilities.draw_utils import center_crop
from yixuan_utilities.hdf5_utils import load_dict_from_hdf5

from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm

from ._model_loader import LoadedModel, load_model
from .helpers.expert_action import expert_action_from_episode
from .obs import BatchedObservation, Observation
from .registry import get_task_spec
from .state import EnvState


class WorldModelEnv:
    """Stable env interface over a frozen latent world model.

    See ``IMPLEMENTATION_REPORT.md`` for the full contract. Briefly:
    - Construct with a task name, the constructor loads the checkpoint.
    - reset() initializes the latent window from a dataset frame, a user
      RGB, or a snapshot.
    - step(action) rolls one latent step forward and returns an Observation.
    - snapshot()/restore() deep-clone the env state.
    - step_batch(actions) runs K parallel H-step rollouts and returns
      latents + decoded RGBs.

    The env is stochastic by design — fresh noise on every dynamics and
    decoder call, matching the keyboard/browser demos. There is no seeding
    or RNG management inside the env.
    """

    # ------------------------------------------------------------------ ctor

    def __init__(
        self,
        task: str,
        device: str | torch.device = "cuda",
        hist_context: int = 10,
        dyn_infer_steps: int | None = None,
        decode_on_step: bool = False,
        repo_root: Path | None = None,
    ) -> None:
        self._task = task
        self._spec = get_task_spec(task)
        self._hist_context = int(hist_context)
        self._decode_on_step = bool(decode_on_step)
        self._closed = False
        self._repo_root = repo_root

        self._loaded: LoadedModel = load_model(
            task,
            device=device,
            repo_root=repo_root,
            n_frames=self._hist_context,
            dyn_infer_steps=dyn_infer_steps,
        )

        # Derived shape constants pulled from the loaded model.
        cfg = self._loaded.cfg
        self._resolution = int(cfg.dataset.resolution)
        self._num_views = len(self._spec.obs_keys)
        self._action_dim = int(cfg.algorithm.action_dim)
        self._num_latent_channel = int(cfg.algorithm.num_latent_channel)
        self._latent_resolution = int(cfg.algorithm.latent_resolution)

        # Empty until reset(). Sliding windows are kept as (T_hist, ...)
        # device tensors; expanded to (1, T_hist, ...) only at call sites
        # that need a batch dim.
        self._latent_window: torch.Tensor | None = None
        self._action_window: torch.Tensor | None = None
        self._step_counter: int = 0

        # Gym-style spaces (informational; we don't drive a gym.Env loop).
        self._action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self._action_dim,), dtype=np.float32
        )
        self._observation_space = gym.spaces.Dict(
            {
                "latent": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(
                        self._num_latent_channel,
                        self._latent_resolution,
                        self._latent_resolution,
                    ),
                    dtype=np.float32,
                ),
                "step": gym.spaces.Box(
                    low=0, high=np.iinfo(np.int64).max, shape=(), dtype=np.int64
                ),
            }
        )

    # ----------------------------------------------------------- properties

    @property
    def task(self) -> str:
        return self._task

    @property
    def device(self) -> torch.device:
        return self._loaded.model.device

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return self._observation_space

    # ---------------------------------------------------------------- reset

    def reset(
        self,
        *,
        init_state: EnvState | None = None,
        init_rgb: np.ndarray | dict[str, np.ndarray] | None = None,
        init_episode_path: str | None = None,
        init_episode_index: int = 0,
        init_window_size: int = 1,
    ) -> Observation:
        """Initialize the env. Returns the first Observation.

        Priority of init paths (only one is used):
        1. init_state — restore-from-snapshot shortcut.
        2. init_rgb — user RGB(s); single ndarray for single-view tasks,
           dict[obs_key, ndarray] otherwise.
        3. init_episode_path — HDF5 episode + init_episode_index.
        4. Fallback to the registry's default episode (frame 0).

        Multi-frame warmup (``init_window_size > 1``)
        ---------------------------------------------
        With ``init_window_size = W`` the env reads W consecutive RGB
        frames ending at ``init_episode_index`` from the HDF5 episode,
        encodes each into ``latent_window``, and fills ``action_window``
        with the matching expert actions. ``init_episode_index`` is the
        LAST frame of the window — e.g. ``init_window_size=10,
        init_episode_index=9`` reads frames 0..9 inclusive.

        ``action_window[i]`` is filled with the action that drove the
        env INTO ``latent_window[i]`` (i.e., the expert action at the
        frame *before* the one being encoded into ``latent_window[i]``).
        For the first slot, when the previous frame falls before the
        start of the episode (``init_episode_index - W < 0``),
        ``action_window[0]`` is filled with zeros — matching the
        existing single-frame reset convention. This alignment makes
        a warmup-from-W equivalent to a single-frame reset followed by
        W-1 ``observe()`` calls (see Test 9 in the smoke script).

        Warmup currently requires ``task == "pusht_cam1"`` because it
        depends on :func:`expert_action_from_episode`, which is only
        verified for that task.

        Validation (raises on first failure):
        - ``init_window_size < 1`` -> ``ValueError``
        - ``init_window_size > hist_context`` -> ``ValueError``
        - warmup on a non-``pusht_cam1`` task -> ``NotImplementedError``
        - warmup with ``init_rgb`` -> ``ValueError``
        - warmup with ``init_state`` -> ``ValueError``
        - warmup with ``init_episode_path is None`` -> ``ValueError``
        - warmup with ``init_episode_index < init_window_size - 1`` ->
          ``ValueError``
        """
        self._check_open()
        self._validate_init_window_size(
            init_window_size,
            init_state=init_state,
            init_rgb=init_rgb,
            init_episode_path=init_episode_path,
            init_episode_index=init_episode_index,
        )

        if init_state is not None:
            self.restore(init_state)
            return self._make_observation(source="reset")

        if init_window_size > 1:
            assert init_episode_path is not None  # validated above
            self._warmup_from_episode(
                episode_path=init_episode_path,
                end_frame_index=init_episode_index,
                window_size=init_window_size,
            )
            self._step_counter = 0
            return self._make_observation(source="reset")

        # Single-frame init (existing behavior — unchanged).
        if init_rgb is not None:
            views = self._normalize_init_rgb(init_rgb)
        else:
            episode_path = init_episode_path or str(
                self._resolve_repo_path(self._spec.default_episode_path)
            )
            views = self._load_rgb_views_from_hdf5(episode_path, init_episode_index)

        with torch.no_grad():
            latent = self._encode_views(views)  # (C_latent, H_lat, W_lat)

        self._latent_window = latent.unsqueeze(0).contiguous()  # (1, C, H, W)
        self._action_window = torch.zeros(
            (1, self._action_dim),
            device=self.device,
            dtype=self._loaded.dtype,
        )
        self._step_counter = 0
        return self._make_observation(source="reset")

    # --------------------------------------------------------------- observe

    def observe(
        self,
        rgb: np.ndarray | dict[str, np.ndarray],
        last_action: np.ndarray | torch.Tensor,
    ) -> tuple[Observation, dict]:
        """Incorporate a real RGB observation into the env state.

        This is the deployment-time counterpart to :meth:`step`. Where
        ``step(action)`` predicts the next latent via the dynamics model
        from an action the planner is ABOUT to take, ``observe(rgb,
        last_action)`` consumes a real RGB observation produced by a
        physical execution and the action that JUST drove the env into
        that observation.

        Both ``latent_window`` and ``action_window`` are extended by one
        slot and trimmed to ``hist_context``. ``step_counter`` increments
        identically to ``step()``. ``snapshot()``/``restore()`` work
        across ``observe()`` unchanged.

        Parameters
        ----------
        rgb: real RGB observation at the current step. Single ``np.ndarray``
            for single-view tasks, ``dict[obs_key, np.ndarray]`` otherwise.
            Each array must be (H, W, 3) uint8 RGB (not BGR).
        last_action: the absolute normalized action ``[-1, 1]^action_dim``
            that drove the env from the previous state into the state
            ``rgb`` shows. Required; passing ``None`` raises ``TypeError``.

        Returns
        -------
        ``(Observation, info)``. ``Observation.rgb`` carries the
        preprocessed input RGB (center-crop + resize to
        ``(resolution, resolution, 3)`` uint8) — NOT a decoder round-trip.
        For multi-view tasks it's a ``dict[obs_key, ndarray]``.
        ``info["source"] == "encoder"``.

        Raises
        ------
        TypeError: ``last_action`` is None.
        RuntimeError: called before ``reset()``.
        ValueError / TypeError: invalid ``rgb`` shape/dtype or
            invalid ``last_action`` shape (delegated to existing
            helpers).
        """
        self._check_open()
        if last_action is None:
            raise TypeError(
                "observe() requires last_action (the action that produced this rgb); "
                "got None"
            )
        if self._latent_window is None or self._action_window is None:
            raise RuntimeError(
                "observe() called before reset(); call env.reset(...) first."
            )

        action_tensor, clipped = self._to_action_tensor(last_action)

        views = self._normalize_init_rgb(rgb)
        preprocessed = [_preprocess_rgb_uint8(v, self._resolution) for v in views]

        with torch.no_grad():
            new_latent = self._encode_preprocessed_views(preprocessed)  # (C, H, W)

        self._latent_window = torch.cat(
            [self._latent_window, new_latent.unsqueeze(0)], dim=0
        )[-self._hist_context :].contiguous()
        self._action_window = torch.cat(
            [self._action_window, action_tensor.unsqueeze(0)], dim=0
        )[-self._hist_context :].contiguous()
        self._step_counter += 1

        if len(preprocessed) == 1:
            obs_rgb: np.ndarray | dict[str, np.ndarray] = preprocessed[0]
        else:
            obs_rgb = {k: arr for k, arr in zip(self._spec.obs_keys, preprocessed, strict=True)}

        obs = self._make_observation(source="encoder", rgb=obs_rgb)
        info = {
            "step": self._step_counter,
            "clipped": clipped,
            "latent_norm": float(new_latent.norm().item()),
            "source": "encoder",
        }
        return obs, info

    # ----------------------------------------------------------------- step

    def step(self, action: np.ndarray | torch.Tensor) -> tuple[Observation, dict]:
        """Roll the latent window forward by exactly one step."""
        self._check_open()
        assert self._latent_window is not None and self._action_window is not None, (
            "step() called before reset()"
        )

        action_tensor, clipped = self._to_action_tensor(action)

        # Build (1, T_hist + 1, A) action tensor: history + the new action.
        full_actions = torch.cat(
            [self._action_window, action_tensor.unsqueeze(0)], dim=0
        )
        action_for_dyn = full_actions.unsqueeze(0)  # (1, T_hist+1, A)

        # Latents: (1, T_hist, C, H, W).
        z_in = self._latent_window.unsqueeze(0)

        with torch.no_grad():
            latent_pred = self._loaded.model.dynamics_forward(
                z_in, action_for_dyn
            )  # (1, 1, C, H, W)

        new_latent = latent_pred[0, 0].detach()  # (C, H, W)
        self._latent_window = torch.cat(
            [self._latent_window, new_latent.unsqueeze(0)], dim=0
        )[-self._hist_context :].contiguous()
        self._action_window = torch.cat(
            [self._action_window, action_tensor.unsqueeze(0)], dim=0
        )[-self._hist_context :].contiguous()
        self._step_counter += 1

        obs = self._make_observation(source="dynamics")
        info = {
            "step": self._step_counter,
            "clipped": clipped,
            "latent_norm": float(new_latent.norm().item()),
            "source": "dynamics",
        }
        return obs, info

    # ----------------------------------------------- snapshot / restore

    def snapshot(self) -> EnvState:
        """Return a frozen, deep-cloned EnvState."""
        self._check_open()
        assert self._latent_window is not None and self._action_window is not None, (
            "snapshot() called before reset()"
        )
        return EnvState(
            latent_window=self._latent_window.detach().clone(),
            action_window=self._action_window.detach().clone(),
            step_counter=self._step_counter,
            task=self._task,
        )

    def restore(self, state: EnvState) -> None:
        """Replace internal state from a snapshot (deep-cloned again on the way in)."""
        self._check_open()
        if state.task != self._task:
            raise ValueError(
                f"Cannot restore: snapshot is for task {state.task!r}, "
                f"env is for {self._task!r}."
            )
        self._latent_window = state.latent_window.detach().clone().to(self.device)
        self._action_window = state.action_window.detach().clone().to(self.device)
        self._step_counter = int(state.step_counter)

    def _snapshot_noclone(self) -> EnvState:
        """Private variant used internally by step_batch — skips the clone since
        the caller (us) guarantees no mutation between snapshot and use."""
        assert self._latent_window is not None and self._action_window is not None
        return EnvState(
            latent_window=self._latent_window,
            action_window=self._action_window,
            step_counter=self._step_counter,
            task=self._task,
        )

    # ----------------------------------------------------------- decoding

    def render(self) -> np.ndarray:
        """Decode the most recent latent to an HWC uint8 RGB frame.

        For single-view tasks this is (resolution, resolution, 3). For
        multi-view tasks it would be (resolution, resolution, 3*V) — all
        shipped tasks are single-view, so V=1 in practice.
        """
        self._check_open()
        assert self._latent_window is not None
        last = self._latent_window[-1].unsqueeze(0)  # (1, C, H, W)
        rgb_chw = self._decode_latents(last)[0]  # (3*V, H_img, W_img) uint8
        return np.transpose(rgb_chw, (1, 2, 0))  # HWC

    def latent_to_rgb(self, latent: torch.Tensor) -> np.ndarray:
        """Decode an arbitrary latent (or batch of latents) to RGB.

        Accepts (C, H, W) or (B, C, H, W). Returns HWC uint8 for a single
        latent or (B, H, W, 3*V) uint8 for a batch.
        """
        self._check_open()
        if latent.ndim == 3:
            batched = latent.unsqueeze(0).to(self.device, dtype=self._loaded.dtype)
            rgb_chw = self._decode_latents(batched)[0]
            return np.transpose(rgb_chw, (1, 2, 0))
        if latent.ndim == 4:
            batched = latent.to(self.device, dtype=self._loaded.dtype)
            rgb_chw = self._decode_latents(batched)  # (B, 3*V, H, W)
            return np.transpose(rgb_chw, (0, 2, 3, 1))
        raise ValueError(
            f"latent_to_rgb expects a (C, H, W) or (B, C, H, W) tensor, got shape {tuple(latent.shape)}"
        )

    # ------------------------------------------------------- goal_preprocess

    def goal_preprocess(
        self, rgb: np.ndarray | dict[str, np.ndarray]
    ) -> np.ndarray:
        """Apply the dataset pipeline's RGB preprocessing.

        Center-crops and resizes each view to (resolution, resolution),
        converts to float32 in [0, 1], stacks views channel-first.

        Returns
        -------
        For single-view tasks: (3, resolution, resolution).
        For multi-view tasks:  (3 * num_views, resolution, resolution).
        Values are in [0, 1]; no encoder, no normalizer is applied here.
        """
        views = self._normalize_init_rgb(rgb)  # list[(H, W, 3) uint8]
        chunks = []
        for arr in views:
            chunks.append(_rgb_to_chw_float01(arr, self._resolution))
        return np.concatenate(chunks, axis=0)

    # ------------------------------------------------------------ step_batch

    def step_batch(
        self, actions: np.ndarray | torch.Tensor
    ) -> BatchedObservation:
        """Run K parallel rollouts of length H from the current state.

        Parameters
        ----------
        actions: (K, H, action_dim) float32 in [-1, 1]. Out-of-range entries
            are silently clipped.

        Returns
        -------
        BatchedObservation with:
            latents: (K, H, C_latent, H_lat, W_lat) on env.device, fp32.
            rgbs:    (K, H, 3 * num_views, resolution, resolution) uint8.

        Side effects: none. The env's own state is untouched.
        """
        self._check_open()
        assert self._latent_window is not None and self._action_window is not None, (
            "step_batch() called before reset()"
        )

        actions_t = torch.as_tensor(
            actions, dtype=self._loaded.dtype, device=self.device
        )
        if actions_t.ndim != 3 or actions_t.shape[-1] != self._action_dim:
            raise ValueError(
                f"step_batch expects (K, H, {self._action_dim}); got {tuple(actions_t.shape)}"
            )
        actions_t = torch.clamp(actions_t, -1.0, 1.0)
        K, H, _ = actions_t.shape

        # Build batched z_0 of shape (K, T_hist, C, H, W)
        z_start = self._latent_window.detach()  # (T_hist, C, H, W)
        z_batched = (
            z_start.unsqueeze(0).expand(K, *z_start.shape).contiguous()
        )  # (K, T_hist, C, H, W)

        # Build batched action of shape (K, T_hist + H, A)
        a_hist = self._action_window.detach()  # (T_hist, A)
        a_hist_batched = (
            a_hist.unsqueeze(0).expand(K, *a_hist.shape).contiguous()
        )  # (K, T_hist, A)
        a_batched = torch.cat([a_hist_batched, actions_t], dim=1)  # (K, T_hist+H, A)

        with torch.no_grad():
            latents = self._loaded.model.dynamics_forward(
                z_batched, a_batched
            )  # (K, H, C, H, W)

        # Decode every (k, h) cell. Flatten the leading two dims for
        # render_img_cm, then reshape.
        flat = latents.reshape(K * H, *latents.shape[2:])
        rgb_chw = self._decode_latents(flat)  # (K*H, 3*V, H_img, W_img) uint8
        rgbs = rgb_chw.reshape(K, H, *rgb_chw.shape[1:])

        return BatchedObservation(
            latents=latents.detach(),
            rgbs=rgbs,
        )

    # ----------------------------------------------------------- shutdown

    def close(self) -> None:
        """Release the model, drop tensors, ask CUDA to release cached memory.

        Idempotent. After close(), all other methods raise.
        """
        if self._closed:
            return
        self._latent_window = None
        self._action_window = None
        self._loaded = None  # type: ignore[assignment]
        torch.cuda.empty_cache()
        self._closed = True

    def as_gym(self) -> Any:
        """Adapter to gymnasium.Env — not implemented in this pass."""
        raise NotImplementedError("Gymnasium adapter not implemented yet.")

    # ============================================================== internals

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("WorldModelEnv has been closed.")

    def _make_observation(
        self,
        *,
        source: str = "dynamics",
        rgb: np.ndarray | dict[str, np.ndarray] | None = None,
    ) -> Observation:
        """Build an Observation for the current env state.

        Parameters
        ----------
        source: producer of the latest latent — one of "reset", "dynamics",
            "encoder". Used purely for default rgb handling: when source is
            "dynamics" and decode_on_step=True the render() path is taken;
            "encoder" callers pass the preprocessed input rgb explicitly;
            "reset" leaves rgb as None unless the caller supplies it.
        rgb: explicit rgb to attach to the Observation. If None, the env
            falls back to the source-specific default. If supplied, it is
            stored as-is.
        """
        assert self._latent_window is not None
        latest = self._latent_window[-1].detach()
        history = self._latent_window.detach()
        if rgb is None and source == "dynamics" and self._decode_on_step:
            rgb_out: np.ndarray | dict[str, np.ndarray] | None = self.render()
        else:
            rgb_out = rgb
        return Observation(
            latent=latest,
            latent_history=history,
            step=self._step_counter,
            rgb=rgb_out,
        )

    def _to_action_tensor(
        self, action: np.ndarray | torch.Tensor
    ) -> tuple[torch.Tensor, bool]:
        """Cast/clip an input action to a 1-D device tensor; report clipping."""
        if isinstance(action, np.ndarray):
            a = torch.as_tensor(action, dtype=self._loaded.dtype, device=self.device)
        elif isinstance(action, torch.Tensor):
            a = action.to(self.device, dtype=self._loaded.dtype)
        else:
            raise TypeError(
                f"action must be numpy.ndarray or torch.Tensor, got {type(action).__name__}"
            )
        if a.shape != (self._action_dim,):
            raise ValueError(
                f"action must have shape ({self._action_dim},); got {tuple(a.shape)}"
            )
        if not torch.isfinite(a).all():
            raise ValueError("action contains non-finite values.")
        out_of_range = bool((a < -1.0).any() or (a > 1.0).any())
        a = torch.clamp(a, -1.0, 1.0)
        return a, out_of_range

    def _normalize_init_rgb(
        self, rgb: np.ndarray | dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        """Validate init_rgb and return a list of per-view (H, W, 3) uint8 arrays."""
        if isinstance(rgb, np.ndarray):
            if self._num_views != 1:
                raise ValueError(
                    f"Task {self._task!r} has {self._num_views} views; pass a dict "
                    f"keyed by {self._spec.obs_keys}."
                )
            return [_validate_rgb(rgb)]
        if isinstance(rgb, dict):
            missing = [k for k in self._spec.obs_keys if k not in rgb]
            if missing:
                raise ValueError(
                    f"init_rgb is missing required keys {missing} for task {self._task!r}."
                )
            return [_validate_rgb(rgb[k]) for k in self._spec.obs_keys]
        raise TypeError(
            f"init_rgb must be a numpy array or dict; got {type(rgb).__name__}"
        )

    def _load_rgb_views_from_hdf5(
        self, episode_path: str, frame_index: int
    ) -> list[np.ndarray]:
        """Replicate the demos' HDF5-to-RGB loading."""
        path = Path(episode_path)
        if not path.exists():
            raise FileNotFoundError(f"Episode HDF5 not found: {path}")
        epi_data, _ = load_dict_from_hdf5(str(path))
        views: list[np.ndarray] = []
        for k in self._spec.obs_keys:
            try:
                imgs = epi_data["obs"]["images"][k]
            except KeyError as e:
                raise KeyError(
                    f"Episode {path} does not contain obs/images/{k}"
                ) from e
            if frame_index >= imgs.shape[0]:
                raise IndexError(
                    f"frame_index={frame_index} out of range for episode of "
                    f"length {imgs.shape[0]} ({path})"
                )
            views.append(np.asarray(imgs[frame_index]))
        return views

    def _validate_init_window_size(
        self,
        init_window_size: int,
        *,
        init_state: EnvState | None,
        init_rgb: np.ndarray | dict[str, np.ndarray] | None,
        init_episode_path: str | None,
        init_episode_index: int,
    ) -> None:
        """Validate the reset() kwargs governing the warmup path."""
        if init_window_size < 1:
            raise ValueError(
                f"init_window_size must be >= 1; got {init_window_size}"
            )
        if init_window_size > self._hist_context:
            raise ValueError(
                f"init_window_size={init_window_size} exceeds hist_context="
                f"{self._hist_context}; the rolling window cannot hold a "
                "warmup longer than its capacity."
            )
        if init_window_size == 1:
            return  # the rest of the checks only apply to multi-frame warmup

        if self._task != "pusht_cam1":
            raise NotImplementedError(
                f"reset(init_window_size > 1) requires expert_action_from_episode, "
                f"which is only verified for 'pusht_cam1'; got task={self._task!r}. "
                "Extend the helper before using warmup on other tasks."
            )
        if init_rgb is not None:
            raise ValueError(
                "init_rgb cannot fill a multi-frame warmup window; pass "
                "init_episode_path or use init_window_size=1."
            )
        if init_state is not None:
            raise ValueError(
                "init_state already encodes a complete window; combining it with "
                "init_window_size > 1 is contradictory. Use init_window_size=1, "
                "or drop init_state."
            )
        if init_episode_path is None:
            raise ValueError(
                "init_window_size > 1 requires init_episode_path; pass an HDF5 "
                "episode to warm up from."
            )
        if init_episode_index < init_window_size - 1:
            raise ValueError(
                f"init_episode_index={init_episode_index} too small for a window "
                f"of size {init_window_size}; need init_episode_index >= "
                f"{init_window_size - 1}."
            )

    def _warmup_from_episode(
        self,
        *,
        episode_path: str,
        end_frame_index: int,
        window_size: int,
    ) -> None:
        """Populate latent_window and action_window from W consecutive frames.

        latent_window[i]    = encode(frame end_frame_index - W + 1 + i)
        action_window[i]    = expert action AT frame (end_frame_index - W + i)
                              if that frame exists; else zeros.

        The action_window convention is "action that drove INTO this latent",
        which matches what observe() appends. This makes a warmup-from-W
        equivalent to a single-frame reset followed by (W-1) observe()
        calls — see Test 9 in the smoke script.
        """
        latents_list: list[torch.Tensor] = []
        actions_list: list[torch.Tensor] = []
        for i in range(window_size):
            frame_idx = end_frame_index - window_size + 1 + i
            views = self._load_rgb_views_from_hdf5(episode_path, frame_idx)
            with torch.no_grad():
                latents_list.append(self._encode_views(views))

            prev_frame_idx = frame_idx - 1
            if prev_frame_idx < 0:
                actions_list.append(
                    torch.zeros(
                        self._action_dim,
                        device=self.device,
                        dtype=self._loaded.dtype,
                    )
                )
            else:
                a_np = expert_action_from_episode(
                    self, episode_path, prev_frame_idx
                )
                actions_list.append(
                    torch.from_numpy(a_np).to(
                        self.device, dtype=self._loaded.dtype
                    )
                )

        self._latent_window = torch.stack(latents_list, dim=0).contiguous()
        self._action_window = torch.stack(actions_list, dim=0).contiguous()

    def _encode_views(self, views: list[np.ndarray]) -> torch.Tensor:
        """Preprocess (center-crop+resize) and run the encoder.

        Mirrors deploy/server.py:151-168 and teleoperate_keyboard.py:723-742.
        """
        preprocessed = [_preprocess_rgb_uint8(v, self._resolution) for v in views]
        return self._encode_preprocessed_views(preprocessed)

    def _encode_preprocessed_views(
        self, preprocessed_views: list[np.ndarray]
    ) -> torch.Tensor:
        """Encode views that have already been center-cropped + resized to
        (resolution, resolution, 3) uint8 HWC.

        Splitting this out lets observe() compute the preprocessed RGB
        once and reuse the same bytes both as the encoder input and as
        the value stored on Observation.rgb — no redundant crop/resize.
        """
        normalizer = self._loaded.model.normalizer
        tensors: list[torch.Tensor] = []
        for arr_u8, obs_key in zip(
            preprocessed_views, self._spec.obs_keys, strict=True
        ):
            chw01 = np.transpose(arr_u8.astype(np.float32) / 255.0, (2, 0, 1))
            t = torch.from_numpy(chw01).unsqueeze(0)  # (1, 3, H, W)
            t = (
                normalizer[obs_key]
                .normalize(t)
                .to(self.device, dtype=self._loaded.dtype)
            )
            tensors.append(t)
        stacked = torch.cat(tensors, dim=1)  # (1, 3*V, H, W)
        latent = self._loaded.model.encoder_forward(stacked)  # (1, C_lat, H_lat, W_lat)
        return latent[0].detach()

    def _decode_latents(self, latents: torch.Tensor) -> np.ndarray:
        """Decode a (B, C_latent, H_lat, W_lat) tensor to (B, 3*V, H, W) uint8."""
        with torch.no_grad():
            rgb = render_img_cm(
                self._loaded.model,
                latents.to(self.device, dtype=self._loaded.dtype),
                self._resolution,
                normalizer=self._loaded.model.normalizer,
                num_views=self._num_views,
            )  # (B, 3*V, H, W) float in [0,1]
        rgb_u8 = (rgb.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        return rgb_u8.detach().cpu().numpy()

    def _resolve_repo_path(self, rel_or_abs: str) -> Path:
        p = Path(rel_or_abs)
        if p.is_absolute():
            return p
        if self._repo_root is not None:
            return (self._repo_root / p).resolve()
        # _model_loader already walked up to the repo root once; recompute.
        from ._model_loader import _find_repo_root

        return (_find_repo_root() / p).resolve()


# ===========================================================================
# Module-level helpers (pure functions, no model access)
# ===========================================================================


def _validate_rgb(arr: np.ndarray) -> np.ndarray:
    """Sanity-check a user-supplied RGB array."""
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(
            f"RGB array must have shape (H, W, 3); got {tuple(arr.shape)}"
        )
    if arr.dtype != np.uint8:
        raise ValueError(
            f"RGB array must be uint8 (RGB, not BGR); got dtype {arr.dtype}"
        )
    return arr


def _preprocess_rgb_uint8(arr: np.ndarray, resolution: int) -> np.ndarray:
    """Center-crop (aspect 1:1) + resize. Returns (resolution, resolution, 3) uint8.

    Shares its preprocessing semantics with the dataset pipeline: see
    `yixuan_utilities.draw_utils.center_crop` (aspect-ratio-driven crop,
    so a 640x480 input with target aspect 1:1 yields a 480x480 center
    crop), then a uniform resize.
    """
    arr = _validate_rgb(arr)
    arr = center_crop(arr, (resolution, resolution))
    arr = cv2.resize(arr, (resolution, resolution), interpolation=cv2.INTER_AREA)
    return arr.copy()


def _rgb_to_chw_float01(arr: np.ndarray, resolution: int) -> np.ndarray:
    """Center-crop + resize + /255 + HWC->CHW. Matches the demo preprocessing."""
    arr = _validate_rgb(arr)
    arr = center_crop(arr, (resolution, resolution))
    arr = cv2.resize(arr, (resolution, resolution), interpolation=cv2.INTER_AREA)
    arr = arr.astype(np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1)).copy()  # contiguous (3, H, W)
