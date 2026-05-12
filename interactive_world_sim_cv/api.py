"""Public API for the pink-T pose detector.

# ============================================================================
# Verbatim replication of aloha's process_episode_wm pose-detection
# preprocessing. Do not reorder, do not inline, do not "simplify."
# Citations point to the original aloha source for provenance; the
# actual code runs from interactive_world_sim_cv/_detection.py, which is
# a verbatim copy of those line ranges. See _detection.py header for the
# pinned aloha SHA.
#
#   1. Upscale to processing resolution with INTER_CUBIC
#      Source: analyze.py:386
#      Code:   frame_resized = cv2.resize(
#                  frame_rgb,
#                  (processing_resolution, processing_resolution),
#                  interpolation=cv2.INTER_CUBIC,
#              )
#
#   2. RGB → BGR
#      Source: analyze.py:387
#      Code:   frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
#
#   3. Build template contour (cached by processing_resolution in our
#      wrapper; aloha caches it outside the function in
#      process_episode_wm at analyze.py:364)
#      Source: analyze.py:90-100 (get_template_contour)
#      Code:   template_contour = get_template_contour(
#                  T_BLOCK_SHAPE,
#                  t_scale = processing_resolution / 512.0,
#              )
#
#   4. Run ICP-based pose estimation
#      Source: analyze.py:390-393 (call) and analyze.py:156-208 (impl)
#      Code:   raw_center, raw_angle, raw_error = estimate_current_pose(
#                  frame_bgr,
#                  template_contour,
#                  scale = processing_resolution / 512.0,
#                  hsv_lower,
#                  hsv_upper,
#              )
#
# Failure: any of (None, None, None) → return None from our wrapper.
# Success: pack (raw_center[0], raw_center[1], sin, cos, raw_angle, raw_error)
# into a TPose.
#
# Sin/cos are derived in the wrapper; aloha returns degrees only.
# angle_deg is UNWRAPPED (loop adds init_angle ∈ {0,30,...,330} to an
# arctan2 in (-180,180]). math.sin / math.cos handle the wrap implicitly.
# No other normalization applied — we return the raw unwrapped angle_deg
# alongside sin/cos so callers can audit.
# ============================================================================
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from typing import Literal

import numpy as np

from . import _pool


@dataclass(frozen=True)
class TPose:
    """Detected pink-T pose in processing-resolution pixel space.

    Attributes
    ----------
    x, y : float
        Pixel coords of the T-block centroid in the processing-resolution
        frame. For the default processing_resolution=512, these are in
        [0, 512). NOT in 128² source space. Callers wanting 128² coords
        must divide by (processing_resolution / source_resolution).
    sin, cos : float
        Decomposition of angle_deg into the unit circle. Provided as a
        convenience; the source of truth is angle_deg.
            sin = math.sin(math.radians(angle_deg))
            cos = math.cos(math.radians(angle_deg))
    angle_deg : float
        Raw aloha output, in degrees, UNWRAPPED. Roughly in [0, 690]
        because aloha's ICP loop adds an init_angle in {0, 30, ..., 330}
        to an arctan2 result in (-180, 180]. No normalization applied.
    error : float
        ICP residual: mean Euclidean distance (pixels in processing
        resolution) between the trimmed-fraction template points and
        their nearest mask-contour neighbors. Lower is better.
        NOT a confidence score. NOT in [0, 1]. NOT comparable across
        different processing_resolution values.
    """

    x: float
    y: float
    sin: float
    cos: float
    angle_deg: float
    error: float


def _validate_rgb(rgb: np.ndarray) -> None:
    if not isinstance(rgb, np.ndarray):
        raise TypeError(f"rgb must be a numpy array; got {type(rgb).__name__}")
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"rgb must be (H, W, 3); got shape {rgb.shape}")
    if rgb.dtype != np.uint8:
        raise ValueError(f"rgb must be uint8; got {rgb.dtype}")


def _validate_rgbs(rgbs: np.ndarray) -> None:
    if not isinstance(rgbs, np.ndarray):
        raise TypeError(f"rgbs must be a numpy array; got {type(rgbs).__name__}")
    if rgbs.ndim != 4 or rgbs.shape[-1] != 3:
        raise ValueError(f"rgbs must be (N, H, W, 3); got shape {rgbs.shape}")
    if rgbs.dtype != np.uint8:
        raise ValueError(f"rgbs must be uint8; got {rgbs.dtype}")


def _validate_processing_resolution(processing_resolution: int) -> None:
    if not isinstance(processing_resolution, int) or processing_resolution <= 0:
        raise ValueError(
            f"processing_resolution must be a positive int; got {processing_resolution!r}"
        )


def _validate_mode(mode: str) -> None:
    if mode not in ("wm", "real"):
        raise ValueError(f"mode must be 'wm' or 'real'; got {mode!r}")


def detect(
    rgb: np.ndarray,
    mode: Literal["wm", "real"],
    processing_resolution: int = 512,
) -> TPose | None:
    """Detect the pink-T in a single RGB frame.

    Parameters
    ----------
    rgb : np.ndarray
        (H, W, 3) uint8 HWC RGB. Internally upscaled to
        (processing_resolution, processing_resolution) with INTER_CUBIC.
    mode : {"wm", "real"}
        Selects the HSV color range. Caller must choose explicitly.
        "wm"   → HSV_LOWER_WM/HSV_UPPER_WM (pink/magenta WM renders)
        "real" → HSV_LOWER_REAL/HSV_UPPER_REAL (darker real T-block)
    processing_resolution : int
        Detection runs at this resolution (default 512, matching aloha's
        WM pipeline). Output (x, y) are in this resolution's pixel space.

    Returns
    -------
    TPose | None
        None on detection failure. Three internal failure paths
        (empty mask / area < 100 / empty template) collapse to None.
    """
    _validate_rgb(rgb)
    _validate_mode(mode)
    _validate_processing_resolution(processing_resolution)
    return _pool._worker_task(rgb, mode, processing_resolution)


def detect_batch(
    rgbs: np.ndarray,
    mode: Literal["wm", "real"],
    processing_resolution: int = 512,
    num_workers: int | None = None,
) -> list[TPose | None]:
    """Detect the pink-T in a batch of RGB frames.

    Parameters
    ----------
    rgbs : np.ndarray
        (N, H, W, 3) uint8 HWC RGB.
    num_workers : int | None
        - None  : sequential, no process pool. Recommended for small N
                  or when called once.
        - int>0 : spawns a fresh ProcessPoolExecutor of that size for
                  THIS CALL ONLY, tears it down on return. Simple but
                  pays full pool startup cost every call. Use
                  DetectorPool for repeated batched calls.

    Returns
    -------
    list[TPose | None]
        Same length as rgbs. Element i is None iff detection failed on
        rgbs[i]. Order preserved.
    """
    _validate_rgbs(rgbs)
    _validate_mode(mode)
    _validate_processing_resolution(processing_resolution)

    if num_workers is None:
        return [_pool._worker_task(rgbs[i], mode, processing_resolution) for i in range(len(rgbs))]

    if not isinstance(num_workers, int) or num_workers < 1:
        raise ValueError(f"num_workers must be None or positive int; got {num_workers!r}")

    with DetectorPool(num_workers) as pool:
        return pool.detect_batch(rgbs, mode, processing_resolution)


class DetectorPool:
    """Persistent process pool for batched detection.

    Use when MPPI calls detect_batch many times across plan iterations —
    spawning a fresh ProcessPoolExecutor per call is expensive
    (~hundreds of ms on Linux fork). DetectorPool keeps workers warm
    between calls; each worker holds its own loaded _detection module
    and per-resolution template-contour lru_cache.

    Example
    -------
        with DetectorPool(num_workers=8) as pool:
            for plan_iter in mppi_loop:
                results = pool.detect_batch(rgbs, mode="wm")

    Workers spawn eagerly on `start()` (called by `__enter__`), so the
    first `detect_batch()` call already has warm workers. Use as a
    context manager to guarantee shutdown.

    Not thread-safe. MPPI is single-controller-thread; that's fine.
    """

    def __init__(self, num_workers: int) -> None:
        if not isinstance(num_workers, int) or num_workers < 1:
            raise ValueError(f"num_workers must be a positive int; got {num_workers!r}")
        self._num_workers = num_workers
        self._executor: ProcessPoolExecutor | None = None

    def start(self) -> None:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self._num_workers)

    def shutdown(self, *, wait: bool = True) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=wait, cancel_futures=True)
            self._executor = None

    def __enter__(self) -> "DetectorPool":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown(wait=True)

    def detect_batch(
        self,
        rgbs: np.ndarray,
        mode: Literal["wm", "real"],
        processing_resolution: int = 512,
    ) -> list[TPose | None]:
        _validate_rgbs(rgbs)
        _validate_mode(mode)
        _validate_processing_resolution(processing_resolution)
        if self._executor is None:
            self.start()
        n = len(rgbs)
        results = list(
            self._executor.map(
                _pool._worker_task,
                (rgbs[i] for i in range(n)),
                repeat(mode, n),
                repeat(processing_resolution, n),
            )
        )
        return results
