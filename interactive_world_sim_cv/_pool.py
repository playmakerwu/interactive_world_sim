"""Worker function and template-contour cache for parallel detection.

This module is process-pool-aware: `_worker_task` is the picklable
callable executed by ProcessPoolExecutor workers and also by the
controller process for sequential `detect()` calls. The lru_cache on
`_cached_template` is per-process — each worker fills its own copy
on first use at a given resolution.

The `_aloha` import is to our local `_detection` module (the
verbatim-copied pose detection). Production code never reaches into
the aloha repo.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Literal

import cv2
import numpy as np

from . import _detection


@lru_cache(maxsize=8)
def _cached_template(processing_resolution: int) -> np.ndarray:
    """Memoize get_template_contour by integer processing_resolution.

    Cache is per-process. maxsize=8 is generous; in practice only one
    or two distinct resolutions appear over the lifetime of a worker.
    Keying on the int (not the float t_scale) keeps the hash stable.
    """
    return _detection.get_template_contour(
        _detection.T_BLOCK_SHAPE,
        processing_resolution / 512.0,
    )


def _hsv_for_mode(mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "wm":
        return _detection.HSV_LOWER_WM, _detection.HSV_UPPER_WM
    if mode == "real":
        return _detection.HSV_LOWER_REAL, _detection.HSV_UPPER_REAL
    raise ValueError(f"mode must be 'wm' or 'real'; got {mode!r}")


def _worker_task(
    rgb: np.ndarray,
    mode: Literal["wm", "real"],
    processing_resolution: int,
):
    """Single-frame detection. Returns TPose or None.

    Picklable by reference. Lazily imports TPose from api to avoid a
    module-load cycle (api imports _pool indirectly via DetectorPool).
    """
    # Lazy import to break the api ↔ _pool cycle. Cheap once api.py is
    # loaded in the parent (workers inherit sys.modules on fork).
    from .api import TPose

    hsv_lower, hsv_upper = _hsv_for_mode(mode)

    # Four-step preprocessing — see api.py header for citations to
    # analyze.py line ranges.
    frame_resized = cv2.resize(
        rgb,
        (processing_resolution, processing_resolution),
        interpolation=cv2.INTER_CUBIC,
    )
    frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
    template_contour = _cached_template(processing_resolution)

    raw_center, raw_angle, raw_error = _detection.estimate_current_pose(
        frame_bgr,
        template_contour,
        processing_resolution / 512.0,
        hsv_lower,
        hsv_upper,
    )

    if raw_center is None:
        return None

    angle_rad = math.radians(raw_angle)
    return TPose(
        x=float(raw_center[0]),
        y=float(raw_center[1]),
        sin=math.sin(angle_rad),
        cos=math.cos(angle_rad),
        angle_deg=float(raw_angle),
        error=float(raw_error),
    )
