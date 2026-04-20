"""Overlay state estimates on top of an RGB image.

The primitive in `render_state_on_image` is the single drawing routine
used by the sanity check, the validation grid, the worst-K viz, and the
RL-rollout hook. Keep it small and side-effect-free so callers can build
multi-overlay canvases by chaining calls.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

ARROW_LEN_PX = 20
CENTER_RADIUS_PX = 3
LINE_THICKNESS = 2
ARROW_TIP_LENGTH = 0.35
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.4
TEXT_THICKNESS = 1


def _to_uint8_rgb(rgb) -> np.ndarray:
    if isinstance(rgb, torch.Tensor):
        arr = rgb.detach().cpu().numpy()
    else:
        arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3) RGB, got shape {arr.shape}")
    if arr.dtype == np.uint8:
        return arr.copy()
    if np.issubdtype(arr.dtype, np.floating):
        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr.astype(np.uint8)


def render_state_on_image(
    rgb,
    cx: float,
    cy: float,
    sin_theta: float,
    cos_theta: float,
    color: tuple[int, int, int] = (255, 0, 0),
    label: str | None = None,
) -> np.ndarray:
    """Draw a center marker + orientation arrow on an RGB image.

    Args:
        rgb: (H, W, 3) image, uint8 or float in [0, 1], np.ndarray or torch.Tensor.
            On the first call, the image is copied — subsequent calls can chain
            on the returned canvas to overlay multiple sources.
        cx, cy: pixel coordinates of the marker center.
        sin_theta, cos_theta: orientation. Direction = (cos θ, sin θ) in
            standard image coordinates (x right, y down).
        color: RGB triple in 0..255.
        label: optional short text drawn next to the marker, same color.

    Returns:
        (H, W, 3) uint8 RGB image with overlay drawn.
    """
    canvas = _to_uint8_rgb(rgb)

    cx_i = int(round(cx))
    cy_i = int(round(cy))
    tip_x = int(round(cx + ARROW_LEN_PX * cos_theta))
    tip_y = int(round(cy + ARROW_LEN_PX * sin_theta))

    cv2.arrowedLine(
        canvas,
        (cx_i, cy_i),
        (tip_x, tip_y),
        color=color,
        thickness=LINE_THICKNESS,
        tipLength=ARROW_TIP_LENGTH,
    )
    cv2.circle(canvas, (cx_i, cy_i), CENTER_RADIUS_PX, color, thickness=-1)

    if label:
        text_org = (cx_i + CENTER_RADIUS_PX + 3, cy_i - CENTER_RADIUS_PX - 3)
        cv2.putText(
            canvas, label, text_org, TEXT_FONT, TEXT_SCALE, color, TEXT_THICKNESS, cv2.LINE_AA
        )

    return canvas
