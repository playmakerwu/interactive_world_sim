"""Unit tests for rl.visualization.state_viz.render_state_on_image."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from rl.visualization.state_viz import render_state_on_image


H = W = 64


def _blank(dtype=np.uint8) -> np.ndarray:
    if dtype == np.uint8:
        return np.full((H, W, 3), 200, dtype=np.uint8)
    if dtype == np.float32:
        return np.full((H, W, 3), 0.8, dtype=np.float32)
    raise ValueError(dtype)


def test_signature_and_defaults():
    sig = inspect.signature(render_state_on_image)
    params = list(sig.parameters.keys())
    assert params == ["rgb", "cx", "cy", "sin_theta", "cos_theta", "color", "label"]
    assert sig.parameters["color"].default == (255, 0, 0)
    assert sig.parameters["label"].default is None


def test_uint8_input_preserves_shape_and_dtype():
    img = _blank()
    out = render_state_on_image(img, cx=32, cy=32, sin_theta=0.0, cos_theta=1.0)
    assert out.shape == (H, W, 3)
    assert out.dtype == np.uint8


def test_float_input_returns_uint8():
    img = _blank(np.float32)
    out = render_state_on_image(img, cx=32, cy=32, sin_theta=0.0, cos_theta=1.0)
    assert out.shape == (H, W, 3)
    assert out.dtype == np.uint8


def test_tensor_input_accepted():
    img = torch.full((H, W, 3), 0.8, dtype=torch.float32)
    out = render_state_on_image(img, cx=32, cy=32, sin_theta=0.0, cos_theta=1.0)
    assert out.shape == (H, W, 3)
    assert out.dtype == np.uint8


def test_input_not_mutated():
    img = _blank()
    img_copy = img.copy()
    _ = render_state_on_image(img, cx=32, cy=32, sin_theta=0.0, cos_theta=1.0)
    np.testing.assert_array_equal(img, img_copy), "input must not be mutated in place"


def test_draws_something_at_center():
    img = _blank()
    out = render_state_on_image(
        img, cx=32, cy=32, sin_theta=0.0, cos_theta=1.0, color=(255, 0, 0)
    )
    # The center marker has radius CENTER_RADIUS_PX=3 and solid red fill.
    # Pixel at (32, 32) must now contain red.
    assert tuple(out[32, 32]) == (255, 0, 0)


def test_arrow_points_in_expected_direction():
    # cos=1, sin=0 => arrow tip to the right of center at (32 + 20, 32)
    img = _blank()
    out = render_state_on_image(
        img, cx=32, cy=32, sin_theta=0.0, cos_theta=1.0, color=(0, 0, 255)
    )
    # Pixel along +x from center should have been drawn on.
    row = out[32]
    assert np.any(np.all(row[:, :] == (0, 0, 255), axis=-1)), "arrow not drawn along +x"

    # cos=0, sin=1 => arrow tip downward at (32, 32 + 20).
    out2 = render_state_on_image(
        img, cx=32, cy=32, sin_theta=1.0, cos_theta=0.0, color=(0, 0, 255)
    )
    col = out2[:, 32]
    assert np.any(np.all(col[:, :] == (0, 0, 255), axis=-1)), "arrow not drawn along +y"


def test_multi_overlay_composability():
    """Calling twice on the same canvas must deposit both markers.

    This is the key property: Phase 1 sanity check chains two overlays
    (REAL green + WM orange) on one image; Phase 3-A validation grid
    chains probe (red) + CV (green)."""
    img = _blank()

    red = (255, 0, 0)
    green = (0, 255, 0)

    out1 = render_state_on_image(img, cx=20, cy=20, sin_theta=0.0, cos_theta=1.0, color=red)
    out2 = render_state_on_image(out1, cx=44, cy=44, sin_theta=0.0, cos_theta=1.0, color=green)

    # Both markers present.
    assert tuple(out2[20, 20]) == red, "first overlay lost after second call"
    assert tuple(out2[44, 44]) == green, "second overlay missing"
    # Original image untouched.
    np.testing.assert_array_equal(img, _blank())


def test_label_renders_without_error():
    img = _blank()
    out = render_state_on_image(
        img, cx=32, cy=32, sin_theta=0.0, cos_theta=1.0, label="probe"
    )
    assert out.shape == (H, W, 3)
    # We don't pixel-check text rendering — fonts vary. Just verify no crash
    # and that the marker is still there.
    assert tuple(out[32, 32]) == (255, 0, 0)


def test_rejects_bad_shape():
    with pytest.raises(ValueError):
        render_state_on_image(
            np.zeros((H, W), dtype=np.uint8), cx=0, cy=0, sin_theta=0.0, cos_theta=1.0
        )
    with pytest.raises(ValueError):
        render_state_on_image(
            np.zeros((H, W, 4), dtype=np.uint8), cx=0, cy=0, sin_theta=0.0, cos_theta=1.0
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
