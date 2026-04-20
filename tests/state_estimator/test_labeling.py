"""Unit tests for rl.labeling.cv_labeler.

Covers:
- One-shot labeling on a real dataset image via REAL preset — output keys
  present, success=True, output ranges sensible.
- Blank image → success=False.
- Angle normalisation is idempotent and stays in (-180, 180].
- (sin theta, cos theta) consistent with theta_rad.
- CVLabeler resolution check rejects mismatched input.
- Torch tensor input accepted.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest
import torch

from rl.labeling.cv_labeler import (
    HSV_PRESETS,
    CVLabeler,
    CVLabelResult,
    label_image,
    normalize_angle_deg,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_IMG = (
    REPO_ROOT / "data" / "mini" / "pusht" / "train" / "episode_0.hdf5"
)

RES = 128


def _center_crop_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(h, w)
    sh = (h - s) // 2
    sw = (w - s) // 2
    return img[sh : sh + s, sw : sw + s]


def _load_dataset_frame(frame_idx: int = 100) -> np.ndarray:
    """Load one RGB frame from the training set, resized to RES x RES."""
    with h5py.File(DATASET_IMG, "r") as f:
        raw = f["obs/images/camera_1_color"][frame_idx]  # (480, 640, 3) uint8 RGB
    cropped = _center_crop_square(raw)
    resized = cv2.resize(cropped, (RES, RES), interpolation=cv2.INTER_AREA)
    return resized


# -----------------------------------------------------------------------------
# Core behaviour
# -----------------------------------------------------------------------------

def test_label_dataset_image_real_preset():
    rgb = _load_dataset_frame()
    result = label_image(rgb, preset="REAL", resolution=RES)
    assert isinstance(result, CVLabelResult)
    assert result.success is True, f"REAL preset should detect T-block in train image; got {result}"

    # required keys per kickoff §Step 2
    keys = {
        "cx", "cy", "theta_rad", "sin_theta", "cos_theta",
        "success", "contour_area", "icp_residual",
    }
    assert keys.issubset(result.as_dict().keys())

    # sensible ranges
    assert 0 <= result.cx < RES
    assert 0 <= result.cy < RES
    assert -np.pi < result.theta_rad <= np.pi
    assert -1.0 <= result.sin_theta <= 1.0
    assert -1.0 <= result.cos_theta <= 1.0
    assert result.contour_area >= 100.0  # passed the reject threshold


def test_sincos_matches_theta_rad():
    rgb = _load_dataset_frame()
    result = label_image(rgb, preset="REAL", resolution=RES)
    assert result.success
    np.testing.assert_allclose(
        result.sin_theta, np.sin(result.theta_rad), atol=1e-6
    )
    np.testing.assert_allclose(
        result.cos_theta, np.cos(result.theta_rad), atol=1e-6
    )
    # unit-norm
    np.testing.assert_allclose(
        result.sin_theta ** 2 + result.cos_theta ** 2, 1.0, atol=1e-6
    )


def test_blank_image_fails():
    """No pink pixels, no contour — success must be False."""
    blank = np.full((RES, RES, 3), 255, dtype=np.uint8)  # pure white
    result = label_image(blank, preset="REAL", resolution=RES)
    assert result.success is False
    assert np.isnan(result.cx) and np.isnan(result.cy)
    assert np.isnan(result.theta_rad)
    # contour count can be 0 or >0 depending on noise; area must be small
    assert result.contour_area < 100


# -----------------------------------------------------------------------------
# Preset handling
# -----------------------------------------------------------------------------

def test_all_three_presets_construct():
    for p in HSV_PRESETS:
        labeler = CVLabeler(preset=p, resolution=RES)
        assert labeler.preset == p
        assert labeler.hsv_lower.shape == (3,)
        assert labeler.hsv_upper.shape == (3,)


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        CVLabeler(preset="not_a_preset", resolution=RES)  # type: ignore[arg-type]


def test_iws_wm_render_defaults_to_real():
    """Until calibration overrides it, iws_wm_render must not be empty."""
    a_lo, a_hi = HSV_PRESETS["iws_wm_render"]
    b_lo, b_hi = HSV_PRESETS["REAL"]
    np.testing.assert_array_equal(a_lo, b_lo)
    np.testing.assert_array_equal(a_hi, b_hi)


# -----------------------------------------------------------------------------
# Input handling
# -----------------------------------------------------------------------------

def test_resolution_mismatch_raises():
    wrong_size = np.zeros((64, 64, 3), dtype=np.uint8)
    labeler = CVLabeler(preset="REAL", resolution=RES)
    with pytest.raises(ValueError):
        labeler.label(wrong_size)


def test_accepts_torch_tensor():
    rgb_np = _load_dataset_frame()
    rgb_tensor = torch.from_numpy(rgb_np.astype(np.float32) / 255.0)
    result = label_image(rgb_tensor, preset="REAL", resolution=RES)
    # behaviour parity with numpy uint8
    result_np = label_image(rgb_np, preset="REAL", resolution=RES)
    assert result.success == result_np.success
    if result.success:
        np.testing.assert_allclose(result.cx, result_np.cx, atol=1.0)
        np.testing.assert_allclose(result.cy, result_np.cy, atol=1.0)


def test_return_masks_option():
    rgb = _load_dataset_frame()
    labeler = CVLabeler(preset="REAL", resolution=RES)
    out = labeler.label(rgb, return_masks=True)
    assert isinstance(out, tuple) and len(out) == 3
    result, raw_mask, post_mask = out
    assert raw_mask.shape == (RES, RES)
    assert post_mask.shape == (RES, RES)
    assert raw_mask.dtype == np.uint8
    # post-morph area is always ≤ raw-mask area (close+open shaves edges)
    assert int((post_mask > 0).sum()) <= int((raw_mask > 0).sum()) + 1


# -----------------------------------------------------------------------------
# Angle normalisation
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        (0.0, 0.0),
        (179.9, 179.9),
        (180.0, 180.0),
        (180.1, -179.9),
        (360.0, 0.0),
        (-180.0, 180.0),
        (-179.9, -179.9),
        (540.0, 180.0),  # 540 % 360 = 180
        (-540.0, 180.0),
    ],
)
def test_normalize_angle_deg_ranges(raw, expected):
    got = normalize_angle_deg(raw)
    np.testing.assert_allclose(got, expected, atol=1e-6)
    assert -180.0 < got <= 180.0, f"{raw} -> {got} out of (-180, 180]"


def test_normalize_angle_is_idempotent():
    for raw in [-720.3, -180.0, -42.0, 0.0, 42.0, 180.0, 721.7]:
        once = normalize_angle_deg(raw)
        twice = normalize_angle_deg(once)
        np.testing.assert_allclose(once, twice, atol=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
