"""CV-based T-block pose labeler.

Wraps a classical HSV + contour + ICP pipeline that produces
(cx, cy, theta) for the PushT T-block given a 128x128 RGB image.

The CV routines (T_BLOCK_SHAPE, HSV constants, detect_t_block_mask,
rotate_points, sample_contour, get_template_contour, trimmed_icp,
estimate_current_pose) are vendored from the supervisor's repo at
~/Documents/aloha/aloha/world_model/eval/analyze.py, attribution below.
Vendoring was chosen over importlib-from-filesystem (the Phase 1
sanity-check approach) so the labeling pipeline is self-contained and
reproducible on cloud / CI without the supervisor's repo present.

Source: aloha/world_model/eval/analyze.py
Project: ALOHA Contributors (Yixuan Wang et al.), playmakerwu/aloha
License: MIT (per supervisor repo's pyproject.toml)
"""

from __future__ import annotations

import multiprocessing
from dataclasses import asdict, dataclass
from typing import Literal

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree


# -----------------------------------------------------------------------------
# Vendored constants
# -----------------------------------------------------------------------------

# T-block shape for pose estimation (8 points, symmetric T with 90-degree
# corners). Defined at 512-px canvas scale in the supervisor's repo.
T_BLOCK_SHAPE = np.array(
    [
        (0, 0),      # bar top-left
        (128, 0),    # bar top-right
        (128, 34),   # bar bottom-right
        (81, 34),    # junction right
        (81, 128),   # stem bottom-right
        (47, 128),   # stem bottom-left
        (47, 34),    # junction left
        (0, 34),     # bar bottom-left
    ],
    dtype=np.float32,
)
T_BLOCK_FILLED_CENTROID = np.array([63.71, 43.93], dtype=np.float32)

# HSV ranges for T-block detection.
# REAL: tuned against real RealSense images of a real pink T-block.
# WM:   tuned against supervisor's WM decoder outputs.
# iws_wm_render: initially copies REAL; filled in by HSV calibration (Step 3)
#   if the per-channel ablation shows an IWS-specific preset improves the
#   drop rate / area recovery meaningfully.
HSV_PRESETS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "REAL": (
        np.array([160, 50, 100], dtype=np.uint8),
        np.array([179, 200, 244], dtype=np.uint8),
    ),
    "WM": (
        np.array([140, 50, 100], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8),
    ),
    "iws_wm_render": (
        np.array([160, 50, 100], dtype=np.uint8),
        np.array([179, 200, 244], dtype=np.uint8),
    ),
}


PresetName = Literal["REAL", "WM", "iws_wm_render"]


# -----------------------------------------------------------------------------
# Vendored CV routines
# -----------------------------------------------------------------------------

def _rotate_points(points: np.ndarray, angle_deg: float) -> np.ndarray:
    angle_rad = np.radians(angle_deg)
    R = np.array(
        [[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]]
    )
    return (R @ points.T).T


def _sample_contour(contour: np.ndarray, num_points: int) -> np.ndarray:
    indices = np.linspace(0, len(contour) - 1, num_points, dtype=int)
    return contour[indices]


def _get_template_contour(template_shape: np.ndarray, scale: float) -> np.ndarray:
    """Template contour centred at origin at the requested scale."""
    template_mask = np.zeros((600, 600), dtype=np.uint8)
    pts = template_shape.copy() * scale + [200, 200]
    cv2.fillPoly(template_mask, [pts.astype(np.int32)], (255,))
    tmpl_cont, _ = cv2.findContours(template_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not tmpl_cont:
        return np.array([])
    template_contour = tmpl_cont[0].reshape(-1, 2).astype(np.float32)
    template_contour = _sample_contour(template_contour, 200)
    return template_contour - template_contour.mean(axis=0)


def _trimmed_icp(
    source: np.ndarray,
    target: np.ndarray,
    target_tree: cKDTree,
    trim_ratio: float = 0.5,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, float, np.ndarray]:
    src = source.copy()
    T_total = np.eye(3)
    n_keep = int(len(src) * trim_ratio)
    prev_error = float("inf")

    for _ in range(max_iterations):
        min_distances, indices = target_tree.query(src)
        keep_idx = np.argsort(min_distances)[:n_keep]
        src_trimmed = src[keep_idx]
        matched_target = target[indices[keep_idx]]

        src_c = src_trimmed.mean(axis=0)
        tgt_c = matched_target.mean(axis=0)

        H = (src_trimmed - src_c).T @ (matched_target - tgt_c)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = tgt_c - R @ src_c
        src = (R @ src.T).T + t

        T = np.eye(3)
        T[:2, :2] = R
        T[:2, 2] = t
        T_total = T @ T_total

        error = min_distances[keep_idx].mean()
        if abs(prev_error - error) < tolerance:
            break
        prev_error = error

    return T_total, error, src


def _detect_t_block_mask(
    frame_bgr: np.ndarray, hsv_lower: np.ndarray, hsv_upper: np.ndarray
) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    return cleaned


def _estimate_current_pose(
    frame_bgr: np.ndarray,
    template_contour: np.ndarray,
    hsv_lower: np.ndarray,
    hsv_upper: np.ndarray,
    *,
    raw_mask_out: list | None = None,
    post_morph_out: list | None = None,
) -> tuple[np.ndarray | None, float | None, float | None, int, float]:
    """Returns (center_xy, angle_deg, icp_residual, contour_count, largest_area).

    If `raw_mask_out` / `post_morph_out` are provided as single-element
    lists, the respective masks are returned via those slots. This keeps
    the public API stable while letting the calibration script harvest
    both raw-mask and post-morph pixel counts.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    raw_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    if raw_mask_out is not None:
        raw_mask_out.append(raw_mask)

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    if post_morph_out is not None:
        post_morph_out.append(cleaned)

    if len(template_contour) == 0:
        return None, None, None, 0, 0.0

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_count = len(contours)
    if not contours:
        return None, None, None, 0, 0.0

    largest = max(contours, key=cv2.contourArea)
    largest_area = float(cv2.contourArea(largest))
    if largest_area < 100:
        return None, None, None, contour_count, largest_area

    mask_contour = largest.reshape(-1, 2).astype(np.float32)
    mask_contour = _sample_contour(mask_contour, 200)
    mask_centroid = mask_contour.mean(axis=0)
    mask_tree = cKDTree(mask_contour)

    best_error = float("inf")
    best_final_contour: np.ndarray | None = None
    best_angle = 0.0

    for init_angle in range(0, 360, 30):
        rotated_tmpl = _rotate_points(template_contour, init_angle)
        init_tmpl = rotated_tmpl + mask_centroid
        T, error, final_contour = _trimmed_icp(
            init_tmpl, mask_contour, mask_tree, trim_ratio=0.5
        )
        if error < best_error:
            best_error = error
            best_final_contour = final_contour
            R = T[:2, :2]
            best_angle = init_angle + np.degrees(np.arctan2(R[1, 0], R[0, 0]))

    if best_final_contour is None:
        return None, None, None, contour_count, largest_area

    aligned_center = best_final_contour.mean(axis=0)
    return aligned_center, best_angle, best_error, contour_count, largest_area


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def normalize_angle_deg(theta_deg: float) -> float:
    """Normalise a raw ICP angle output into (-180, 180]."""
    wrapped = (theta_deg + 180.0) % 360.0 - 180.0
    # `(-180 % 360) - 180 == -180`, but we want the half-open (-180, 180].
    return wrapped if wrapped > -180.0 else 180.0


def _rgb_to_bgr_u8(rgb: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert any supported RGB input to BGR uint8 (the CV pipeline's input)."""
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3) RGB, got shape {arr.shape}")
    if arr.dtype == np.uint8:
        rgb_u8 = arr
    elif np.issubdtype(arr.dtype, np.floating):
        rgb_u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    else:
        rgb_u8 = arr.astype(np.uint8)
    return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)


@dataclass
class CVLabelResult:
    success: bool
    cx: float
    cy: float
    theta_rad: float
    sin_theta: float
    cos_theta: float
    contour_count: int
    contour_area: float
    icp_residual: float
    # exposed for debugging / calibration diagnostics
    theta_deg_raw: float  # the raw estimator output (pre-normalisation)
    theta_deg: float      # the normalised angle ((-180, 180])
    preset: str

    def as_dict(self) -> dict:
        return asdict(self)


FAIL_RESULT_TEMPLATE = CVLabelResult(
    success=False,
    cx=float("nan"),
    cy=float("nan"),
    theta_rad=float("nan"),
    sin_theta=float("nan"),
    cos_theta=float("nan"),
    contour_count=0,
    contour_area=0.0,
    icp_residual=float("nan"),
    theta_deg_raw=float("nan"),
    theta_deg=float("nan"),
    preset="",
)


# ── Worker-pool plumbing for label_batch ─────────────────────────────
#
# The CV pipeline is CPU-bound (HSV mask → contour → trimmed-ICP per
# 30-degree start angle), so threading does not help past the GIL.
# Multiprocessing does, but we MUST use the spawn context: the parent
# process holds CUDA state (the WM lives on GPU), and forking with live
# CUDA contexts corrupts both parent and child. The two functions below
# are module-level so they're picklable through spawn; the labeler is
# reconstructed inside each worker rather than pickled, which avoids
# pickling cv2 / kdtree state and makes worker init order-independent.

_WORKER_LABELER: "CVLabeler | None" = None


def _worker_init(preset: "PresetName", resolution: int) -> None:
    global _WORKER_LABELER
    _WORKER_LABELER = CVLabeler(preset=preset, resolution=resolution)


def _worker_label(image: np.ndarray) -> dict:
    assert _WORKER_LABELER is not None, "_worker_init was not called"
    return _WORKER_LABELER.label(image).as_dict()


class CVLabeler:
    """Reusable CV labeler bound to a preset + canvas resolution.

    The template contour depends on the resolution (scale = res / 512), so
    we pre-compute it at construction time to avoid repeating the work
    for every frame in a bulk label run.
    """

    def __init__(self, preset: PresetName = "REAL", resolution: int = 128):
        if preset not in HSV_PRESETS:
            raise ValueError(
                f"unknown preset '{preset}'; known: {list(HSV_PRESETS.keys())}"
            )
        self.preset = preset
        self.resolution = resolution
        self.hsv_lower, self.hsv_upper = HSV_PRESETS[preset]
        self._t_scale = resolution / 512.0
        self._template_contour = _get_template_contour(T_BLOCK_SHAPE, self._t_scale)

    def label(
        self,
        rgb: np.ndarray | torch.Tensor,
        *,
        return_masks: bool = False,
    ) -> CVLabelResult | tuple[CVLabelResult, np.ndarray, np.ndarray]:
        """Run the CV pipeline on one RGB image.

        Args:
            rgb: (H, W, 3) image in RGB, uint8 or float32 [0,1] or torch.Tensor.
                 Must match `self.resolution` on H and W.
            return_masks: if True, also return (raw_mask, post_morph_mask).

        Returns a CVLabelResult. If `return_masks`, returns a 3-tuple.
        """
        bgr = _rgb_to_bgr_u8(rgb)
        if bgr.shape[0] != self.resolution or bgr.shape[1] != self.resolution:
            raise ValueError(
                f"image shape {bgr.shape[:2]} does not match "
                f"labeler resolution {self.resolution}"
            )

        raw_out: list = []
        post_out: list = []
        center, angle_deg_raw, residual, n_contours, area = _estimate_current_pose(
            bgr,
            self._template_contour,
            self.hsv_lower,
            self.hsv_upper,
            raw_mask_out=raw_out,
            post_morph_out=post_out,
        )

        raw_mask = raw_out[0]
        post_morph = post_out[0]

        if center is None:
            result = CVLabelResult(
                success=False,
                cx=float("nan"),
                cy=float("nan"),
                theta_rad=float("nan"),
                sin_theta=float("nan"),
                cos_theta=float("nan"),
                contour_count=n_contours,
                contour_area=area,
                icp_residual=float("nan") if residual is None else float(residual),
                theta_deg_raw=float("nan"),
                theta_deg=float("nan"),
                preset=self.preset,
            )
        else:
            theta_deg = normalize_angle_deg(float(angle_deg_raw))
            theta_rad = np.deg2rad(theta_deg)
            result = CVLabelResult(
                success=True,
                cx=float(center[0]),
                cy=float(center[1]),
                theta_rad=float(theta_rad),
                sin_theta=float(np.sin(theta_rad)),
                cos_theta=float(np.cos(theta_rad)),
                contour_count=n_contours,
                contour_area=float(area),
                icp_residual=float(residual),
                theta_deg_raw=float(angle_deg_raw),
                theta_deg=float(theta_deg),
                preset=self.preset,
            )

        if return_masks:
            return result, raw_mask, post_morph
        return result

    def label_batch(
        self,
        images: list[np.ndarray],
        n_workers: int = 0,
    ) -> list[dict]:
        """Label a batch of RGB images, returning ordered ``CVLabelResult.as_dict()``.

        Args:
            images: list of (H, W, 3) RGB arrays matching ``self.resolution``.
            n_workers: 0 → sequential loop in this process (no pool overhead;
                this is the default, drop-in equivalent to the per-image
                ``label`` loop callers used to write).
                ≥1 → spawn ``n_workers`` processes via the ``spawn`` context.

        Output order is guaranteed to match input order (we use ``pool.map``,
        not ``imap_unordered``). Per-image semantics are bit-exact with
        ``self.label(img).as_dict()`` — no preset, ICP-tolerance, or
        morphology-kernel knob is touched in this method.

        TODO(perf): pool reuse across plan_step calls. Right now we
        recreate the pool every call, which costs one spawn-per-worker
        per call (~tens of ms per worker on Linux). Acceptable while we
        validate correctness; revisit if the profiler shows pool-init
        as the dominant CV cost at small N.
        """
        if n_workers <= 0:
            return [self.label(img).as_dict() for img in images]
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(
            processes=int(n_workers),
            initializer=_worker_init,
            initargs=(self.preset, self.resolution),
        ) as pool:
            return pool.map(_worker_label, images)


def label_image(
    rgb: np.ndarray | torch.Tensor,
    preset: PresetName = "REAL",
    resolution: int = 128,
) -> CVLabelResult:
    """One-shot labeling convenience wrapper — prefer `CVLabeler` for bulk."""
    labeler = CVLabeler(preset=preset, resolution=resolution)
    return labeler.label(rgb)
