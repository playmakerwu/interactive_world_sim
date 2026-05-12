"""Verbatim-copied pink-T pose detection from aloha.

Source: /home/yiru-wu/Documents/aloha/aloha/world_model/eval/analyze.py
Reference SHA: dc5b117113a064b7e24b7fd69a174618065c0ff5
Copy date: 2026-05-12

DO NOT modify the algorithmic content of this file. If aloha is updated
upstream and we want the new behavior, re-copy rather than editing in
place. The bitwise equivalence test in scripts/smoke_detector.py guards
against accidental drift.

Top-level definitions have been reordered to put leaves first
(dependency order). The interior of each function/constant is
byte-identical to the source. Only the trailing blank-line padding
between top-level definitions may differ — semantic content is
unchanged.

Symbols copied:
    T_BLOCK_SHAPE              analyze.py:43-52
    T_BLOCK_FILLED_CENTROID    analyze.py:55
    HSV_LOWER_REAL             analyze.py:59
    HSV_UPPER_REAL             analyze.py:60
    HSV_LOWER_WM               analyze.py:63
    HSV_UPPER_WM               analyze.py:64
    sample_contour             analyze.py:84-87
    rotate_points              analyze.py:74-81
    trimmed_icp                analyze.py:103-140
    get_template_contour       analyze.py:90-100
    detect_t_block_mask        analyze.py:143-153
    estimate_current_pose      analyze.py:156-208

Imports trimmed to the detection path only: cv2, numpy, scipy.spatial.cKDTree.
"""

import cv2
import numpy as np
from scipy.spatial import cKDTree


# Verbatim from analyze.py:43-52
# T-block shape for pose estimation (8 points, symmetric T with 90-degree corners)
T_BLOCK_SHAPE = np.array([
    (0, 0),      # bar top-left
    (128, 0),    # bar top-right
    (128, 34),   # bar bottom-right
    (81, 34),    # junction right
    (81, 128),   # stem bottom-right
    (47, 128),   # stem bottom-left
    (47, 34),    # junction left
    (0, 34),     # bar bottom-left
], dtype=np.float32)

# Verbatim from analyze.py:55
# Filled pixel centroid of T_BLOCK_SHAPE
T_BLOCK_FILLED_CENTROID = np.array([63.71, 43.93], dtype=np.float32)

# Verbatim from analyze.py:57-64
# HSV ranges for T-block detection
# REAL: Real T-block RGB (172, 88, 99) -> HSV (176, 125, 172)
HSV_LOWER_REAL = np.array([160, 50, 100])
HSV_UPPER_REAL = np.array([179, 200, 244])

# WM: Pink/magenta T-block in world model renders
HSV_LOWER_WM = np.array([140, 50, 100])
HSV_UPPER_WM = np.array([179, 255, 255])


# Verbatim from analyze.py:84-87
def sample_contour(contour: np.ndarray, num_points: int) -> np.ndarray:
    """Uniformly sample points from contour."""
    indices = np.linspace(0, len(contour) - 1, num_points, dtype=int)
    return contour[indices]


# Verbatim from analyze.py:74-81
def rotate_points(points: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate points around origin by angle_deg degrees."""
    angle_rad = np.radians(angle_deg)
    R = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    return (R @ points.T).T


# Verbatim from analyze.py:103-140
def trimmed_icp(source, target, target_tree, trim_ratio=0.5, max_iterations=100, tolerance=1e-6):
    """Trimmed ICP - only uses the best-matching fraction of points."""
    src = source.copy()
    T_total = np.eye(3)
    n_keep = int(len(src) * trim_ratio)
    prev_error = float('inf')

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


# Verbatim from analyze.py:90-100
def get_template_contour(template_shape: np.ndarray, scale: float) -> np.ndarray:
    """Get template contour centered at origin."""
    template_mask = np.zeros((600, 600), dtype=np.uint8)
    pts = template_shape.copy() * scale + [200, 200]
    cv2.fillPoly(template_mask, [pts.astype(np.int32)], (255,))
    tmpl_cont, _ = cv2.findContours(template_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not tmpl_cont:
        return np.array([])
    template_contour = tmpl_cont[0].reshape(-1, 2).astype(np.float32)
    template_contour = sample_contour(template_contour, 200)
    return template_contour - template_contour.mean(axis=0)


# Verbatim from analyze.py:143-153
def detect_t_block_mask(frame: np.ndarray, hsv_lower: np.ndarray, hsv_upper: np.ndarray) -> np.ndarray:
    """Detect T-block using color thresholding."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

    return cleaned_mask


# Verbatim from analyze.py:156-208
def estimate_current_pose(
    frame: np.ndarray,
    template_contour: np.ndarray,
    scale: float,
    hsv_lower: np.ndarray,
    hsv_upper: np.ndarray,
) -> tuple[np.ndarray | None, float | None, float | None]:
    """Estimate current T-block pose from frame using ICP.

    Returns:
        center, angle, error (or None, None, None if detection fails)
    """
    mask = detect_t_block_mask(frame, hsv_lower, hsv_upper)

    if len(template_contour) == 0:
        return None, None, None

    # Find contours in mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None, None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 100:
        return None, None, None

    mask_contour = largest.reshape(-1, 2).astype(np.float32)
    mask_contour = sample_contour(mask_contour, 200)
    mask_centroid = mask_contour.mean(axis=0)

    mask_tree = cKDTree(mask_contour)

    best_error = float('inf')
    best_final_contour = None
    best_angle = 0

    for init_angle in range(0, 360, 30):
        rotated_tmpl = rotate_points(template_contour, init_angle)
        init_tmpl = rotated_tmpl + mask_centroid

        T, error, final_contour = trimmed_icp(init_tmpl, mask_contour, mask_tree, trim_ratio=0.5)

        if error < best_error:
            best_error = error
            best_final_contour = final_contour
            R = T[:2, :2]
            best_angle = init_angle + np.degrees(np.arctan2(R[1, 0], R[0, 0]))

    if best_final_contour is None:
        return None, None, None

    aligned_center = best_final_contour.mean(axis=0)
    return aligned_center, best_angle, best_error
