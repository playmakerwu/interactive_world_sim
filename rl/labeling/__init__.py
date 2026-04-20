"""Offline CV labeling for the state probe.

Vendors the supervisor's classical pose estimator (HSV + contour + ICP)
so the labeling pipeline does not depend on the supervisor's repo being
present on the filesystem at runtime.
"""

from rl.labeling.cv_labeler import (
    HSV_PRESETS,
    CVLabelResult,
    CVLabeler,
    label_image,
    normalize_angle_deg,
)

__all__ = [
    "CVLabeler",
    "CVLabelResult",
    "HSV_PRESETS",
    "label_image",
    "normalize_angle_deg",
]
