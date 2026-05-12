"""interactive_world_sim_cv — pink-T pose detector.

Public API:
    TPose          — frozen dataclass holding (x, y, sin, cos, angle_deg, error)
    detect         — single-frame detection
    detect_batch   — batched detection, sequential or one-shot pool
    DetectorPool   — persistent process pool for repeated batched detection

The core detection algorithm lives in _detection.py, which is a verbatim
copy from aloha's analyze.py. See _detection.py's header for the pinned
aloha SHA. Production code does not import from the aloha repo at runtime.
"""

from .api import DetectorPool, TPose, detect, detect_batch

__all__ = ["DetectorPool", "TPose", "detect", "detect_batch"]
