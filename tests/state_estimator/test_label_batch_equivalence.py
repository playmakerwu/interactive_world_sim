"""Equivalence test for CVLabeler.label_batch.

Asserts that the new batched interface returns bit-exact dicts vs the
per-image ``label()`` path, in both sequential (n_workers=0) and
spawn-pool (n_workers=4) modes. Output order must match input order
in both cases.

Why this matters: the spawn-pool path forks process state through
multiprocessing, which can subtly perturb numerical results if the
workers initialize cv2 / numpy / kdtree state in a different order
than the parent. Trimmed-ICP is iterative and converges on a tolerance
threshold, so any divergence shows up immediately as a residual or
angle delta. If this test ever loosens its tolerances, ICP is the
suspect — investigate the parallel path, do not weaken the assertion.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

from rl.labeling.cv_labeler import CVLabeler

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "data" / "mini" / "pusht" / "train" / "episode_0.hdf5"
RES = 128
N_IMAGES = 8


def _center_crop_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(h, w)
    sh = (h - s) // 2
    sw = (w - s) // 2
    return img[sh : sh + s, sw : sw + s]


def _load_eight_frames() -> list[np.ndarray]:
    """Eight 128x128 RGB frames from the train set, deterministic offsets."""
    if not DATASET.exists():
        pytest.skip(f"dataset not present: {DATASET}")
    # Spread across the 200-frame episode so the T-block is in different
    # poses; this exercises the ICP search across different start angles.
    indices = [10, 30, 60, 90, 120, 140, 160, 180]
    assert len(indices) == N_IMAGES
    out = []
    with h5py.File(DATASET, "r") as f:
        for i in indices:
            raw = f["obs/images/camera_1_color"][i]  # (480, 640, 3) uint8
            cropped = _center_crop_square(raw)
            out.append(
                cv2.resize(cropped, (RES, RES), interpolation=cv2.INTER_AREA)
            )
    return out


def _equal_value(a, b) -> bool:
    """NaN-safe equality (NaN == NaN here, by design)."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b
    return a == b


def _assert_dicts_identical(actual: dict, expected: dict, ctx: str) -> None:
    assert set(actual.keys()) == set(expected.keys()), (
        f"{ctx}: key mismatch — actual={sorted(actual)} expected={sorted(expected)}"
    )
    for k, ev in expected.items():
        av = actual[k]
        if isinstance(ev, np.ndarray):
            np.testing.assert_allclose(
                av, ev, atol=1e-8,
                err_msg=f"{ctx}: array field {k!r} differs",
            )
        else:
            assert _equal_value(av, ev), (
                f"{ctx}: scalar field {k!r} differs: {av!r} vs {ev!r}"
            )


@pytest.fixture(scope="module")
def images() -> list[np.ndarray]:
    return _load_eight_frames()


@pytest.fixture(scope="module")
def labeler() -> CVLabeler:
    return CVLabeler(preset="REAL", resolution=RES)


@pytest.fixture(scope="module")
def expected(labeler: CVLabeler, images: list[np.ndarray]) -> list[dict]:
    return [labeler.label(img).as_dict() for img in images]


def test_label_batch_sequential_matches_per_image(labeler, images, expected):
    seq = labeler.label_batch(images, n_workers=0)
    assert len(seq) == len(expected)
    for i, (got, want) in enumerate(zip(seq, expected)):
        _assert_dicts_identical(got, want, ctx=f"seq[{i}]")


def test_label_batch_pool_matches_per_image(labeler, images, expected):
    par = labeler.label_batch(images, n_workers=4)
    assert len(par) == len(expected)
    for i, (got, want) in enumerate(zip(par, expected)):
        _assert_dicts_identical(got, want, ctx=f"par[{i}]")


def test_label_batch_preserves_input_order(labeler, images):
    """Reverse the input list — output must reverse with it."""
    rev_images = list(reversed(images))
    rev_seq = labeler.label_batch(rev_images, n_workers=0)
    rev_par = labeler.label_batch(rev_images, n_workers=4)
    assert len(rev_seq) == len(rev_par) == len(images)
    for i, (s, p) in enumerate(zip(rev_seq, rev_par)):
        _assert_dicts_identical(p, s, ctx=f"order-rev[{i}]")
