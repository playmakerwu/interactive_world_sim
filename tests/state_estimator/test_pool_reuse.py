"""Tests for CVLabeler persistent worker-pool reuse (cv-pool-reuse branch).

These tests assert that the lazily-spawned pool is reused across successive
``label_batch(n_workers > 0)`` calls, recreated when ``n_workers`` changes,
torn down cleanly by ``close()``, and that pool-reused calls remain bit-exact
identical to the per-image ``label()`` path. The most important assertion is
``test_repeated_calls_bit_exact`` — pool reuse breaking determinism is the
primary correctness risk.

Each test constructs its own CVLabeler and calls ``close()`` in teardown so
worker processes do not accumulate across tests (especially important under
larger ``n_workers`` values).
"""

from __future__ import annotations

import math
import time
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
    if not DATASET.exists():
        pytest.skip(f"dataset not present: {DATASET}")
    indices = [10, 30, 60, 90, 120, 140, 160, 180]
    out: list[np.ndarray] = []
    with h5py.File(DATASET, "r") as f:
        for i in indices:
            raw = f["obs/images/camera_1_color"][i]
            cropped = _center_crop_square(raw)
            out.append(cv2.resize(cropped, (RES, RES), interpolation=cv2.INTER_AREA))
    return out


def _equal_value(a, b) -> bool:
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
def expected_results(images: list[np.ndarray]) -> list[dict]:
    """Compute the per-image label() reference once for the module."""
    ref = CVLabeler(preset="REAL", resolution=RES)
    try:
        return [ref.label(img).as_dict() for img in images]
    finally:
        ref.close()


@pytest.fixture
def labeler():
    """Per-test labeler with explicit close() on teardown so pool workers
    do not accumulate across tests."""
    lab = CVLabeler(preset="REAL", resolution=RES)
    try:
        yield lab
    finally:
        lab.close()


def test_pool_persists_across_calls(labeler, images):
    """Two label_batch(n_workers=4) calls reuse the same pool object (id check)
    AND the second call is dramatically faster (timing assertion). If the
    second call is comparable to the first, the pool isn't actually reused
    even if ids match."""
    assert labeler._persistent_pool is None
    assert labeler._pool_n_workers is None

    t0 = time.perf_counter()
    out1 = labeler.label_batch(images, n_workers=4)
    t1 = time.perf_counter()
    first_pool = labeler._persistent_pool
    first_call_wall = t1 - t0
    assert first_pool is not None
    assert labeler._pool_n_workers == 4
    assert len(out1) == len(images)

    t2 = time.perf_counter()
    out2 = labeler.label_batch(images, n_workers=4)
    t3 = time.perf_counter()
    second_pool = labeler._persistent_pool
    second_call_wall = t3 - t2

    # Pool object identity preserved across calls.
    assert second_pool is first_pool, "pool object changed between successive calls"
    assert labeler._pool_n_workers == 4

    # Second call must be < 30% of first-call wall (spawn cost amortized).
    assert second_call_wall < first_call_wall * 0.3, (
        f"second call ({second_call_wall*1000:.1f} ms) not amortized vs "
        f"first call ({first_call_wall*1000:.1f} ms) — pool likely not reused"
    )


def test_pool_cleanup(labeler, images):
    """After close(), _persistent_pool is None and worker child PIDs are gone."""
    labeler.label_batch(images, n_workers=4)
    assert labeler._persistent_pool is not None
    pool = labeler._persistent_pool
    worker_pids = [p.pid for p in pool._pool]  # type: ignore[attr-defined]
    assert all(pid is not None for pid in worker_pids)

    labeler.close()
    assert labeler._persistent_pool is None
    assert labeler._pool_n_workers is None

    # Calling close() a second time must be a no-op (idempotent).
    labeler.close()
    assert labeler._persistent_pool is None

    # Verify workers are no longer alive. psutil if available, else os.kill probe.
    try:
        import psutil

        for pid in worker_pids:
            assert pid is not None
            # After pool.join, the worker should be reaped.
            assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE  # noqa: E501
            if psutil.pid_exists(pid):
                # Zombie tolerated only as a transient state — give it a beat
                # then re-check.
                time.sleep(0.2)
                assert not psutil.pid_exists(pid), (
                    f"worker pid {pid} still alive after close()+join()"
                )
    except ImportError:
        # psutil missing — fall back to os.kill 0 (signal probe)
        import os
        for pid in worker_pids:
            assert pid is not None
            try:
                os.kill(pid, 0)
                # Reachable means worker is still alive (or zombie).
                # Sleep briefly and try again — pool.join should have reaped.
                time.sleep(0.2)
                try:
                    os.kill(pid, 0)
                    pytest.fail(f"worker pid {pid} still reachable after close()+join()")
                except ProcessLookupError:
                    pass  # reaped after the brief sleep — acceptable
            except ProcessLookupError:
                pass  # already reaped — best case


def test_pool_recreates_on_worker_count_change(labeler, images, capsys):
    """When n_workers changes, the old pool is torn down and a new one spawned.
    The pool object identity differs across the change."""
    labeler.label_batch(images, n_workers=4)
    pool_v4 = labeler._persistent_pool
    pids_v4 = [p.pid for p in pool_v4._pool]  # type: ignore[attr-defined]
    assert labeler._pool_n_workers == 4

    labeler.label_batch(images, n_workers=8)
    pool_v8 = labeler._persistent_pool
    assert labeler._pool_n_workers == 8

    assert pool_v8 is not pool_v4, "pool not recreated when n_workers changed"

    # Stderr should carry the visibility warning (A2).
    captured = capsys.readouterr()
    assert "pool recreating" in captured.err, (
        f"expected 'pool recreating' in stderr; got: {captured.err!r}"
    )
    assert "4 -> 8" in captured.err

    # v4 workers should be dead.
    time.sleep(0.2)  # let the join settle
    try:
        import psutil
        for pid in pids_v4:
            assert pid is not None
            assert not psutil.pid_exists(pid), (
                f"old (n=4) worker pid {pid} still alive after pool recreation"
            )
    except ImportError:
        import os
        for pid in pids_v4:
            assert pid is not None
            try:
                os.kill(pid, 0)
                pytest.fail(f"old (n=4) worker pid {pid} still reachable after recreate")
            except ProcessLookupError:
                pass


def test_equivalence_with_persistent_pool(labeler, images, expected_results):
    """label_batch(n_workers=4) result must be bit-exact with per-image label()
    even after the pool has been used multiple times (pool-reuse must not
    perturb numerical output)."""
    # Warm: do a first call, then test a second call against the per-image ref.
    labeler.label_batch(images, n_workers=4)
    out = labeler.label_batch(images, n_workers=4)
    assert len(out) == len(expected_results)
    for i, (got, want) in enumerate(zip(out, expected_results)):
        _assert_dicts_identical(got, want, ctx=f"persistent-pool[{i}]")


def test_repeated_calls_bit_exact(labeler, images):
    """5 successive label_batch(n_workers=4) calls produce IDENTICAL outputs
    each call. Pool reuse must not introduce per-call drift (e.g. worker
    accumulating state, RNG drift, kdtree cache divergence)."""
    outs: list[list[dict]] = [
        labeler.label_batch(images, n_workers=4) for _ in range(5)
    ]
    assert len(outs) == 5
    ref = outs[0]
    for k, out in enumerate(outs[1:], start=1):
        assert len(out) == len(ref)
        for i, (got, want) in enumerate(zip(out, ref)):
            _assert_dicts_identical(got, want, ctx=f"call[{k}].frame[{i}]")
