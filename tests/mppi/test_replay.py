"""Smoke test for scripts/wm_interactive_replay.py::run_replay.

Uses 3 random actions starting from a synthetic random unit-norm latent.
Asserts the per-step files are produced and have the right shape/keys.
Does not assert correctness of CV or WM outputs themselves — those have
their own tests.

CUDA + WM checkpoint required.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

CKPT_PATH = Path("outputs/pusht_cam1/checkpoints/best.ckpt")
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _have_gpu_and_ckpt() -> bool:
    return torch.cuda.is_available() and CKPT_PATH.exists()


@pytest.fixture(scope="module")
def replay_dir(tmp_path_factory):
    if not _have_gpu_and_ckpt():
        pytest.skip("CUDA + WM checkpoint required for replay smoke test")

    # Import the module via spec since scripts/ isn't a package.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "wm_interactive_replay",
        REPO_ROOT / "scripts" / "wm_interactive_replay.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from rl.labeling.cv_labeler import CVLabeler
    from rl.models.world_model import DifferentiableDynamics

    device = "cuda:0"
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    wm = DifferentiableDynamics(str(CKPT_PATH), device=device)
    labeler = CVLabeler(preset="REAL", resolution=128)

    # Use the saved z_goal as a stable, in-distribution starting latent —
    # a fully-random latent would have norm != 32 and may behave weirdly.
    z0 = torch.load(
        REPO_ROOT / "tests" / "goal_selection" / "z_goal.pt",
        weights_only=False, map_location=device,
    ).float()
    actions = torch.randn(3, 4, device=device) * 0.05  # 3 small random actions

    out = tmp_path_factory.mktemp("wm_replay_smoke")
    summary = mod.run_replay(z0, actions, wm, labeler, out, fps=2)
    yield out, summary


def test_replay_writes_per_step_files(replay_dir):
    out, _summary = replay_dir
    # 3 actions -> steps 0..3
    for t in range(4):
        png = out / f"step_{t:02d}.png"
        meta = out / f"step_{t:02d}_meta.json"
        assert png.exists(), f"missing {png}"
        assert meta.exists(), f"missing {meta}"
    # Aggregate artefacts
    for fname in ("replay.mp4", "action_norms.png", "latent_drift.png", "summary.json"):
        assert (out / fname).exists(), f"missing {fname}"


def test_replay_meta_schema(replay_dir):
    out, _summary = replay_dir
    required_keys = {
        "t", "action", "action_norm", "cv_state", "cv_success",
        "contour_area", "icp_residual", "latent_norm",
        "latent_cosine_sim_to_z0", "latent_cosine_sim_to_prev",
    }
    for t in range(4):
        meta = json.loads((out / f"step_{t:02d}_meta.json").read_text())
        assert set(meta.keys()) == required_keys, (
            f"step {t} keys mismatch: {set(meta.keys())} vs {required_keys}"
        )
        # cosine sims to z0: t=0 must be 1.0; later may drift.
        if t == 0:
            assert meta["latent_cosine_sim_to_z0"] == pytest.approx(1.0, abs=1e-5)
            assert meta["latent_cosine_sim_to_prev"] is None
            assert meta["action"] is None  # no action led to z_0
        else:
            assert isinstance(meta["latent_cosine_sim_to_prev"], float)
            assert isinstance(meta["action"], list) and len(meta["action"]) == 4


def test_replay_png_shape(replay_dir):
    import cv2
    out, _summary = replay_dir
    img = cv2.imread(str(out / "step_00.png"))
    # Upscaled by UPSCALE=2 -> 256x256x3
    assert img is not None
    assert img.shape == (256, 256, 3), img.shape


def test_replay_summary_schema(replay_dir):
    _out, summary = replay_dir
    for k in (
        "n_steps", "first_cv_fail_step", "n_cv_fails",
        "first_drift_below_0_95", "first_drift_below_0_90",
        "min_cos_z0", "final_cos_z0",
        "min_latent_norm", "max_latent_norm",
        "min_action_norm", "max_action_norm", "mean_action_norm",
    ):
        assert k in summary, f"summary missing key: {k}"
    assert summary["n_steps"] == 3
