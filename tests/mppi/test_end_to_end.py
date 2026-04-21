"""Step 5 smoke test: minimal MPPI run end to end.

Uses a tiny config (N=8, H=3, 5 control steps) — should complete under
a minute on CUDA. Asserts non-NaN outputs and that the summary.json has
the expected keys. Does NOT assert convergence; that's for the full run.

CUDA + WM ckpt are required. There is no CPU fallback for the full WM.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

CKPT_PATH = Path("outputs/pusht_cam1/checkpoints/best.ckpt")
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _have_gpu_and_ckpt() -> bool:
    return torch.cuda.is_available() and CKPT_PATH.exists()


@pytest.fixture(scope="module")
def smoke_run_dir(tmp_path_factory):
    if not _have_gpu_and_ckpt():
        pytest.skip("CUDA + WM checkpoint required")
    run_name = f"smoke_test_{tmp_path_factory.mktemp('mppi').name}"

    cmd = [
        sys.executable, "scripts/run_mppi.py",
        "--run_name", run_name,
        "--initial_state", "mini/val/0/0",
        "--N", "8",
        "--H", "3",
        "--sigma", "0.1",
        "--control_steps", "5",
        "--seed", "0",
    ]
    result = subprocess.run(
        cmd, cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    out_dir = REPO_ROOT / "outputs" / "mppi" / run_name
    yield out_dir

    # cleanup: remove the tmp run outputs after the test module exits.
    # Only delete if under outputs/mppi/ (belt-and-suspenders safety).
    if out_dir.exists() and out_dir.is_relative_to(REPO_ROOT / "outputs" / "mppi"):
        import shutil
        shutil.rmtree(out_dir)


def test_e2e_produces_expected_files(smoke_run_dir):
    expected = [
        "trajectory.mp4",
        "trajectory_overlay.mp4",
        "trajectory_latents.pt",
        "action_history.pt",
        "reward_curve.png",
        "summary.json",
        "rollout_samples_step_00.png",  # SNAPSHOT_STEPS[0] falls within 5 steps
    ]
    for fname in expected:
        assert (smoke_run_dir / fname).exists(), f"missing: {fname}"


def test_e2e_summary_schema(smoke_run_dir):
    with open(smoke_run_dir / "summary.json") as f:
        summary = json.load(f)
    for key in (
        "run_name", "config", "goal_state", "initial_state_measured",
        "final_state_measured", "final_pos_distance_px", "final_angle_error_deg",
        "final_angle_sim", "success_strict", "success_cos",
        "flipped_convergence", "reward_trajectory_stats", "per_step",
        "wall_time_s", "wall_per_step_s",
    ):
        assert key in summary, f"summary.json missing key: {key}"
    assert len(summary["per_step"]) == 6  # 5 control + initial
    assert all(row["reward"] is not None for row in summary["per_step"])


def test_e2e_trajectory_tensors_not_nan(smoke_run_dir):
    latents = torch.load(smoke_run_dir / "trajectory_latents.pt", map_location="cpu")
    actions = torch.load(smoke_run_dir / "action_history.pt", map_location="cpu")
    assert latents.shape[0] == 6       # initial + 5 steps
    assert actions.shape == (5, 4)     # 5 steps, action_dim=4
    assert torch.all(torch.isfinite(latents))
    assert torch.all(torch.isfinite(actions))
