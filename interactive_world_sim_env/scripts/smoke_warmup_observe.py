"""Smoke tests for reset(init_window_size=W) and observe().

Runs nine concrete checks against pusht_cam1 and prints PASS / FAIL with
the actual outcome. Failures are NOT silently swallowed — each test sets
its own ``ok`` flag and the script exits 1 if any failed.
"""

from __future__ import annotations

import dataclasses
import sys
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import h5py
import numpy as np
import torch

from interactive_world_sim_env import EnvState, WorldModelEnv
from interactive_world_sim_env.helpers.expert_action import (
    expert_action_from_episode,
)

EPISODE = str(_REPO_ROOT / "data" / "mini" / "pusht" / "val" / "episode_0.hdf5")
CAM_KEY = "camera_1_color"


def _expected_action_dim() -> int:
    return 4  # pusht_cam1


def _read_rgb(t: int) -> np.ndarray:
    with h5py.File(EPISODE, "r") as f:
        return np.asarray(f[f"obs/images/{CAM_KEY}"][t])


def _check(name: str, fn) -> bool:
    print(f"\n--- {name} ---")
    try:
        ok, detail = fn()
    except Exception as exc:
        print(f"  EXCEPTION: {exc!r}")
        traceback.print_exc(limit=2)
        print("  FAIL")
        return False
    status = "PASS" if ok else "FAIL"
    if detail:
        print(f"  {detail}")
    print(f"  {status}")
    return ok


# ----------------------------------------------------------------- tests


def test_1_warmup_shapes() -> tuple[bool, str]:
    env = WorldModelEnv("pusht_cam1")
    try:
        env.reset(
            init_episode_path=EPISODE,
            init_episode_index=9,
            init_window_size=10,
        )
        snap = env.snapshot()
        lw, aw = snap.latent_window.shape, snap.action_window.shape
        ok = lw == (10, 4, 32, 32) and aw == (10, 4) and snap.step_counter == 0
        return ok, f"latent_window={lw} action_window={aw} step={snap.step_counter}"
    finally:
        env.close()


def test_2_warmup_then_step() -> tuple[bool, str]:
    env = WorldModelEnv("pusht_cam1")
    try:
        env.reset(
            init_episode_path=EPISODE,
            init_episode_index=9,
            init_window_size=10,
        )
        obs, info = env.step(np.zeros(4, dtype=np.float32))
        ok = (
            obs.latent_history.shape == (10, 4, 32, 32)
            and info["step"] == 1
            and info["source"] == "dynamics"
        )
        return ok, (
            f"latent_history={tuple(obs.latent_history.shape)} "
            f"info={ {'step': info['step'], 'source': info['source']} }"
        )
    finally:
        env.close()


def test_3_cold_vs_warmup_shape_consistency() -> tuple[bool, str]:
    env_cold = WorldModelEnv("pusht_cam1")
    env_warm = WorldModelEnv("pusht_cam1")
    try:
        env_cold.reset(init_episode_path=EPISODE, init_episode_index=0)
        for _ in range(10):
            env_cold.step(np.zeros(4, dtype=np.float32))
        env_warm.reset(
            init_episode_path=EPISODE,
            init_episode_index=9,
            init_window_size=10,
        )
        env_warm.step(np.zeros(4, dtype=np.float32))
        s_cold, s_warm = env_cold.snapshot(), env_warm.snapshot()
        ok = (
            s_cold.latent_window.shape == s_warm.latent_window.shape
            and s_cold.latent_window.shape == (10, 4, 32, 32)
            and s_cold.step_counter == 10
            and s_warm.step_counter == 1
        )
        return ok, (
            f"cold step_counter={s_cold.step_counter} "
            f"warm step_counter={s_warm.step_counter} "
            f"both latent_window={s_cold.latent_window.shape}"
        )
    finally:
        env_cold.close()
        env_warm.close()


def test_4_reset_step_step_observe_step() -> tuple[bool, str]:
    env = WorldModelEnv("pusht_cam1")
    try:
        env.reset(init_episode_path=EPISODE, init_episode_index=0)
        env.step(np.zeros(4, dtype=np.float32))
        env.step(np.zeros(4, dtype=np.float32))
        rgb = _read_rgb(3)
        obs, info_obs = env.observe(rgb, last_action=np.zeros(4, dtype=np.float32))
        env.step(np.zeros(4, dtype=np.float32))
        # observed latent norm should be in the encoder's natural range
        # (~sqrt(num_views * H_lat * W_lat) per design = 32 for pusht_cam1).
        norm_ok = 20.0 <= info_obs["latent_norm"] <= 64.0
        ok = (
            info_obs["step"] == 3
            and info_obs["source"] == "encoder"
            and env.snapshot().step_counter == 4
            and norm_ok
        )
        return ok, (
            f"observe info step={info_obs['step']} latent_norm={info_obs['latent_norm']:.2f} "
            f"source={info_obs['source']} (norm_ok={norm_ok})"
        )
    finally:
        env.close()


def test_5_cold_observe_vs_warmup_latents_equal() -> tuple[bool, str]:
    """Latent-only equivalence — encoder is deterministic, so bitwise equal."""
    env_obs = WorldModelEnv("pusht_cam1")
    env_warm = WorldModelEnv("pusht_cam1")
    try:
        env_obs.reset(init_episode_path=EPISODE, init_episode_index=0)
        for t in range(1, 10):
            rgb_t = _read_rgb(t)
            a = expert_action_from_episode(env_obs, EPISODE, t - 1)
            env_obs.observe(rgb_t, last_action=a)
        env_warm.reset(
            init_episode_path=EPISODE,
            init_episode_index=9,
            init_window_size=10,
        )
        lw_obs = env_obs.snapshot().latent_window
        lw_warm = env_warm.snapshot().latent_window
        ok = bool(torch.equal(lw_obs, lw_warm))
        return ok, f"torch.equal(latent_windows)={ok}  shape={tuple(lw_obs.shape)}"
    finally:
        env_obs.close()
        env_warm.close()


def test_6a_warmup_non_pusht_raises() -> tuple[bool, str]:
    env = WorldModelEnv("pusht_cam1")
    try:
        original_task = env._task
        env._task = "bimanual_sweep_cam0"  # local monkey-patch; no other ckpt available
        try:
            env.reset(
                init_episode_path=EPISODE,
                init_episode_index=9,
                init_window_size=10,
            )
            return False, "did not raise"
        except NotImplementedError as e:
            return True, f"raised NotImplementedError: {e}"
        finally:
            env._task = original_task
    finally:
        env.close()


def test_6b_warmup_with_init_rgb_raises() -> tuple[bool, str]:
    env = WorldModelEnv("pusht_cam1")
    try:
        try:
            env.reset(
                init_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
                init_window_size=10,
                init_episode_path=EPISODE,
                init_episode_index=9,
            )
        except ValueError as e:
            return True, f"raised ValueError: {e}"
        return False, "did not raise"
    finally:
        env.close()


def test_6c_warmup_with_init_state_raises() -> tuple[bool, str]:
    env = WorldModelEnv("pusht_cam1")
    try:
        env.reset(init_episode_path=EPISODE, init_episode_index=0)
        snap = env.snapshot()
        try:
            env.reset(
                init_state=snap,
                init_window_size=10,
                init_episode_path=EPISODE,
                init_episode_index=9,
            )
        except ValueError as e:
            return True, f"raised ValueError: {e}"
        return False, "did not raise"
    finally:
        env.close()


def test_6d_warmup_without_episode_path_raises() -> tuple[bool, str]:
    env = WorldModelEnv("pusht_cam1")
    try:
        try:
            env.reset(init_window_size=10, init_episode_index=9)
        except ValueError as e:
            return True, f"raised ValueError: {e}"
        return False, "did not raise"
    finally:
        env.close()


def test_6e_warmup_index_too_small_raises() -> tuple[bool, str]:
    env = WorldModelEnv("pusht_cam1")
    try:
        try:
            env.reset(
                init_episode_path=EPISODE,
                init_episode_index=3,
                init_window_size=10,
            )
        except ValueError as e:
            return True, f"raised ValueError: {e}"
        return False, "did not raise"
    finally:
        env.close()


def test_6f_zero_window_raises() -> tuple[bool, str]:
    env = WorldModelEnv("pusht_cam1")
    try:
        try:
            env.reset(init_episode_path=EPISODE, init_window_size=0)
        except ValueError as e:
            return True, f"raised ValueError: {e}"
        return False, "did not raise"
    finally:
        env.close()


def test_6g_window_over_hist_context_raises() -> tuple[bool, str]:
    env = WorldModelEnv("pusht_cam1")
    try:
        try:
            env.reset(
                init_episode_path=EPISODE,
                init_episode_index=11,
                init_window_size=11,  # hist_context default = 10
            )
        except ValueError as e:
            return True, f"raised ValueError: {e}"
        return False, "did not raise"
    finally:
        env.close()


def test_6h_observe_missing_action_raises() -> tuple[bool, str]:
    env = WorldModelEnv("pusht_cam1")
    try:
        env.reset(init_episode_path=EPISODE, init_episode_index=0)
        try:
            env.observe(_read_rgb(1), last_action=None)  # type: ignore[arg-type]
        except TypeError as e:
            return True, f"raised TypeError: {e}"
        return False, "did not raise"
    finally:
        env.close()


def test_6i_observe_before_reset_raises() -> tuple[bool, str]:
    env = WorldModelEnv("pusht_cam1")
    try:
        try:
            env.observe(_read_rgb(0), last_action=np.zeros(4, dtype=np.float32))
        except RuntimeError as e:
            return True, f"raised RuntimeError: {e}"
        return False, "did not raise"
    finally:
        env.close()


def test_7_snapshot_restore_across_observe() -> tuple[bool, str]:
    env = WorldModelEnv("pusht_cam1")
    try:
        env.reset(init_episode_path=EPISODE, init_episode_index=0)
        env.step(np.zeros(4, dtype=np.float32))
        env.observe(_read_rgb(2), last_action=np.zeros(4, dtype=np.float32))
        snap = env.snapshot()
        env.step(np.zeros(4, dtype=np.float32))
        s_after_step = env.snapshot()
        env.restore(snap)
        env.step(np.zeros(4, dtype=np.float32))
        s_after_restore = env.snapshot()
        # After restore+step, step_counter must equal that of the original
        # "snap step" trajectory; shapes match.
        ok = (
            s_after_step.step_counter == s_after_restore.step_counter
            and s_after_step.latent_window.shape == s_after_restore.latent_window.shape
        )
        return ok, (
            f"step_counter after_step={s_after_step.step_counter} "
            f"after_restore+step={s_after_restore.step_counter} "
            f"shape={tuple(s_after_step.latent_window.shape)}"
        )
    finally:
        env.close()


def test_8_cross_task_restore_after_observe_raises() -> tuple[bool, str]:
    env = WorldModelEnv("pusht_cam1")
    try:
        env.reset(init_episode_path=EPISODE, init_episode_index=0)
        env.observe(_read_rgb(1), last_action=np.zeros(4, dtype=np.float32))
        snap = env.snapshot()
        bad = dataclasses.replace(snap, task="bimanual_sweep_cam0")
        try:
            env.restore(bad)
        except ValueError as e:
            return True, f"raised ValueError: {e}"
        return False, "did not raise"
    finally:
        env.close()


def test_9_warmup_full_equivalence() -> tuple[bool, str]:
    """Both latent_window and action_window must be bitwise-equal."""
    env_warm = WorldModelEnv("pusht_cam1")
    env_obs = WorldModelEnv("pusht_cam1")
    try:
        env_warm.reset(
            init_episode_path=EPISODE,
            init_episode_index=9,
            init_window_size=10,
        )

        env_obs.reset(init_episode_path=EPISODE, init_episode_index=0)
        for t in range(1, 10):
            rgb_t = _read_rgb(t)
            a = expert_action_from_episode(env_obs, EPISODE, t - 1)
            env_obs.observe(rgb_t, last_action=a)

        s1, s2 = env_warm.snapshot(), env_obs.snapshot()
        lat_eq = bool(torch.equal(s1.latent_window, s2.latent_window))
        act_eq = bool(torch.equal(s1.action_window, s2.action_window))
        ok = lat_eq and act_eq
        return ok, (
            f"torch.equal(latent_window)={lat_eq}  "
            f"torch.equal(action_window)={act_eq}  "
            f"warm shape={tuple(s1.latent_window.shape)}"
        )
    finally:
        env_warm.close()
        env_obs.close()


# ----------------------------------------------------------------- driver

TESTS = [
    ("test_1: warmup_shapes", test_1_warmup_shapes),
    ("test_2: warmup_then_step", test_2_warmup_then_step),
    ("test_3: cold_vs_warmup_shape_consistency", test_3_cold_vs_warmup_shape_consistency),
    ("test_4: reset_step_step_observe_step", test_4_reset_step_step_observe_step),
    ("test_5: cold_observe_vs_warmup_latents_equal", test_5_cold_observe_vs_warmup_latents_equal),
    ("test_6a: warmup_non_pusht_raises", test_6a_warmup_non_pusht_raises),
    ("test_6b: warmup_with_init_rgb_raises", test_6b_warmup_with_init_rgb_raises),
    ("test_6c: warmup_with_init_state_raises", test_6c_warmup_with_init_state_raises),
    ("test_6d: warmup_without_episode_path_raises", test_6d_warmup_without_episode_path_raises),
    ("test_6e: warmup_index_too_small_raises", test_6e_warmup_index_too_small_raises),
    ("test_6f: zero_window_raises", test_6f_zero_window_raises),
    ("test_6g: window_over_hist_context_raises", test_6g_window_over_hist_context_raises),
    ("test_6h: observe_missing_action_raises", test_6h_observe_missing_action_raises),
    ("test_6i: observe_before_reset_raises", test_6i_observe_before_reset_raises),
    ("test_7: snapshot_restore_across_observe", test_7_snapshot_restore_across_observe),
    ("test_8: cross_task_restore_after_observe_raises", test_8_cross_task_restore_after_observe_raises),
    ("test_9: warmup_full_equivalence", test_9_warmup_full_equivalence),
]


def main() -> int:
    results: list[tuple[str, bool]] = []
    for name, fn in TESTS:
        ok = _check(name, fn)
        results.append((name, ok))
    print("\n========== summary ==========")
    n_pass = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{n_pass} / {len(results)} tests passed")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
