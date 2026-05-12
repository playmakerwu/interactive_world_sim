"""7-test smoke suite for interactive_world_sim_cv.

Tests:
  1.  Bitwise equivalence: real frame, mode='real'.
      Compares our detect() output against running aloha's
      estimate_current_pose directly via spec_from_file_location.
      The ONLY place in this package that touches the aloha repo.
      Skipped (loud message, not a failure) if aloha is not available.

  1b. Bitwise equivalence: decoded frame, mode='wm'. Same idea on a
      decoded frame from a fresh imagination rollout.

  2.  Detect 10 decoded frames without exception; print error magnitudes.

  3.  Batched vs sequential equivalence on 20 decoded frames, num_workers=4.

  4.  Wrong-mode HSV: real frame with mode='wm'. Confirms the two HSV
      ranges do real work — wrong-mode result should be either None or
      visibly different from the correct-mode result.

  5.  Empty-frame failure: all-zeros image returns None.

  6.  Speed benchmark on 100 decoded frames: sequential vs one-shot
      pool vs persistent pool. Flag if persistent/(a) < 1.5×.

Run from the repo root:

    python interactive_world_sim_cv/scripts/smoke_detector.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402

from interactive_world_sim_cv import (  # noqa: E402
    DetectorPool,
    TPose,
    detect,
    detect_batch,
)
from interactive_world_sim_cv import _detection as _our_detection  # noqa: E402


EPISODE = str(_REPO_ROOT / "data" / "mini" / "pusht" / "val" / "episode_0.hdf5")
N_DECODED = 100
PROCESSING_RES = 512


# --------------------------------------------------------------- printing


_RESULTS: dict[str, str] = {}


def _banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _pass(name: str, detail: str = "") -> None:
    msg = f"PASS  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    _RESULTS[name] = "PASS"


def _fail(name: str, detail: str = "") -> None:
    msg = f"FAIL  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    _RESULTS[name] = "FAIL"


def _skip(name: str, reason: str) -> None:
    print(f"SKIP  {name}  ({reason})")
    _RESULTS[name] = f"SKIP ({reason})"


# --------------------------------------------------------------- setup


def _load_real_frame(t: int = 0) -> np.ndarray:
    """Load camera_1_color[t] from the episode HDF5, preprocess to 128² RGB.

    Mirrors WorldModelEnv's preprocessing: center-crop to 480×480, resize
    to 128×128 with INTER_AREA. Returns uint8 HWC RGB.
    """
    with h5py.File(EPISODE, "r") as f:
        raw = np.asarray(f["obs/images/camera_1_color"][t])  # (480, 640, 3) RGB uint8
    h, w = raw.shape[:2]
    lo = (w - h) // 2
    cropped = raw[:, lo : lo + h]
    return cv2.resize(cropped, (128, 128), interpolation=cv2.INTER_AREA)


def _generate_decoded_frames(n: int) -> np.ndarray:
    """Warmup 10 frames + n imagination steps; return (n, 128, 128, 3) uint8 RGB."""
    from interactive_world_sim_env import WorldModelEnv
    from interactive_world_sim_env.helpers.expert_action import (
        expert_action_from_episode,
    )

    env = WorldModelEnv("pusht_cam1")
    env.reset(
        init_episode_path=EPISODE,
        init_episode_index=9,
        init_window_size=10,
    )
    frames = np.empty((n, 128, 128, 3), dtype=np.uint8)
    for offset in range(n):
        t = 10 + offset
        action_t = expert_action_from_episode(env, EPISODE, t - 1)
        env.step(action_t)
        frames[offset] = env.render()
    env.close()
    # Clear GPU state before tests that may fork workers.
    try:
        import gc

        import torch  # type: ignore

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return frames


def _load_aloha_reference():
    """Load aloha's analyze.py via spec_from_file_location.

    Returns the module, or None if the aloha repo is not available.
    THIS IS THE ONLY PLACE IN THE PACKAGE THAT TOUCHES THE ALOHA REPO.
    Production code (api.py, _pool.py, _detection.py) does NOT.
    """
    aloha_root = os.environ.get("ALOHA_REPO_ROOT", "/home/yiru-wu/Documents/aloha")
    analyze_path = Path(aloha_root) / "aloha" / "world_model" / "eval" / "analyze.py"
    if not analyze_path.exists():
        return None, str(analyze_path)
    spec = importlib.util.spec_from_file_location("_aloha_reference", str(analyze_path))
    if spec is None or spec.loader is None:
        return None, str(analyze_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_aloha_reference"] = mod
    spec.loader.exec_module(mod)
    return mod, str(analyze_path)


# --------------------------------------------------------------- tests


def _bitwise_equivalence(name: str, rgb: np.ndarray, mode: str, ref_mod) -> None:
    if ref_mod is None:
        _skip(name, "aloha reference not found at $ALOHA_REPO_ROOT or default path")
        return

    hsv_lower, hsv_upper = (
        (ref_mod.HSV_LOWER_WM, ref_mod.HSV_UPPER_WM)
        if mode == "wm"
        else (ref_mod.HSV_LOWER_REAL, ref_mod.HSV_UPPER_REAL)
    )

    # Aloha reference path (the four steps, applied verbatim from analyze.py
    # process_episode_wm at lines 386-393):
    ref_resized = cv2.resize(
        rgb, (PROCESSING_RES, PROCESSING_RES), interpolation=cv2.INTER_CUBIC
    )
    ref_bgr = cv2.cvtColor(ref_resized, cv2.COLOR_RGB2BGR)
    ref_t_scale = PROCESSING_RES / 512.0
    ref_tmpl = ref_mod.get_template_contour(ref_mod.T_BLOCK_SHAPE, ref_t_scale)
    ref_center, ref_angle, ref_error = ref_mod.estimate_current_pose(
        ref_bgr, ref_tmpl, ref_t_scale, hsv_lower, hsv_upper
    )

    # Our path:
    ours = detect(rgb, mode=mode, processing_resolution=PROCESSING_RES)

    # Both None?
    if ref_center is None and ours is None:
        _pass(name, "both detectors returned None")
        return
    if ref_center is None or ours is None:
        _fail(
            name,
            f"divergence: ours={ours!r}  ref_center={ref_center!r}",
        )
        return

    print(f"  ours.x       = {ours.x:.6f}    ref_center[0] = {float(ref_center[0]):.6f}")
    print(f"  ours.y       = {ours.y:.6f}    ref_center[1] = {float(ref_center[1]):.6f}")
    print(f"  ours.angle_deg = {ours.angle_deg:.6f}    ref_angle     = {float(ref_angle):.6f}")
    print(f"  ours.error   = {ours.error:.6f}    ref_error     = {float(ref_error):.6f}")

    if not np.array_equal(np.array([ours.x, ours.y]), ref_center):
        _fail(name, "(x, y) not bitwise equal")
        return
    if ours.angle_deg != float(ref_angle):
        _fail(name, f"angle_deg not bitwise equal: {ours.angle_deg} vs {float(ref_angle)}")
        return
    if ours.error != float(ref_error):
        _fail(name, f"error not bitwise equal: {ours.error} vs {float(ref_error)}")
        return

    _pass(name, "(x, y, angle_deg, error) bitwise equal")


def test_2_no_exceptions(decoded_rgbs: np.ndarray) -> None:
    name = "test 2: detect 10 decoded frames without exception"
    _banner(name)
    errors = []
    for i in range(10):
        r = detect(decoded_rgbs[i], mode="wm", processing_resolution=PROCESSING_RES)
        if r is None:
            print(f"  frame {i}: None")
        else:
            print(
                f"  frame {i}: x={r.x:6.2f} y={r.y:6.2f} "
                f"angle={r.angle_deg:7.2f}° error={r.error:.4f}"
            )
            errors.append(r.error)
    if not errors:
        _fail(name, "all 10 frames returned None")
        return
    print(
        f"  errors min/median/max = {min(errors):.4f} / "
        f"{float(np.median(errors)):.4f} / {max(errors):.4f}"
    )
    _pass(name, f"{len(errors)}/10 detected without exception")


def test_3_batched_equivalence(decoded_rgbs: np.ndarray) -> None:
    name = "test 3: batched vs sequential equivalence (N=20, num_workers=4)"
    _banner(name)
    rgbs = decoded_rgbs[:20]
    seq = [detect(rgbs[i], mode="wm", processing_resolution=PROCESSING_RES) for i in range(20)]
    par = detect_batch(rgbs, mode="wm", processing_resolution=PROCESSING_RES, num_workers=4)
    if len(seq) != len(par):
        _fail(name, f"length mismatch: seq={len(seq)} par={len(par)}")
        return
    mismatches = []
    for i, (a, b) in enumerate(zip(seq, par)):
        if a != b:
            mismatches.append((i, a, b))
    if mismatches:
        print(f"  mismatches: {len(mismatches)}")
        for i, a, b in mismatches[:3]:
            print(f"    [{i}] seq={a!r}  par={b!r}")
        _fail(name, f"{len(mismatches)}/20 frames differed")
        return
    print(f"  all 20 elements equal (TPose == TPose, None == None)")
    _pass(name, "batched and sequential produced identical TPose lists")


def _make_synthetic_magenta_t(size: int = 128) -> np.ndarray:
    """RGB magenta T on black background. After RGB→BGR→HSV, hue ≈ 150,
    which is INSIDE WM [140, 179] but OUTSIDE REAL [160, 179]. Saturation
    is also 255, outside REAL [50, 200]. So mode='real' should reject it
    and mode='wm' should detect it — a definitive HSV-range diagnostic.
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # Scale T_BLOCK_SHAPE (128px bbox) down to fit and center it.
    scale = (size - 16) / 128.0  # leave a margin
    pts = _our_detection.T_BLOCK_SHAPE * scale
    # Center on the image
    pts -= pts.mean(axis=0)
    pts += np.array([size / 2, size / 2])
    cv2.fillPoly(img, [pts.astype(np.int32)], (255, 0, 255))  # RGB magenta
    return img


def test_4_wrong_mode_hsv(real_rgb: np.ndarray, decoded_rgb: np.ndarray) -> None:
    name = "test 4: wrong-mode HSV (3 probes)"
    _banner(name)

    # Probe 1: real_rgb with mode='wm' vs mode='real'.
    # Note: HSV_LOWER_WM=[140,50,100], HSV_UPPER_WM=[179,255,255] is a
    # set-theoretic SUPERSET of HSV_LOWER_REAL=[160,50,100],
    # HSV_UPPER_REAL=[179,200,244]. On a real frame whose T-hue lies in
    # the REAL range (and thus also in WM), both modes detect the same
    # blob and produce identical TPose. Math, not bug.
    real_real = detect(real_rgb, mode="real", processing_resolution=PROCESSING_RES)
    real_wm = detect(real_rgb, mode="wm", processing_resolution=PROCESSING_RES)
    real_diff = (real_real != real_wm)
    print(f"  probe 1 — real frame, mode='real': {real_real}")
    print(f"  probe 1 — real frame, mode='wm':   {real_wm}")
    print(f"  probe 1 differ? {real_diff}")

    # Probe 2: decoded_rgb (WM render) with mode='real' vs mode='wm'.
    # If the decoded T's hue lies in [140, 160) — inside WM, outside
    # REAL — REAL mode would fail or differ. In practice this codebase's
    # decoder renders the T in the [160, 179] range with moderate S/V,
    # so both ranges accept it. Observed identically to probe 1.
    dec_wm = detect(decoded_rgb, mode="wm", processing_resolution=PROCESSING_RES)
    dec_real = detect(decoded_rgb, mode="real", processing_resolution=PROCESSING_RES)
    dec_diff = (dec_wm != dec_real)
    print(f"  probe 2 — decoded frame, mode='wm':   {dec_wm}")
    print(f"  probe 2 — decoded frame, mode='real': {dec_real}")
    print(f"  probe 2 differ? {dec_diff}")

    # Probe 3: synthetic magenta T (RGB 255,0,255 → HSV 150,255,255).
    # H=150 is OUTSIDE REAL [160,179]; S=255 is OUTSIDE REAL [50,200].
    # Both individually rule out REAL. WM should detect; REAL should
    # return None. This definitively proves the HSV ranges differ.
    synth = _make_synthetic_magenta_t(size=128)
    synth_wm = detect(synth, mode="wm", processing_resolution=PROCESSING_RES)
    synth_real = detect(synth, mode="real", processing_resolution=PROCESSING_RES)
    synth_diff = (synth_wm != synth_real)
    print(f"  probe 3 — synthetic magenta T, mode='wm':   {synth_wm}")
    print(f"  probe 3 — synthetic magenta T, mode='real': {synth_real}")
    print(f"  probe 3 differ? {synth_diff}")

    # The HSV ranges do real work iff they produce different results on
    # at least one input. Probe 3 (synthetic) is constructed to expose
    # the difference unambiguously.
    if real_diff or dec_diff or synth_diff:
        which = []
        if real_diff:
            which.append("probe 1")
        if dec_diff:
            which.append("probe 2")
        if synth_diff:
            which.append("probe 3")
        _pass(
            name,
            f"HSV ranges differ on: {', '.join(which)}. "
            "Note: on natural frames in this codebase, ranges happen to coincide; "
            "synthetic probe confirms they are not redundant in general.",
        )
        return
    _fail(
        name,
        "HSV ranges produced identical results on all three probes — "
        "including the synthetic magenta — which would mean the mode "
        "parameter is not reaching the detection code.",
    )


def test_5_empty_frame() -> None:
    name = "test 5: empty-frame failure"
    _banner(name)
    empty = np.zeros((128, 128, 3), dtype=np.uint8)
    for mode in ("wm", "real"):
        r = detect(empty, mode=mode, processing_resolution=PROCESSING_RES)
        print(f"  mode={mode!r}: result = {r}")
        if r is not None:
            _fail(name, f"mode={mode!r} should have returned None")
            return
    _pass(name, "both modes returned None on zeros image")


def test_6_speed(decoded_rgbs: np.ndarray) -> None:
    name = "test 6: speed (N=100, mode='wm')"
    _banner(name)
    rgbs = decoded_rgbs[:100]

    t0 = time.perf_counter()
    seq = [detect(rgbs[i], mode="wm", processing_resolution=PROCESSING_RES) for i in range(100)]
    t_seq = time.perf_counter() - t0
    print(f"  (a) sequential          : {t_seq*1000:7.1f} ms")

    t0 = time.perf_counter()
    one_shot = detect_batch(rgbs, mode="wm", processing_resolution=PROCESSING_RES, num_workers=8)
    t_one = time.perf_counter() - t0
    print(
        f"  (b) one-shot pool n=8   : {t_one*1000:7.1f} ms  ratio (a)/(b) = {t_seq/t_one:.2f}×"
    )

    with DetectorPool(num_workers=8) as pool:
        # Warm-up call (don't time)
        _ = pool.detect_batch(rgbs[:8], mode="wm", processing_resolution=PROCESSING_RES)
        t0 = time.perf_counter()
        persistent = pool.detect_batch(rgbs, mode="wm", processing_resolution=PROCESSING_RES)
        t_persist = time.perf_counter() - t0
    ratio = t_seq / t_persist if t_persist > 0 else float("inf")
    print(
        f"  (c) persistent pool n=8 : {t_persist*1000:7.1f} ms  ratio (a)/(c) = {ratio:.2f}×"
    )

    # Equivalence between (a), (b), (c) at index 0 sanity:
    if seq != one_shot or seq != persistent:
        _fail(name, "results differ between modes — non-deterministic detection?")
        return

    if ratio < 1.5:
        print(
            f"  WARNING: persistent/(a) = {ratio:.2f}× < 1.5×.\n"
            f"  Pool overhead is dominating; per-frame work may be too small "
            f"to amortize IPC, or num_workers is wrong for this machine."
        )
        _pass(name, f"benchmark ran, but speedup {ratio:.2f}× below 1.5× target")
    else:
        _pass(name, f"persistent speedup {ratio:.2f}×")


# --------------------------------------------------------------- main


def main() -> int:
    print("=" * 72)
    print("interactive_world_sim_cv — 7-test smoke suite")
    print("=" * 72)
    print(f"  episode    : {EPISODE}")
    print(f"  N decoded  : {N_DECODED}")
    print(f"  processing : {PROCESSING_RES}×{PROCESSING_RES}")

    # Setup: load real frame and generate decoded frames.
    print("\n[setup] loading real frame...")
    real_rgb = _load_real_frame(t=0)
    print(f"  real_rgb: shape={real_rgb.shape} dtype={real_rgb.dtype}")

    print(f"[setup] generating {N_DECODED} decoded frames "
          f"(10-frame warmup + {N_DECODED} imagination steps)...")
    decoded_rgbs = _generate_decoded_frames(N_DECODED)
    print(f"  decoded_rgbs: shape={decoded_rgbs.shape} dtype={decoded_rgbs.dtype}")

    # Load aloha reference (for tests 1 and 1b only).
    print("\n[setup] loading aloha reference for bitwise tests...")
    ref_mod, ref_path = _load_aloha_reference()
    if ref_mod is None:
        print(f"  aloha not found at {ref_path}; tests 1 and 1b will be SKIPPED.")
    else:
        print(f"  aloha loaded from {ref_path}")

    # ------ Tests ------
    _banner("test 1: bitwise equivalence on real frame (mode='real')")
    _bitwise_equivalence(
        "test 1: bitwise on real frame (mode='real')",
        real_rgb, "real", ref_mod,
    )

    _banner("test 1b: bitwise equivalence on decoded frame (mode='wm')")
    _bitwise_equivalence(
        "test 1b: bitwise on decoded frame (mode='wm')",
        decoded_rgbs[0], "wm", ref_mod,
    )

    test_2_no_exceptions(decoded_rgbs)
    test_3_batched_equivalence(decoded_rgbs)
    test_4_wrong_mode_hsv(real_rgb, decoded_rgbs[0])
    test_5_empty_frame()
    test_6_speed(decoded_rgbs)

    # ------ Summary ------
    _banner("SUMMARY")
    n_pass = sum(1 for v in _RESULTS.values() if v == "PASS")
    n_fail = sum(1 for v in _RESULTS.values() if v == "FAIL")
    n_skip = sum(1 for v in _RESULTS.values() if v.startswith("SKIP"))
    for name, status in _RESULTS.items():
        print(f"  {status:<28s}  {name}")
    print()
    print(f"  TOTAL: {len(_RESULTS)}   PASS: {n_pass}   FAIL: {n_fail}   SKIP: {n_skip}")
    if n_skip:
        print(
            "\n  NOTE: at least one test was SKIPPED — typically the bitwise\n"
            "  equivalence tests, which require the aloha repo at\n"
            "  $ALOHA_REPO_ROOT (or default /home/yiru-wu/Documents/aloha)."
        )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
