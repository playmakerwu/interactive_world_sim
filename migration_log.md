# Migration Log: Supervisor's CV pose estimator → IWS state-probe pipeline

Source: `~/Documents/aloha/aloha/world_model/eval/analyze.py`
Target: `rl/labeling/cv_labeler.py` (Branch A — `state-probe-training`)

Each entry below records a deviation from the supervisor's defaults, the
observed symptom that motivated the change, and the downstream impact.
Format described in §7.1 of the task brief.

---

## 2026-04-19 — Phase 1 sanity check setup
- **What in the supervisor repo**: `analyze.py` exposes `HSV_LOWER_REAL/UPPER_REAL`
  (lines 60–61, intended for real RealSense images at 480px) and
  `HSV_LOWER_WM/UPPER_WM` (lines 64–65, intended for their WM render distribution).
- **Changed to**: nothing yet — sanity check runs both presets unmodified against
  our `decode(z_goal) → 128×128 RGB` and reports which (if either) cleanly isolates
  the T-block. Calibration of a third `iws_wm_render` preset is a deliberate
  Phase 2/3 decision point.
- **Why**: Section 3.1 of the task brief flags HSV mismatch as the dominant risk
  for label quality. We do not assume either preset transfers; we measure first.
- **Evidence**: `tests/state_estimator/sanity_outputs/` after running
  `tests/state_estimator/sanity_check.py` (mask images per preset + JSON metrics).
- **Outcome**: see 2026-04-20 entry below.

## 2026-04-20 — Phase 1 sanity-check observations on z_goal
- **Setup**: `decode(z_goal)` → 128×128 RGB, supervisor's `estimate_current_pose`
  with both presets unmodified. WM ckpt: `outputs/pusht_cam1/checkpoints/best.ckpt`.
- **Both presets succeeded with identical numbers**: mask_pixels=476, largest
  contour area=415 px (one connected component), center=(55.51, 62.70) px,
  angle=+1.53°, ICP residual=0.289. WM range is a strict superset of REAL
  (H∈[140,179] vs [160,179], S/V upper higher), and our T-block pixels happen
  to fall entirely in the REAL subset.
- **Mask quality**: clean — single contiguous T-shape blob, no fragmentation,
  no spurious detections from the orange gripper tips (they fall outside the
  H≥140 cutoff). Visually the masked region matches the pink T-block in the
  decoded image. Marker + orientation arrow in `02_annotated_real.png` and
  `02_annotated_wm.png` land on the T-block.
- **Coverage concern (deferred to Phase 2)**: T-block area at 128×128 with
  scale=128/512 should be ≈471 px (from `T_BLOCK_SHAPE` total area). We get
  415 px — ~88% coverage. Likely conservative HSV cut on the V channel is
  shaving edge pixels. ICP residual 0.289 is low so this doesn't break the
  pose estimate at the goal, but on out-of-distribution decoder frames a
  mildly tighter mask could push the contour below the 100-px reject threshold.
- **Verdict**: REAL preset is sufficient for the goal frame. No threshold
  changes yet. Re-evaluate during HSV calibration on ~20 in-distribution
  decoded frames in Phase 2.

## 2026-04-20 — GPU contention resolved at run time
- **State at Phase 1 start**: 1.7 GiB free of 12 GiB; concurrent CoinRun
  training pid 54706 holding 10 GiB. Sanity check held off per §8.1.
- **State at run time**: 5.6 GiB free. WM load consumed ~560 MiB; decode of
  one (1, 4, 32, 32) latent consumed an additional ~330 MiB peak. Total WM
  resident footprint ≈ 890 MiB. Decode wall time 0.74 s (after 4.3 s ckpt load).
- **Implication for Phase 2 batch sizing**: with WM at ~900 MiB and the
  user's CoinRun training holding the bulk of VRAM, our budget for probe
  training peak allocation under contention is ~3 GiB. Stick to the §8.2
  starting batch of 16; measure before scaling.

## 2026-04-19 — Loading supervisor module without ROS
- **What in the supervisor repo**: `aloha/__init__.py` line 8 imports
  `RealAlohaEnv`, which transitively requires ROS. A naive `import aloha.world_model.eval.analyze`
  would fail in our `iws` conda env.
- **Changed to**: load `analyze.py` via `importlib.util.spec_from_file_location`,
  bypassing the package `__init__.py` entirely (see
  `tests/state_estimator/sanity_check.py::_load_supervisor_module`).
- **Why**: `analyze.py` itself only depends on `cv2/numpy/scipy/shapely/matplotlib`,
  all available in the `iws` env. Avoids polluting the env with ROS deps.
- **Evidence**: sanity check imports the module successfully and reads `T_BLOCK_SHAPE`
  / `HSV_LOWER_*` / `estimate_current_pose` from it.
- **Outcome**: Works for Phase 1. Phase 3-A will copy the relevant CV functions
  into `rl/labeling/cv_labeler.py` with attribution rather than keep the
  filesystem-path-coupled importlib hack in production code.
