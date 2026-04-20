"""HSV calibration for the CV T-block labeler.

Samples 20 diverse frames across the 5 training episodes, decodes each
through the frozen WM (Option A per design doc §1.4 — match the exact
distribution the probe will see at RL time), runs the CV pipeline
under several preset and ablation variants, and produces the quantitative
evidence required by design doc §1.3 to pick a labeling preset.

Variants evaluated:
- Two baselines: REAL and WM presets (from the supervisor's repo).
- Three single-channel relaxations on top of REAL: H widened +-5 deg,
  S lower -> 0, V upper -> 255. These isolate which channel (if any)
  is responsible for the Phase 1 88% area-coverage shortfall.

Outputs (tests/state_estimator/calibration_outputs/):
- calibration_table.csv   : one row per (frame, variant) with raw_mask_px,
                             post_morph_px, contour_area, icp_residual,
                             cv_success, theta_deg, cx, cy.
- agreement_table.csv     : per-frame REAL-vs-WM comparison: n_pixels_wm_only,
                             pose_delta_px, pose_delta_deg, agreement_flag.
- per_channel_relaxation.csv : variant summary stats (mean/median area
                             recovery vs REAL baseline).
- calibration_grid_{variant}.png : 4x5 grid of decoded-RGB + annotated
                             overlays under each variant.
- calibration_summary.md  : written summary with the three mandatory
                             numbers (drop rate, area recovery,
                             agreement rate) and a proposed preset.

Run from repo root in the iws env:
    conda run -n iws python scripts/calibrate_hsv.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.labeling.cv_labeler import (  # noqa: E402
    HSV_PRESETS,
    CVLabeler,
    _detect_t_block_mask,
    _estimate_current_pose,
    _get_template_contour,
    T_BLOCK_SHAPE,
    normalize_angle_deg,
)
from rl.models.world_model import DifferentiableDynamics  # noqa: E402
from rl.visualization.state_viz import render_state_on_image  # noqa: E402

CKPT_PATH = REPO_ROOT / "outputs" / "pusht_cam1" / "checkpoints" / "best.ckpt"
TRAIN_DIR = REPO_ROOT / "data" / "mini" / "pusht" / "train"
OUT_DIR = REPO_ROOT / "tests" / "state_estimator" / "calibration_outputs"

RESOLUTION = 128
OBS_KEY = "camera_1_color"
DEVICE = "cuda:0"
SEED = 42
UPSCALE = 4  # overlay legibility

# Theoretical T-block post-morph area at the 128 canvas scale =
# T_BLOCK_SHAPE polygon area x (128/512)^2 ~ 471 px.
TSCALE = RESOLUTION / 512.0
THEORETICAL_AREA = float(
    cv2.contourArea(T_BLOCK_SHAPE.astype(np.float32)) * (TSCALE ** 2)
)

# Per-channel relaxations on top of REAL.
REAL_LO, REAL_HI = HSV_PRESETS["REAL"]
WM_LO, WM_HI = HSV_PRESETS["WM"]
VARIANTS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "REAL":           (REAL_LO, REAL_HI),
    "WM":             (WM_LO, WM_HI),
    "REAL+H_wide5":   (
        np.array([max(0, REAL_LO[0] - 5), REAL_LO[1], REAL_LO[2]], dtype=np.uint8),
        np.array([min(179, REAL_HI[0] + 5), REAL_HI[1], REAL_HI[2]], dtype=np.uint8),
    ),
    "REAL+S_lo0":     (
        np.array([REAL_LO[0], 0, REAL_LO[2]], dtype=np.uint8),
        REAL_HI.copy(),
    ),
    "REAL+V_hi255":   (
        REAL_LO.copy(),
        np.array([REAL_HI[0], REAL_HI[1], 255], dtype=np.uint8),
    ),
}


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _center_crop_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(h, w)
    return img[(h - s) // 2 : (h - s) // 2 + s, (w - s) // 2 : (w - s) // 2 + s]


def _preprocess_for_encode(raw_rgb: np.ndarray) -> torch.Tensor:
    cropped = _center_crop_square(raw_rgb)
    resized = cv2.resize(cropped, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)


def _pick_frames(episodes: list[Path], n: int = 20) -> list[tuple[int, int]]:
    """Stratified sample: n / len(episodes) frames per episode, evenly
    spaced within each episode. Returns (episode_id, t_idx) pairs."""
    per_ep = n // len(episodes)
    assert per_ep * len(episodes) == n, f"{n} frames not divisible by {len(episodes)} episodes"
    picks: list[tuple[int, int]] = []
    rng = np.random.default_rng(SEED)
    for ep_id, ep_path in enumerate(episodes):
        with h5py.File(ep_path, "r") as f:
            n_frames = f["action"].shape[0]
        # evenly spaced positions with a small rng jitter so repeated
        # calibration runs can sample diverse frames if desired.
        base = np.linspace(0.05, 0.95, per_ep)
        jitter = rng.uniform(-0.02, 0.02, size=per_ep)
        fracs = np.clip(base + jitter, 0.0, 0.99)
        for frac in fracs:
            picks.append((ep_id, int(frac * (n_frames - 1))))
    return picks


def _decode_all(wm: DifferentiableDynamics, picks: list[tuple[int, int]]) -> list[np.ndarray]:
    """Encode+decode all picks; return the decoded uint8 RGB per frame."""
    decoded = []
    for ep_id, t_idx in picks:
        ep_path = TRAIN_DIR / f"episode_{ep_id}.hdf5"
        with h5py.File(ep_path, "r") as f:
            raw = f[f"obs/images/{OBS_KEY}"][t_idx]
        img_pre = _preprocess_for_encode(raw).to(DEVICE)
        with torch.no_grad():
            z = wm.encode(img_pre)
            rgb = wm.decode(z, RESOLUTION)
        rgb_np = rgb[0].permute(1, 2, 0).detach().cpu().float().numpy()
        decoded.append(np.clip(rgb_np * 255.0, 0, 255).astype(np.uint8))
    return decoded


def _run_variant(
    rgb_u8: np.ndarray, hsv_lo: np.ndarray, hsv_hi: np.ndarray, template_contour: np.ndarray
) -> dict:
    """Run one CV pass and return dict of per-frame metrics."""
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    raw_out: list = []
    post_out: list = []
    center, angle_raw, residual, n_contours, area = _estimate_current_pose(
        bgr, template_contour, hsv_lo, hsv_hi,
        raw_mask_out=raw_out, post_morph_out=post_out,
    )
    raw_mask = raw_out[0]
    post_morph = post_out[0]

    row = {
        "raw_mask_px": int((raw_mask > 0).sum()),
        "post_morph_px": int((post_morph > 0).sum()),
        "contour_count": n_contours,
        "contour_area": float(area),
        "cv_success": center is not None,
    }
    if center is not None:
        theta_deg = normalize_angle_deg(float(angle_raw))
        row.update({
            "cx": float(center[0]),
            "cy": float(center[1]),
            "theta_deg": theta_deg,
            "theta_deg_raw": float(angle_raw),
            "icp_residual": float(residual),
        })
    else:
        row.update({
            "cx": float("nan"), "cy": float("nan"),
            "theta_deg": float("nan"), "theta_deg_raw": float("nan"),
            "icp_residual": float("nan") if residual is None else float(residual),
        })
    return row, raw_mask, post_morph


def _agreement_flag(
    post_real: np.ndarray, post_wm: np.ndarray,
    pose_real: dict, pose_wm: dict,
) -> tuple[str, int, int, float, float]:
    real_only = int(((post_real > 0) & (post_wm == 0)).sum())
    wm_only = int(((post_wm > 0) & (post_real == 0)).sum())
    if not pose_real["cv_success"] or not pose_wm["cv_success"]:
        return "one_failed", real_only, wm_only, float("nan"), float("nan")
    d_px = float(np.hypot(pose_real["cx"] - pose_wm["cx"], pose_real["cy"] - pose_wm["cy"]))
    d_deg_raw = abs(pose_real["theta_deg"] - pose_wm["theta_deg"])
    d_deg = min(d_deg_raw, 360.0 - d_deg_raw)
    pixelwise_same = (real_only == 0) and (wm_only == 0)
    if pixelwise_same:
        return "identical", real_only, wm_only, d_px, d_deg
    if d_px < 1.0 and d_deg < 1.0:
        return "mask_differs_but_pose_agrees", real_only, wm_only, d_px, d_deg
    return "mask_and_pose_differ", real_only, wm_only, d_px, d_deg


def _build_grid(
    tiles: list[np.ndarray], cols: int, tag: str
) -> np.ndarray:
    rows = (len(tiles) + cols - 1) // cols
    h, w = tiles[0].shape[:2]
    pad = np.full((h, w, 3), 255, dtype=np.uint8)
    out_rows = []
    for r in range(rows):
        row_tiles = []
        for c in range(cols):
            idx = r * cols + c
            row_tiles.append(tiles[idx] if idx < len(tiles) else pad)
        out_rows.append(np.concatenate(row_tiles, axis=1))
    return np.concatenate(out_rows, axis=0)


def _tile(rgb_u8: np.ndarray, caption: str, overlays: list[dict]) -> np.ndarray:
    canvas = rgb_u8.copy()
    for ov in overlays:
        if ov["success"]:
            canvas = render_state_on_image(
                canvas, ov["cx"], ov["cy"],
                np.sin(np.deg2rad(ov["theta_deg"])),
                np.cos(np.deg2rad(ov["theta_deg"])),
                color=ov["color"], label=ov.get("label"),
            )
    big = cv2.resize(canvas, (canvas.shape[1] * UPSCALE, canvas.shape[0] * UPSCALE),
                     interpolation=cv2.INTER_NEAREST)
    h_pad = 28
    out = np.full((big.shape[0] + h_pad, big.shape[1], 3), 255, dtype=np.uint8)
    out[h_pad:, :] = big
    cv2.putText(out, caption, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    episodes = sorted(TRAIN_DIR.glob("episode_*.hdf5"))
    assert len(episodes) == 5, f"expected 5 train episodes, got {len(episodes)}"

    picks = _pick_frames(episodes, n=20)
    print(f"sampled {len(picks)} frames across {len(episodes)} episodes")
    for ep_id, t in picks:
        print(f"  ep={ep_id} t={t}")

    free, _ = torch.cuda.mem_get_info(0)
    print(f"free GPU mem before WM load: {free / 1024**2:.0f} MiB")

    t0 = time.time()
    wm = DifferentiableDynamics(str(CKPT_PATH), device=DEVICE)
    print(f"WM loaded in {time.time() - t0:.1f}s")

    decoded = _decode_all(wm, picks)
    print(f"decoded {len(decoded)} frames")

    template_contour = _get_template_contour(T_BLOCK_SHAPE, TSCALE)

    # ---- run each variant on each frame, collect metrics + masks ----
    variant_rows: dict[str, list[dict]] = {k: [] for k in VARIANTS}
    variant_masks_post: dict[str, list[np.ndarray]] = {k: [] for k in VARIANTS}

    for (ep_id, t_idx), rgb in zip(picks, decoded):
        for vname, (lo, hi) in VARIANTS.items():
            row, _raw, post = _run_variant(rgb, lo, hi, template_contour)
            row.update({"episode": ep_id, "t_idx": t_idx, "variant": vname,
                        "hsv_lo": list(map(int, lo)), "hsv_hi": list(map(int, hi))})
            variant_rows[vname].append(row)
            variant_masks_post[vname].append(post)

    # ---- calibration_table.csv ----
    with (OUT_DIR / "calibration_table.csv").open("w", newline="") as f:
        cols = ["frame_id", "episode", "t_idx", "variant",
                "hsv_lo_h", "hsv_lo_s", "hsv_lo_v", "hsv_hi_h", "hsv_hi_s", "hsv_hi_v",
                "raw_mask_px", "post_morph_px", "contour_count", "contour_area",
                "cv_success", "cx", "cy", "theta_deg", "theta_deg_raw", "icp_residual"]
        w = csv.writer(f); w.writerow(cols)
        fid = 0
        for i, (ep_id, t_idx) in enumerate(picks):
            for vname in VARIANTS:
                r = variant_rows[vname][i]
                w.writerow([
                    fid, ep_id, t_idx, vname,
                    r["hsv_lo"][0], r["hsv_lo"][1], r["hsv_lo"][2],
                    r["hsv_hi"][0], r["hsv_hi"][1], r["hsv_hi"][2],
                    r["raw_mask_px"], r["post_morph_px"], r["contour_count"],
                    f"{r['contour_area']:.2f}", int(r["cv_success"]),
                    f"{r['cx']:.3f}" if r["cv_success"] else "",
                    f"{r['cy']:.3f}" if r["cv_success"] else "",
                    f"{r['theta_deg']:.3f}" if r["cv_success"] else "",
                    f"{r['theta_deg_raw']:.3f}" if r["cv_success"] else "",
                    f"{r['icp_residual']:.6f}" if r["cv_success"] else "",
                ])
                fid += 1

    # ---- agreement_table.csv (REAL vs WM) ----
    agreement_rows = []
    for i, (ep_id, t_idx) in enumerate(picks):
        real_row = variant_rows["REAL"][i]
        wm_row = variant_rows["WM"][i]
        flag, real_only, wm_only, d_px, d_deg = _agreement_flag(
            variant_masks_post["REAL"][i], variant_masks_post["WM"][i],
            real_row, wm_row,
        )
        agreement_rows.append({
            "episode": ep_id, "t_idx": t_idx,
            "n_pixels_real_only": real_only,
            "n_pixels_wm_only": wm_only,
            "pose_delta_px": d_px, "pose_delta_deg": d_deg,
            "agreement_flag": flag,
        })

    with (OUT_DIR / "agreement_table.csv").open("w", newline="") as f:
        cols = ["episode", "t_idx", "n_pixels_real_only", "n_pixels_wm_only",
                "pose_delta_px", "pose_delta_deg", "agreement_flag"]
        w = csv.writer(f); w.writerow(cols)
        for r in agreement_rows:
            w.writerow([r["episode"], r["t_idx"], r["n_pixels_real_only"],
                        r["n_pixels_wm_only"],
                        f"{r['pose_delta_px']:.3f}" if np.isfinite(r["pose_delta_px"]) else "",
                        f"{r['pose_delta_deg']:.3f}" if np.isfinite(r["pose_delta_deg"]) else "",
                        r["agreement_flag"]])

    # ---- per_channel_relaxation.csv ----
    # For each variant, stats: drop_rate, mean/median post_morph_px,
    # mean area recovery vs REAL baseline.
    real_area = np.array([r["post_morph_px"] for r in variant_rows["REAL"]])
    summary_rows = []
    for vname in VARIANTS:
        areas = np.array([r["post_morph_px"] for r in variant_rows[vname]])
        successes = np.array([r["cv_success"] for r in variant_rows[vname]])
        drop_rate = 1.0 - successes.mean()
        area_recovery_vs_real_pct_pts = float(
            (areas - real_area).mean() / THEORETICAL_AREA * 100.0
        )
        residuals = np.array(
            [r["icp_residual"] for r in variant_rows[vname] if r["cv_success"]]
        )
        summary_rows.append({
            "variant": vname,
            "drop_rate": drop_rate,
            "mean_post_morph_px": float(areas.mean()),
            "median_post_morph_px": float(np.median(areas)),
            "area_frac_of_theoretical": float(areas.mean() / THEORETICAL_AREA),
            "area_recovery_vs_real_pct_pts": area_recovery_vs_real_pct_pts,
            "mean_icp_residual": float(residuals.mean()) if len(residuals) else float("nan"),
            "p95_icp_residual": float(np.percentile(residuals, 95)) if len(residuals) else float("nan"),
        })

    with (OUT_DIR / "per_channel_relaxation.csv").open("w", newline="") as f:
        cols = ["variant", "drop_rate", "mean_post_morph_px", "median_post_morph_px",
                "area_frac_of_theoretical", "area_recovery_vs_real_pct_pts",
                "mean_icp_residual", "p95_icp_residual"]
        w = csv.writer(f); w.writerow(cols)
        for s in summary_rows:
            w.writerow([s["variant"],
                        f"{s['drop_rate']:.4f}",
                        f"{s['mean_post_morph_px']:.1f}",
                        f"{s['median_post_morph_px']:.1f}",
                        f"{s['area_frac_of_theoretical']:.4f}",
                        f"{s['area_recovery_vs_real_pct_pts']:+.2f}",
                        f"{s['mean_icp_residual']:.4f}",
                        f"{s['p95_icp_residual']:.4f}"])

    # ---- visualization grids per variant ----
    for vname in VARIANTS:
        tiles = []
        for i, (ep_id, t_idx) in enumerate(picks):
            row = variant_rows[vname][i]
            overlays = [{
                "success": row["cv_success"],
                "cx": row.get("cx", 0), "cy": row.get("cy", 0),
                "theta_deg": row.get("theta_deg", 0),
                "color": (0, 220, 0),
                "label": vname,
            }]
            cap = (f"ep{ep_id} t={t_idx}"
                   + (f" area={row['post_morph_px']}px" if True else ""))
            if row["cv_success"]:
                cap += (f" cxy=({row['cx']:.0f},{row['cy']:.0f})"
                        f" th={row['theta_deg']:+.0f}d"
                        f" res={row['icp_residual']:.2f}")
            else:
                cap += " CV FAIL"
            tiles.append(_tile(decoded[i], cap, overlays))
        grid = _build_grid(tiles, cols=4, tag=vname)
        cv2.imwrite(
            str(OUT_DIR / f"calibration_grid_{vname}.png"),
            cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
        )

    # Also a dual-overlay (REAL green + WM orange) to visualise agreement.
    dual_tiles = []
    for i, (ep_id, t_idx) in enumerate(picks):
        rr = variant_rows["REAL"][i]
        wr = variant_rows["WM"][i]
        overlays = [
            {"success": rr["cv_success"], "cx": rr.get("cx", 0), "cy": rr.get("cy", 0),
             "theta_deg": rr.get("theta_deg", 0), "color": (0, 220, 0), "label": "R"},
            {"success": wr["cv_success"], "cx": wr.get("cx", 0), "cy": wr.get("cy", 0),
             "theta_deg": wr.get("theta_deg", 0), "color": (0, 120, 255), "label": "W"},
        ]
        flag = agreement_rows[i]["agreement_flag"]
        cap = f"ep{ep_id} t={t_idx} {flag}"
        dual_tiles.append(_tile(decoded[i], cap, overlays))
    cv2.imwrite(
        str(OUT_DIR / "calibration_grid_REAL_vs_WM.png"),
        cv2.cvtColor(_build_grid(dual_tiles, cols=4, tag="dual"), cv2.COLOR_RGB2BGR),
    )

    # ---- calibration_summary.md ----
    agreement_counts: dict[str, int] = {}
    for r in agreement_rows:
        agreement_counts[r["agreement_flag"]] = agreement_counts.get(r["agreement_flag"], 0) + 1
    identical_rate = agreement_counts.get("identical", 0) / len(agreement_rows)

    # Decision rule (design doc §1.2): chosen preset must satisfy
    #  (1) drop_rate <= 10%
    #  (2) median post_morph_px >= 90% of theoretical area
    #  (3) no new false positives (human check on the grids above)
    preset_picks = []
    for s in summary_rows:
        passes_1 = s["drop_rate"] <= 0.10
        passes_2 = s["median_post_morph_px"] >= 0.90 * THEORETICAL_AREA
        preset_picks.append({**s, "passes_drop": passes_1, "passes_area": passes_2})

    # Pick the candidate that passes (1) and (2) with the smallest drop_rate
    # (tiebreak on largest area recovery). Report the ranking; the human
    # decides from the grid whether any candidate introduces false positives.
    passers = [p for p in preset_picks if p["passes_drop"] and p["passes_area"]]
    if passers:
        ranked = sorted(passers, key=lambda p: (p["drop_rate"], -p["area_recovery_vs_real_pct_pts"]))
        rec = ranked[0]["variant"]
    else:
        # fallback to whichever has the smallest drop rate
        ranked = sorted(preset_picks, key=lambda p: p["drop_rate"])
        rec = ranked[0]["variant"]

    with (OUT_DIR / "calibration_summary.md").open("w") as f:
        f.write("# HSV Calibration Summary\n\n")
        f.write(f"Frames: 20 (stratified across 5 train episodes, 4 per ep)\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Theoretical T-block area at 128 canvas: {THEORETICAL_AREA:.1f} px\n\n")

        f.write("## The three mandatory numbers (design doc §1.3)\n\n")
        f.write("| variant | drop rate | median area (frac of theoretical) | area recovery vs REAL (pct pts) |\n")
        f.write("|---|---|---|---|\n")
        for s in summary_rows:
            f.write(
                f"| {s['variant']} | {s['drop_rate']:.1%} | "
                f"{s['mean_post_morph_px']:.1f} px ({s['area_frac_of_theoretical']:.1%}) | "
                f"{s['area_recovery_vs_real_pct_pts']:+.2f} |\n"
            )

        f.write(f"\nREAL-vs-WM agreement on 20 frames: **{identical_rate:.1%} pixel-identical**\n")
        f.write("Agreement breakdown:\n")
        for flag, n in sorted(agreement_counts.items()):
            f.write(f"- {flag}: {n}\n")

        f.write("\n## ICP residual distribution (successful frames only)\n\n")
        f.write("| variant | mean | p95 |\n|---|---|---|\n")
        for s in summary_rows:
            f.write(f"| {s['variant']} | {s['mean_icp_residual']:.3f} | {s['p95_icp_residual']:.3f} |\n")

        f.write("\n## Decision\n\n")
        f.write(f"**Proposed preset: `{rec}`**\n\n")
        f.write("Pass against the design-doc §1.2 decision criteria:\n\n")
        f.write("| variant | drop rate <= 10%% | median area >= 90%% theoretical |\n|---|---|---|\n")
        for p in preset_picks:
            f.write(f"| {p['variant']} | {'pass' if p['passes_drop'] else 'FAIL'} | {'pass' if p['passes_area'] else 'FAIL'} |\n")

        f.write("\nFalse-positive check (criterion 3) requires visual review of the "
                f"grid PNGs in {OUT_DIR.name}/ — flagged to the user at Checkpoint 3.\n")

    # ---- results.json: machine-readable digest ----
    with (OUT_DIR / "results.json").open("w") as f:
        json.dump({
            "theoretical_area_px": THEORETICAL_AREA,
            "n_frames": len(picks),
            "picks": [{"episode": ep, "t_idx": t} for ep, t in picks],
            "summary": summary_rows,
            "agreement_counts": agreement_counts,
            "identical_rate": identical_rate,
            "recommended_preset": rec,
        }, f, indent=2)

    # ---- terminal summary ----
    print("\n" + "=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)
    print(f"theoretical area (128 canvas): {THEORETICAL_AREA:.1f} px")
    print()
    print(f"{'variant':<16} {'drop%':>6} {'mean_area':>10} {'frac':>7} {'+/-pp':>7}")
    for s in summary_rows:
        print(f"{s['variant']:<16} "
              f"{s['drop_rate']*100:5.1f}% "
              f"{s['mean_post_morph_px']:9.1f} "
              f"{s['area_frac_of_theoretical']:6.1%} "
              f"{s['area_recovery_vs_real_pct_pts']:+6.2f}")
    print()
    print(f"REAL-vs-WM pixel-identical: {identical_rate:.1%} of {len(agreement_rows)}")
    for flag, n in sorted(agreement_counts.items()):
        print(f"  {flag}: {n}")
    print(f"\nrecommended preset: {rec}")
    print(f"artefacts: {OUT_DIR}")


if __name__ == "__main__":
    main()
