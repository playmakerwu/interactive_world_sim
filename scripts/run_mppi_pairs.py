"""Run MPPI for 10 (start, goal) pairs. Each pair gets:
  - a unique state_goal.pt computed from the goal frame
  - a full MPPI run starting from the start frame
  - artifacts under outputs/mppi/pairs_run/pair_NN/

Skips a pair if its CV fails on the goal frame (rare but possible for
heavily-flipped poses). Runs sequentially so the GPU isn't contended.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.labeling.cv_labeler import CVLabeler  # noqa: E402
from rl.models.world_model import DifferentiableDynamics  # noqa: E402

CKPT = REPO_ROOT / "outputs" / "pusht_cam1" / "checkpoints" / "best.ckpt"
RES = 128
OBS_KEY = "camera_1_color"

# 10 distinct pairs covering different start/end poses.
# Format: (start_spec, goal_spec) where each spec is "split/ep/frame".
# Goal frames: prefer mid-episode (T already pushed) for variety.
PAIRS: list[tuple[str, str]] = [
    ("mini/val/0/0",   "mini/val/0/100"),  # within-episode push (ep0)
    ("mini/val/1/0",   "mini/val/1/100"),  # within-episode push (ep1)
    ("mini/val/2/0",   "mini/val/2/100"),  # within-episode push (ep2)
    ("mini/val/3/0",   "mini/val/3/100"),  # within-episode push (ep3)
    ("mini/val/4/0",   "mini/val/4/100"),  # within-episode push (ep4)
    ("mini/val/0/100", "mini/val/2/0"),    # cross-episode
    ("mini/val/1/100", "mini/val/3/0"),    # cross-episode
    ("mini/val/2/100", "mini/val/4/0"),    # cross-episode
    ("mini/val/3/100", "mini/val/0/0"),    # cross-episode
    ("mini/val/4/100", "mini/val/1/0"),    # cross-episode
]


def _preprocess_rgb(raw: np.ndarray) -> np.ndarray:
    h, w = raw.shape[:2]
    s = min(h, w)
    cr = raw[(h - s) // 2 : (h - s) // 2 + s, (w - s) // 2 : (w - s) // 2 + s]
    resized = cv2.resize(cr, (RES, RES), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def _load_frame(spec: str) -> np.ndarray:
    parts = spec.split("/")
    dataset, split, ep, t = parts
    p = REPO_ROOT / "data" / dataset / "pusht" / split / f"episode_{int(ep)}.hdf5"
    with h5py.File(p, "r") as f:
        return f[f"obs/images/{OBS_KEY}"][int(t)]


def main() -> None:
    out_root = REPO_ROOT / "outputs" / "mppi" / "pairs_run"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading WM from {CKPT}")
    wm = DifferentiableDynamics(str(CKPT), device="cuda:0")
    labeler = CVLabeler(preset="REAL", resolution=RES)

    # ── pre-compute goal state_goal.pt for each pair ──
    pair_dirs: list[Path] = []
    print(f"\nPre-computing {len(PAIRS)} goal state_goal.pt files…")
    for k, (start_spec, goal_spec) in enumerate(PAIRS):
        pair_dir = out_root / f"pair_{k:02d}"
        pair_dir.mkdir(exist_ok=True)
        # Encode and decode the goal frame -> CV -> state_goal
        raw = _load_frame(goal_spec)
        pre = _preprocess_rgb(raw)
        pre_t = torch.from_numpy(pre).permute(2, 0, 1).unsqueeze(0).to("cuda:0")
        with torch.no_grad():
            z_goal = wm.encode(pre_t)                       # (1, 4, 32, 32)
            rgb_goal = wm.decode(z_goal, resolution=RES)
        rgb_u8 = (rgb_goal.clamp(0, 1).cpu().numpy()[0] * 255).astype(np.uint8)
        rgb_u8 = rgb_u8.transpose(1, 2, 0)
        lbl = labeler.label(rgb_u8)
        if not lbl.success:
            print(f"  pair {k}: CV FAIL on goal {goal_spec} — SKIP")
            (pair_dir / "skip_reason.txt").write_text(
                f"CV failed on goal frame {goal_spec}\n"
            )
            continue
        state_goal = {
            "state": torch.tensor(
                [lbl.cx, lbl.cy, lbl.sin_theta, lbl.cos_theta]
            ),
            "cx": lbl.cx, "cy": lbl.cy,
            "sin_theta": lbl.sin_theta, "cos_theta": lbl.cos_theta,
            "theta_rad": float(np.deg2rad(lbl.theta_deg)),
            "theta_deg": float(lbl.theta_deg),
            "resolution": RES,
            "meta": {
                "preset": "REAL",
                "source_spec": goal_spec,
                "icp_residual": lbl.icp_residual,
                "contour_area": lbl.contour_area,
            },
        }
        torch.save(state_goal, pair_dir / "state_goal.pt")
        cv2.imwrite(
            str(pair_dir / "goal_decoded.png"),
            cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR),
        )
        pair_dirs.append(pair_dir)
        print(
            f"  pair {k}: start={start_spec}  goal={goal_spec}  "
            f"-> ({lbl.cx:.1f}, {lbl.cy:.1f}, {lbl.theta_deg:+.1f} deg)"
        )

    # ── release WM memory before subprocess MPPI runs ──
    del wm, labeler
    torch.cuda.empty_cache()

    # ── run MPPI for each valid pair as a subprocess ──
    print(f"\nRunning {len(pair_dirs)} MPPI episodes…")
    for k, (start_spec, goal_spec) in enumerate(PAIRS):
        pair_dir = out_root / f"pair_{k:02d}"
        if not (pair_dir / "state_goal.pt").exists():
            continue
        run_name = f"pairs_run/pair_{k:02d}"
        goal_rel = (pair_dir / "state_goal.pt").relative_to(REPO_ROOT)
        cmd = [
            sys.executable, "scripts/run_mppi.py",
            "--run_name", run_name,
            "--initial_state", start_spec,
            "--goal_path", str(goal_rel),
            "--action_source", "gaussian",
            "--N", "16", "--H", "10", "--sigma", "0.1",
            "--control_steps", "50", "--seed", "0",
        ]
        print(f"\n=== pair {k}: {start_spec} -> {goal_spec} ===")
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"  FAILED (rc={proc.returncode}):\n{proc.stderr[-500:]}")
            (pair_dir / "skip_reason.txt").write_text(proc.stderr[-2000:])
            continue
        # Last summary lines from stdout
        for line in proc.stdout.splitlines()[-7:]:
            print(f"  {line}")
        print(f"  wall: {time.time() - t0:.1f}s")

    # ── pair manifest ──
    manifest = []
    for k, (start_spec, goal_spec) in enumerate(PAIRS):
        pair_dir = out_root / f"pair_{k:02d}"
        sf = pair_dir / "summary.json"
        entry = {
            "k": k,
            "start": start_spec,
            "goal": goal_spec,
            "summary_path": (
                str(sf.relative_to(REPO_ROOT)) if sf.exists() else None
            ),
        }
        if sf.exists():
            s = json.loads(sf.read_text())
            entry["final_pos_distance_px"] = s.get("final_pos_distance_px")
            entry["final_angle_sim"] = s.get("final_angle_sim")
            entry["min_cos_to_z0"] = s.get("min_latent_cosine_sim_to_z0")
            entry["n_cv_fail"] = s.get("cv_failures_along_trajectory")
        else:
            sr = pair_dir / "skip_reason.txt"
            entry["skipped"] = sr.read_text().strip() if sr.exists() else "unknown"
        manifest.append(entry)
    (out_root / "pairs_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {out_root}/pairs_manifest.json")


if __name__ == "__main__":
    main()
