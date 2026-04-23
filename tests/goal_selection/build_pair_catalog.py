"""Build a curated (start, goal) pair catalog for multi-run MPPI evaluation.

Produces:
  tests/goal_selection/pair_catalog.json
  tests/goal_selection/pair_goals/pair_{A,B,C,D}_goal.pt
  tests/goal_selection/pair_catalog_viz.png

Four pairs covering distinct angular regimes:
  pair_A: near-axis goal (theta ~ 0  +/- 10)
  pair_B: 45-degree goal (theta ~ 45 +/- 10)
  pair_C: 90-degree goal (theta ~ 90 +/- 10)
  pair_D: flipped goal   (theta ~ 180 +/- 15)

Constraints applied to every pair:
  * start and goal in different val episodes
  * each pair's start episode is unique across the 4 pairs
  * start-goal pixel distance > 50 (>40 for pair_D)
  * both poses CV-valid with icp_residual < 0.5
  * both poses interior (>= 15 px from any frame edge)

Pipeline:
  1. CV-survey every (episode, frame) in data/mini/pusht/val (CPU only)
  2. Filter to candidates meeting interior + icp constraints
  3. Greedy pair selection across the 4 angular targets
  4. WM round-trip on each chosen GOAL frame so the saved goal CV pose
     comes from the decoded latent (same path MPPI's reward will travel).
     Start CV poses are kept as raw (start frames are encoded into the WM
     at run time, no persisted state needed).
  5. Persist catalog JSON, individual goal .pt files, and a 4x2 viz grid.

Run from repo root:
    conda run -n iws python tests/goal_selection/build_pair_catalog.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.labeling.cv_labeler import CVLabeler  # noqa: E402

# ─── Config ─────────────────────────────────────────────────────────────

DATA_DIR = REPO_ROOT / "data" / "mini" / "pusht" / "val"
EPISODES = [0, 1, 2, 3, 4]
WM_CKPT = REPO_ROOT / "outputs" / "pusht_cam1" / "checkpoints" / "best.ckpt"

OUTPUT_JSON = REPO_ROOT / "tests" / "goal_selection" / "pair_catalog.json"
GOAL_DIR = REPO_ROOT / "tests" / "goal_selection" / "pair_goals"
VIZ_PATH = REPO_ROOT / "tests" / "goal_selection" / "pair_catalog_viz.png"

RESOLUTION = 128
EDGE_MARGIN = 15
MAX_ICP = 0.5
CAMERA_KEY = "camera_1_color"

# Angular targets and per-pair distance thresholds. Order matters for
# greedy selection: hardest (180-degree) first so we don't starve it.
PAIR_SPECS = [
    {"id": "pair_D", "desc": "flipped goal (theta ~ 180)",
     "target_deg": 180.0, "tol_deg": 15.0, "min_dist": 40.0},
    {"id": "pair_C", "desc": "90-degree goal",
     "target_deg": 90.0,  "tol_deg": 10.0, "min_dist": 50.0},
    {"id": "pair_B", "desc": "45-degree goal",
     "target_deg": 45.0,  "tol_deg": 10.0, "min_dist": 50.0},
    {"id": "pair_A", "desc": "near-axis goal (theta ~ 0)",
     "target_deg": 0.0,   "tol_deg": 10.0, "min_dist": 50.0},
]


# ─── Helpers ────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    episode: int
    frame: int
    cx: float
    cy: float
    sin_theta: float
    cos_theta: float
    theta_deg: float
    contour_area: float
    icp_residual: float


def _preprocess_frame(raw_hwc_u8: np.ndarray, resolution: int) -> np.ndarray:
    """Center-crop square + resize to (resolution, resolution, 3) uint8 RGB.

    Mirrors env.pusht_wm_env._preprocess_rgb_uint8 so the CV survey sees
    the same pixels the WM will see after env.load_initial_from_hdf5.
    """
    h, w = raw_hwc_u8.shape[:2]
    s = min(h, w)
    cropped = raw_hwc_u8[(h - s) // 2 : (h - s) // 2 + s,
                         (w - s) // 2 : (w - s) // 2 + s]
    return cv2.resize(cropped, (resolution, resolution), interpolation=cv2.INTER_AREA)


def _episode_path(ep: int) -> Path:
    return DATA_DIR / f"episode_{ep}.hdf5"


def _angular_diff_deg(a: float, b: float) -> float:
    """Smallest unsigned angular distance on the [0, 180] half-circle."""
    d = (a - b) % 360.0
    if d > 180.0:
        d = 360.0 - d
    return abs(d)


def _euclid(c1: Candidate, c2: Candidate) -> float:
    return float(np.hypot(c1.cx - c2.cx, c1.cy - c2.cy))


def _interior(c: Candidate) -> bool:
    return (
        EDGE_MARGIN <= c.cx <= RESOLUTION - EDGE_MARGIN
        and EDGE_MARGIN <= c.cy <= RESOLUTION - EDGE_MARGIN
    )


# ─── Stage 1: CV survey ────────────────────────────────────────────────

def cv_survey() -> tuple[list[Candidate], dict[tuple[int, int], np.ndarray]]:
    """Run CV on every frame of every val episode.

    Returns (candidates_passing_filters, raw_frames_lookup). The raw
    frames lookup is keyed by (episode, frame) and used later for viz +
    goal round-trip.
    """
    labeler = CVLabeler(preset="REAL", resolution=RESOLUTION)
    candidates: list[Candidate] = []
    raw_frames: dict[tuple[int, int], np.ndarray] = {}

    n_total = 0
    n_success = 0
    n_passed = 0

    for ep in EPISODES:
        path = _episode_path(ep)
        if not path.exists():
            raise FileNotFoundError(f"missing val episode: {path}")
        with h5py.File(str(path), "r") as f:
            frames = f[f"obs/images/{CAMERA_KEY}"][:]  # (T, H, W, 3) uint8
        T = frames.shape[0]
        for t in range(T):
            n_total += 1
            pre = _preprocess_frame(frames[t], RESOLUTION)  # (128, 128, 3) u8
            res = labeler.label(pre)
            if not res.success:
                continue
            n_success += 1
            cand = Candidate(
                episode=ep, frame=t,
                cx=res.cx, cy=res.cy,
                sin_theta=res.sin_theta, cos_theta=res.cos_theta,
                theta_deg=res.theta_deg,
                contour_area=res.contour_area,
                icp_residual=res.icp_residual,
            )
            if not _interior(cand):
                continue
            if cand.icp_residual >= MAX_ICP:
                continue
            n_passed += 1
            candidates.append(cand)
            raw_frames[(ep, t)] = pre

    print(f"CV survey: {n_total} frames -> {n_success} success "
          f"-> {n_passed} pass interior+icp filters")
    return candidates, raw_frames


# ─── Stage 2: greedy pair selection ────────────────────────────────────

def select_pairs(candidates: list[Candidate]) -> dict[str, dict]:
    """Greedy assignment of starts to goals across the 4 pair specs.

    Constraints enforced here:
      * goal angle within target +/- tol
      * pair distance > min_dist
      * start_episode != goal_episode
      * start_episode unique across pairs
    """
    chosen: dict[str, dict] = {}
    used_start_episodes: set[int] = set()

    for spec in PAIR_SPECS:
        target = spec["target_deg"]
        tol = spec["tol_deg"]
        min_dist = spec["min_dist"]
        # Sort goal candidates by closeness to target angle so we prefer
        # the cleanest exemplar.
        goal_cands = sorted(
            (c for c in candidates if _angular_diff_deg(c.theta_deg, target) <= tol),
            key=lambda c: _angular_diff_deg(c.theta_deg, target),
        )
        if not goal_cands:
            raise RuntimeError(
                f"{spec['id']}: no candidates within {tol} deg of {target}. "
                f"Available angles in pool: "
                f"{sorted({round(c.theta_deg, 1) for c in candidates})}"
            )

        picked = None
        for goal in goal_cands:
            # find a start in a different episode AND in an episode
            # not yet used, with sufficient distance.
            start_cands = [
                s for s in candidates
                if s.episode != goal.episode
                and s.episode not in used_start_episodes
                and _euclid(s, goal) >= min_dist
            ]
            if not start_cands:
                continue
            # prefer the start with the largest distance (most demanding pair)
            start = max(start_cands, key=lambda s: _euclid(s, goal))
            picked = (start, goal)
            break

        if picked is None:
            raise RuntimeError(
                f"{spec['id']}: no (start, goal) tuple satisfies all constraints. "
                f"Goal pool size at this target: {len(goal_cands)}; "
                f"start episodes already used: {sorted(used_start_episodes)}"
            )

        start, goal = picked
        used_start_episodes.add(start.episode)
        chosen[spec["id"]] = {
            "spec": spec,
            "start": start,
            "goal": goal,
            "distance_px": _euclid(start, goal),
            "angular_difference_deg": _angular_diff_deg(start.theta_deg, goal.theta_deg),
        }
        print(f"  {spec['id']}: start ep{start.episode} f{start.frame} "
              f"theta={start.theta_deg:+6.1f}deg | "
              f"goal ep{goal.episode} f{goal.frame} theta={goal.theta_deg:+6.1f}deg | "
              f"dist={_euclid(start, goal):.1f}px")
    return chosen


# ─── Stage 3: round-trip goals through the WM ──────────────────────────

def round_trip_goals(
    chosen: dict[str, dict],
    raw_frames: dict[tuple[int, int], np.ndarray],
) -> dict[str, dict]:
    """For each chosen goal frame, encode->decode->CV-label.

    Saves a state_goal.pt-style payload to GOAL_DIR/pair_X_goal.pt and
    returns a dict keyed by pair_id with the round-tripped CV pose +
    the decoded RGB (for viz).
    """
    # Lazy import so the CV-survey-only code path doesn't pay the WM
    # construction cost.
    from env.pusht_wm_env import PushTWMEnv  # noqa: E402

    if not WM_CKPT.exists():
        raise FileNotFoundError(f"WM checkpoint missing: {WM_CKPT}")

    print(f"Loading WM from {WM_CKPT}")
    t0 = time.time()
    env = PushTWMEnv(str(WM_CKPT), device="cuda:0")
    print(f"  WM loaded in {time.time() - t0:.1f}s")

    labeler = CVLabeler(preset="REAL", resolution=RESOLUTION)
    out: dict[str, dict] = {}

    GOAL_DIR.mkdir(parents=True, exist_ok=True)

    for pair_id, info in chosen.items():
        goal = info["goal"]
        raw = raw_frames[(goal.episode, goal.frame)]  # (128, 128, 3) u8 RGB

        # encode + decode
        rgb_t = torch.from_numpy(raw.astype(np.float32) / 255.0).permute(2, 0, 1)
        z = env.encode(rgb_t)               # (C, H_lat, W_lat)
        rgb_dec = env.decode(z)             # (3, H, W) float in [0,1]
        rgb_dec_u8 = (rgb_dec.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        rgb_dec_hwc = rgb_dec_u8.transpose(1, 2, 0)

        # CV on the decoded frame
        res = labeler.label(rgb_dec_hwc)
        if not res.success:
            raise RuntimeError(
                f"{pair_id}: CV failed on round-tripped goal frame "
                f"(ep{goal.episode} f{goal.frame}). Pre-round-trip CV was "
                f"successful (theta={goal.theta_deg:.1f}); decoder distorted "
                f"the T enough to break detection. Pick a different exemplar."
            )

        state_pt_path = GOAL_DIR / f"{pair_id}_goal.pt"
        payload = {
            "state": torch.tensor(
                [res.cx, res.cy, res.sin_theta, res.cos_theta], dtype=torch.float32,
            ),
            "cx": float(res.cx),
            "cy": float(res.cy),
            "sin_theta": float(res.sin_theta),
            "cos_theta": float(res.cos_theta),
            "theta_rad": float(res.theta_rad),
            "theta_deg": float(res.theta_deg),
            "resolution": RESOLUTION,
            "meta": {
                "preset": "REAL",
                "hsv_lower": labeler.hsv_lower.tolist(),
                "hsv_upper": labeler.hsv_upper.tolist(),
                "ckpt": str(WM_CKPT),
                "source_hdf5": str(_episode_path(goal.episode)),
                "source_frame": int(goal.frame),
                "raw_cv_theta_deg": float(goal.theta_deg),
                "round_tripped": True,
                "contour_area": float(res.contour_area),
                "icp_residual": float(res.icp_residual),
            },
        }
        torch.save(payload, state_pt_path)

        out[pair_id] = {
            "round_tripped_cx": float(res.cx),
            "round_tripped_cy": float(res.cy),
            "round_tripped_theta_deg": float(res.theta_deg),
            "round_tripped_icp_residual": float(res.icp_residual),
            "decoded_rgb_u8": rgb_dec_hwc,
            "state_pt_path": str(state_pt_path.relative_to(REPO_ROOT)),
        }
        print(
            f"  {pair_id}: round-trip pose "
            f"({res.cx:.1f}, {res.cy:.1f}, {res.theta_deg:+.1f}deg) "
            f"vs raw ({goal.cx:.1f}, {goal.cy:.1f}, {goal.theta_deg:+.1f}deg)"
        )
    return out


# ─── Stage 4: catalog + viz ─────────────────────────────────────────────

def write_catalog(
    chosen: dict[str, dict],
    rt_info: dict[str, dict],
) -> dict:
    catalog: dict = {}
    for pair_id, info in chosen.items():
        spec = info["spec"]
        start = info["start"]
        goal = info["goal"]
        rt = rt_info[pair_id]
        catalog[pair_id] = {
            "description": spec["desc"],
            "start": {
                "hdf5_path": str(_episode_path(start.episode).relative_to(REPO_ROOT)),
                "frame_idx": int(start.frame),
                "cx": float(start.cx),
                "cy": float(start.cy),
                "sin_theta": float(start.sin_theta),
                "cos_theta": float(start.cos_theta),
                "theta_deg": float(start.theta_deg),
                "icp_residual": float(start.icp_residual),
            },
            "goal": {
                "hdf5_path": str(_episode_path(goal.episode).relative_to(REPO_ROOT)),
                "frame_idx": int(goal.frame),
                "raw_cx": float(goal.cx),
                "raw_cy": float(goal.cy),
                "raw_theta_deg": float(goal.theta_deg),
                "round_tripped_cx": rt["round_tripped_cx"],
                "round_tripped_cy": rt["round_tripped_cy"],
                "round_tripped_theta_deg": rt["round_tripped_theta_deg"],
                "round_tripped_icp_residual": rt["round_tripped_icp_residual"],
                "state_pt_path": rt["state_pt_path"],
            },
            "distance_px": round(info["distance_px"], 3),
            "angular_difference_deg": round(info["angular_difference_deg"], 2),
            "target_angle_deg": float(spec["target_deg"]),
            "target_angle_tol_deg": float(spec["tol_deg"]),
        }
    OUTPUT_JSON.write_text(json.dumps(catalog, indent=2))
    print(f"Wrote catalog: {OUTPUT_JSON}")
    return catalog


def render_viz(
    chosen: dict[str, dict],
    rt_info: dict[str, dict],
    raw_frames: dict[tuple[int, int], np.ndarray],
) -> None:
    """4x2 grid: row = pair, left = start raw, right = goal round-tripped."""
    from rl.visualization.state_viz import render_state_on_image  # noqa: E402

    pair_order = ["pair_A", "pair_B", "pair_C", "pair_D"]
    fig, axes = plt.subplots(4, 2, figsize=(7.5, 14))

    for row, pair_id in enumerate(pair_order):
        info = chosen[pair_id]
        start = info["start"]
        goal = info["goal"]
        rt = rt_info[pair_id]

        # left: start raw + start CV overlay
        start_canvas = render_state_on_image(
            raw_frames[(start.episode, start.frame)],
            cx=start.cx, cy=start.cy,
            sin_theta=start.sin_theta, cos_theta=start.cos_theta,
            color=(0, 220, 0), label=f"start θ={start.theta_deg:+.0f}",
        )
        axes[row, 0].imshow(start_canvas)
        axes[row, 0].set_title(
            f"{pair_id}  start  ep{start.episode} f{start.frame}"
        )
        axes[row, 0].axis("off")

        # right: goal round-tripped + round-tripped CV overlay (red)
        goal_canvas = render_state_on_image(
            rt["decoded_rgb_u8"],
            cx=rt["round_tripped_cx"], cy=rt["round_tripped_cy"],
            sin_theta=np.sin(np.deg2rad(rt["round_tripped_theta_deg"])),
            cos_theta=np.cos(np.deg2rad(rt["round_tripped_theta_deg"])),
            color=(220, 0, 0),
            label=f"goal θ={rt['round_tripped_theta_deg']:+.0f}",
        )
        axes[row, 1].imshow(goal_canvas)
        axes[row, 1].set_title(
            f"{pair_id}  goal (RT)  ep{goal.episode} f{goal.frame}\n"
            f"dist={info['distance_px']:.1f}px  "
            f"Δθ={info['angular_difference_deg']:.0f}°"
        )
        axes[row, 1].axis("off")

    fig.suptitle("MPPI multi-pair catalog (start raw vs goal WM round-trip)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(VIZ_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote viz: {VIZ_PATH}")


# ─── main ──────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Stage 1: CV survey ===")
    candidates, raw_frames = cv_survey()
    if len(candidates) < 50:
        raise RuntimeError(
            f"Only {len(candidates)} CV candidates passed filters; expected >=50. "
            f"Aborting before pair selection."
        )

    print("\n=== Stage 2: greedy pair selection ===")
    chosen = select_pairs(candidates)

    print("\n=== Stage 3: WM round-trip on goals ===")
    rt_info = round_trip_goals(chosen, raw_frames)

    print("\n=== Stage 4: persist catalog + viz ===")
    write_catalog(chosen, rt_info)
    render_viz(chosen, rt_info, raw_frames)
    print("Done.")


if __name__ == "__main__":
    main()
