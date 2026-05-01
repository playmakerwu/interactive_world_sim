"""Build easy same-episode (start, goal) pairs for local MPPI diagnostics.

Produces:
  tests/goal_selection/easy_pair_catalog.json
  tests/goal_selection/easy_pair_goals/easy_pair_{1..5}_goal.pt
  tests/goal_selection/easy_pair_catalog_viz.png

Selection target:
  * frame gap exactly 5
  * same val episode
  * 4 px <= translation distance <= 18 px
  * |delta theta| <= 30 deg
  * both raw frames CV-valid, interior, and ICP residual < 0.5

Episode 0 in the mini val split is effectively static under this gap
(max center displacement is about 1 px), so the builder documents it as a
zero-qualifying fallback and selects a second pair from another episode to
still produce 5 total tasks.

Run from repo root:
    conda run -n iws python tests/goal_selection/build_easy_pair_catalog.py
"""

from __future__ import annotations

import json
import math
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

DATA_DIR = REPO_ROOT / "data" / "mini" / "pusht" / "val"
EPISODES = [0, 1, 2, 3, 4]
WM_CKPT = REPO_ROOT / "outputs" / "pusht_cam1" / "checkpoints" / "best.ckpt"

OUTPUT_JSON = REPO_ROOT / "tests" / "goal_selection" / "easy_pair_catalog.json"
GOAL_DIR = REPO_ROOT / "tests" / "goal_selection" / "easy_pair_goals"
VIZ_PATH = REPO_ROOT / "tests" / "goal_selection" / "easy_pair_catalog_viz.png"

RESOLUTION = 128
EDGE_MARGIN = 15
MAX_ICP = 0.5
FRAME_GAP = 5
MIN_DIST = 4.0
MAX_DIST = 18.0
MAX_ABS_DTHETA = 30.0
CAMERA_KEY = "camera_1_color"


@dataclass(frozen=True)
class Pose:
    episode: int
    frame: int
    cx: float
    cy: float
    sin_theta: float
    cos_theta: float
    theta_deg: float
    contour_area: float
    icp_residual: float


@dataclass(frozen=True)
class EasyPair:
    source_episode: int
    start: Pose
    goal: Pose
    distance_px: float
    delta_theta_deg: float


def _episode_path(ep: int) -> Path:
    return DATA_DIR / f"episode_{ep}.hdf5"


def _preprocess_frame(raw_hwc_u8: np.ndarray, resolution: int = RESOLUTION) -> np.ndarray:
    """Mirror env.pusht_wm_env._preprocess_rgb_uint8, returning uint8 HWC RGB."""
    h, w = raw_hwc_u8.shape[:2]
    s = min(h, w)
    cropped = raw_hwc_u8[(h - s) // 2 : (h - s) // 2 + s,
                         (w - s) // 2 : (w - s) // 2 + s]
    return cv2.resize(cropped, (resolution, resolution), interpolation=cv2.INTER_AREA)


def _interior(pose: Pose) -> bool:
    return (
        EDGE_MARGIN <= pose.cx <= RESOLUTION - EDGE_MARGIN
        and EDGE_MARGIN <= pose.cy <= RESOLUTION - EDGE_MARGIN
    )


def _signed_angle_diff_deg(a: float, b: float) -> float:
    return float((b - a + 180.0) % 360.0 - 180.0)


def _distance(a: Pose, b: Pose) -> float:
    return float(math.hypot(a.cx - b.cx, a.cy - b.cy))


def _pose_to_json(pose: Pose) -> dict:
    return {
        "hdf5_path": str(_episode_path(pose.episode).relative_to(REPO_ROOT)),
        "frame_idx": int(pose.frame),
        "cx": float(pose.cx),
        "cy": float(pose.cy),
        "sin_theta": float(pose.sin_theta),
        "cos_theta": float(pose.cos_theta),
        "theta_deg": float(pose.theta_deg),
        "contour_area": float(pose.contour_area),
        "icp_residual": float(pose.icp_residual),
    }


def _load_episode_frames(ep: int) -> np.ndarray:
    path = _episode_path(ep)
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(str(path), "r") as f:
        return f[f"obs/images/{CAMERA_KEY}"][:]


def cv_survey() -> tuple[dict[int, list[Pose | None]], dict[tuple[int, int], np.ndarray]]:
    """Run raw-frame CV over all mini val episodes."""
    labeler = CVLabeler(preset="REAL", resolution=RESOLUTION)
    by_episode: dict[int, list[Pose | None]] = {}
    raw_frames: dict[tuple[int, int], np.ndarray] = {}

    for ep in EPISODES:
        frames = _load_episode_frames(ep)
        poses: list[Pose | None] = []
        n_valid = 0
        for frame_idx, raw in enumerate(frames):
            pre = _preprocess_frame(raw)
            raw_frames[(ep, frame_idx)] = pre
            res = labeler.label(pre)
            if not res.success:
                poses.append(None)
                continue
            pose = Pose(
                episode=ep,
                frame=frame_idx,
                cx=float(res.cx),
                cy=float(res.cy),
                sin_theta=float(res.sin_theta),
                cos_theta=float(res.cos_theta),
                theta_deg=float(res.theta_deg),
                contour_area=float(res.contour_area),
                icp_residual=float(res.icp_residual),
            )
            if pose.icp_residual >= MAX_ICP or not _interior(pose):
                poses.append(None)
                continue
            n_valid += 1
            poses.append(pose)
        by_episode[ep] = poses
        print(f"episode {ep}: {n_valid}/{len(poses)} raw poses pass CV/interior/ICP")
    return by_episode, raw_frames


def _qualifying_pairs_for_episode(episode: int, poses: list[Pose | None]) -> list[EasyPair]:
    out: list[EasyPair] = []
    for start_idx in range(0, len(poses) - FRAME_GAP):
        start = poses[start_idx]
        goal = poses[start_idx + FRAME_GAP]
        if start is None or goal is None:
            continue
        dist = _distance(start, goal)
        dtheta = _signed_angle_diff_deg(start.theta_deg, goal.theta_deg)
        if not (MIN_DIST <= dist <= MAX_DIST):
            continue
        if abs(dtheta) > MAX_ABS_DTHETA:
            continue
        out.append(
            EasyPair(
                source_episode=episode,
                start=start,
                goal=goal,
                distance_px=dist,
                delta_theta_deg=dtheta,
            )
        )
    return out


def select_easy_pairs(
    by_episode: dict[int, list[Pose | None]],
) -> tuple[list[EasyPair], dict]:
    """Pick first qualifying pair per episode, then documented fallbacks."""
    per_episode: dict[int, list[EasyPair]] = {
        ep: _qualifying_pairs_for_episode(ep, poses)
        for ep, poses in by_episode.items()
    }

    selected: list[EasyPair] = []
    fallback_notes: list[dict] = []
    for ep in EPISODES:
        if per_episode[ep]:
            selected.append(per_episode[ep][0])
        else:
            max_gap_dist = _max_gap_distance(by_episode[ep])
            fallback_notes.append({
                "episode": ep,
                "reason": (
                    f"zero qualifying frame-gap-{FRAME_GAP} pairs under "
                    f"{MIN_DIST:g}-{MAX_DIST:g}px and |delta theta| <= "
                    f"{MAX_ABS_DTHETA:g} deg"
                ),
                "max_frame_gap_distance_px": round(max_gap_dist, 3),
            })

    used = {(p.source_episode, p.start.frame, p.goal.frame) for p in selected}
    fallback_pool: list[EasyPair] = []
    for ep in EPISODES:
        for pair in per_episode[ep]:
            key = (pair.source_episode, pair.start.frame, pair.goal.frame)
            if key not in used:
                fallback_pool.append(pair)
    fallback_pool.sort(key=lambda p: (p.source_episode, p.start.frame))

    while len(selected) < 5 and fallback_pool:
        pair = fallback_pool.pop(0)
        selected.append(pair)
        fallback_notes.append({
            "episode": pair.source_episode,
            "reason": "additional fallback pair selected to reach 5 total tasks",
            "start_frame": int(pair.start.frame),
            "goal_frame": int(pair.goal.frame),
        })

    if len(selected) != 5:
        raise RuntimeError(f"expected 5 easy pairs, found only {len(selected)}")

    metadata = {
        "selection_criteria": {
            "same_episode": True,
            "frame_gap": FRAME_GAP,
            "distance_px": [MIN_DIST, MAX_DIST],
            "abs_delta_theta_deg_max": MAX_ABS_DTHETA,
            "max_icp_residual": MAX_ICP,
            "edge_margin_px": EDGE_MARGIN,
        },
        "qualifying_pairs_per_episode": {
            str(ep): len(per_episode[ep]) for ep in EPISODES
        },
        "fallback_notes": fallback_notes,
        "unique_source_episodes": sorted({p.source_episode for p in selected}),
    }

    print("Selected easy pairs:")
    for i, pair in enumerate(selected, start=1):
        print(
            f"  easy_pair_{i}: ep{pair.source_episode} "
            f"f{pair.start.frame}->{pair.goal.frame} "
            f"dist={pair.distance_px:.1f}px "
            f"delta_theta={pair.delta_theta_deg:+.1f}deg"
        )
    if fallback_notes:
        print("Fallback notes:")
        for note in fallback_notes:
            print(f"  {note}")
    return selected, metadata


def _max_gap_distance(poses: list[Pose | None]) -> float:
    max_dist = 0.0
    for start_idx in range(0, len(poses) - FRAME_GAP):
        start = poses[start_idx]
        goal = poses[start_idx + FRAME_GAP]
        if start is None or goal is None:
            continue
        max_dist = max(max_dist, _distance(start, goal))
    return max_dist


def round_trip_goals(
    pairs: list[EasyPair],
    raw_frames: dict[tuple[int, int], np.ndarray],
) -> dict[str, dict]:
    """Round-trip every chosen goal frame through the WM and save goal .pt files."""
    from env.pusht_wm_env import PushTWMEnv  # noqa: E402

    if not WM_CKPT.exists():
        raise FileNotFoundError(WM_CKPT)

    GOAL_DIR.mkdir(parents=True, exist_ok=True)
    labeler = CVLabeler(preset="REAL", resolution=RESOLUTION)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    print(f"Loading WM from {WM_CKPT}")
    t0 = time.time()
    env = PushTWMEnv(str(WM_CKPT), device="cuda:0")
    print(f"  WM loaded in {time.time() - t0:.1f}s")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    out: dict[str, dict] = {}
    for i, pair in enumerate(pairs, start=1):
        pair_id = f"easy_pair_{i}"
        raw = raw_frames[(pair.goal.episode, pair.goal.frame)]
        rgb_t = torch.from_numpy(raw.astype(np.float32) / 255.0).permute(2, 0, 1)
        z = env.encode(rgb_t)
        rgb_dec = env.decode(z)
        rgb_dec_u8 = (rgb_dec.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        rgb_dec_hwc = rgb_dec_u8.transpose(1, 2, 0)

        res = labeler.label(rgb_dec_hwc)
        if not res.success:
            raise RuntimeError(
                f"{pair_id}: CV failed on round-tripped goal "
                f"ep{pair.goal.episode} f{pair.goal.frame}"
            )
        if res.icp_residual >= MAX_ICP:
            raise RuntimeError(
                f"{pair_id}: round-tripped goal ICP residual "
                f"{res.icp_residual:.3f} >= {MAX_ICP}"
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
                "source_hdf5": str(_episode_path(pair.goal.episode)),
                "source_frame": int(pair.goal.frame),
                "raw_cv_cx": float(pair.goal.cx),
                "raw_cv_cy": float(pair.goal.cy),
                "raw_cv_theta_deg": float(pair.goal.theta_deg),
                "round_tripped": True,
                "contour_area": float(res.contour_area),
                "icp_residual": float(res.icp_residual),
            },
        }
        torch.save(payload, state_pt_path)

        rt_dist = math.hypot(pair.start.cx - res.cx, pair.start.cy - res.cy)
        rt_dtheta = _signed_angle_diff_deg(pair.start.theta_deg, res.theta_deg)
        out[pair_id] = {
            "round_tripped_cx": float(res.cx),
            "round_tripped_cy": float(res.cy),
            "round_tripped_theta_deg": float(res.theta_deg),
            "round_tripped_icp_residual": float(res.icp_residual),
            "round_tripped_distance_px": round(float(rt_dist), 3),
            "round_tripped_delta_theta_deg": round(float(rt_dtheta), 2),
            "decoded_rgb_u8": rgb_dec_hwc,
            "state_pt_path": str(state_pt_path.relative_to(REPO_ROOT)),
        }
        print(
            f"  {pair_id}: raw goal ({pair.goal.cx:.1f}, {pair.goal.cy:.1f}, "
            f"{pair.goal.theta_deg:+.1f}deg) -> RT ({res.cx:.1f}, {res.cy:.1f}, "
            f"{res.theta_deg:+.1f}deg)"
        )
    return out


def write_catalog(
    pairs: list[EasyPair],
    rt_info: dict[str, dict],
    metadata: dict,
) -> dict:
    catalog: dict = {"_metadata": metadata}
    for i, pair in enumerate(pairs, start=1):
        pair_id = f"easy_pair_{i}"
        rt = rt_info[pair_id]
        catalog[pair_id] = {
            "description": "same-episode frame-gap-5 gentle push",
            "start": _pose_to_json(pair.start),
            "goal": {
                **_pose_to_json(pair.goal),
                "raw_cx": float(pair.goal.cx),
                "raw_cy": float(pair.goal.cy),
                "raw_theta_deg": float(pair.goal.theta_deg),
                "round_tripped_cx": rt["round_tripped_cx"],
                "round_tripped_cy": rt["round_tripped_cy"],
                "round_tripped_theta_deg": rt["round_tripped_theta_deg"],
                "round_tripped_icp_residual": rt["round_tripped_icp_residual"],
                "state_pt_path": rt["state_pt_path"],
            },
            "frame_gap": FRAME_GAP,
            "distance_px": round(float(pair.distance_px), 3),
            "delta_theta_deg": round(float(pair.delta_theta_deg), 2),
            "abs_delta_theta_deg": round(abs(float(pair.delta_theta_deg)), 2),
            "round_tripped_distance_px": rt["round_tripped_distance_px"],
            "round_tripped_delta_theta_deg": rt["round_tripped_delta_theta_deg"],
        }
    OUTPUT_JSON.write_text(json.dumps(catalog, indent=2))
    print(f"Wrote catalog: {OUTPUT_JSON}")
    return catalog


def render_viz(
    pairs: list[EasyPair],
    rt_info: dict[str, dict],
    raw_frames: dict[tuple[int, int], np.ndarray],
) -> None:
    from rl.visualization.state_viz import render_state_on_image  # noqa: E402

    fig, axes = plt.subplots(5, 2, figsize=(7.5, 17.0))
    for row, pair in enumerate(pairs):
        pair_id = f"easy_pair_{row + 1}"
        rt = rt_info[pair_id]

        start_canvas = render_state_on_image(
            raw_frames[(pair.start.episode, pair.start.frame)],
            cx=pair.start.cx,
            cy=pair.start.cy,
            sin_theta=pair.start.sin_theta,
            cos_theta=pair.start.cos_theta,
            color=(0, 220, 0),
            label="start",
        )
        axes[row, 0].imshow(start_canvas)
        axes[row, 0].set_title(
            f"{pair_id} start ep{pair.source_episode} f{pair.start.frame}"
        )
        axes[row, 0].axis("off")

        goal_canvas = render_state_on_image(
            rt["decoded_rgb_u8"],
            cx=rt["round_tripped_cx"],
            cy=rt["round_tripped_cy"],
            sin_theta=np.sin(np.deg2rad(rt["round_tripped_theta_deg"])),
            cos_theta=np.cos(np.deg2rad(rt["round_tripped_theta_deg"])),
            color=(220, 0, 0),
            label="goal",
        )
        axes[row, 1].imshow(goal_canvas)
        axes[row, 1].set_title(
            f"{pair_id} goal RT f{pair.goal.frame}\n"
            f"raw dist={pair.distance_px:.1f}px, "
            f"delta theta={pair.delta_theta_deg:+.1f}deg"
        )
        axes[row, 1].axis("off")

    fig.suptitle("Easy MPPI pairs: start raw vs goal WM round-trip", fontsize=12)
    fig.tight_layout()
    fig.savefig(VIZ_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote viz: {VIZ_PATH}")


def main() -> None:
    print("=== Stage 1: raw CV survey ===")
    by_episode, raw_frames = cv_survey()

    print("\n=== Stage 2: easy pair selection ===")
    pairs, metadata = select_easy_pairs(by_episode)

    print("\n=== Stage 3: WM round-trip on goals ===")
    rt_info = round_trip_goals(pairs, raw_frames)

    print("\n=== Stage 4: persist catalog + viz ===")
    write_catalog(pairs, rt_info, metadata)
    render_viz(pairs, rt_info, raw_frames)
    print("Done.")


if __name__ == "__main__":
    main()
