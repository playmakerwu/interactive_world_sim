"""Bulk-label the PushT replay buffer with the CV pose estimator.

Per design doc §1.4 (Option A): for each RGB frame in the dataset,
encode through the frozen WM to get a latent, decode that latent back
to RGB (matching the distribution the probe will see at RL time),
label the decoded RGB with the CV pipeline, and store (latent, label)
pairs grouped by split and tagged with episode id.

Quality filters (per design doc §1.6):
- success=False from the CV pipeline   -> drop, reason=cv_failed
- contour_count == 0                    -> drop, reason=no_contour
- contour_area < 100                    -> drop, reason=contour_too_small
                                          (also reflected in cv_failed above)
- contour_area > 2 * theoretical        -> drop, reason=contour_too_large
- icp_residual > RESIDUAL_MAX           -> drop, reason=icp_diverged

RESIDUAL_MAX = max(p95_calibration * 1.5, 0.5) — floors a too-tight
threshold from dropping valid in-distribution frames.

Outputs in outputs/state_probe/labels/:
- labels_train.pt, labels_val.pt : dicts per design doc §1.6
- drops.csv                      : one row per dropped frame
- meta.json                      : counts, RESIDUAL_MAX, preset, wall time
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

from rl.labeling.cv_labeler import CVLabeler  # noqa: E402
from rl.models.world_model import DifferentiableDynamics  # noqa: E402

# Constants
CKPT_PATH = REPO_ROOT / "outputs" / "pusht_cam1" / "checkpoints" / "best.ckpt"
DATA_ROOT = REPO_ROOT / "data" / "mini" / "pusht"
OUT_DIR = REPO_ROOT / "outputs" / "state_probe" / "labels"

PRESET = "REAL"
RESOLUTION = 128
OBS_KEY = "camera_1_color"
DEVICE = "cuda:0"

ENCODE_BATCH = 32
DECODE_BATCH = 8       # conservative: WM decode activations are the spiky term

# From per_channel_relaxation.csv: p95 ICP residual = 0.4361
# RESIDUAL_MAX = max(p95 * 1.5, 0.5)
RESIDUAL_MAX = max(0.4361 * 1.5, 0.5)  # = 0.65415

# Theoretical T-block area at 128 canvas = 471.8 px; allow up to 2x
# before declaring a contour an obvious false positive.
CONTOUR_AREA_MAX = 2.0 * 471.8  # = 943.6


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _center_crop_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(h, w)
    return img[(h - s) // 2 : (h - s) // 2 + s, (w - s) // 2 : (w - s) // 2 + s]


def _preprocess_rgb(raw_rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 -> (128, 128, 3) float32 in [0, 1]."""
    cropped = _center_crop_square(raw_rgb)
    resized = cv2.resize(cropped, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def _to_rgb_u8(img_f32_nhwc: torch.Tensor) -> np.ndarray:
    """(B, 3, H, W) float [0, 1] -> (B, H, W, 3) uint8 RGB, on CPU."""
    arr = img_f32_nhwc.permute(0, 2, 3, 1).detach().cpu().float().numpy()
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def label_one_episode(
    wm: DifferentiableDynamics,
    labeler: CVLabeler,
    ep_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[dict]]:
    """Label every frame in one episode.

    Returns (latents, labels, kept_t_idx, dropped_records)
      latents: (N_kept, 4, 32, 32) float32
      labels:  (N_kept, 4) float32 with columns [cx, cy, sin, cos]
      kept_t_idx: list of original frame indices that survived filters
      dropped_records: list of dicts for drops.csv
    """
    with h5py.File(ep_path, "r") as f:
        frames = f[f"obs/images/{OBS_KEY}"][()]  # (T, H, W, 3) uint8 RGB

    T = frames.shape[0]
    preprocessed = np.stack([_preprocess_rgb(frames[t]) for t in range(T)], axis=0)
    # (T, 128, 128, 3) -> (T, 3, 128, 128) tensor
    frames_tensor = torch.from_numpy(preprocessed).permute(0, 3, 1, 2).contiguous()

    # -------------------- encode all frames --------------------
    latents_list: list[torch.Tensor] = []
    for i in range(0, T, ENCODE_BATCH):
        batch = frames_tensor[i : i + ENCODE_BATCH].to(DEVICE)
        with torch.no_grad():
            z = wm.encode(batch)
        latents_list.append(z.detach().cpu().float())
    latents_all = torch.cat(latents_list, dim=0)  # (T, 4, 32, 32)

    # -------------------- decode + label -----------------------
    kept_latents: list[torch.Tensor] = []
    kept_labels: list[torch.Tensor] = []
    kept_t_idx: list[int] = []
    dropped: list[dict] = []

    for i in range(0, T, DECODE_BATCH):
        z_batch = latents_all[i : i + DECODE_BATCH].to(DEVICE)
        with torch.no_grad():
            rgb_batch = wm.decode(z_batch, RESOLUTION)  # (b, 3, 128, 128) in [0,1]
        rgb_u8 = _to_rgb_u8(rgb_batch)

        for j, frame in enumerate(rgb_u8):
            t = i + j
            if t >= T:
                break
            result = labeler.label(frame)

            drop_reason = None
            if not result.success:
                # CV returned None; bucket by whichever sub-reason applies
                if result.contour_count == 0:
                    drop_reason = "no_contour"
                elif result.contour_area < 100:
                    drop_reason = "contour_too_small"
                else:
                    drop_reason = "cv_failed"
            elif result.contour_area > CONTOUR_AREA_MAX:
                drop_reason = "contour_too_large"
            elif result.icp_residual > RESIDUAL_MAX:
                drop_reason = "icp_diverged"

            if drop_reason is not None:
                dropped.append({
                    "t_idx": t,
                    "reason": drop_reason,
                    "cv_success": int(result.success),
                    "contour_area": result.contour_area,
                    "icp_residual": (result.icp_residual if result.success else float("nan")),
                })
                continue

            kept_latents.append(latents_all[t])
            kept_labels.append(torch.tensor([
                result.cx, result.cy, result.sin_theta, result.cos_theta
            ], dtype=torch.float32))
            kept_t_idx.append(t)

    if kept_latents:
        lat_t = torch.stack(kept_latents, dim=0)
        lab_t = torch.stack(kept_labels, dim=0)
    else:
        lat_t = torch.empty((0, 4, 32, 32), dtype=torch.float32)
        lab_t = torch.empty((0, 4), dtype=torch.float32)

    return lat_t, lab_t, kept_t_idx, dropped


def label_split(
    split: str,
    wm: DifferentiableDynamics,
    labeler: CVLabeler,
    ckpt_mtime: float,
) -> dict:
    split_dir = DATA_ROOT / split
    episodes = sorted(split_dir.glob("episode_*.hdf5"))
    assert episodes, f"no episodes in {split_dir}"

    all_latents: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_episode_ids: list[int] = []
    all_t_idx: list[int] = []
    all_drops: list[dict] = []

    per_ep_stats: list[dict] = []
    t_start = time.time()

    for ep_path in episodes:
        ep_id = int(ep_path.stem.split("_")[1])
        lat, lab, kept_t, drops = label_one_episode(wm, labeler, ep_path)

        n_total = lat.shape[0] + len(drops)
        n_drop = len(drops)
        drop_rate = n_drop / n_total if n_total else 0.0

        print(
            f"  ep{ep_id}: kept {lat.shape[0]}/{n_total}  "
            f"(drop {n_drop}, {drop_rate*100:.1f}%)"
        )
        per_ep_stats.append({
            "episode": ep_id,
            "n_total": n_total,
            "n_kept": lat.shape[0],
            "n_dropped": n_drop,
            "drop_rate": drop_rate,
        })

        all_latents.append(lat)
        all_labels.append(lab)
        all_episode_ids.extend([ep_id] * lat.shape[0])
        all_t_idx.extend(kept_t)
        for d in drops:
            d["episode"] = ep_id
            d["split"] = split
        all_drops.extend(drops)

    wall_s = time.time() - t_start

    # assemble tensors
    if all_latents and sum(l.shape[0] for l in all_latents) > 0:
        latents = torch.cat(all_latents, dim=0)
        labels = torch.cat(all_labels, dim=0)
    else:
        latents = torch.empty((0, 4, 32, 32), dtype=torch.float32)
        labels = torch.empty((0, 4), dtype=torch.float32)

    ep_id_tensor = torch.tensor(all_episode_ids, dtype=torch.int64)
    t_idx_tensor = torch.tensor(all_t_idx, dtype=torch.int64)

    out = {
        "episode": ep_id_tensor,
        "t_idx": t_idx_tensor,
        "latents": latents,
        "labels": labels,
        "meta": {
            "preset": PRESET,
            "hsv_lower": labeler.hsv_lower.tolist(),
            "hsv_upper": labeler.hsv_upper.tolist(),
            "residual_max": RESIDUAL_MAX,
            "contour_area_max": CONTOUR_AREA_MAX,
            "resolution": RESOLUTION,
            "obs_key": OBS_KEY,
            "ckpt": str(CKPT_PATH),
            "ckpt_mtime": ckpt_mtime,
            "split": split,
            "n_total": sum(s["n_total"] for s in per_ep_stats),
            "n_kept": sum(s["n_kept"] for s in per_ep_stats),
            "n_dropped": sum(s["n_dropped"] for s in per_ep_stats),
            "drop_rate": (
                sum(s["n_dropped"] for s in per_ep_stats)
                / max(1, sum(s["n_total"] for s in per_ep_stats))
            ),
            "per_episode": per_ep_stats,
            "wall_s": wall_s,
        },
    }

    return out, all_drops


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    free, total = torch.cuda.mem_get_info(0)
    print(f"free GPU mem before WM load: {free / 1024**2:.0f} / {total / 1024**2:.0f} MiB")

    t0 = time.time()
    wm = DifferentiableDynamics(str(CKPT_PATH), device=DEVICE)
    print(f"WM loaded in {time.time() - t0:.1f}s")
    ckpt_mtime = CKPT_PATH.stat().st_mtime

    labeler = CVLabeler(preset=PRESET, resolution=RESOLUTION)
    print(f"labeler preset={PRESET} HSV={labeler.hsv_lower.tolist()}-{labeler.hsv_upper.tolist()}")
    print(f"RESIDUAL_MAX={RESIDUAL_MAX:.4f}  CONTOUR_AREA_MAX={CONTOUR_AREA_MAX:.1f} px")

    all_drops: list[dict] = []
    split_metas: dict[str, dict] = {}
    wall_total_t0 = time.time()

    for split in ("train", "val"):
        print(f"\n--- labelling {split} split ---")
        out, drops = label_split(split, wm, labeler, ckpt_mtime)
        torch.save(out, OUT_DIR / f"labels_{split}.pt")
        split_metas[split] = out["meta"]
        all_drops.extend(drops)

    wall_total = time.time() - wall_total_t0

    # drops.csv
    with (OUT_DIR / "drops.csv").open("w", newline="") as f:
        cols = ["split", "episode", "t_idx", "reason",
                "cv_success", "contour_area", "icp_residual"]
        w = csv.writer(f); w.writerow(cols)
        for d in all_drops:
            w.writerow([d["split"], d["episode"], d["t_idx"], d["reason"],
                        d["cv_success"], f"{d['contour_area']:.2f}",
                        "" if (isinstance(d["icp_residual"], float) and np.isnan(d["icp_residual"])) else f"{d['icp_residual']:.4f}"])

    # meta.json
    with (OUT_DIR / "meta.json").open("w") as f:
        json.dump({
            "preset": PRESET,
            "residual_max": RESIDUAL_MAX,
            "contour_area_max": CONTOUR_AREA_MAX,
            "wall_total_s": wall_total,
            "splits": split_metas,
        }, f, indent=2)

    # ---- terminal summary ----
    print("\n" + "=" * 70)
    print("BULK LABELING SUMMARY")
    print("=" * 70)
    print(f"preset: {PRESET}")
    print(f"RESIDUAL_MAX: {RESIDUAL_MAX:.4f}")
    print(f"wall (total): {wall_total:.1f}s ({wall_total/60:.2f} min)")
    print()
    for split in ("train", "val"):
        m = split_metas[split]
        print(
            f"{split}: {m['n_kept']}/{m['n_total']} kept "
            f"(drop {m['n_dropped']}, {m['drop_rate']*100:.2f}%) "
            f"wall {m['wall_s']:.1f}s"
        )
        for s in m["per_episode"]:
            print(
                f"   ep{s['episode']}: {s['n_kept']}/{s['n_total']} kept "
                f"(drop {s['n_dropped']}, {s['drop_rate']*100:.2f}%)"
            )
    print()
    print(f"outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
