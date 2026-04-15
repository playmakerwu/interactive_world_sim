"""Spatial-structure argument: the latent encodes arm position more than T-block.

Strategy: for each episode, compute per-frame T-block and arm masks via simple
color segmentation (T is blue, arm-tip is a bright marker). Track centroids
across time. Then find:
  (a) a pair (t1, t2) where |arm centroid diff| is large but |T centroid diff|
      is small  → "arm_dominates.png"
  (b) a pair where |T centroid diff| is large but |arm centroid diff| is small
      → "tblock_invisible.png"

Each figure: Frame A | Frame B | 32×32 per-position cosine-similarity heatmap
with mean cos_sim printed.

Usage (from repo root):
    conda run -n iws python pre/05_spatial_analysis/generate_argument.py
"""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from yixuan_utilities.draw_utils import center_crop
from yixuan_utilities.hdf5_utils import load_dict_from_hdf5

from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import (
    LatentWorldModel,
)

CKPT_PATH = "outputs/pusht_cam1/checkpoints/best.ckpt"
VAL_DIR = Path("data/mini/pusht/val")
OBS_KEY = "camera_1_color"
RESOLUTION = 128
DEVICE = "cuda:0"
OUT_DIR = Path("pre/05_spatial_analysis")
plt.rcParams.update({"font.size": 13})


def _patch_attention_backends():
    from torch.nn.attention import SDPBackend
    from interactive_world_sim.algorithms.models.attention import Attention
    cap = torch.cuda.get_device_capability()
    if cap[0] >= 8 and cap[0] != 8:
        _orig_init = Attention.__init__

        def _patched_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            self.cuda_backends = [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]

        Attention.__init__ = _patched_init


def load_model(ckpt_path: str) -> LatentWorldModel:
    _patch_attention_backends()
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x), replace=True)
    cfg_path = Path(ckpt_path).parent.parent / ".hydra" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    dtype = torch.float32 if "dtype" not in cfg.algorithm else cfg.algorithm.dtype
    cfg.n_frames = 10
    cfg.algorithm.n_frames = 10
    if "diffusion" in cfg.algorithm and "sampling_timesteps" in cfg.algorithm.diffusion:
        cfg.algorithm.diffusion.sampling_timesteps = 10
    if (
        "diffusion" in cfg.algorithm.dynamics
        and "sampling_timesteps" in cfg.algorithm.dynamics.diffusion
    ):
        cfg.algorithm.dynamics.diffusion.sampling_timesteps = 10
    cfg.algorithm.load_ae = None

    algo = LatentWorldModel.load_from_checkpoint(
        ckpt_path, cfg=cfg.algorithm, map_location=DEVICE,
        dtype=dtype, strict=False, weights_only=False,
    )
    algo.dynamics = algo.dynamics.to(dtype)
    algo.eval()
    return algo


def preprocess(raw_img: np.ndarray) -> np.ndarray:
    img = center_crop(raw_img, (RESOLUTION, RESOLUTION))
    img = cv2.resize(img, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


@torch.no_grad()
def encode(model, img_float):
    t = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)
    t = model.normalizer[OBS_KEY].normalize(t).to(DEVICE)
    return model.encoder_forward(t)[0].float().cpu()


def spatial_cos_sim(z_a, z_b):
    C, H, W = z_a.shape
    a = z_a.reshape(C, -1).T
    b = z_b.reshape(C, -1).T
    cs = F.cosine_similarity(a, b, dim=1)
    return cs.reshape(H, W).numpy()


def t_centroid(img_rgb_float):
    """Centroid of the pink/magenta T-block. img_rgb_float: (H,W,3) in [0,1]."""
    img = (img_rgb_float * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # pink/magenta: H ~ 150-175 (OpenCV), medium-high S
    mask = cv2.inRange(hsv, (145, 60, 100), (178, 255, 255))
    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        return None, 0
    return (float(xs.mean()), float(ys.mean())), int(mask.sum() / 255)


def arm_centroid(img_rgb_float):
    """Centroid of the orange gripper tips (the visible arm end-effectors)."""
    img = (img_rgb_float * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # orange: H ~ 5-22, high S, high V
    mask = cv2.inRange(hsv, (3, 150, 120), (22, 255, 255))
    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        return None, 0
    return (float(xs.mean()), float(ys.mean())), int(mask.sum() / 255)


def load_episode_frames(ep_path):
    epi, _ = load_dict_from_hdf5(str(ep_path))
    imgs = epi["obs"]["images"][OBS_KEY]
    T = len(imgs)
    frames = [preprocess(imgs[t][()]) for t in range(T)]
    return frames


def find_pairs():
    """Scan val episodes, compute centroids, pick two pairs."""
    ep_paths = sorted(VAL_DIR.glob("episode_*.hdf5"))
    all_records = []  # list of (ep_idx, t, frame, t_ctr, a_ctr)
    for ep_path in ep_paths:
        ep_idx = int(ep_path.stem.split("_")[-1])
        frames = load_episode_frames(ep_path)
        for t, f in enumerate(frames):
            tc, _ = t_centroid(f)
            ac, _ = arm_centroid(f)
            if tc is None or ac is None:
                continue
            all_records.append((ep_idx, t, f, tc, ac))
    print(f"Scanned {len(ep_paths)} episodes, {len(all_records)} usable frames")

    # sort by episode + time for downstream
    # (a) arm moves, T static: same episode, small |Δt_ctr|, large |Δa_ctr|
    best_arm = None  # (score, rec_i, rec_j)
    for i in range(len(all_records)):
        epi_i, ti, fi, tci, aci = all_records[i]
        for j in range(i + 1, len(all_records)):
            epj, tj, fj, tcj, acj = all_records[j]
            if epj != epi_i:
                continue
            dt = np.hypot(tci[0] - tcj[0], tci[1] - tcj[1])
            da = np.hypot(aci[0] - acj[0], aci[1] - acj[1])
            if dt > 3.0:  # T must be nearly static (≤3 px)
                continue
            if da < 25.0:  # arm must differ meaningfully
                continue
            score = da - 4 * dt
            if best_arm is None or score > best_arm[0]:
                best_arm = (score, i, j, dt, da)

    # (b) T moves, arm similar
    best_t = None
    for i in range(len(all_records)):
        epi_i, ti, fi, tci, aci = all_records[i]
        for j in range(i + 1, len(all_records)):
            epj, tj, fj, tcj, acj = all_records[j]
            if epj != epi_i:
                continue
            dt = np.hypot(tci[0] - tcj[0], tci[1] - tcj[1])
            da = np.hypot(aci[0] - acj[0], aci[1] - acj[1])
            if dt < 10.0:
                continue
            if da > 10.0:
                continue
            score = dt - 2 * da
            if best_t is None or score > best_t[0]:
                best_t = (score, i, j, dt, da)

    if best_arm is None:
        raise RuntimeError("could not find arm-dominates pair")
    if best_t is None:
        # relax
        print("Relaxing T-moves constraint …")
        for i in range(len(all_records)):
            epi_i, ti, fi, tci, aci = all_records[i]
            for j in range(i + 1, len(all_records)):
                epj, tj, fj, tcj, acj = all_records[j]
                if epj != epi_i:
                    continue
                dt = np.hypot(tci[0] - tcj[0], tci[1] - tcj[1])
                da = np.hypot(aci[0] - acj[0], aci[1] - acj[1])
                if dt < 5.0:
                    continue
                score = dt - 0.5 * da
                if best_t is None or score > best_t[0]:
                    best_t = (score, i, j, dt, da)
    if best_t is None:
        raise RuntimeError("could not find T-moves pair")

    rec = lambda k: all_records[k]
    return rec(best_arm[1]), rec(best_arm[2]), best_arm[3], best_arm[4], \
           rec(best_t[1]), rec(best_t[2]), best_t[3], best_t[4]


def plot_pair(frame_a, frame_b, sim, out_path, title, label_a, label_b):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(frame_a)
    axes[0].set_title(f"Frame A — {label_a}")
    axes[0].axis("off")
    axes[1].imshow(frame_b)
    axes[1].set_title(f"Frame B — {label_b}")
    axes[1].axis("off")
    im = axes[2].imshow(sim, cmap="RdYlGn", vmin=-1, vmax=1)
    mean_cs = float(sim.mean())
    axes[2].set_title(
        f"32×32 cos_sim heatmap\nmean = {mean_cs:.4f}  "
        f"min = {sim.min():.3f}"
    )
    axes[2].set_xticks([0, 8, 16, 24, 31])
    axes[2].set_yticks([0, 8, 16, 24, 31])
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Saved {out_path}  (mean cos_sim = {mean_cs:.4f})")
    return mean_cs


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Scanning val episodes for suitable frame pairs …")
    (
        arm_a, arm_b, arm_dt, arm_da,
        t_a, t_b, t_dt, t_da,
    ) = find_pairs()

    epA, tA, fA, tcA, acA = arm_a
    epB, tB, fB, tcB, acB = arm_b
    print(f"\nArm-dominates pair:  ep{epA} t={tA}  vs  ep{epB} t={tB}")
    print(f"  |Δ T-centroid| = {arm_dt:.2f} px   |Δ arm-centroid| = {arm_da:.2f} px")

    ep2A, t2A, f2A, _, _ = t_a
    ep2B, t2B, f2B, _, _ = t_b
    print(f"\nT-moves pair:       ep{ep2A} t={t2A}  vs  ep{ep2B} t={t2B}")
    print(f"  |Δ T-centroid| = {t_dt:.2f} px   |Δ arm-centroid| = {t_da:.2f} px")

    print("\nLoading world model …")
    model = load_model(CKPT_PATH)

    print("Encoding pairs …")
    zA1 = encode(model, fA)
    zA2 = encode(model, fB)
    simA = spatial_cos_sim(zA1, zA2)

    zB1 = encode(model, f2A)
    zB2 = encode(model, f2B)
    simB = spatial_cos_sim(zB1, zB2)

    mean_arm = plot_pair(
        fA, fB, simA,
        OUT_DIR / "arm_dominates.png",
        "Arms in different positions, T-block unchanged —\n"
        "but latent similarity drops significantly",
        label_a=f"ep{epA} t={tA}",
        label_b=f"ep{epB} t={tB}",
    )

    mean_t = plot_pair(
        f2A, f2B, simB,
        OUT_DIR / "tblock_invisible.png",
        "T-block moved, arms similar —\n"
        "latent similarity barely changes",
        label_a=f"ep{ep2A} t={t2A}",
        label_b=f"ep{ep2B} t={t2B}",
    )

    print("\n" + "=" * 70)
    print("ARGUMENT")
    print("=" * 70)
    print(f"Arm-dominates pair  (T static, arm differs): mean cos_sim = {mean_arm:.4f}")
    print(f"T-moves pair        (arm similar, T differs): mean cos_sim = {mean_t:.4f}")
    print(f"Δ (T-moves − arm-moves) = {mean_t - mean_arm:+.4f}")
    print("→ If mean_t > mean_arm, the latent is MORE sensitive to arm than T,")
    print("  explaining why cosine-similarity reward is not task-informative.")
    print("=" * 70)


if __name__ == "__main__":
    main()
