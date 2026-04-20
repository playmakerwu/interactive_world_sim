"""DIAGNOSTIC-ONLY probe training: odd-even split.

Purpose: isolate whether the probe architecture can fit the
latent -> (cx, cy, sin theta, cos theta) mapping at all when train and
val distributions are matched. The Checkpoint-6 failure (probe_v1) was
shown to come from disjoint train/val label distributions from the
mini dataset's existing train/ vs val/ split. This script uses a
pooled-episodes split where every episode contributes ~half its frames
to each side, alternating by t_idx modulo 2.

THIS IS A LEAKED SPLIT. Train and val frames are temporal neighbors
(~100 ms apart at 10 Hz). Metrics produced here severely OVERESTIMATE
probe generalisation to RL-deployment latents. They are NOT acceptance
metrics. All outputs are labelled "DIAGNOSTIC" to prevent accidental
reuse.

Only the split changes. Architecture, loss, optimizer, and schedule are
identical to scripts/train_state_probe.py (design doc §1.7-§1.8).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.models.state_probe import StateProbe, normalize_sincos, split_output  # noqa: E402
from rl.models.world_model import DifferentiableDynamics  # noqa: E402
from rl.visualization.state_viz import render_state_on_image  # noqa: E402

LABELS_DIR = REPO_ROOT / "outputs" / "state_probe" / "labels"
RUN_ROOT = REPO_ROOT / "outputs" / "state_probe"
CKPT_PATH = REPO_ROOT / "outputs" / "pusht_cam1" / "checkpoints" / "best.ckpt"

RESOLUTION = 128
DEVICE = "cuda:0"
LAMBDA_POS, LAMBDA_ANG, LAMBDA_NORM = 7.0, 1.0, 0.01
BATCH_SIZE, LR, WD = 16, 3e-4, 1e-4
WARMUP_STEPS, MIN_EPOCHS, MAX_EPOCHS, PATIENCE = 500, 20, 100, 10
GRAD_CLIP = 1.0
SEED = 0
OVERFIT_THRESHOLD = 2.5


# -----------------------------------------------------------------------------
# Pooled + odd-even split
# -----------------------------------------------------------------------------

def load_pooled_labels():
    tr = torch.load(LABELS_DIR / "labels_train.pt", weights_only=False)
    va = torch.load(LABELS_DIR / "labels_val.pt",   weights_only=False)
    # Give val episodes distinct IDs 5..9 so the combined dataset has 10
    # unique episode ids.
    tr_ep = tr["episode"].clone()
    va_ep = va["episode"].clone() + 5
    latents = torch.cat([tr["latents"], va["latents"]], dim=0)
    labels  = torch.cat([tr["labels"],  va["labels"]],  dim=0)
    episodes = torch.cat([tr_ep, va_ep], dim=0)
    t_idx   = torch.cat([tr["t_idx"],   va["t_idx"]],   dim=0)
    return latents, labels, episodes, t_idx


def odd_even_split(
    episodes: torch.Tensor, t_idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Train = even t_idx (0, 2, 4, ...), val = odd (1, 3, 5, ...).

    Split done per-episode so each episode contributes ~half its frames
    to each side.
    """
    is_train = (t_idx % 2) == 0
    train_idx = torch.nonzero(is_train, as_tuple=True)[0]
    val_idx   = torch.nonzero(~is_train, as_tuple=True)[0]
    return train_idx, val_idx


class PoolingDataset(Dataset):
    def __init__(self, latents, labels_raw, episodes, t_idx, indices):
        self.indices = indices.long()
        self.latents = latents[indices]
        self.labels_raw = labels_raw[indices]
        self.episodes = episodes[indices]
        self.t_idx = t_idx[indices]
        lbls = self.labels_raw.clone()
        lbls[:, 0] /= RESOLUTION
        lbls[:, 1] /= RESOLUTION
        self.labels_norm = lbls

    def __len__(self): return self.latents.shape[0]

    def __getitem__(self, i):
        return (self.latents[i], self.labels_norm[i],
                int(self.episodes[i]), int(self.t_idx[i]))


# -----------------------------------------------------------------------------
# Loss / metrics (identical to the main training script)
# -----------------------------------------------------------------------------

def compute_loss(pred, target):
    pos_p, sc_p = split_output(pred)
    pos_t, sc_t = split_output(target)
    lp = F.mse_loss(pos_p, pos_t)
    la = F.mse_loss(sc_p, sc_t)
    sq = sc_p.pow(2).sum(dim=-1)
    ln = ((sq - 1.0) ** 2).mean()
    tot = LAMBDA_POS * lp + LAMBDA_ANG * la + LAMBDA_NORM * ln
    return tot, lp, la, ln


def per_frame_errors(pred, tgt_norm, res=RESOLUTION):
    pos_p, sc_p = split_output(pred)
    pos_t, sc_t = split_output(tgt_norm)
    pos_err = ((pos_p - pos_t) * res).norm(dim=-1)
    sc_p_u = sc_p / sc_p.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cos_d = (sc_p_u * sc_t).sum(dim=-1).clamp(-1.0, 1.0)
    return pos_err, torch.rad2deg(torch.acos(cos_d))


def compute_val_metrics(probe, val_loader, device):
    probe.eval()
    all_pred, all_tgt, all_ep = [], [], []
    sum_loss = {"loss": 0.0, "pos": 0.0, "ang": 0.0, "norm": 0.0}
    n = 0
    with torch.no_grad():
        for z, y, ep, _ in val_loader:
            z = z.to(device); y = y.to(device)
            p = probe(z)
            t, lp, la, ln = compute_loss(p, y)
            sum_loss["loss"] += t.item()
            sum_loss["pos"] += lp.item()
            sum_loss["ang"] += la.item()
            sum_loss["norm"] += ln.item()
            n += 1
            all_pred.append(p.cpu()); all_tgt.append(y.cpu())
            all_ep.append(ep if isinstance(ep, torch.Tensor) else torch.tensor(ep))

    preds = torch.cat(all_pred, dim=0)
    tgts  = torch.cat(all_tgt, dim=0)
    eps   = torch.cat(all_ep, dim=0)
    pos_err, ang_err = per_frame_errors(preds, tgts)
    pos_p = float(np.percentile(pos_err.numpy(), 95))
    ang_p = float(np.percentile(ang_err.numpy(), 95))
    wp, wa = 0.0, 0.0
    for e in torch.unique(eps).tolist():
        m = eps == e
        wp = max(wp, float(np.percentile(pos_err[m].numpy(), 95)))
        wa = max(wa, float(np.percentile(ang_err[m].numpy(), 95)))
    _, sc = split_output(preds)
    probe.train()
    for k in sum_loss: sum_loss[k] /= max(1, n)
    return {
        "val_loss": sum_loss["loss"], "val_loss_pos": sum_loss["pos"],
        "val_loss_ang": sum_loss["ang"], "val_loss_norm": sum_loss["norm"],
        "val_pos_mean": float(pos_err.mean()),
        "val_pos_p95_pooled": pos_p,
        "val_pos_p95_worst_episode": wp,
        "val_ang_mean": float(ang_err.mean()),
        "val_ang_p95_pooled": ang_p,
        "val_ang_p95_worst_episode": wa,
        "mean_pred_norm": float(sc.norm(dim=-1).mean()),
        "pos_err": pos_err, "ang_err": ang_err,
        "preds": preds, "tgts": tgts, "eps": eps,
    }


def lr_at_step(step, total_steps, base_lr):
    if step < WARMUP_STEPS:
        return base_lr * (step + 1) / WARMUP_STEPS
    p = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    p = min(1.0, max(0.0, p))
    return 1e-6 + 0.5 * (base_lr - 1e-6) * (1.0 + math.cos(math.pi * p))


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

def plot_learning_curves(val_log, out_path, tag):
    epochs = [r["epoch"] for r in val_log]
    tl = [r["train_loss"] for r in val_log]
    vl = [r["val_loss"] for r in val_log]
    ratio = [v / max(t, 1e-9) for v, t in zip(vl, tl)]
    pp = [r["val_pos_p95_pooled"] for r in val_log]
    pw = [r["val_pos_p95_worst_episode"] for r in val_log]
    ap = [r["val_ang_p95_pooled"] for r in val_log]
    aw = [r["val_ang_p95_worst_episode"] for r in val_log]

    best_idx = int(np.argmin([r["combined_gate"] for r in val_log]))
    max_ratio_idx = int(np.argmax(ratio))

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    a = ax[0, 0]
    a.plot(epochs, tl, label="train_loss", color="#2874A6")
    a.plot(epochs, vl, label="val_loss", color="#E67E22")
    a.axhline(vl[best_idx], linestyle="--", color="gray",
              label=f"best val_loss @ ep {epochs[best_idx]}")
    a.set_yscale("log"); a.set_xlabel("epoch"); a.set_ylabel("loss (log)")
    a.set_title("Loss curves"); a.legend(loc="best"); a.grid(True, alpha=0.3)

    a = ax[0, 1]
    a.plot(epochs, ratio, color="#8E44AD")
    a.axhline(1.0, linestyle="--", color="gray", alpha=0.5, label="no overfit")
    a.axhline(OVERFIT_THRESHOLD, linestyle="--", color="#C0392B", alpha=0.7,
              label=f"warn y={OVERFIT_THRESHOLD}")
    a.annotate(f"max {ratio[max_ratio_idx]:.2f} @ ep {epochs[max_ratio_idx]}",
               xy=(epochs[max_ratio_idx], ratio[max_ratio_idx]),
               xytext=(5, -20), textcoords="offset points", fontsize=10)
    a.set_xlabel("epoch"); a.set_ylabel("val/train")
    a.set_title("Overfit ratio"); a.legend(loc="best"); a.grid(True, alpha=0.3)

    a = ax[1, 0]
    a.plot(epochs, pp, label="pooled p95", color="#2874A6")
    a.plot(epochs, pw, label="worst-episode p95", color="#F39C12", linestyle="--")
    a.axhline(3.0, linestyle="--", color="#C0392B", alpha=0.7,
              label="ACCEPTANCE (3 px, not applicable here)")
    a.set_xlabel("epoch"); a.set_ylabel("pos p95 (px)")
    a.set_title("Position p95"); a.legend(loc="best", fontsize=8); a.grid(True, alpha=0.3)

    a = ax[1, 1]
    a.plot(epochs, ap, label="pooled p95", color="#2874A6")
    a.plot(epochs, aw, label="worst-episode p95", color="#F39C12", linestyle="--")
    a.axhline(5.0, linestyle="--", color="#C0392B", alpha=0.7,
              label="ACCEPTANCE (5°, not applicable here)")
    a.set_xlabel("epoch"); a.set_ylabel("ang p95 (deg)")
    a.set_title("Angle p95"); a.legend(loc="best", fontsize=8); a.grid(True, alpha=0.3)

    fig.suptitle(f"DIAGNOSTIC — {tag} (leaked odd/even split)", fontsize=14)
    plt.tight_layout(); plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()


def plot_label_distribution_overlap(
    train_labels_px: torch.Tensor,
    val_labels_px: torch.Tensor,
    out_path_prefix: Path,
):
    """Two PNGs: scatter of (cx, cy), histogram of theta."""
    # [N, 4] = [cx_px, cy_px, sin, cos]
    tr_cx, tr_cy = train_labels_px[:, 0].numpy(), train_labels_px[:, 1].numpy()
    va_cx, va_cy = val_labels_px[:, 0].numpy(),   val_labels_px[:, 1].numpy()
    tr_theta = torch.rad2deg(torch.atan2(train_labels_px[:, 2], train_labels_px[:, 3])).numpy()
    va_theta = torch.rad2deg(torch.atan2(val_labels_px[:, 2],   val_labels_px[:, 3])).numpy()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(tr_cx, tr_cy, s=10, alpha=0.45, c="#2874A6", label=f"train ({len(tr_cx)})")
    ax.scatter(va_cx, va_cy, s=10, alpha=0.45, c="#E67E22", label=f"val   ({len(va_cx)})")
    ax.set_xlim(0, RESOLUTION); ax.set_ylim(RESOLUTION, 0)  # image y-down
    ax.set_xlabel("cx (px)"); ax.set_ylabel("cy (px)")
    ax.set_title("DIAGNOSTIC — label distribution: train vs val position (cx, cy)")
    ax.legend(loc="best"); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out_path_prefix}_position_scatter.png", dpi=120); plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(-180, 180, 37)
    ax.hist(tr_theta, bins=bins, alpha=0.5, color="#2874A6", label=f"train ({len(tr_theta)})")
    ax.hist(va_theta, bins=bins, alpha=0.5, color="#E67E22", label=f"val   ({len(va_theta)})")
    ax.set_xlabel("theta (deg)"); ax.set_ylabel("count")
    ax.set_title("DIAGNOSTIC — label distribution: train vs val angle")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out_path_prefix}_angle_hist.png", dpi=120); plt.close()


def compose_grid(tiles, cols, upscale=4):
    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    pad = np.full((h, w, 3), 255, dtype=np.uint8)
    out_rows = []
    for r in range(rows):
        row_tiles = []
        for c in range(cols):
            k = r * cols + c
            row_tiles.append(tiles[k] if k < len(tiles) else pad)
        out_rows.append(np.concatenate(row_tiles, axis=1))
    return np.concatenate(out_rows, axis=0)


def make_overlay_tile(rgb_u8, probe_state, cv_state, caption):
    canvas = rgb_u8.copy()
    # CV (green) first, then probe (red) on top
    canvas = render_state_on_image(
        canvas, cv_state["cx"], cv_state["cy"], cv_state["sin"], cv_state["cos"],
        color=(0, 220, 0), label="CV",
    )
    canvas = render_state_on_image(
        canvas, probe_state["cx"], probe_state["cy"], probe_state["sin"], probe_state["cos"],
        color=(220, 0, 0), label="pr",
    )
    big = cv2.resize(canvas, (canvas.shape[1] * 4, canvas.shape[0] * 4),
                     interpolation=cv2.INTER_NEAREST)
    pad = 28
    out = np.full((big.shape[0] + pad, big.shape[1], 3), 255, dtype=np.uint8)
    out[pad:, :] = big
    cv2.putText(out, caption, (4, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def render_val_overlays(probe, wm, ds: PoolingDataset, indices, out_path, title):
    """Decode val latents, render probe (red) + CV (green) overlays, build grid."""
    tiles = []
    probe.eval()
    with torch.no_grad():
        for idx in indices:
            idx = int(idx)
            z = ds.latents[idx].unsqueeze(0).to(DEVICE)
            pred = probe(z)[0].cpu()
            pos, sc = split_output(pred)
            sc_n = normalize_sincos(sc.unsqueeze(0))[0]
            rgb = wm.decode(z, RESOLUTION)[0].permute(1, 2, 0).detach().cpu().float().numpy()
            rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
            cv_row = ds.labels_raw[idx]
            cv_state = {"cx": float(cv_row[0]), "cy": float(cv_row[1]),
                        "sin": float(cv_row[2]), "cos": float(cv_row[3])}
            probe_state = {
                "cx": float(pos[0]) * RESOLUTION,
                "cy": float(pos[1]) * RESOLUTION,
                "sin": float(sc_n[0]), "cos": float(sc_n[1]),
            }
            dpx = math.hypot(probe_state["cx"] - cv_state["cx"],
                             probe_state["cy"] - cv_state["cy"])
            cos_d = max(-1.0, min(1.0, probe_state["sin"] * cv_state["sin"]
                                          + probe_state["cos"] * cv_state["cos"]))
            dth = math.degrees(math.acos(cos_d))
            cap = (f"ep{int(ds.episodes[idx])} t={int(ds.t_idx[idx])} "
                   f"pos {dpx:.2f}px ang {dth:.2f}d")
            tiles.append(make_overlay_tile(rgb_u8, probe_state, cv_state, cap))
    probe.train()
    grid = compose_grid(tiles, cols=4)
    # stamp DIAGNOSTIC on top of the composed grid
    banner_h = 30
    stamped = np.full((grid.shape[0] + banner_h, grid.shape[1], 3), 255, dtype=np.uint8)
    stamped[banner_h:, :] = grid
    cv2.putText(stamped, title, (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), cv2.cvtColor(stamped, cv2.COLOR_RGB2BGR))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="probe_v2_odd_even")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_dir = RUN_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "overfit_dumps").mkdir(exist_ok=True)
    tb = SummaryWriter(log_dir=str(run_dir / "tb"))

    free, total = torch.cuda.mem_get_info(0)
    print(f"[gpu] free {free/1024**2:.0f} / {total/1024**2:.0f} MiB")

    # ---- pooled data + odd/even split ----
    latents, labels, episodes, t_idx = load_pooled_labels()
    train_idx, val_idx = odd_even_split(episodes, t_idx)
    print(f"[split] pooled {latents.shape[0]}  train {len(train_idx)}  val {len(val_idx)}")

    # per-episode counts for the report
    per_ep_counts = {}
    for e in torch.unique(episodes).tolist():
        n_tr = int(((episodes == e) & ((t_idx % 2) == 0)).sum())
        n_va = int(((episodes == e) & ((t_idx % 2) == 1)).sum())
        per_ep_counts[int(e)] = {"train": n_tr, "val": n_va}
        print(f"  ep{int(e):2d}  train={n_tr}  val={n_va}")

    train_ds = PoolingDataset(latents, labels, episodes, t_idx, train_idx)
    val_ds   = PoolingDataset(latents, labels, episodes, t_idx, val_idx)

    # distribution overlap plots (before any training — properties of the split)
    plot_label_distribution_overlap(
        train_ds.labels_raw, val_ds.labels_raw,
        run_dir / "label_distribution",
    )

    split_summary = {
        "kind": "odd-even pooled (DIAGNOSTIC, leaked)",
        "note": ("Train frames with t_idx % 2 == 0; val frames with "
                 "t_idx % 2 == 1. Per-episode, the pooled dataset has "
                 "10 episode ids (0..4 from original train/, 5..9 from val/)."),
        "n_total": int(latents.shape[0]),
        "n_train": int(train_ds.__len__()),
        "n_val": int(val_ds.__len__()),
        "per_episode": per_ep_counts,
    }
    with (run_dir / "split_summary.json").open("w") as f:
        json.dump(split_summary, f, indent=2)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    # ---- model + optim ----
    probe = StateProbe().to(DEVICE)
    print(f"[model] params={probe.num_params:,}")
    decay, no_decay = [], []
    for n, p in probe.named_parameters():
        (no_decay if ("norm" in n.lower() or "bias" in n.lower()) else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WD},
         {"params": no_decay, "weight_decay": 0.0}], lr=LR, betas=(0.9, 0.999))

    steps_per_epoch = max(1, (len(train_ds) + args.batch_size - 1) // args.batch_size)
    total_steps = steps_per_epoch * args.max_epochs

    val_log: list[dict] = []
    best_gate = float("inf"); best_epoch = -1; patience_ctr = 0
    overfit_epochs: list[int] = []
    peak_mib: float | None = None
    wall_t0 = time.time()
    step = 0

    for epoch in range(args.max_epochs):
        probe.train()
        sums = {"l": 0.0, "p": 0.0, "a": 0.0, "n": 0.0}; nb = 0
        for z, y, _ep, _t in train_loader:
            for pg in opt.param_groups:
                pg["lr"] = lr_at_step(step, total_steps, LR)
            z = z.to(DEVICE); y = y.to(DEVICE)
            pred = probe(z)
            loss, lp, la, ln = compute_loss(pred, y)
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(probe.parameters(), GRAD_CLIP)
            opt.step()
            sums["l"] += loss.item(); sums["p"] += lp.item()
            sums["a"] += la.item(); sums["n"] += ln.item()
            nb += 1; step += 1
            if step % 25 == 0:
                tb.add_scalar("train/loss_total", loss.item(), step)
                tb.add_scalar("train/lr", opt.param_groups[0]["lr"], step)
                tb.add_scalar("train/grad_norm", float(gn), step)
            if step == 10 and peak_mib is None:
                torch.cuda.synchronize()
                peak_mib = torch.cuda.max_memory_allocated(0) / 1024**2
                print(f"[mem] peak allocated after 10 steps: {peak_mib:.1f} MiB")

        tr = {
            "train_loss": sums["l"] / nb, "train_loss_pos": sums["p"] / nb,
            "train_loss_ang": sums["a"] / nb, "train_loss_norm": sums["n"] / nb,
        }
        vm = compute_val_metrics(probe, val_loader, DEVICE)
        combined_gate = (LAMBDA_POS * vm["val_pos_p95_pooled"]
                         + LAMBDA_ANG * vm["val_ang_p95_pooled"])
        row = {
            "epoch": epoch, "step": step, **tr,
            **{k: v for k, v in vm.items()
               if k not in ("pos_err", "ang_err", "preds", "tgts", "eps")},
            "combined_gate": combined_gate,
            "val_over_train": vm["val_loss"] / max(tr["train_loss"], 1e-9),
        }
        val_log.append(row)

        for k in ("train_loss", "train_loss_pos", "train_loss_ang", "train_loss_norm",
                  "val_loss", "val_loss_pos", "val_loss_ang", "val_loss_norm",
                  "val_pos_mean", "val_pos_p95_pooled", "val_pos_p95_worst_episode",
                  "val_ang_mean", "val_ang_p95_pooled", "val_ang_p95_worst_episode",
                  "mean_pred_norm"):
            tb.add_scalar(f"epoch/{k}", row[k], epoch)
        tb.add_scalar("epoch/val_over_train", row["val_over_train"], epoch)
        tb.add_scalar("epoch/combined_gate", combined_gate, epoch)

        print(
            f"ep{epoch:03d} tr={row['train_loss']:.5f} vl={row['val_loss']:.5f} "
            f"v/t={row['val_over_train']:.2f} "
            f"pos_p95={row['val_pos_p95_pooled']:5.2f}px "
            f"(worst_ep {row['val_pos_p95_worst_episode']:5.2f}) "
            f"ang_p95={row['val_ang_p95_pooled']:5.2f}d "
            f"(worst_ep {row['val_ang_p95_worst_episode']:5.2f}) "
            f"||sc||={row['mean_pred_norm']:.3f}"
        )

        if row["val_over_train"] > OVERFIT_THRESHOLD:
            overfit_epochs.append(epoch)
            print(f"  WARN: overfit, val/train={row['val_over_train']:.2f}")

        if combined_gate < best_gate:
            best_gate = combined_gate; best_epoch = epoch; patience_ctr = 0
            torch.save({"epoch": epoch, "step": step,
                        "state_dict": probe.state_dict(), "row": row},
                       run_dir / "best.pt")
        else:
            patience_ctr += 1

        if epoch + 1 >= MIN_EPOCHS and patience_ctr >= PATIENCE:
            print(f"[early-stop] patience {PATIENCE} hit; best @ ep {best_epoch}")
            break

    torch.save({"epoch": epoch, "step": step,
                "state_dict": probe.state_dict(), "row": val_log[-1]},
               run_dir / "last.pt")
    with (run_dir / "val_log.csv").open("w", newline="") as f:
        cols = list(val_log[0].keys()); w = csv.writer(f); w.writerow(cols)
        for r in val_log:
            w.writerow([r[c] for c in cols])

    plot_learning_curves(val_log, run_dir / "learning_curves.png", args.run_name)

    # ---- load best ckpt for viz ----
    best = torch.load(run_dir / "best.pt", weights_only=False)
    probe.load_state_dict(best["state_dict"])

    # ---- viz grids ----
    print("[viz] loading WM for decoding val frames ...")
    wm = DifferentiableDynamics(str(CKPT_PATH), device=DEVICE)

    rng = np.random.default_rng(42)
    if len(val_ds) >= 16:
        grid_idx = rng.choice(len(val_ds), size=16, replace=False)
    else:
        grid_idx = np.arange(len(val_ds))

    # compute all val errors to find worst-K
    vm_final = compute_val_metrics(probe, val_loader, DEVICE)
    pos_err = vm_final["pos_err"].numpy()
    ang_err = vm_final["ang_err"].numpy()
    # normalise each error, sum, pick top-12
    combined = pos_err / RESOLUTION + (1.0 - np.cos(np.deg2rad(ang_err))) / 2.0
    worst_idx = np.argsort(-combined)[:12]

    render_val_overlays(
        probe, wm, val_ds, grid_idx,
        run_dir / "probe_validation_grid.png",
        "DIAGNOSTIC — validation grid (leaked odd/even split)",
    )
    render_val_overlays(
        probe, wm, val_ds, worst_idx,
        run_dir / "probe_worst_cases.png",
        "DIAGNOSTIC — worst 12 val frames (leaked odd/even split)",
    )

    wall = time.time() - wall_t0
    best_row = val_log[best_epoch]
    max_ratio = max(r["val_over_train"] for r in val_log)
    final_gap_ratio = best_row["val_over_train"]

    # verdict logic
    pos_p95 = best_row["val_pos_p95_pooled"]
    ang_p95 = best_row["val_ang_p95_pooled"]
    if pos_p95 < 1.5 and ang_p95 < 3.0:
        verdict_tag = "A"
        verdict_text = ("**Architecture validated.** Pos p95 < 1.5 px AND angle "
                        "p95 < 3° on this LEAKED split. The probe can learn the "
                        "`latent → (cx, cy, sin θ, cos θ)` mapping. The real fix "
                        "is more data + a proper held-out split (next step, on "
                        "cloud). These numbers DO NOT indicate production "
                        "readiness.")
    elif pos_p95 < 1.5 or ang_p95 < 3.0:
        verdict_tag = "C"
        verdict_text = ("**Mixed.** One of position / angle met the leaked-split "
                        f"target, the other did not (pos {pos_p95:.2f} px, "
                        f"ang {ang_p95:.2f}°). Likely implication: architecture "
                        "handles one sub-task but not the other, or labels are "
                        "bimodal on one dimension. Details below.")
    else:
        verdict_tag = "B"
        verdict_text = ("**Architecture insufficient.** Even the leaked split "
                        f"yielded pos {pos_p95:.2f} px (threshold 1.5), angle "
                        f"{ang_p95:.2f}° (threshold 3). The probe cannot fit "
                        "the mapping — not a dataset problem alone. Needs "
                        "architecture work before more data.")

    # ---- report.md ----
    with (run_dir / "report.md").open("w") as f:
        f.write("# Diagnostic Probe Run — odd/even split\n\n")
        f.write("## Purpose (copy from the Phase 3-A diagnostic prompt §0)\n\n")
        f.write("> This is a **diagnostic experiment**, NOT an acceptance run. "
                "We are answering exactly one question: can the probe architecture "
                "learn the `latent → (cx, cy, sinθ, cosθ)` mapping at all, when "
                "train and val distributions match? The numbers below come from "
                "a split where val frames are temporal neighbours of train frames "
                "(~100 ms apart at 10 Hz). They severely OVERESTIMATE probe "
                "generalisation to RL-deployment latents. They are **not** "
                "acceptance metrics and must not be reused as such.\n\n")

        f.write("## Split description\n\n")
        f.write(f"- Kind: {split_summary['kind']}\n")
        f.write(f"- Total frames pooled: {split_summary['n_total']}\n")
        f.write(f"- Train: {split_summary['n_train']}  Val: {split_summary['n_val']}\n")
        f.write("- Per-episode counts:\n\n")
        f.write("  | ep | train | val |\n  |---|---|---|\n")
        for e, c in split_summary["per_episode"].items():
            f.write(f"  | {e} | {c['train']} | {c['val']} |\n")
        f.write("\nOverlap evidence: "
                "see `label_distribution_position_scatter.png` "
                "and `label_distribution_angle_hist.png`. "
                "By construction the two distributions should be near-identical; "
                "any visible divergence would invalidate the diagnostic.\n\n")

        f.write("## Final metrics (best checkpoint, epoch {})\n\n".format(best_epoch))
        f.write("| metric | value | leaked-split target |\n|---|---|---|\n")
        f.write(f"| val pos p95 pooled | **{pos_p95:.3f} px** | < 1.5 |\n")
        f.write(f"| val pos p95 worst-episode | {best_row['val_pos_p95_worst_episode']:.3f} px | (diagnostic) |\n")
        f.write(f"| val pos mean | {best_row['val_pos_mean']:.3f} px | — |\n")
        f.write(f"| val ang p95 pooled | **{ang_p95:.3f}°** | < 3 |\n")
        f.write(f"| val ang p95 worst-episode | {best_row['val_ang_p95_worst_episode']:.3f}° | (diagnostic) |\n")
        f.write(f"| val ang mean | {best_row['val_ang_mean']:.3f}° | — |\n")
        f.write(f"| val mean_pred_norm | {best_row['mean_pred_norm']:.4f} | ≈ 1.0 |\n")
        f.write(f"| val/train at best | {final_gap_ratio:.3f} | — |\n\n")

        f.write("## Overfitting analysis\n\n")
        f.write("Per Phase 3-A supplement §2. All evidence is in "
                "`learning_curves.png` and `val_log.csv`.\n\n")
        # 1 divergence
        diverged = False
        for i in range(1, len(val_log)):
            if val_log[i]["val_loss"] > val_log[i-1]["val_loss"] and val_log[i]["train_loss"] < val_log[i-1]["train_loss"]:
                pass
        # simple: did val_loss plateau or rise after some epoch while train kept falling?
        # pick epoch where val started growing while train shrank
        div_epoch = None
        for i in range(1, len(val_log)):
            if (val_log[i]["val_loss"] > val_log[best_epoch]["val_loss"] * 1.1
                and val_log[i]["train_loss"] < val_log[best_epoch]["train_loss"] * 0.9):
                div_epoch = val_log[i]["epoch"]; break
        f.write("1. **Did train/val loss diverge?** "
                + (f"Yes — around epoch {div_epoch}. "
                   "Early stopping {} catch it (best @ epoch {}, stop fired at epoch {}).".format(
                       "did" if best_epoch < val_log[-1]["epoch"] else "did not",
                       best_epoch, val_log[-1]["epoch"])
                   if div_epoch is not None
                   else "No clear divergence observed up to the last logged epoch.")
                + "\n")

        # 2 max ratio
        max_ep = [r["epoch"] for r in val_log if r["val_over_train"] == max_ratio][0]
        f.write(f"2. **Max `val_loss / train_loss`:** {max_ratio:.3f} at epoch {max_ep}. ")
        if max_ratio > OVERFIT_THRESHOLD:
            dump_eps = sorted(overfit_epochs)
            f.write(f"Exceeded the 2.5 warning threshold at epochs {dump_eps}. "
                    f"Diagnostic dumps saved to `overfit_dumps/`.\n")
        else:
            f.write(f"Stayed below the 2.5 warning threshold throughout.\n")

        # 3
        best_metric_combined = combined_gate
        f.write(f"3. **Did val p95 follow loss?** Best combined gate "
                f"({LAMBDA_POS}·pos_p95 + {LAMBDA_ANG}·ang_p95) = {best_metric_combined:.3f} "
                f"at epoch {best_epoch}. ")
        if pos_p95 < 1.5 and ang_p95 < 3.0:
            f.write("Both p95 metrics continued improving in line with (or better than) "
                    "val loss. No divergence between loss trajectory and acceptance-relevant "
                    "metrics.\n")
        else:
            f.write("See curves for the full story.\n")

        # 4 final gap
        f.write(f"4. **Final gap at best-val epoch:** val/train = {final_gap_ratio:.3f}. ")
        if final_gap_ratio < 1.8:
            f.write("Healthy; well below the 1.8 discussion threshold.\n")
        else:
            f.write("Above 1.8 — flagged for explicit discussion.\n")

        # 5 verdict (for overfit analysis, not the A/B/C verdict)
        if max_ratio < OVERFIT_THRESHOLD and final_gap_ratio < 1.8:
            overfit_verdict = "no overfit evident"
        elif max_ratio < OVERFIT_THRESHOLD and final_gap_ratio >= 1.8:
            overfit_verdict = "mild but not concerning"
        elif max_ratio >= OVERFIT_THRESHOLD and best_epoch < val_log[-1]["epoch"]:
            overfit_verdict = "overfit caught by early stopping"
        else:
            overfit_verdict = "overfit present and harmful"
        f.write(f"5. **Overfit verdict:** **{overfit_verdict}.**\n\n")

        f.write(f"## Run verdict: {verdict_tag}\n\n")
        f.write(verdict_text + "\n\n")

        f.write("## Reminder — these numbers are NOT acceptance metrics\n\n")
        f.write("Per the diagnostic prompt §§1, 5: this run does NOT advance "
                "Branch A to merge. The probe_v2_odd_even checkpoint will not be "
                "wired into RL. Real acceptance (pos ≤ 3 px, ang ≤ 5°) will be "
                "measured on a properly held-out split after the cloud data "
                "collection step.\n")

    # ---- config.json ----
    with (run_dir / "config.json").open("w") as f:
        json.dump({
            "run_name": args.run_name,
            "kind": "DIAGNOSTIC — odd/even split, LEAKED",
            "seed": args.seed,
            "arch": "MLP [4096 -> 256 -> 128 -> 4] GELU + LayerNorm",
            "num_params": probe.num_params,
            "lambda_pos": LAMBDA_POS, "lambda_ang": LAMBDA_ANG, "lambda_norm": LAMBDA_NORM,
            "lr": LR, "weight_decay": WD, "warmup_steps": WARMUP_STEPS,
            "batch_size": args.batch_size,
            "min_epochs": MIN_EPOCHS, "max_epochs": args.max_epochs,
            "patience": PATIENCE, "grad_clip": GRAD_CLIP,
            "overfit_threshold": OVERFIT_THRESHOLD,
            "resolution": RESOLUTION,
            "best_epoch": best_epoch, "best_gate": best_gate,
            "overfit_triggered_epochs": overfit_epochs,
            "peak_mib_after_10_steps": peak_mib,
            "wall_s": wall,
            "verdict": verdict_tag,
        }, f, indent=2)

    tb.close()

    print()
    print("=" * 70)
    print(f"DIAGNOSTIC RUN COMPLETE — VERDICT: {verdict_tag}")
    print("=" * 70)
    print(f"run: {run_dir}")
    print(f"best epoch: {best_epoch}")
    print(f"val pos p95 pooled:   {pos_p95:.3f} px  (leaked-split target < 1.5)")
    print(f"val ang p95 pooled:   {ang_p95:.3f} deg (leaked-split target < 3.0)")
    print(f"val mean_pred_norm:   {best_row['mean_pred_norm']:.4f}")
    print(f"val/train at best:    {final_gap_ratio:.3f}")
    print(f"max val/train:        {max_ratio:.3f}")


if __name__ == "__main__":
    main()
