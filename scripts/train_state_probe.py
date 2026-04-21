"""Train the latent -> T-block-pose probe on the bulk-labeled dataset.

Per design doc §1.7-§1.8 revised + Phase 3-A supplement:
- MLP [4096 -> 256 -> 128 -> 4], GELU, LayerNorm, ~1.08M params.
- Loss: lambda_pos * MSE(pos_norm) + lambda_ang * MSE(sin_cos)
       + lambda_norm * ((sin^2 + cos^2) - 1)^2
  with lambda_pos=7, lambda_ang=1, lambda_norm=0.01.
- AdamW, lr=3e-4, cosine warmup 500 steps, weight_decay=1e-4.
- Batch 16, max 100 epochs, min 20, early-stop patience 10 on
  lambda_pos * val_pos_p95_pooled + lambda_ang * val_angle_p95_pooled.
- Overfit flag at val_loss / train_loss > 2.5 (warn + dump, continue).
- Logs per-epoch: train/val total + per-term losses, pooled + worst-
  episode p95 for position and angle, mean_pred_norm.
- Saves best.pt by pooled-p95 combined metric.
- Produces learning_curves.png (2x2 panel) after training per the
  Phase 3-A supplement.

Outputs: outputs/state_probe/<run_name>/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.models.state_probe import StateProbe, split_output  # noqa: E402

LABELS_DIR = REPO_ROOT / "outputs" / "state_probe" / "labels"
RUN_ROOT = REPO_ROOT / "outputs" / "state_probe"

RESOLUTION = 128
DEVICE = "cuda:0"

# Loss / reward weights (design doc §1.7)
LAMBDA_POS = 7.0
LAMBDA_ANG = 1.0
LAMBDA_NORM = 0.1

# Training config (design doc §1.8)
BATCH_SIZE = 16
LR = 3e-4
WD = 1e-4
WARMUP_STEPS = 500
MIN_EPOCHS = 20
MAX_EPOCHS = 100
PATIENCE = 10
GRAD_CLIP = 1.0
SEED = 0

# Monitoring
OVERFIT_THRESHOLD = 2.5  # val_loss / train_loss trigger


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class LabeledLatentDataset(Dataset):
    """Wraps the tensors saved by scripts/label_replay_buffer.py.

    Returns (latent, label_norm) where label_norm = [cx/RES, cy/RES,
    sin, cos] so the position component is already in [0, 1] for
    unit-free MSE."""

    def __init__(self, path: Path):
        d = torch.load(path, map_location="cpu", weights_only=False)
        self.latents = d["latents"].float()
        self.labels_raw = d["labels"].float()  # [cx_px, cy_px, sin, cos]
        self.episode = d["episode"].long()
        self.t_idx = d["t_idx"].long()
        self.meta = d["meta"]

        lbls = self.labels_raw.clone()
        lbls[:, 0] = lbls[:, 0] / RESOLUTION
        lbls[:, 1] = lbls[:, 1] / RESOLUTION
        self.labels_norm = lbls

    def __len__(self) -> int:
        return self.latents.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        return (
            self.latents[i],
            self.labels_norm[i],
            int(self.episode[i]),
            int(self.t_idx[i]),
        )


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------

def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (total, loss_pos, loss_ang, loss_norm), each a scalar tensor."""
    pos_pred, sincos_pred = split_output(pred)
    pos_tgt, sincos_tgt = split_output(target)
    loss_pos = F.mse_loss(pos_pred, pos_tgt)
    loss_ang = F.mse_loss(sincos_pred, sincos_tgt)
    # soft unit-norm: ((sin^2 + cos^2) - 1)^2, averaged over batch
    sq = sincos_pred.pow(2).sum(dim=-1)
    loss_norm = ((sq - 1.0) ** 2).mean()
    total = LAMBDA_POS * loss_pos + LAMBDA_ANG * loss_ang + LAMBDA_NORM * loss_norm
    return total, loss_pos, loss_ang, loss_norm


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def per_frame_errors(
    pred: torch.Tensor, target_norm: torch.Tensor, resolution: int = RESOLUTION
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (pos_err_px, ang_err_deg) per frame.

    Angle error via min(|dtheta|, 360 - |dtheta|) where dtheta is
    derived from (sin, cos) predictions and targets.
    """
    pos_pred, sincos_pred = split_output(pred)
    pos_tgt, sincos_tgt = split_output(target_norm)

    dpx = (pos_pred - pos_tgt) * resolution
    pos_err = torch.norm(dpx, dim=-1)

    # Normalise (sin, cos) to unit for honest angle comparison.
    s_p = sincos_pred / torch.clamp_min(torch.norm(sincos_pred, dim=-1, keepdim=True), 1e-8)
    s_t = sincos_tgt  # targets are already unit-norm
    cos_d = (s_p * s_t).sum(dim=-1).clamp(-1.0, 1.0)
    dtheta_deg = torch.rad2deg(torch.acos(cos_d))
    # dtheta_deg in [0, 180], already wrap-corrected.
    return pos_err, dtheta_deg


def compute_val_metrics(
    probe: nn.Module,
    val_loader: DataLoader,
    device: str,
) -> dict:
    probe.eval()
    all_pred, all_tgt, all_ep = [], [], []
    total_losses = {"loss": 0.0, "pos": 0.0, "ang": 0.0, "norm": 0.0}
    n_batches = 0
    with torch.no_grad():
        for z, y, ep, _t in val_loader:
            z = z.to(device)
            y = y.to(device)
            pred = probe(z)
            tot, lp, la, ln = compute_loss(pred, y)
            total_losses["loss"] += tot.item()
            total_losses["pos"]  += lp.item()
            total_losses["ang"]  += la.item()
            total_losses["norm"] += ln.item()
            n_batches += 1
            all_pred.append(pred.cpu())
            all_tgt.append(y.cpu())
            all_ep.append(ep.clone().detach() if isinstance(ep, torch.Tensor)
                           else torch.tensor(ep))

    preds = torch.cat(all_pred, dim=0)
    tgts = torch.cat(all_tgt, dim=0)
    eps = torch.cat(all_ep, dim=0)

    pos_err, ang_err = per_frame_errors(preds, tgts)

    # pooled p95
    pos_p95 = float(np.percentile(pos_err.numpy(), 95))
    ang_p95 = float(np.percentile(ang_err.numpy(), 95))
    # per-episode p95, then take the worst-episode value
    ep_worst_pos = 0.0
    ep_worst_ang = 0.0
    for ep_id in torch.unique(eps).tolist():
        mask = (eps == ep_id)
        ep_pos_p95 = float(np.percentile(pos_err[mask].numpy(), 95))
        ep_ang_p95 = float(np.percentile(ang_err[mask].numpy(), 95))
        ep_worst_pos = max(ep_worst_pos, ep_pos_p95)
        ep_worst_ang = max(ep_worst_ang, ep_ang_p95)

    _, sincos_pred = split_output(preds)
    mean_pred_norm = float(sincos_pred.norm(dim=-1).mean())

    for k in total_losses:
        total_losses[k] /= max(1, n_batches)

    probe.train()
    return {
        "val_loss": total_losses["loss"],
        "val_loss_pos": total_losses["pos"],
        "val_loss_ang": total_losses["ang"],
        "val_loss_norm": total_losses["norm"],
        "val_pos_mean": float(pos_err.mean()),
        "val_pos_p95_pooled": pos_p95,
        "val_pos_p95_worst_episode": ep_worst_pos,
        "val_ang_mean": float(ang_err.mean()),
        "val_ang_p95_pooled": ang_p95,
        "val_ang_p95_worst_episode": ep_worst_ang,
        "mean_pred_norm": mean_pred_norm,
    }


# -----------------------------------------------------------------------------
# LR schedule
# -----------------------------------------------------------------------------

def lr_at_step(step: int, total_steps: int, base_lr: float) -> float:
    if step < WARMUP_STEPS:
        return base_lr * (step + 1) / WARMUP_STEPS
    # cosine to min 1e-6
    progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    progress = min(1.0, max(0.0, progress))
    min_lr = 1e-6
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# -----------------------------------------------------------------------------
# Learning-curves plot (supplement)
# -----------------------------------------------------------------------------

def plot_learning_curves(val_log: list[dict], out_path: Path) -> None:
    """Produce the 2x2 panel PNG required by the Phase 3-A supplement."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [r["epoch"] for r in val_log]
    train_loss = [r["train_loss"] for r in val_log]
    val_loss = [r["val_loss"] for r in val_log]
    ratio = [v / max(t, 1e-9) for v, t in zip(val_loss, train_loss)]
    pos_p = [r["val_pos_p95_pooled"] for r in val_log]
    pos_w = [r["val_pos_p95_worst_episode"] for r in val_log]
    ang_p = [r["val_ang_p95_pooled"] for r in val_log]
    ang_w = [r["val_ang_p95_worst_episode"] for r in val_log]

    best_idx = int(np.argmin([r["combined_gate"] for r in val_log]))
    best_ep = epochs[best_idx]
    best_val = val_loss[best_idx]

    max_ratio_idx = int(np.argmax(ratio))
    max_ratio = ratio[max_ratio_idx]
    max_ratio_ep = epochs[max_ratio_idx]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # top-left: loss curves (log y)
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, label="train_loss", color="#2874A6")
    ax.plot(epochs, val_loss, label="val_loss", color="#E67E22")
    ax.axhline(best_val, linestyle="--", color="gray", alpha=0.7,
               label=f"best val_loss @ epoch {best_ep}")
    ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss (log)")
    ax.set_title("Loss curves")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3)

    # top-right: val/train ratio
    ax = axes[0, 1]
    ax.plot(epochs, ratio, color="#8E44AD")
    ax.axhline(1.0, linestyle="--", color="gray", alpha=0.5, label="no overfit (y=1)")
    ax.axhline(OVERFIT_THRESHOLD, linestyle="--", color="#C0392B", alpha=0.7,
               label=f"warn threshold (y={OVERFIT_THRESHOLD})")
    ax.annotate(
        f"max {max_ratio:.2f} @ ep {max_ratio_ep}",
        xy=(max_ratio_ep, max_ratio), xytext=(5, -20),
        textcoords="offset points", fontsize=10,
    )
    ax.set_xlabel("epoch"); ax.set_ylabel("val_loss / train_loss")
    ax.set_title("Overfit ratio")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3)

    # bottom-left: pos p95
    ax = axes[1, 0]
    ax.plot(epochs, pos_p, label="pooled p95", color="#2874A6")
    ax.plot(epochs, pos_w, label="worst-episode p95", color="#F39C12",
            linestyle="--")
    ax.axhline(3.0, linestyle="--", color="#C0392B", alpha=0.7,
               label="acceptance (3 px)")
    ax.set_xlabel("epoch"); ax.set_ylabel("pos error (px)")
    ax.set_title("Position p95")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3)

    # bottom-right: ang p95
    ax = axes[1, 1]
    ax.plot(epochs, ang_p, label="pooled p95", color="#2874A6")
    ax.plot(epochs, ang_w, label="worst-episode p95", color="#F39C12",
            linestyle="--")
    ax.axhline(5.0, linestyle="--", color="#C0392B", alpha=0.7,
               label="acceptance (5°)")
    ax.set_xlabel("epoch"); ax.set_ylabel("ang error (deg)")
    ax.set_title("Angle p95")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3)

    fig.suptitle(f"State probe — {out_path.parent.name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str,
                        default=f"run_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_dir = RUN_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "overfit_dumps").mkdir(exist_ok=True)
    tb_dir = run_dir / "tb"
    tb = SummaryWriter(log_dir=str(tb_dir))

    free, total = torch.cuda.mem_get_info(0)
    print(f"[gpu] free {free/1024**2:.0f} / {total/1024**2:.0f} MiB")

    # --- data ---
    train_ds = LabeledLatentDataset(LABELS_DIR / "labels_train.pt")
    val_ds   = LabeledLatentDataset(LABELS_DIR / "labels_val.pt")
    print(f"[data] train {len(train_ds)} val {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    # --- model + optim ---
    probe = StateProbe().to(DEVICE)
    print(f"[model] params={probe.num_params:,}")

    decay, no_decay = [], []
    for n, p in probe.named_parameters():
        (no_decay if ("norm" in n.lower() or "bias" in n.lower()) else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WD},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=LR, betas=(0.9, 0.999),
    )

    steps_per_epoch = max(1, (len(train_ds) + args.batch_size - 1) // args.batch_size)
    total_steps = steps_per_epoch * args.max_epochs

    # --- training loop ---
    train_log: list[dict] = []
    val_log: list[dict] = []

    best_gate = float("inf")
    best_epoch = -1
    patience_ctr = 0
    overfit_triggered_epochs: list[int] = []
    peak_mib_after_10_steps: float | None = None
    wall_t0 = time.time()
    step = 0

    for epoch in range(args.max_epochs):
        probe.train()
        sum_loss = sum_pos = sum_ang = sum_norm = 0.0
        n_batches = 0
        for z, y, _ep, _t in train_loader:
            for pg in opt.param_groups:
                pg["lr"] = lr_at_step(step, total_steps, LR)

            z = z.to(DEVICE)
            y = y.to(DEVICE)
            pred = probe(z)
            loss, lp, la, ln = compute_loss(pred, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(probe.parameters(), GRAD_CLIP)
            opt.step()

            sum_loss += loss.item()
            sum_pos  += lp.item()
            sum_ang  += la.item()
            sum_norm += ln.item()
            n_batches += 1
            step += 1

            # log per-step scalars to TB (sparser to avoid TB bloat)
            if step % 25 == 0:
                tb.add_scalar("train/loss_total", loss.item(), step)
                tb.add_scalar("train/loss_pos", lp.item(), step)
                tb.add_scalar("train/loss_ang", la.item(), step)
                tb.add_scalar("train/loss_norm", ln.item(), step)
                tb.add_scalar("train/lr", opt.param_groups[0]["lr"], step)
                tb.add_scalar("train/grad_norm", float(gn), step)

            if step == 10 and peak_mib_after_10_steps is None:
                torch.cuda.synchronize()
                peak_mib_after_10_steps = torch.cuda.max_memory_allocated(0) / 1024**2
                print(f"[mem] peak allocated after 10 steps: {peak_mib_after_10_steps:.1f} MiB")

        train_metrics = {
            "train_loss": sum_loss / n_batches,
            "train_loss_pos": sum_pos / n_batches,
            "train_loss_ang": sum_ang / n_batches,
            "train_loss_norm": sum_norm / n_batches,
        }
        val_metrics = compute_val_metrics(probe, val_loader, DEVICE)
        combined_gate = (LAMBDA_POS * val_metrics["val_pos_p95_pooled"]
                         + LAMBDA_ANG * val_metrics["val_ang_p95_pooled"])

        row = {
            "epoch": epoch,
            "step": step,
            **train_metrics,
            **val_metrics,
            "combined_gate": combined_gate,
            "val_over_train": val_metrics["val_loss"] / max(train_metrics["train_loss"], 1e-9),
        }
        val_log.append(row)

        # TB per-epoch scalars
        for k in ("train_loss", "train_loss_pos", "train_loss_ang", "train_loss_norm"):
            tb.add_scalar(f"epoch/{k}", row[k], epoch)
        for k in ("val_loss", "val_loss_pos", "val_loss_ang", "val_loss_norm",
                  "val_pos_mean", "val_pos_p95_pooled", "val_pos_p95_worst_episode",
                  "val_ang_mean", "val_ang_p95_pooled", "val_ang_p95_worst_episode",
                  "mean_pred_norm"):
            tb.add_scalar(f"epoch/{k}", row[k], epoch)
        tb.add_scalar("epoch/val_over_train", row["val_over_train"], epoch)
        tb.add_scalar("epoch/combined_gate", combined_gate, epoch)

        print(
            f"ep{epoch:03d} "
            f"tr={row['train_loss']:.5f} "
            f"vl={row['val_loss']:.5f} "
            f"v/t={row['val_over_train']:.2f} "
            f"pos_p95={row['val_pos_p95_pooled']:5.2f}px "
            f"(worst_ep {row['val_pos_p95_worst_episode']:5.2f}) "
            f"ang_p95={row['val_ang_p95_pooled']:5.2f}deg "
            f"(worst_ep {row['val_ang_p95_worst_episode']:5.2f}) "
            f"||sc||={row['mean_pred_norm']:.3f}"
        )

        # overfit flag
        if row["val_over_train"] > OVERFIT_THRESHOLD:
            overfit_triggered_epochs.append(epoch)
            print(f"  WARN: overfit flag triggered, val/train={row['val_over_train']:.2f}")
            # diagnostic dump: first 64 val predictions + targets
            with torch.no_grad():
                z_d, y_d, ep_d, t_d = next(iter(val_loader))
                z_d = z_d[:64].to(DEVICE)
                y_d = y_d[:64]
                ep_d = ep_d[:64] if isinstance(ep_d, torch.Tensor) else torch.tensor(ep_d[:64])
                t_d = t_d[:64] if isinstance(t_d, torch.Tensor) else torch.tensor(t_d[:64])
                pred_d = probe(z_d).cpu()
            torch.save({
                "epoch": epoch, "val_over_train": row["val_over_train"],
                "pred": pred_d, "target": y_d, "episode": ep_d, "t_idx": t_d,
            }, run_dir / f"overfit_dumps/epoch_{epoch:03d}.pt")

        # best / patience
        if combined_gate < best_gate:
            best_gate = combined_gate
            best_epoch = epoch
            patience_ctr = 0
            torch.save({
                "epoch": epoch, "step": step,
                "state_dict": probe.state_dict(),
                "row": row,
            }, run_dir / "best.pt")
        else:
            patience_ctr += 1

        # early stop
        if epoch + 1 >= MIN_EPOCHS and patience_ctr >= PATIENCE:
            print(f"[early-stop] patience {PATIENCE} hit; best @ ep {best_epoch}")
            break

    # final: save last + logs
    torch.save({"epoch": epoch, "step": step,
                "state_dict": probe.state_dict(), "row": val_log[-1]},
               run_dir / "last.pt")

    # csv log
    with (run_dir / "val_log.csv").open("w", newline="") as f:
        cols = list(val_log[0].keys())
        w = csv.writer(f); w.writerow(cols)
        for r in val_log:
            w.writerow([r[c] for c in cols])

    # Phase 3-A supplement: learning_curves.png
    plot_learning_curves(val_log, run_dir / "learning_curves.png")

    # config + meta
    wall = time.time() - wall_t0
    with (run_dir / "config.json").open("w") as f:
        json.dump({
            "run_name": args.run_name,
            "seed": args.seed,
            "arch": "MLP [4096 -> 256 -> 128 -> 4] GELU + LayerNorm",
            "num_params": probe.num_params,
            "lambda_pos": LAMBDA_POS,
            "lambda_ang": LAMBDA_ANG,
            "lambda_norm": LAMBDA_NORM,
            "lr": LR, "weight_decay": WD,
            "warmup_steps": WARMUP_STEPS,
            "batch_size": args.batch_size,
            "min_epochs": MIN_EPOCHS, "max_epochs": args.max_epochs,
            "patience": PATIENCE,
            "grad_clip": GRAD_CLIP,
            "overfit_threshold": OVERFIT_THRESHOLD,
            "resolution": RESOLUTION,
            "best_epoch": best_epoch,
            "best_gate": best_gate,
            "overfit_triggered_epochs": overfit_triggered_epochs,
            "peak_mib_after_10_steps": peak_mib_after_10_steps,
            "wall_s": wall,
            "train_meta": train_ds.meta,
            "val_meta": val_ds.meta,
        }, f, indent=2, default=str)

    tb.close()

    # terminal summary
    br = val_log[best_epoch]
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"run: {run_dir}")
    print(f"wall: {wall:.1f}s ({wall/60:.2f} min)")
    print(f"peak allocated after 10 steps: {peak_mib_after_10_steps:.1f} MiB")
    print(f"epochs ran: {len(val_log)} / {args.max_epochs}")
    print(f"best ep: {best_epoch}  (combined gate {best_gate:.4f})")
    print()
    print(f"best val pos p95 pooled:        {br['val_pos_p95_pooled']:.3f} px")
    print(f"best val pos p95 worst-episode: {br['val_pos_p95_worst_episode']:.3f} px")
    print(f"best val ang p95 pooled:        {br['val_ang_p95_pooled']:.3f} deg")
    print(f"best val ang p95 worst-episode: {br['val_ang_p95_worst_episode']:.3f} deg")
    print(f"best val mean_pred_norm:        {br['mean_pred_norm']:.4f}")
    print()
    print(f"val/train ratio at best epoch:  {br['val_over_train']:.3f}")
    print(f"max val/train ratio observed:   {max(r['val_over_train'] for r in val_log):.3f}")
    print(f"overfit flag triggered epochs:  {overfit_triggered_epochs or 'none'}")


if __name__ == "__main__":
    main()
