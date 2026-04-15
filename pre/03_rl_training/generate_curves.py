"""Generate training-curve plots from tensorboard logs in rl/outputs/logs/.

Usage (from repo root):
    conda run -n iws python pre/03_rl_training/generate_curves.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

LOG_DIR = Path("rl/outputs/logs")
OUT_DIR = Path("pre/03_rl_training")
plt.rcParams.update({"font.size": 14})


def load_scalars(log_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load all scalar tags from the most recent event file in log_dir."""
    event_files = sorted(log_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No event files in {log_dir}")
    event_file = event_files[-1]
    print(f"Loading {event_file}")
    ea = event_accumulator.EventAccumulator(
        str(event_file), size_guidance={"scalars": 0},
    )
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    print(f"Tags: {tags}")
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tag in tags:
        evs = ea.Scalars(tag)
        steps = np.array([e.step for e in evs])
        vals = np.array([e.value for e in evs])
        out[tag] = (steps, vals)
    return out


def smooth(y: np.ndarray, w: int = 20) -> np.ndarray:
    if len(y) < w or w <= 1:
        return y
    kernel = np.ones(w) / w
    pad = w // 2
    padded = np.concatenate(
        [np.full(pad, y[0]), y, np.full(pad, y[-1])]
    )
    sm = np.convolve(padded, kernel, mode="same")
    return sm[pad : pad + len(y)]


def _first_tag(data: dict, names: list[str]) -> str | None:
    for n in names:
        if n in data:
            return n
    return None


def plot_curve(
    data: dict, tag_candidates: list[str], out_path: Path,
    title: str, ylabel: str, annotation: str | None = None,
    color: str = "tab:blue",
):
    tag = _first_tag(data, tag_candidates)
    if tag is None:
        print(f"⚠ No tag matched {tag_candidates}; skipping {out_path.name}")
        return
    steps, vals = data[tag]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, vals, color=color, alpha=0.3, linewidth=1, label="raw")
    ax.plot(steps, smooth(vals, w=max(1, len(vals) // 25)),
            color=color, linewidth=2.5, label="smoothed")
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    if annotation:
        ax.text(
            0.02, 0.03, annotation,
            transform=ax.transAxes, fontsize=11,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.85),
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    return vals


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_scalars(LOG_DIR)

    # actor loss
    plot_curve(
        data,
        ["actor_loss"],
        OUT_DIR / "actor_loss_curve.png",
        title="Actor loss over training",
        ylabel="actor_loss  (−λ-return, lower = better)",
        color="tab:blue",
    )

    # critic loss
    plot_curve(
        data,
        ["critic_loss"],
        OUT_DIR / "critic_loss_curve.png",
        title="Critic loss over training",
        ylabel="critic_loss  (MSE of twin value heads)",
        color="tab:orange",
    )

    # reward (cosine similarity)
    reward_tag = _first_tag(data, ["mean_reward_goal", "mean_reward"])
    if reward_tag is not None:
        steps, vals = data[reward_tag]
        init_r = float(vals[: max(1, len(vals) // 20)].mean())
        final_r = float(vals[-max(1, len(vals) // 20):].mean())
        ann = (
            f"Reward: {init_r:.4f} → {final_r:.4f}  (Δ = {final_r - init_r:+.4f})\n"
            "Algorithm works, but improvement is marginal because\n"
            "cosine similarity is not task-informative."
        )
        plot_curve(
            data,
            [reward_tag],
            OUT_DIR / "reward_curve.png",
            title="Reward (cosine similarity to goal) over training",
            ylabel="mean reward",
            annotation=ann,
            color="tab:green",
        )

    # lambda return
    plot_curve(
        data,
        ["mean_lambda_return"],
        OUT_DIR / "lambda_return_curve.png",
        title="Mean λ-return over training",
        ylabel="λ-return",
        color="tab:purple",
    )

    print("\nTraining-curve plots complete.")


if __name__ == "__main__":
    main()
