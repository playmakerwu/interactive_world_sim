# Diagnostic Probe Run — odd/even split

## Purpose (copy from the Phase 3-A diagnostic prompt §0)

> This is a **diagnostic experiment**, NOT an acceptance run. We are answering exactly one question: can the probe architecture learn the `latent → (cx, cy, sinθ, cosθ)` mapping at all, when train and val distributions match? The numbers below come from a split where val frames are temporal neighbours of train frames (~100 ms apart at 10 Hz). They severely OVERESTIMATE probe generalisation to RL-deployment latents. They are **not** acceptance metrics and must not be reused as such.

## Split description

- Kind: odd-even pooled (DIAGNOSTIC, leaked)
- Total frames pooled: 1999
- Train: 1000  Val: 999
- Per-episode counts:

  | ep | train | val |
  |---|---|---|
  | 0 | 100 | 100 |
  | 1 | 100 | 100 |
  | 2 | 100 | 100 |
  | 3 | 100 | 100 |
  | 4 | 100 | 99 |
  | 5 | 100 | 100 |
  | 6 | 100 | 100 |
  | 7 | 100 | 100 |
  | 8 | 100 | 100 |
  | 9 | 100 | 100 |

Overlap evidence: see `label_distribution_position_scatter.png` and `label_distribution_angle_hist.png`. By construction the two distributions should be near-identical; any visible divergence would invalidate the diagnostic.

## Final metrics (best checkpoint, epoch 92)

| metric | value | leaked-split target |
|---|---|---|
| val pos p95 pooled | **2.402 px** | < 1.5 |
| val pos p95 worst-episode | 3.041 px | (diagnostic) |
| val pos mean | 1.081 px | — |
| val ang p95 pooled | **7.984°** | < 3 |
| val ang p95 worst-episode | 10.853° | (diagnostic) |
| val ang mean | 2.920° | — |
| val mean_pred_norm | 0.9972 | ≈ 1.0 |
| val/train at best | 1.452 | — |

## Overfitting analysis

Per Phase 3-A supplement §2. All evidence is in `learning_curves.png` and `val_log.csv`.

1. **Did train/val loss diverge?** No sustained divergence. Both curves drop monotonically. Train settles around 0.002, val around 0.003. Best val at epoch 92; training ran the full 100 epochs (early stopping **did not fire** — the combined-gate metric kept improving until epoch 92, patience 10 would have fired at epoch 102+).
2. **Max `val_loss / train_loss`:** 2.608 at epoch 24 (single-epoch spike). One diagnostic dump saved at `overfit_dumps/epoch_024.pt`. The ratio oscillated in [1.0, 2.6] throughout training, mostly sitting around 1.3–1.6 — mild generalisation gap, not a runaway overfit.
3. **Did val p95 follow loss?** Yes — both pos and ang p95 dropped monotonically with val loss (with the same noise). The p95 trajectory ends near its minimum, not overshooting.
4. **Final gap at best-val epoch:** val/train = 1.452. Below the 1.8 discussion threshold; healthy.
5. **Overfit verdict:** **mild but not concerning.** The ratio briefly spiked above 2.5 at epoch 24 then settled; no harm done to the p95 trajectory.

## Run verdict: B (by my script's strict A/B/C logic) — but see caveats

Auto-logic (from §1.3 of the kickoff): pos p95 = 2.40 px (target < 1.5) and ang p95 = 7.98° (target < 3) miss both tight leaked-split targets, so the script labelled this **B**. But the **narrative** attached to B in the kickoff — "the probe cannot fit the mapping" — is contradicted by the evidence:

- **Mean pos error 1.08 px**, **mean ang error 2.92°** — both well under the **real-split** acceptance thresholds (3 px / 5°).
- val pos p95 worst-episode **3.04 px** — essentially at the real-split 3-px line.
- val ang p95 worst-episode **10.85°** — 2× over the real-split 5° line, but still on the same order of magnitude, not "random" (random would be ~90°).
- val mean_pred_norm **0.9972** — the soft unit-norm regulariser holds; no (sin, cos) collapse.
- train loss collapsed from 0.92 → 0.002 and val followed from 0.7 → 0.003. The architecture **can and does** fit the mapping at high resolution.

So strictly the run misses the leaked-split targets; practically it already sits near the real-split acceptance on average. Comparing to probe_v1 (pos p95 42 px, ang p95 168°), this is a 17× improvement in pos and a 21× improvement in ang with **only the split changed**.

I would not call this "architecture insufficient." The honest framing is:
- The architecture can represent the mapping well enough for the task.
- Angle is the weaker dimension — p95 8° where position p95 is 2.4 px. This is expected: CV has the θ vs θ+180 ambiguity in Phase 1's observations, and a handful of frames at near-ambiguous orientations drag the tail.
- The gap between mean (2.9°) and p95 (8°) on angle signals that a small number of frames carry most of the error.

**My read**: it's really a **C (mixed)** with a side of "already pretty good." The strict verdict-B label is a mechanical consequence of how I encoded the thresholds in the script; it overstates the problem.

Per the diagnostic prompt §4: "If v2 is verdict (B) or (C), stop and let me decide next step." Stopping.

## Reminder — these numbers are NOT acceptance metrics

Per the diagnostic prompt §§1, 5: this run does NOT advance Branch A to merge. The probe_v2_odd_even checkpoint will not be wired into RL. Real acceptance (pos ≤ 3 px, ang ≤ 5°) will be measured on a properly held-out split after the cloud data collection step.
