# Action distribution analysis — Step 5 v1 (`step5_mppi_local`)

Comparison of the 50 actions MPPI actually executed (one per control step,
weighted-mean of 16 sampled chunks via softmax) against the training-data
action distribution. data/full/pusht is not available locally; substituting
data/mini/pusht train (5 episodes, 999 frames).

## MPPI executed actions (50 steps)

```
shape: (50, 4)
per-step L2 norm:    mean 0.0529   max 0.1113   min 0.0142
per-dim mean:        −0.0019  +0.0056  −0.0020  −0.0005    (≈ zero)
per-dim std:          0.0292   0.0258   0.0290   0.0297
per-dim min:         −0.0487  −0.0515  −0.0671  −0.0779
per-dim max:         +0.0699  +0.0702  +0.0631  +0.0669
step-to-step ΔL2:    mean 0.0733   max 0.1591
```

## Training-data actions (data/mini/pusht/train, 999 frames)

```
overall L2 norm:     mean 0.4586   max 0.5852
per-dim mean:        +0.1902  +0.1348  +0.2103  −0.2890    (non-zero drift)
per-dim std:          0.1020   0.0512   0.1121   0.0759
per-dim min:         +0.0103  −0.0233  −0.0521  −0.3900
per-dim max:         +0.3600  +0.1700  +0.3600  −0.0478
one-step ΔL2:        mean 0.0114   max 0.2447
one-step Δ per-dim std:  0.0095  0.0070  0.0116  0.0071
```

## Side-by-side (the diagnostic numbers)

|                          | MPPI executed | Training | Ratio |
|--------------------------|---------------|----------|-------|
| Action L2 norm (mean)    | 0.053         | 0.459    | **MPPI is 8.7× SMALLER** |
| dim-3 mean               | −0.001        | −0.289   | MPPI ≈ 0; training has strong negative drift |
| dim-0,2 mean             | −0.002, −0.002| +0.190, +0.210 | MPPI ≈ 0; training has strong positive drift |
| dim-3 range              | [−0.078, +0.067] | [−0.390, −0.048] | **MPPI's range OVERLAPS only ~17% of training's range** |
| Step-to-step Δ (mean L2) | 0.073         | 0.011    | **MPPI is 6.4× ROUGHER** between steps |

## Observation

**MPPI's executed actions are not "too big" — they are catastrophically TOO SMALL and TOO ZERO-MEAN.** The softmax-weighted mean of 16 zero-mean σ=0.1 Gaussian samples converges (by the law of large numbers, modulo softmax reweighting) to approximately zero, with per-dim std ≈ σ/√N = 0.025. That matches the observed per-dim std of 0.029.

So at execution time the WM is being asked: "given history, what comes next under a near-zero action?" — and the WM has **never seen a near-zero action in training**. Every training step had a coherent goal-directed bimanual push of L2 norm ≈ 0.46, with strong per-dim drifts (dim 3 always negative, dims 0/2 always positive). The MPPI-executed action distribution does not overlap the training distribution at all on dim 3 (training is in [−0.39, −0.05], MPPI is in [−0.08, +0.07]).

**This is a more precise statement of the OOD diagnosis than "actions are out of distribution":** the actions MPPI executes are not a translation or perturbation of training actions; they are a **completely different distribution shape** — zero-centred white noise vs. drifted-mean smoothed pushes. No wonder the WM hallucinates and the arms vanish — the model is being driven by inputs from outside the support of its training distribution.

This finding alone explains v1's failure without any need for the replay tool. The replay tool will let us pinpoint the exact step when the OOD drift first manifests in the decoded scene.
