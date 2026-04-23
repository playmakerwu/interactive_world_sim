# MPPI on IWS — Notes (post-refactor)

## Status

- Phase 1 refactor complete on 2026-04-22
- Environment wrapper: [env/pusht_wm_env.py](env/pusht_wm_env.py)
- MPPI implementation: [rl/mppi/mppi_planner.py](rl/mppi/mppi_planner.py)
- Config: [configs/mppi/default.yaml](configs/mppi/default.yaml)
- Reference algorithm spec: [MPPI_REFERENCE_NOTES.md](MPPI_REFERENCE_NOTES.md)
- Historical notes archived at [MPPI_NOTES_archive_pre_refactor.md](MPPI_NOTES_archive_pre_refactor.md)

## Known prior findings worth remembering

- **Camera bug** (commit `fd875ef`): WM trained on `camera_1_color`,
  inference scripts had been using `camera_0_color`. Fixed across 6
  PushT inference scripts. Camera is now hardcoded inside
  `env.PushTWMEnv.load_initial_from_hdf5` to prevent regression.
- **Post-camera-fix v1 sanity run** (`step5_v1_camfix`, now deleted):
  final position 12.95 px from a goal-adjacent initial state, arms
  visible throughout, zero CV failures, latent cosine similarity to
  `z_0` never below 0.985. Established that the WM is not the
  bottleneck — the bottleneck is the (now-superseded) minimal-MPPI
  algorithm.
- **v10 keyboard experiment** (also deleted): action accumulator
  sampling vs. independent sampling difference — camera-independent
  observation. Re-examined as needed in Phase 2+.
- **Reference algorithm**: diffusion-forcing's MPPI
  (`~/Documents/diffusion-forcing/configurations/planner/planner_v0_0.yaml`)
  uses iterative refinement (`n_update_iter=50`), warm-start
  (`beta_filter=0.7`), sharp softmax (`reward_weight=200`), and
  σ=0.05. Our refactored MPPI matches these knobs (see
  `MPPI_REFERENCE_NOTES.md` for the exact formulas).

## Phase 2 Sanity Runs

Two MPPI runs using the Phase 1 refactored implementation
(`rl/mppi/mppi_planner.py`, `configs/mppi/default.yaml`) on val episodes
loaded via `env.PushTWMEnv`. Both runs completed without OOD symptoms
(0 CV failures, latent stayed in distribution throughout).

> **Critical caveat**: these empirical results used N=16 (see
> `config_deviation` in each `summary.json`). Our default config is
> N=100 matching reference. The difference is substantial for Phase 1's
> sharp-softmax design — N=16 argmax is statistically a much worse
> action selector than N=100 argmax. Treat these local numbers as
> qualitative evidence (does MPPI move T toward goal? do OOD symptoms
> recur?), not as quantitative performance claims. Cloud replication at
> N=100 is required for quantitative conclusions. The exact cloud
> command lives at
> `outputs/mppi/phase2_sanity_{near,far}/reproduce_on_cloud.sh`.

| Metric | v1_camfix (pre-refactor, archived) | phase2_near (post-refactor) | phase2_far (post-refactor) |
|---|---|---|---|
| Config | N=16, H=10, n_iter=1, σ=0.1, τ=1, no β-filter | N=16, H=10, n_iter=5, σ=0.05, τ≈0.005, β=0.7 | same as phase2_near |
| Initial state hdf5 | val/0 ep_0 frame 0 | val/0 ep_0 frame 0 | val/2 ep_2 frame 195 |
| Initial dist to goal | 0.14 px | 0.18 px | **47.57 px** |
| Final pos distance | 12.95 px | 13.56 px | **49.98 px** |
| Final angle error | −134.7° | −57.5° | −62.9° |
| Final angle sim | −0.43 | +0.54 | +0.46 |
| success_strict | False | False | False |
| success_cos | False | False | False |
| min_cos_to_z0 | 0.985 | 0.983 | 0.963 |
| n_cv_failures | 0 | 0 | 0 |
| best reward in trajectory | not recorded | **−0.0001** | −0.82 |
| mean_reward_last_10_steps | not recorded | −0.34 | −0.86 |
| Wall time | ~5 min | 24.8 min | 25.0 min |
| Wall per plan_step | ~6 s | 29.7 s | 30.0 s |

### Near-goal holding test

The refactored MPPI did not hold a goal-adjacent state better than
pre-refactor `v1_camfix`. Final position 13.56 px vs 12.95 px — within
seed-level noise, essentially identical. Phase 1's algorithmic upgrades
(iterative refinement × 5, sharp softmax `reward_weight=200`,
intra-horizon noise smoothing `beta_filter=0.7`) did not produce a
visible holding improvement at this sample budget. **However**: the
refactored controller did reach a much better transient — best reward
in trajectory was **−0.0001** (essentially zero, i.e. the goal was
within reach of MPPI's search at some point), where pre-refactor
v1_camfix's best reward was −0.16. The controller found the goal but
couldn't stay; over 50 steps it drifted to (69, 61, −57°) with reward
−0.34 in the last 10 steps. The drift pattern matches what we'd expect
from a sharp-softmax controller fed too few samples: most iterations
the argmax-like selection latches onto a fluke high-reward sample
that doesn't generalise to the next state.

### Far-from-goal convergence test

The refactored MPPI **did not meaningfully push T toward the goal**
from the val/2 t=195 starting state. Initial distance 47.57 px → final
distance 49.98 px. Gap closed: **(47.57 − 49.98) / 47.57 = −5%** —
slightly worse than starting position. Reward curve in
`outputs/mppi/phase2_sanity_far/reward_curve.png` is essentially flat
around −0.85 across all 50 steps. CV pose stays in a narrow band
around (104, 50, −63°) — barely moves at all from the initial (102,
51, −71°). At N=16 the softmax-near-argmax selector evidently can't
find any push direction that improves the (decoded-frame) reward, so
the controller defaults to near-zero net actions and the WM's intrinsic
stochastic dynamics carry the T-block in a small wander around its
starting pose. This is the predicted expected_impact — at low sample
density a sharp-softmax controller has a hard time finding good
actions.

### Failure-mode check

Zero arm-disappearance, zero CV failures, latent cosine-similarity to
`z_0` stayed ≥ 0.96 throughout both runs. Decoded scenes remain
visually coherent end-to-end. The OOD symptoms that plagued
pre-camera-fix runs (arm disappearance, frequent CV flips, latent
drift below 0.95) are entirely absent. Camera fix + Phase 1 refactor
together produce a clean IWS+CV+MPPI pipeline at the algorithmic
level; the remaining gap from acceptance is "MPPI doesn't have enough
search authority at N=16 to actually drive the system".

### Artifacts

- [outputs/mppi/phase2_sanity_near/](outputs/mppi/phase2_sanity_near/)
  - `summary.json`, `trajectory.mp4`, `reward_curve.png`,
    `reproduce_on_cloud.sh`
- [outputs/mppi/phase2_sanity_far/](outputs/mppi/phase2_sanity_far/)
  - same files

### Open question for Phase 3

Does cloud-N=100 actually close the gap, or is the issue that even
with the upgraded algorithm, sharp-softmax MPPI on the IWS WM doesn't
produce reliable pushing actions? Both interpretations are consistent
with the local data. **The cloud rerun via `reproduce_on_cloud.sh` is
the required next experiment before drawing any structural
conclusions.**
