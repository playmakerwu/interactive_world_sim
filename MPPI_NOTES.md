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
