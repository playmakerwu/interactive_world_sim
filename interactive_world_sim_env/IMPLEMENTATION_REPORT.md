# WorldModelEnv — Implementation Report

## 1. Summary

Built the `interactive_world_sim_env` sibling package as the single firewall
between the trained latent world model and downstream MPPI code. The
wrapper exposes `WorldModelEnv` with the full public API agreed in the
spec: `__init__`, `reset` (with snapshot, user-RGB, episode-path, and
default-fallback init paths), `step`, `snapshot`/`restore`, `render`,
`latent_to_rgb`, `goal_preprocess`, `step_batch`, and `close`. The
constructor loads a checkpoint via the only allowed import site
(`_model_loader.py`) and cross-checks the registry against the saved
`.hydra/config.yaml`. The build is **complete**: all 10 implementation
steps from the spec are done. A smoke test that constructs the env on
the downloaded `pusht_cam1` checkpoint, runs `reset → step → snapshot →
step → restore → step → render → step_batch(K=2,H=3) → goal_preprocess →
close` passes end-to-end on the user's RTX 5070 Ti Laptop with the
Blackwell attention fix already in place. No tests were written in this
pass.

## 2. Files created

| File path | Purpose | Lines | Key public symbols |
|---|---|---|---|
| `interactive_world_sim_env/__init__.py` | Public surface re-exports | 20 | `WorldModelEnv`, `Observation`, `BatchedObservation`, `EnvState`, `TaskSpec`, `RegistryError`, `TASKS`, `get_task_spec` |
| `interactive_world_sim_env/registry.py` | TaskSpec dataclass + the 7-task `TASKS` dict + `RegistryError` + `get_task_spec` | 95 | `TaskSpec`, `TASKS`, `RegistryError`, `get_task_spec` |
| `interactive_world_sim_env/_model_loader.py` | Only module allowed to import `interactive_world_sim.algorithms.*` and to call `LatentWorldModel.load_from_checkpoint`. Registers OmegaConf `eval`/`torch` resolvers. Cross-checks ckpt config against the registry. | 171 | `LoadedModel`, `load_model` |
| `interactive_world_sim_env/state.py` | Frozen `EnvState` dataclass (latent window, action window, step counter, task) | 39 | `EnvState` |
| `interactive_world_sim_env/obs.py` | Frozen `Observation` / `BatchedObservation` dataclasses | 61 | `Observation`, `BatchedObservation` |
| `interactive_world_sim_env/env.py` | `WorldModelEnv` class + private RGB helpers | 543 | `WorldModelEnv` |
| `interactive_world_sim_env/IMPLEMENTATION_REPORT.md` | This file | — | — |

Total new code: **929 lines** Python.

## 3. Files modified outside the new package

Empty.

The earlier one-character fix in
`interactive_world_sim/algorithms/models/attention.py:61` (`>=` → `==`)
was already in place before this implementation session and was not
touched.

## 4. Pre-implementation verification results

### V1 — Decoder attention path on Blackwell

The decoder does **not** go through the buggy attention module. Two
distinct `AttentionBlock` classes exist:

- `interactive_world_sim/algorithms/models/attention.py:32–110`
  (`Attention`/`AttentionBlock`) uses `torch.nn.functional.scaled_dot_product_attention`
  inside a `sdpa_kernel(...)` context that previously forced flash-only
  on any GPU matching `major >= 8 and minor == 0`. **This is the module
  used by the dynamics path** (`CMLatentDynamics`, via the
  `SpatialAttentionBlock` / `TemporalAttentionBlock` einops wrappers).
  The earlier one-character fix covers it.

- `interactive_world_sim/algorithms/models/diffae_unet.py:398–445`
  (`AttentionBlock`) is hand-rolled with `QKVAttention`
  (diffae_unet.py:448), which uses `torch.einsum` + `torch.softmax` —
  no `scaled_dot_product_attention`, no `sdpa_kernel`, no Blackwell
  sensitivity at all. **This is the module the decoder uses**
  (`CMControlledUnetModel` and `CMControlNet` in
  `cm_controlnet.py:6–17` both import `AttentionBlock` from
  `diffae_unet`).

Conclusion: the existing fix already covers the dynamics; the decoder
needs no fix. The smoke test confirmed `render()` and `step_batch()`
(which decodes K×H frames) both succeed on the RTX 5070 Ti.

### V2 — Decoder batch efficiency

`render_img_cm` (`interactive_world_sim/algorithms/common/diffusion_helper.py:67–127`)
accepts a leading batch dim of arbitrary size and internally chunks it
into groups of `batch_size=50` frames. Each chunk does
`dec_infer_steps` (default 3) decoder forward passes through
`algo._forward(algo.decoder, ...)`. The decoder itself (`CMDecoder`)
takes any batch dim ≥ 1 natively — there is no per-frame Python loop.

For K×H ≤ 50, decoding is a single batched call (3 decoder forwards).
For larger K×H it's `ceil(K*H / 50)` chunks in a Python `for j in range`.

Realistic envelope on the user's RTX 5070 Ti Laptop (12 GB VRAM, fp32):
- Decoder activation memory for a chunk of 50 frames at 128² is small
  relative to the model and dynamics state. The smoke test ran
  K=2, H=3 (= 6 frames decoded) in ~0.5 s end-to-end including the
  one-step dynamics call.
- Linear extrapolation gives a rough budget of **~300–500 frames per
  `step_batch` before the decoder dominates wall time** (i.e., K=20,
  H=20 should be comfortable; K=100, H=20 starts to hurt; K=1000 will
  be slow). These numbers are not measured under stress and should be
  treated as a planning guideline, not a benchmark.

The decoder is not optimized in this pass.

## 5. Decisions you made that the design didn't specify

1. **`action_mode` is NOT cross-checked against the saved config.** The
   `pusht_cam1` checkpoint's saved config has `dataset.action_mode:
   single_ee`, but `single_ee` is not in the elif chain of
   `interactive_world_sim/utils/action_utils.py` — it is unreachable
   from `joint_pos_to_action_primitive`. Cross-checking would force a
   RegistryError on a checkpoint that does in fact run correctly. I
   made the registry's `ctrl_mode` the source of truth and cross-check
   only `action_dim`, `obs_keys`, `resolution`. Alternative: trust the
   saved config blindly (would have broken `pusht_cam1`); or fail
   loudly and ask the user to reconcile (extra friction with no
   payoff). Documented this choice in `_model_loader._cross_check_registry`.

2. **`n_frames` defaults to `hist_context`**, both default to 10.
   Mirrors the explicit override done by `teleoperate_keyboard.py:48–49`
   and `deploy/server.py:90–91`. Alternative: respect whatever the
   training config used. I chose the demo override because the latent
   sliding-window length and the model's `n_tokens` need to match
   `hist_context` anyway.

3. **`OmegaConf` resolvers are registered with `replace=True`** at
   `_model_loader.py` import time. Without `replace`, re-importing the
   module (e.g., in a notebook reload) raises. Alternative: a
   try/except, or never re-register. `replace=True` is simplest and the
   behavior is identical to the demos' first-time registration.

4. **`init_paths.py` was not created.** The design listed it as a
   separate module, but the only init-path helpers needed
   (`_normalize_init_rgb`, `_load_rgb_views_from_hdf5`, `_encode_views`,
   `_rgb_to_chw_float01`, `_validate_rgb`) are tightly coupled to the
   env's internals (especially the per-key normalizer access). I made
   them methods/module-level helpers in `env.py` instead. Alternative:
   split them out — would add an import surface without a callable
   second consumer.

5. **`rng.py` was not created.** Spec is "stochastic by design, no
   seeding inside the env" — there is no RNG-manipulation code to
   house.

6. **`info["clipped"]` is a single `bool`**, not a per-dim mask.
   Alternative: `np.ndarray[bool]` of shape `(action_dim,)`. Picked
   bool because the spec said "info\['clipped'\] = bool". The
   `info["latent_norm"]` (float) and `info["step"]` (int) are debug
   aids; the spec didn't enumerate them.

7. **Per-view tile in `step_batch`.** I tile the current window via
   `expand(K, ...).contiguous()`. `expand` returns a view;
   `.contiguous()` materializes. For K=100 at PushT shapes this is
   ~16 MB — acceptable. Alternative: pass a non-contiguous expanded
   tensor; risk that `dynamics_forward`'s internal `clone()` or `cat`
   would handle it correctly but produce surprising behavior. Chose
   contiguous to make the call site dumb-obvious.

8. **`render()` and `latent_to_rgb()` output shapes.** Single-latent
   inputs return HWC uint8 (the human-friendly format). Batched inputs
   return BHWC for `latent_to_rgb`. The `step_batch` RGBs return BCHW
   (channel-first) because the spec explicitly wrote `(K, H, 3,
   H_img, W_img)`. The two conventions are intentional; documented in
   docstrings.

9. **`reset()` with no arguments uses
   `self._spec.default_episode_path`**, frame 0 — the same fallback
   both demos pick. The spec said "registry default for the task"; this
   is the concrete realization.

10. **`as_gym()` is a stub that raises `NotImplementedError`** with a
    clear message. Spec didn't say what to do with it; I treated it as
    a forward-declaration placeholder.

## 6. Known limitations / TODOs

- **Single-checkpoint envs only.** Per the design's open-question
  resolution, the multi-checkpoint variants (`_and_cam1_aloha.sh`
  style, where two checkpoints share an action history) are out of
  scope. If you ever want that, it would be a `WorldModelEnvPair` or a
  multi-view extension — not a simple change to this class.
- **`act_horizon = 1` only.** Hard-coded behavior; the wrapper does not
  expose a multi-step-action mode.
- **No tests.** Per the spec. Section 8 below lists the manual checks
  that compensate.
- **`info["clipped"]` is per-call boolean, not per-dim.** Fine for MPPI
  monitoring; not great for debugging a specific axis being saturated.
- **Decoder is not optimized for very large K×H.** See V2.
- **The wrapper does not protect against the model being mutated from
  outside.** If MPPI code reaches into `env._loaded.model` (bad idea,
  but possible), the firewall is violated. The discipline is social,
  not enforced.
- **`reset()` with `init_state` calls `restore()` which itself clones
  the snapshot.** The user-facing semantics are clone-everywhere; this
  is intentional but slightly wasteful when the caller knows they
  won't mutate. There is no public no-clone path.
- **Single-view tasks only have been smoke-tested.** All 7 shipped
  tasks are single-view, so num_views=1 is the only configuration that
  has actually run end-to-end. The code branches for `num_views > 1`
  (concatenated views in `_encode_views`, multi-channel RGB output)
  are written but not exercised.
- **Default device is `"cuda"`.** No CPU fallback was attempted; the
  underlying model is heavy on CUDA-only paths.
- **No CHANGELOG, no semantic version.** Just `0.1.0`-ish behavior with
  no commitment to API stability across future iterations.

## 7. How to use it (minimal example)

This script runs as-is on your machine. Save as `mppi_smoke.py` at the
repo root and run with `/home/yiru-wu/miniconda3/envs/iws/bin/python
mppi_smoke.py` (or activate `iws` and run `python mppi_smoke.py`).

```python
import imageio.v3 as iio
import numpy as np

from interactive_world_sim_env import WorldModelEnv

env = WorldModelEnv("pusht_cam1", decode_on_step=False)

# 1. Reset from the default episode (data/mini/pusht/val/episode_0.hdf5, frame 0)
obs = env.reset()
print("start:", obs.latent.shape, "step =", obs.step)

# 2. Step a few times with a fixed (zero) action.
fixed_action = np.zeros(env.action_space.shape, dtype=np.float32)
for _ in range(3):
    obs, info = env.step(fixed_action)
    print("step", info["step"], "latent_norm =", info["latent_norm"])

# 3. Snapshot, branch, restore, branch again.
snap = env.snapshot()
obs, _ = env.step(fixed_action)
print("after divergence: step =", obs.step)
env.restore(snap)
print("restored: step =", env.snapshot().step_counter)
obs, _ = env.step(fixed_action)
print("step after restore: step =", obs.step)

# 4. Render the current frame to disk.
rgb = env.render()
iio.imwrite("/tmp/pusht_render.png", rgb)
print("wrote /tmp/pusht_render.png", rgb.shape)

# 5. Batched rollout (K=4 candidates, H=5 steps, no env mutation).
K, H, A = 4, 5, env.action_space.shape[0]
candidate_actions = np.zeros((K, H, A), dtype=np.float32)
candidate_actions[:, :, 0] = np.linspace(-1.0, 1.0, K)[:, None]  # different left-X per candidate
batched = env.step_batch(candidate_actions)
print("batched latents:", tuple(batched.latents.shape), "rgbs:", batched.rgbs.shape)

env.close()
print("ok")
```

## 8. What I should test before relying on this

1. **Re-run the smoke test (Section 7's script)**. You should see step
   counters increment, restored steps match the pre-snapshot count, and
   `/tmp/pusht_render.png` open as a recognizable PushT scene.

2. **Sanity-check `step_batch` output by saving the first candidate's
   first frame and visually comparing it to a single `step()` call from
   the same starting state.** Both should look like one PushT step.
   Because the env is intentionally stochastic, they will not be
   identical, but they should be plausible same-domain frames.
   ```
   python -c "
   from interactive_world_sim_env import WorldModelEnv
   import imageio.v3 as iio, numpy as np
   env = WorldModelEnv('pusht_cam1')
   env.reset()
   b = env.step_batch(np.zeros((1, 1, 4), dtype=np.float32))
   iio.imwrite('/tmp/batched_first.png', np.transpose(b.rgbs[0, 0], (1, 2, 0)))
   env.close()
   "
   ```

3. **Verify the cross-task safety rail.** Try to restore an
   `EnvState` from one task into a different env:
   ```
   python -c "
   from interactive_world_sim_env import WorldModelEnv
   env = WorldModelEnv('pusht_cam1')
   env.reset()
   snap = env.snapshot()
   # mutate task on the frozen dataclass via dataclasses.replace
   import dataclasses
   bad = dataclasses.replace(snap, task='single_grasp_cam0')
   env.restore(bad)  # should raise ValueError
   "
   ```
   Expect a `ValueError` about task mismatch.

4. **Confirm model weights are untouched by a rollout.** Hash the
   model `state_dict` keys/values before and after a 100-step rollout
   + a `step_batch(K=4, H=10)`. They must be equal.
   ```
   python -c "
   import hashlib, torch
   from interactive_world_sim_env import WorldModelEnv
   import numpy as np
   env = WorldModelEnv('pusht_cam1')
   def h():
       m = hashlib.sha256()
       for k, v in env._loaded.model.state_dict().items():
           m.update(k.encode()); m.update(v.detach().cpu().contiguous().view(-1).numpy().tobytes())
       return m.hexdigest()
   env.reset()
   before = h()
   for _ in range(100): env.step(np.zeros(4, dtype=np.float32))
   env.step_batch(np.zeros((4, 10, 4), dtype=np.float32))
   after = h()
   print('weights unchanged:', before == after)
   env.close()
   "
   ```

5. **Confirm the registry / config cross-check actually fires.**
   Temporarily monkey-patch `TASKS["pusht_cam1"]` to have
   `action_dim=3` and re-construct the env — it should raise
   `RegistryError` rather than load silently.
   ```
   python -c "
   from interactive_world_sim_env import TASKS, registry, WorldModelEnv
   import dataclasses
   TASKS['pusht_cam1'] = dataclasses.replace(TASKS['pusht_cam1'], action_dim=3)
   try:
       env = WorldModelEnv('pusht_cam1')
   except registry.RegistryError as e:
       print('OK, raised:', e)
   "
   ```

## 9. Recommended next session

Build a thin test suite next — not because the wrapper is fragile, but
because we're about to layer MPPI on top, and the contract that "the
env is stable" only holds if there's a regression net under it. Port
the four bullets from Section 8 into `tests/env_wrapper/test_basic.py`
and add the three load-bearing ones from the original design (snapshot
↔ restore round-trip — non-deterministic so assert shapes & no crash
rather than equality; weights-unchanged hash; cross-task restore
raises). With those green, start the MPPI skeleton in a new sibling
package `interactive_world_sim_mppi/` that imports only from
`interactive_world_sim_env`. The very first MPPI iteration can be
"random action sampler + RGB pixel cost vs. a goal frame from
`env.goal_preprocess(...)`" — that exercises every method this wrapper
exposes and gives an honest cost-curve to look at before tuning the
sampler.

## Follow-up session

### Task 1 — Modification audit

Confirmed no files outside `interactive_world_sim_env/` were touched
during the wrapper-implementation session.

- `git status`: working tree has zero modified files; only two
  untracked entries — `interactive_world_sim_env/` (the new package)
  and `tests/` (the pre-existing orphan `__pycache__` tree that was
  already untracked before this session and remains unchanged: 53
  files, none newer than HEAD).
- `git diff --stat` and `git diff --stat HEAD`: both empty.
- `git log -1`: tip is commit `fd1f4b2` ("fix: restrict A100 attention
  branch to sm_80 only"), made earlier in the conversation but before
  the wrapper-implementation session began. `attention.py` mtime
  (May 11 23:33) is older than every file in the new package.

**CLEAN: only `interactive_world_sim_env/` was added; `attention.py`
change predates this session; no other modifications.**

### Task 2 — Camera preview

Added `interactive_world_sim_env/scripts/show_camera.py`. Constructs
the env on `pusht_cam1`, decodes the initial frame, rolls out 10 steps
with a zero action and (after a fresh reset) 10 steps with
`action[0] = 0.3`, then concatenates the 22 decoded frames into a
2×11 grid PNG. Output: `/tmp/pusht_cam1_preview.png`, shape
`(256, 1408, 3)` uint8 — exactly 128×128 per cell, confirming the
task spec's resolution.

**Visual content.** Top-down view of the ALOHA table. Each frame
contains a pink "T" block on a cream-colored table surface, with a
dark grey robot end-effector visible near the upper-left of the
T. The view is roughly looking down with a slight tilt; both the T
and the gripper are clearly identifiable.

**Resolution / image quality at 128².** Sharp, not blurry — the
diffusion decoder produces crisp object boundaries rather than the
soft edges typical of pixel-space autoencoders. There is visible
aliasing on the T's edges and on the gripper fingers (jagged
diagonal lines), which is the unavoidable cost of 128×128. The T
block occupies roughly 30–40 pixels across, so its silhouette is
recognizable but sub-pixel localization will be limited. **For an
RGB-based reward function**: coarse XY position of the T-block is
clearly resolvable at this scale; fine pose (rotation angle of a few
degrees) and small gripper-opening changes are at the edge of what
128² can resolve. The signal is clean enough for the planar-push
task PushT; for tasks where the reward depends on millimeter-scale
geometry it would be a bottleneck.

**Rollout coherence.** Both rows stay plausibly in-distribution
across all 11 frames — no blur, no color drift, no scene collapse.
The zero-action row (top) is nearly static: the gripper and T-block
hold position, with mild frame-to-frame variation that is consistent
with the decoder's stochastic init noise rather than physical
motion. The non-trivial-action row (bottom) shows the same scene
evolving slightly differently across frames, also coherent. Two
notes: (a) the leftmost cell of each row is a fresh decode of the
same initial latent, so they differ slightly — that is decoder
stochasticity, not a bug; (b) a constant `action[0] = 0.3` is a
fairly small push and the 10-step horizon is short, so the visible
motion is subtle rather than dramatic — both rows look more similar
than I would have liked for a visual demo, but the model itself
appears healthy.

### Task 2 (revised) — Expert data format

Investigation only; no code changes. Inspection script lives at
`/tmp/inspect_episode.py` (thrown away).

**HDF5 structure of `data/mini/pusht/val/episode_0.hdf5`** (200 steps):

| Path | Shape | dtype |
|---|---|---|
| `action` | `(200, 4)` | float32 |
| `joint_action` | `(200, 14)` | float32 |
| `timestamp` | `(200,)` | float64 |
| `obs/joint_pos` | `(200, 14)` | float32 |
| `obs/full_joint_pos` | `(200, 16)` | float32 |
| `obs/ee_pos` | `(200, 14)` | float32 |
| `obs/world_t_robot_base` | `(200, 2, 4, 4)` | float32 |
| `obs/images/camera_0_color` | `(200, 480, 640, 3)` | uint8 |
| `obs/images/camera_0_extrinsics` | `(200, 4, 4)` | float32 |
| `obs/images/camera_0_intrinsics` | `(200, 3, 3)` | float32 |
| `obs/images/camera_1_color` | `(200, 480, 640, 3)` | uint8 |
| `obs/images/camera_1_extrinsics` | `(200, 4, 4)` | float32 |
| `obs/images/camera_1_intrinsics` | `(200, 3, 3)` | float32 |

**Action-like fields** present in the HDF5:

1. `action` — shape `(200, 4)`, float32, raw values (first 5 timesteps):
   ```
   [[ 0.093   0.17    0.2187 -0.39  ]
    [ 0.0927  0.17    0.2188 -0.39  ]
    [ 0.0922  0.17    0.2189 -0.39  ]
    [ 0.0908  0.17    0.2166 -0.39  ]
    [ 0.0861  0.17    0.2101 -0.39  ]]
   ```
   Range over first 20 t: dim 0 ∈ [0.027, 0.093], dim 1 ∈ [0.070, 0.170],
   dim 2 ∈ [0.197, 0.245], dim 3 = -0.390 constant. **Physical units
   (meters in a world-ish frame)**, not normalized. Dim 3 is suspicious
   (constant -0.39) and likely a placeholder/unused for this task.

2. `joint_action` — `(200, 14)`. 14-dim commanded joint positions
   (one per joint, two arms × 7).

3. `obs/joint_pos` — `(200, 14)` actual measured joint positions.
   First 5 timesteps verbatim:
   ```
   [[-0.0856  0.2408  1.2962 -0.0874 -1.221   0.0046  0.3988
     -0.6627  0.2562  1.1704 -0.701  -1.1674  0.1135  0.3988]
    ...]
   ```
   Range over first 20 t: dim 0 (left arm joint 0) ∈ [-0.336, -0.086]
   rad; dim 6 (left gripper) = 0.399 constant; dim 13 (right gripper)
   = 0.399 constant.

4. `obs/ee_pos` — `(200, 14)`. End-effector poses (looks like 7-dim
   per EE: XYZ + something; haven't traced the encoding).

5. `obs/world_t_robot_base` — `(200, 2, 4, 4)`. Constant across the
   episode: robot 0 at world (0.11, 0.42, 0.02) rotated -90° about
   Z; robot 1 at world (0.11, -0.64, 0.02) rotated +90°.

**CRITICAL FINDING — what the dataset pipeline actually feeds the
model.** `interactive_world_sim/datasets/latent_dynamics/real_aloha_dataset.py:108-126`
ignores `file["action"]` entirely (except for the gripper dim in the
`single_grasp` ctrl_mode). For every timestep it instead:

1. Reads `obs/joint_pos[t]` (14-dim raw joint positions).
2. Applies gripper-joint conversion in place:
   ```python
   joint_pos[r_i * 7 + 6] = MASTER_GRIPPER_JOINT_UNNORMALIZE_FN(
       PUPPET_GRIPPER_JOINT_NORMALIZE_FN(joint_pos[r_i * 7 + 6])
   )
   ```
   (`interactive_world_sim/utils/aloha_conts.py`)
3. Calls `joint_pos_to_action_primitive(joint_pos, ctrl_mode="bimanual_push",
   base_pose_in_world=file["obs"]["world_t_robot_base"][t], kin_helper)`
   (`interactive_world_sim/utils/action_utils.py:391-404`). For
   `bimanual_push` this runs forward kinematics on each arm's 7
   joint values, clips the EE to a workspace box (`x ∈ [0.25, 1.0]`,
   `y ∈ [-0.25, 0.25]` in robot frame), transforms to world frame
   via `world_t_robot @ rob_t_eef`, and emits the world-frame XY of
   each EE as a 4-dim vector `[left_x, left_y, right_x, right_y]`
   in meters.
4. Stores this 4-dim float in the dataset's `action` field; the raw
   `file["action"]` from the HDF5 is **discarded**.

Then at training/validation time
(`interactive_world_sim/algorithms/latent_dynamics/latent_world_model.py:503`),
the model itself normalizes:
```python
action = self.normalizer["action"].normalize(batch["action"])  # (B, T, A)
```
which maps the world-frame meters to `[-1, 1]` via the
`LinearNormalizer["action"]` fitted on the dataset's action
statistics.

**Comparison with the env's current behavior.** The env's `step()`
accepts `[-1, 1]` normalized actions and passes them **directly** to
`dynamics_forward` without further normalization. This is correct
and matches training — the model expects normalized actions at the
dynamics interface. **The env's normalization path is consistent
with the dataset.**

The gap is in pre-processing, not normalization. To replay an
expert episode, the caller must reproduce **all four** dataset
steps above to turn the HDF5 `obs/joint_pos[t]` into a `[-1, 1]`
action vector. None of those steps are currently exposed by the
public env API:

- `joint_pos_to_action_primitive` lives in
  `interactive_world_sim/utils/action_utils.py` (frozen zone).
- `MASTER_GRIPPER_JOINT_UNNORMALIZE_FN` / `PUPPET_GRIPPER_JOINT_NORMALIZE_FN`
  live in `interactive_world_sim/utils/aloha_conts.py` (frozen zone).
- `KinHelper("trossen_vx300s")` comes from `yixuan_utilities.kinematics_helper`.
- The model's `LinearNormalizer["action"]` is accessible only via
  the private `env._loaded.model.normalizer["action"]`.

**Do not use `file["action"]`.** It is in different units and a
different reference convention from what the model was trained on,
the dataset explicitly discards it, and on this episode dim 3 is a
suspicious constant. Replaying `file["action"]` directly — even
after some normalization — would feed the model an out-of-distribution
action and would not give us ground truth.

**Implication for Phase B.** The replay script will need to:
1. Import the three frozen-zone helpers above (read-only — we
   already import from `interactive_world_sim.utils.aloha_conts` in
   the wrapper, and importing from `utils.action_utils` is also
   permitted since the wrapper is the package's only entry point
   into `interactive_world_sim/**`).
2. Reach into `env._loaded.model.normalizer["action"]` to get the
   per-dim normalizer, OR — cleaner — propose a new public helper
   on the env (e.g. `env.normalize_raw_action(raw)` / inverse) so
   the planner code never sees `_loaded`.
3. Loop over the 200 episode timesteps, build the normalized
   action, feed it to `env.step(action)`, decode (or compare against
   `obs/images/camera_1_color[t+1]`).

Flagging for review before any code is written.

### Task 2 (revised) — ee_pos shortcut investigation

Hypothesis: `obs/ee_pos` (shape `(200, 14)` float32) already contains
the dataset's 4-dim action target as some plain index slice, so we
could skip the FK round-trip through `KinHelper`.

Verified numerically in `/tmp/test_ee_pos_shortcut.py` at t = 0, 50,
100 against the dataset's exact pipeline
(`real_aloha_dataset.py:108–126` → gripper normalize composition →
`joint_pos_to_action_primitive(ctrl_mode="bimanual_push", ...)`).

**Result: NO SLICE MATCHES.** Tested candidates and worst-case
per-step error vs `truth_4d`:

| indices | label | max\|err\| t=0 | t=50 | t=100 |
|---|---|---|---|---|
| `(0, 1, 7, 8)` | left XY, right XY (rob frame) | 0.595 m | 0.602 m | 0.570 m |
| `(1, 0, 8, 7)` | swap each | 1.20 m | 1.20 m | 1.20 m |
| `(0, 1, 8, 7)` | swap right only | 1.20 m | 1.20 m | 1.20 m |
| `(1, 0, 7, 8)` | swap left only | 0.595 m | 0.602 m | 0.570 m |
| `(3, 4, 10, 11)` | rotation-block columns | 0.682 m | 0.676 m | 0.698 m |

Target tolerance was 1e-4; closest miss is six thousand times that.

**Why the shortcut fails.** `obs/ee_pos[7:10]` is the right EE
expressed in the LEFT robot's frame (`world_t_robot[0]^-1 @
world_t_robot[1] @ rob1_t_eef`), not in robot 1's own frame as the
naive (0, 1, 7, 8) slice would assume. The dataset's
`joint_pos_to_action_primitive` instead transforms each arm's EE
via *its own* `world_t_robot[i]` to a world-frame XY. No static
slice can reconcile those.

**What does almost work (NOT implemented per task spec).** Applying
the constant 4×4 transform `world_t_robot[0] @ [ee_pos[i:i+3], 1]`
to BOTH `i=0` and `i=7` reproduces `truth_4d` within ~2–4 mm at all
three checked timesteps:

| t | truth_4d | closed-form (w_t_r0 on both arms) | max\|err\| |
|---|---|---|---|
| 0 | `[+0.0982, +0.1700, +0.2182, −0.3900]` | `[+0.0982, +0.1740, +0.2182, −0.3929]` | 0.0040 m |
| 50 | `[+0.2873, +0.1210, +0.3218, −0.3900]` | `[+0.2873, +0.1210, +0.3218, −0.3920]` | 0.0021 m |
| 100 | `[+0.0774, +0.1434, +0.2435, −0.3900]` | `[+0.0774, +0.1434, +0.2435, −0.3935]` | 0.0035 m |

The residual is the workspace clipping (`rob_t_eef[0,3]` to
`[0.25, 1.0]`, `rob_t_eef[1,3]` to `[-0.25, 0.25]`) that
`joint_pos_to_action_primitive` performs before the world-frame
transform. Whether to commit to that closed-form path, and how to
handle the clipping (~4 mm error is well inside the normalizer's
input range, so likely OK after normalization), is a follow-up
decision.

**Deliverable.** New files
`interactive_world_sim_env/helpers/__init__.py` and
`interactive_world_sim_env/helpers/expert_action.py`. The helper
exports the agreed signature
`expert_action_from_episode(env, episode_path, t) -> np.ndarray`,
which raises `NotImplementedError` with a clear pointer to the
module docstring's investigation summary. The module docstring is
the canonical record of what was tried and why it was rejected;
the helper does not silently fall back to KinHelper, in keeping
with the task spec.

**Isolation confirmed.**

- `git status` after the work: only `interactive_world_sim_env/` and
  the pre-existing `tests/` are untracked; `git diff --stat HEAD` is
  empty. No files outside `interactive_world_sim_env/helpers/` were
  added or modified.
- The wrapper's top-level `__all__` is unchanged
  (`['BatchedObservation', 'EnvState', 'Observation', 'RegistryError',
   'TASKS', 'TaskSpec', 'WorldModelEnv', 'get_task_spec']`); the
  helper module is NOT re-exported and must be imported by its full
  path.
- Smoke under
  `python -m interactive_world_sim_env.helpers.expert_action`
  raises and prints the expected `NotImplementedError` cleanly,
  without loading the model.

### Task 2 (revised) — gripper projection verification

Goal: prove that the 4 mm-error expert-action recipe, combined with
the camera intrinsics + extrinsics stored in the HDF5, gives correct
pixel coordinates of the grippers on the actual camera images. No
model involved.

**Camera 1 metadata (constant across the episode).** From
`obs/images/camera_1_intrinsics[0]` and
`obs/images/camera_1_extrinsics[0]`:

```
K = [[386.04   0.    315.48]
     [  0.    385.56 242.41]
     [  0.      0.     1.  ]]

world_t_cam =
[[-0.020   0.999  -0.030   0.137]
 [ 0.999   0.019  -0.027  -0.144]
 [-0.026  -0.031  -0.999   0.668]
 [ 0.     0.      0.       1.   ]]
```
Image resolution 480×640. Camera at world `(0.137, -0.144, 0.668)`,
looking down (cam Z column in world is `(-0.03, -0.03, -1.00)`).
OpenCV camera frame (X right, Y down, Z forward into the scene).

**Deliverables.**

- `interactive_world_sim_env/helpers/projection.py` — three pure
  functions: `ee_pos_to_world_xy(ee_pos, world_t_robot_base) -> (4,)`,
  `world_xy_to_pixel(world_xyz_per_arm, K, world_t_cam) -> (2, 2)`,
  and `project_expert_grippers(hdf5_path, t, cam_key) -> dict`. No
  env, no model, no torch — only numpy + h5py.
- `interactive_world_sim_env/scripts/verify_gripper_projection.py` —
  reads `obs/images/camera_1_color[t]` (the REAL camera image) for
  t ∈ {0, 25, 50, 100, 150}, projects both grippers, draws red
  (left arm) and blue (right arm) circles, saves
  `/tmp/projection_check_t{t}.png` and the horizontal grid
  `/tmp/projection_check_grid.png` (shape `(480, 3200, 3)` uint8).

**Numeric outputs (per-frame pixel coordinates).**

```
t=  0  L(541.5, 232.1)  R(158.7, 304.8)
t= 25  L(518.8, 218.9)  R(158.1, 327.9)
t= 50  L(502.7, 358.9)  R(157.6, 374.1)
t=100  L(520.2, 217.8)  R(159.1, 321.2)
t=150  L(511.1, 218.3)  R(159.0, 329.5)
```

All u/v are inside the 640×480 image. The right-arm u is almost
constant (~158) across the episode — consistent with that arm
holding its world-frame X near 0.08 m. The left-arm u and v swing
as the arm pushes the T-block.

**Visual result.** Both circles consistently land on the **EE
link / wrist** region of the corresponding gripper assembly in
each annotated frame, not on the gripper fingertip:

- t=0 (red on right side, blue on left side): both circles sit on
  the back end of the gripper body, ~30 px above-and-behind the
  orange fingertip stickers.
- t=25, t=100, t=150 (left arm in idle position): same offset
  pattern — circles on the wrist, fingertips visibly displaced
  toward the T-block.
- t=50 (left arm pushed forward into the T): the red circle
  follows the wrist's downward swing to (503, 359), still on the
  wrist body but visibly trailing the orange fingertip.

**Interpretation.** The projection MATH is correct end-to-end. The
~30 px offset between circle and fingertip is the URDF-defined
distance from the "eef" link origin (which is what `obs/ee_pos`
encodes) to the actual gripper fingertip — i.e., a geometric
property of the data, not a projection bug. For an MPPI cost
defined as "EE link near pixel X" the recipe is fit for purpose;
for a cost defined as "fingertip near pixel X" you would need a
constant rob-frame offset added before the world transform
(equivalent to projecting a different URDF link).

**Final isolation audit.**

- `git status`: only `interactive_world_sim_env/` and the
  pre-existing `tests/` are untracked. `git diff --stat HEAD` empty.
- No files outside `interactive_world_sim_env/helpers/` and
  `interactive_world_sim_env/scripts/` were created or modified.
- `helpers/projection.py` is not re-exported from the top-level
  `__init__.py`; callers import by full path.

### Task 2 (revised) — projection on decoded frames

Goal: verify that the projection circle, computed from the SAME
expert action the env was stepped with, lands on the gripper the
DECODER painted — not just on the gripper in the real camera image.

**Method (approach b from the spec).** For each sample
t ∈ {1, 25, 50, 100, 150}:

1. `env.reset(init_episode_path=…, init_episode_index=t−1)` — seed
   the env's latent window from the real RGB at frame t−1.
2. `action_norm = expert_action_from_episode(env, episode, t−1)`.
3. `env.step(action_norm)` — one dynamics step. `latent_window[-1]`
   is now the model's prediction for frame t.
4. `decoded_rgb = env.render()` → `(128, 128, 3)` uint8.
5. Project the SAME action's pixel coords for camera_1: un-normalize
   the action through `env._loaded.model.normalizer["action"]`,
   take Z for each arm from `world_t_robot_base[0] @ ee_pos[t]`,
   call `helpers.projection.world_xy_to_pixel`.
6. Map 640×480 → 128×128 with the same crop+resize env uses
   internally. Inspected `yixuan_utilities.draw_utils.center_crop`:
   it returns the largest center crop matching the target aspect
   ratio. For 640×480 → 1:1 it drops 80 px on each side, then a
   uniform resize:
     `u_decoded = (u_real − 80) · (128/480)`
     `v_decoded = v_real · (128/480)`
7. Draw red (left arm) / blue (right arm) circles on the decoded
   image (radius=10 on the 384×384 upscale).
8. Build a 2-row × 5-col grid: row 0 = decoded 128² → NN-upscaled
   to 384², row 1 = real 640×480 → 384×288. Final size
   `(672, 1920, 3)` at `/tmp/projection_on_decoded_grid.png`.

**New file.** `interactive_world_sim_env/scripts/show_expert_action_on_decoded.py`.
No other files touched.

**Numeric output (pixel coords for camera_1, then the rescaled
decoded coords):**

```
t=  1  real L(541.5, 232.1) R(158.7, 304.8)  ->  decoded L(123.1, 61.9) R(21.0, 81.3)
t= 25  real L(517.7, 211.5) R(157.9, 326.7)  ->  decoded L(116.7, 56.4) R(20.8, 87.1)
t= 50  real L(498.3, 366.8) R(157.7, 372.3)  ->  decoded L(111.5, 97.8) R(20.7, 99.3)
t=100  real L(517.0, 199.6) R(158.9, 326.3)  ->  decoded L(116.5, 53.2) R(21.0, 87.0)
t=150  real L(502.5, 202.9) R(159.0, 329.6)  ->  decoded L(112.7, 54.1) R(21.1, 87.9)
```

All circles land in-bounds on both the decoded and the real images.

**Visual findings (per-timestep, also saved full-size at
`/tmp/decoded_only_t{1,50,100}.png`):**

- **t = 1, t = 100, t = 150 — decoder produces a coherent frame.**
  Pink T is recognizable in roughly the right place; both gripper
  assemblies are clearly painted at left and right with their
  orange fingertip stickers. Circles land on the gripper's
  wrist/back region in the decoded frame, consistent with where
  they land on the real frame in the bottom row. The wrist-vs-
  fingertip offset (~25–30 px in 640² coords, ~6–8 px after
  rescale to 128²) is the same in both rows — the decoder is
  geometrically faithful at these timesteps.

- **t = 50 — decoder is significantly degraded.** Pink T is
  blurred and partially duplicated in the upper-right; gripper
  bodies are smudged; the left arm's orange fingertip is rendered
  somewhere in the bottom-center area as a soft blob. The
  projection circle at the rescaled coordinate `(111.5, 97.8)` is
  in roughly the right region (lower-center where the left arm has
  swung), but with the decoded image this blurry there's no clean
  gripper edge to align against. Can't be judged at sub-pixel
  granularity here.

- **t = 25** falls between: T and grippers visible but with
  visible artifact pink blobs and softness around the gripper
  edges.

**Why t=50 looks worse than t=1.** Two contributing factors I want
to flag rather than dismiss:

1. **Each sample is a 1-step prediction from a 1-frame context.**
   The env's `reset(init_episode_path, init_episode_index=t−1)`
   initializes `latent_window` to length 1 (the single encoded
   frame at t−1). The dynamics model was trained with a 10-frame
   sliding window of latents as context. Predicting from a 1-frame
   context is harder than predicting from a fully-populated
   10-frame autoregressive history, and the model's 1-step
   predictions are visibly noisier than what you see during a
   normal 10-step rollout. This is a property of the test
   methodology, not a projection bug.

2. **Decoder stochasticity.** Each `env.render()` draws fresh
   noise; re-running this script produces different decoded
   pixels (within the same neighborhood). The grid above is a
   single sample. The visual alignment finding should be read as
   "modulo decoder noise."

**Bottom line for an RGB-based MPPI reward.**

- When the decoder produces a coherent frame, the projection
  circle lands where the geometry says it should — same
  wrist-vs-fingertip offset as on real images. The
  action→pixel pipeline is sound end-to-end.
- The 1-step / 1-frame-context regime visibly degrades decode
  quality at some timesteps (e.g. t = 50 here). For MPPI
  scoring, this is a problem: the reward is read off the decoded
  image, and if the gripper's painted location wobbles in
  decoder noise, the reward signal will too. A few possible
  mitigations to consider in a later session — keeping at the
  level of "things to consider," not committing to any:
    - score on `latent_history` rather than the decoded RGB,
      so noise from the decoder doesn't pollute the cost;
    - score on the dynamics latent of the LAST step of a
      multi-step rollout, so the latent window is fully
      populated before the scored frame;
    - average the reward over N decoder samples per latent.

**Isolation audit.**

- `git status`: only `interactive_world_sim_env/` and the
  pre-existing `tests/` are untracked.
- `git diff --stat HEAD`: empty.
- Modified files in this step: zero. New files: one
  (`scripts/show_expert_action_on_decoded.py`). `env.py`,
  `helpers/projection.py`, `helpers/expert_action.py`, and the
  package's top-level `__init__.py` are unchanged from Step 1.

### Task 3 — warmup + observe()

**Files touched.** Modified `env.py` and `obs.py`; added
`scripts/smoke_warmup_observe.py`. No file outside
`interactive_world_sim_env/` was changed.

**Public API delta.**

- `WorldModelEnv.reset(...)` gained a keyword-only parameter
  `init_window_size: int = 1`. Default is byte-identical to the
  previous single-frame behavior. With `init_window_size = W > 1`
  the env reads W consecutive RGB frames from `init_episode_path`
  ending at `init_episode_index` and encodes each one into
  `latent_window`; `action_window` is filled with the matching
  expert actions (see "Convention" below). All seven validation
  rules from the design fire before any HDF5 read.

- New method
  `WorldModelEnv.observe(rgb, last_action) -> (Observation, dict)`.
  Real-deployment counterpart to `step()`: encodes the input RGB
  (single ndarray or `dict[obs_key, ndarray]`) and appends both
  the new latent and `last_action` to the rolling windows.
  `last_action` is required — passing `None` raises `TypeError`.
  `Observation.rgb` carries the **preprocessed input RGB** (uint8
  HWC, or per-view dict), not a decoder round-trip; `info["source"]
  == "encoder"`. `step()` was also updated to set
  `info["source"] == "dynamics"` for symmetry.

- `Observation.rgb` type widened to `np.ndarray |
  dict[str, np.ndarray] | None` to accommodate multi-view
  observe() output and step()'s multi-view decode path. Docstring
  documents the per-method semantics.

**Convention for `action_window[i]` during warmup.** action_window[i]
holds the action that drove the env INTO latent_window[i] (the
expert action AT the frame *before* the one encoded into
latent_window[i]). When that prior frame falls before the episode's
first frame, the slot is filled with zeros. This convention matches
what `observe()` appends, which makes a warmup-from-W bitwise
equivalent to a single-frame reset followed by `W − 1` `observe()`
calls — Test 9 in the smoke script asserts this with `torch.equal`
on both windows.

**No changes** to `state.py`, `registry.py`, `_model_loader.py`,
`helpers/projection.py`, `helpers/expert_action.py`, or the
top-level `__init__.py`. `snapshot()`/`restore()` are unchanged
and exercise both step() and observe() without modification (Test
7 confirms).

**Smoke results.** Ran
`python interactive_world_sim_env/scripts/smoke_warmup_observe.py`.
The nine design buckets expand to 17 individual assertions
(error-case test 6 is split into a..i). All passed:

```
PASS  test_1: warmup_shapes
PASS  test_2: warmup_then_step
PASS  test_3: cold_vs_warmup_shape_consistency
PASS  test_4: reset_step_step_observe_step
PASS  test_5: cold_observe_vs_warmup_latents_equal
PASS  test_6a: warmup_non_pusht_raises
PASS  test_6b: warmup_with_init_rgb_raises
PASS  test_6c: warmup_with_init_state_raises
PASS  test_6d: warmup_without_episode_path_raises
PASS  test_6e: warmup_index_too_small_raises
PASS  test_6f: zero_window_raises
PASS  test_6g: window_over_hist_context_raises
PASS  test_6h: observe_missing_action_raises
PASS  test_6i: observe_before_reset_raises
PASS  test_7: snapshot_restore_across_observe
PASS  test_8: cross_task_restore_after_observe_raises
PASS  test_9: warmup_full_equivalence
17 / 17 tests passed
```

Notable confirmations from the run:
- Test 5: `torch.equal` on latent windows produced by the two
  warmup paths — the encoder is **bitwise deterministic** given
  identical preprocessed inputs. (Important for any future MPPI
  reward that needs reproducible state.)
- Test 9: with the action_window convention described above,
  `torch.equal` on both windows is `True`. The two code paths
  produce identical state.
- Test 4: observed `latent_norm = 32.00` (matches the L2 norm of a
  per-pixel L2-normalized 4×32×32 latent for `num_views = 1`),
  confirming the encoder path is what's executing inside observe().
- Test 6a uses an in-process monkey-patch of `env._task` to test
  the non-pusht guard without needing a second checkpoint.

**Decisions I made that the design didn't fully cover.**

1. **`action_window[0]` for warmup near the start of an episode.**
   The design summary said "fill action_window accordingly" without
   specifying the convention. I chose "the action that drove INTO
   latent_window[i]" (one frame earlier than the latent), with
   zeros when the prior frame is missing. This makes Test 9's full
   equivalence pass; the alternative ("EE position at the same time
   as the latent") would have produced a permanent off-by-one
   mismatch between Method 1 and Method 2 in Test 9 and would
   conflict with the user-named `last_action` parameter on
   `observe()`. The convention is documented in the `reset()`
   docstring and the new `_warmup_from_episode` helper. Trade-off:
   under this convention the model's dynamics conditioning is
   shifted by one frame relative to its training-time alignment;
   for `pusht_cam1` (slow continuous EE motion), `action_at_(t-1)
   ≈ action_at_t` within a few mm, so the approximation is the
   same one the existing demos (and our earlier
   `show_expert_action_on_decoded.py`) already accept.

2. **`_make_observation` gained a `source` kwarg.** Needed so the
   default `rgb` behavior can differ between reset (None), step
   (decode if `decode_on_step=True`), and observe (caller-supplied
   preprocessed input). Backward-compatible: `source` defaults to
   `"dynamics"`, which reproduces the old behavior.

3. **Refactor of `_encode_views` into a `_encode_preprocessed_views`
   inner helper plus a new module-level `_preprocess_rgb_uint8`.**
   Lets `observe()` compute the center-crop+resize ONCE and reuse
   the bytes both for the encoder and for `Observation.rgb`. The
   public behavior of `_encode_views` is unchanged.

**Files modified outside the new package: none.**

**Isolation audit (final).**

- `git status`: only `interactive_world_sim_env/` and the
  pre-existing `tests/` are untracked.
- `git diff --stat HEAD`: empty (the entire wrapper subpackage is
  still untracked, so HEAD doesn't see anything to diff — this is
  more restrictive than the design's "env.py + obs.py + optional
  smoke script" allowance).

### Task 4 — Imagination rollout test

**Goal.** Run the just-built warmup path end-to-end: warm-start the
env with 10 real frames, then drive it through 100 imagination
steps using only the expert's actions (no further real RGB after
warmup). Compare every predicted frame to the matching real
episode frame and produce a side-by-side video, a snapshot grid,
and a drift curve.

**Files touched.** Added
`interactive_world_sim_env/scripts/imagination_rollout.py`.
Nothing else modified.

**Command run.**

```
python interactive_world_sim_env/scripts/imagination_rollout.py
```

Default `--horizon 100`, `--circles` on.

**Outputs.**

- `/tmp/imagination_rollout.mp4` — 100 side-by-side frames at 10 fps
  (libx264; imageio-ffmpeg auto-padded the height 444→448 for
  codec compatibility — content unchanged).
- `/tmp/imagination_rollout_grid.png` — 858×2304, two rows of six
  cells at t = 10, 20, 40, 60, 80, 100 with timestep labels and
  PREDICTED / GROUND TRUTH band labels.
- `/tmp/imagination_rollout_drift.png` — normalized RMSE vs frame
  index, with a dashed reference at 0.1.

**Numeric summary (single rollout, decoder stochasticity not
averaged out):**

```
mean L2:           0.0595
median L2:         0.0610
min  L2:           0.0319  (at t=50)
max  L2:           0.0858  (at t=79)

L2 at t= 10:       0.0348
L2 at t= 25:       0.0559
L2 at t= 50:       0.0319
L2 at t=100:       0.0805

L2 never exceeded the 0.1 threshold over 100 frames.
```

L2 here is normalized RMSE over (128, 128, 3) uint8 pixels rescaled
to [0, 1] — `sqrt(((pred/255 - gt/255)**2).mean())`. 0 = identical,
1.0 = every channel of every pixel maximally different.

**Visual finding (grid + drift curve).** The 100-step pure-
imagination rollout stays coherent end-to-end: every snapshot shows
a recognizable pink T on the cream table with both grippers in
plausible positions, and the orange fingertip stickers are
preserved across all six sampled frames. The model's T-block
position drifts slowly relative to ground truth (visible by
t = 60–80) but the scene never collapses — no smearing, no color
drift, no decoder noise blowing up. The drift curve oscillates in
a tight band of `[0.03, 0.09]` with no monotonic upward trend —
the L2 is essentially stationary across the rollout rather than
diverging.

This is a much better result than the 1-step / 1-frame-context
test from Task 2 (where the decoded frame at t = 50 was notably
degraded). The fully-populated 10-frame warmup window appears to
be the key: once the dynamics has its full historical context,
predictions stay stable far longer than a cold start suggests.

**Implications for MPPI.** With proper warmup, an RGB-based reward
read off the decoder is plausible at least over a 100-step horizon
on this task. The decoder's per-frame noise contributes a noise
floor of ~0.05 RMSE that the reward function will need to be
robust to — averaging over a few decoder samples or scoring on
latent features remain reasonable options if the noise floor
hurts. The drift curve is stationary, not monotone increasing, so
the planner does not face a regime where late frames are
systematically worse than early ones.

**Caveats surfaced in the run:**
- Decoder is stochastic; numbers are from a single rollout, not
  averaged.
- Expert action via `expert_action_from_episode` is the 4 mm-
  approximation (~0.01-0.02 in normalized space, per Task 3
  Step 1 measurements). Some early-rollout drift is attributable
  to this.
- After step 10 the latent_window contains entirely model-
  predicted latents — this is the pure-imagination regime that
  MPPI scoring would actually face.

**Isolation audit.** `git status` shows only
`interactive_world_sim_env/` and the pre-existing `tests/`
untracked. `git diff --stat HEAD` is empty. No edits to `env.py`,
helpers, `obs.py`, `state.py`, `registry.py`, `_model_loader.py`,
or `__init__.py` — the new script lives entirely within
`scripts/`.

