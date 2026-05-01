# MPPI Bottleneck Analysis

Drafted 2026-04-25 on local laptop GPU.

## Section 1 — Methodology

I profiled one MPPI `plan_step` for the easy-pair sanity configuration:

- Pair: `easy_pair_2`
- Initial: `data/mini/pusht/val/episode_2.hdf5`, frame `2`
- Goal: `tests/goal_selection/easy_pair_goals/easy_pair_2_goal.pt`
- Config: `N=32`, `H=10`, `n_update_iter=30`, `control_steps=1`, `seed=0`
- Memory knob: `decode_batch_size=8`, because batch-32 decode OOMs on the local 11.5 GiB GPU
- Environment: conda env `/home/yiru-wu/miniconda3/envs/iws`

Instrumentation was profiling-only and lived under `/tmp`, not in production code:

- `/tmp/profile_mppi_plan_step.py`: full 30-iteration synchronized component profile.
- `/tmp/mppi_plan_step_profile.json`: raw timing output.
- `/tmp/profile_mppi_short.py`: 3-iteration validation profile without GPU-util sampling.

Timers used `time.perf_counter()` plus `torch.cuda.synchronize()` around GPU components so rollout/decode time is assigned to the component that launched it, rather than deferred to the next CPU sync. This makes component attribution cleaner but increases wall time versus the normal runner. For production wall time, I used the existing run output:

- `outputs/mppi/easy_iter30_local_sanity/summary.json`
- Reported `wall_time_s = 346.1` for one control step / one plan step.

I also sampled GPU utilization during the full synchronized profile via `nvidia-smi` every 0.5s. That sampling is diagnostic only; it likely adds overhead and should not be treated as production timing.

## Section 2 — Breakdown of One Plan Step

Production runner wall time for one plan step:

| Run | Config | Wall time |
|---|---:|---:|
| `outputs/mppi/easy_iter30_local_sanity` | `N=32`, `n_update_iter=30`, `control_steps=1`, `decode_batch_size=8` | `346.1s` |

Synchronized component profile for the same logical plan step:

| Component | Wall time per profiled plan_step | % of profiled total | Calls per plan_step | Already batched? |
|---|---:|---:|---:|---|
| WM dynamics rollout | `481.1s` | `56.4%` | `30` rollout calls, each batched over `N=32`, horizon `H=10` | yes |
| WM decode | `121.1s` | `14.2%` | `120` decode calls = `30 iters × 4 chunks`; each chunk has 8 latents | partial; batch-32 OOMs locally |
| CV labeling | `241.7s` | `28.4%` | `960` OpenCV/ICP calls = `30 × 32` | no |
| GPU to CPU frame prep | `0.47s` | `0.06%` | `120` chunks | n/a |
| Sampling/clipping | `0.11s` | `0.01%` | `30` calls | yes |
| Softmax/weighted mean | `0.15s` | `0.02%` | `30` calls | yes |
| Reward math | `0.02s` | `<0.01%` | `30` calls | yes |
| Iteration logging | `0.01s` | `<0.01%` | `30` calls | yes |
| Unattributed Python overhead | `7.74s` | `0.9%` | loop/bookkeeping | n/a |
| **Total, synchronized profile** | **`852.6s`** | **`100%`** |  |  |

Short validation profile over 3 iterations, no GPU-util sampler:

| Component | Time for 3 iters | Per iter |
|---|---:|---:|
| WM dynamics rollout | `47.5s` | `15.8s` |
| WM decode | `10.9s` | `3.6s` |
| CV labeling | `12.1s` | `4.0s` |
| Sampling + reward + softmax | `0.17s` | `0.06s` |
| Total | `70.8s` | `23.6s` |

GPU utilization during the full synchronized profile:

- Samples: `1396`
- Average GPU utilization: `79.9%`
- Max GPU utilization: `100%`
- Average memory used by `nvidia-smi`: `7447 MiB`
- Max memory used by `nvidia-smi`: `11725 MiB`
- PyTorch peak allocated memory: `5493 MiB`

Important interpretation note: the synchronized component profiles are slower than the normal runner because they force extra synchronization and include GPU-util sampling. The production runner’s `346.1s` is the correct wall-clock number for local user experience; the component profiles identify relative cost structure.

## Section 3 — Which Hypothesis Matches?

**Hypothesis A: CV labeling is unbatched and dominates.**

Partly supported, but not dominant. CV is unbatched and expensive: `960` sequential OpenCV/ICP calls. In the full synchronized profile it took `241.7s` (`28.4%`); in the short validation profile it took `4.0s/iter`. It is a real bottleneck, but rollout is larger.

**Hypothesis B: Decode is the bottleneck even though it should be parallel.**

Contradicted for local `decode_batch_size=8`. Decode took `121.1s` (`14.2%`) in the full profile and `3.6s/iter` in the short profile. Decode is not free and batch-32 decode OOMs locally, but it is not the largest cost after chunking.

**Hypothesis C: GPU-CPU sync barriers at every CV call.**

Supported as a structural serialization issue. The actual copy/prep time is tiny (`0.47s` total), but CV requires decoded images on CPU, so GPU decode must finish before CPU CV can begin. There is no overlap between rollout, decode, and CV in the current implementation.

**Hypothesis D: Memory allocator overhead at iteration boundaries.**

Inconclusive, weakly supported for occasional outliers. Full-profile iteration times had repeated outliers: iter `1=60.5s`, `7=59.4s`, `11=49.5s`, `23=57.3s`; the median was `24.7s`. However PyTorch allocated/reserved memory after each iteration stayed flat (`~195 MiB allocated`, `232 MiB reserved` after `empty_cache`), with no monotonic growth. This does not look like a simple leak.

**Hypothesis E: Python loop dispatch overhead from inner-loop kernel launches.**

Contradicted. Pure sampling, softmax, reward math, logging, and unattributed Python overhead are all under `1%` of synchronized profile time. Average GPU utilization was about `80%`, not a Python-dispatch starvation pattern.

**Hypothesis F: The decoder itself is the architectural ceiling.**

Contradicted as phrased. Decode is a ceiling for local batch size because batch-32 OOMs, but it is not the largest measured time. The architectural ceiling is more accurately the combination of WM dynamics rollout plus CPU CV reward evaluation.

Comparison to Phase 2:

- `outputs/mppi/phase2_sanity_far/summary.json`: `N=16`, `n_update_iter=5`, `control_steps=50`, wall `1501.4s`.
- Phase 2 per control step: `1501.4 / 50 = 30.0s`.
- Candidate finals per Phase 2 plan step: `16 × 5 = 80`.
- Current sanity candidate finals per plan step: `32 × 30 = 960`.
- Candidate ratio: `960 / 80 = 12.0×`.
- Wall ratio: `346.1 / 30.0 = 11.5×`.

Conclusion: relative to the current refactored Phase 2 runner, local runtime scales approximately linearly with `N × n_update_iter`; I do not see strong evidence of super-linear slowdown. The run is painful because `960` final states per plan step is a lot of WM rollout/decode/CV work, not because the algorithmic bookkeeping exploded.

## Section 4 — Concrete Optimization Opportunities

1. **Run the iter=30 experiment on cloud GPU**

- What it changes: no algorithm change; run on L40S/cloud hardware and likely avoid `decode_batch_size=8`.
- Estimated speedup: likely `2×–5×` locally observed wall, plus fewer OOM constraints. Exact speedup needs one cloud sanity run.
- Effort: `0.5–1h`.
- Risk: low; same code path and artifacts.
- Dependencies: cloud checkout must include Stage A easy-pair artifacts and the memory-only decode chunking hook if local-style chunking is still used.

2. **Reduce local sanity scope**

- What it changes: use `control_steps=1`, fewer seeds, or lower `N`/`n_update_iter` for local diagnostics.
- Estimated speedup: linear with candidate finals; e.g. `n_update_iter=10` is about `3×` faster than `30`.
- Effort: `0.1h`.
- Risk: scientific risk only; less evidence about deep refinement.
- Dependencies: none.

3. **Parallelize CV labeling across CPU workers**

- What it changes: run the `960` OpenCV/ICP labels through a process/thread pool or batched worker queue.
- Estimated speedup: CV portion could improve `2×–6×`; overall speedup likely `1.2×–1.4×` if rollout remains dominant.
- Effort: `4–8h`.
- Risk: medium. Need preserve output ordering and avoid OpenCV/scipy thread oversubscription. Tests needed for exact reward/log structure.
- Dependencies: stable serialization of frames/states; may need a worker pool lifecycle in `PushTWMEnv` or runner.

4. **Optimize or restructure WM rollout**

- What it changes: target the largest measured component, `env.rollout(z0_batch, act_seqs)` called `30` times per plan step.
- Estimated speedup: unknown; if rollout can improve `2×`, overall local speedup could approach `1.5×–2×`.
- Effort: `1–3 days` to profile model internals properly, more to implement.
- Risk: high. The consistency tests protect MPPI logic, but dynamics changes risk changing generated latents/rewards.
- Dependencies: torch profiler/Nsight profile of `DifferentiableDynamics.rollout` and the latent dynamics model.

5. **Tune decode chunk size**

- What it changes: try `decode_batch_size=12` or `16` on local GPU to reduce decode-call overhead while staying below OOM.
- Estimated speedup: small, maybe `1.05×–1.15×` overall, because decode is only `14–15%`.
- Effort: `0.5h`.
- Risk: low except OOM.
- Dependencies: current `decode_batch_size` hook.

6. **Replace CV reward with a faster learned or vectorized proxy**

- What it changes: avoid CPU OpenCV/ICP in the inner MPPI loop.
- Estimated speedup: removes up to `17–28%` local measured time and removes sync barrier.
- Effort: days to weeks depending on reward proxy.
- Risk: high. It changes the diagnostic target: no longer "frozen WM + classical CV reward."
- Dependencies: validated state estimator/probe or GPU-native CV.

## Section 5 — Recommended Next Action

Run the selected `iter=30` easy-pair experiment on cloud before optimizing code.

Reason: the local bottleneck is not an easy Python bug. The dominant measured costs are WM rollout and CPU CV, and runtime scales roughly linearly with the number of candidate finals (`11.5×` wall for `12×` more candidates versus Phase 2). Local optimization work is likely to save at most a modest factor unless we rewrite rollout or replace CV, both of which risk changing the experiment. A cloud run preserves the diagnostic semantics and is the lowest-risk way to answer the scientific question first.

## Section 6 — Things Tried That Did Not Help

- **Batch-32 decode without chunking**: failed with CUDA OOM on the local 11.5 GiB GPU.
- **`decode_batch_size=8`**: avoided OOM, but did not make the full local run fast enough for 6 sequential runs.
- **`PYTORCH_ALLOC_CONF=expandable_segments:True`**: useful for allocator behavior/OOM mitigation, but did not remove the main runtime cost.
- **Full synchronized component profiling with `nvidia-smi` sampling**: useful for component attribution, but too intrusive for production wall time (`852.6s` profiled vs `346.1s` normal runner).
