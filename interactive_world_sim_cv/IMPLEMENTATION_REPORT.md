# interactive_world_sim_cv — Implementation Report (Phase 3)

A firewalled, self-contained wrapper around aloha's pink-T pose detector,
prepared for use as an MPPI reward source.

The package contains a **verbatim copy** of aloha's pose-detection code
(`_detection.py`). Production code does not import from the aloha repo
at runtime — see "Approach (b) rationale" below. The aloha repo at
`/home/yiru-wu/Documents/aloha` is **reference only**.

## 1. Source provenance

| | |
|---|---|
| Aloha source file | `aloha/world_model/eval/analyze.py` |
| Aloha repo path (at copy time) | `/home/yiru-wu/Documents/aloha` |
| Aloha SHA (`git rev-parse HEAD`) | `dc5b117113a064b7e24b7fd69a174618065c0ff5` |
| Aloha tip commit message | `Add dim6 left-arm absolute position monkey patch, fix render/wdmdl scripts` (2026-01-27) |
| Copy date | 2026-05-12 |

If aloha is updated upstream and we want the new behavior, **re-copy**
the relevant blocks rather than editing `_detection.py` in place. The
bitwise equivalence tests (`scripts/smoke_detector.py` tests 1 and 1b)
guard against accidental drift.

## 2. Files

```
interactive_world_sim_cv/
├── __init__.py                #  16 lines  — re-export TPose, detect, detect_batch, DetectorPool
├── api.py                     # 247 lines  — public API + four-step preprocessing header comment
├── _detection.py              # 195 lines  — verbatim copy from analyze.py
├── _pool.py                   #  92 lines  — worker function, template cache, hsv_for_mode
├── py.typed                   #   0 lines  — PEP 561 marker
├── IMPLEMENTATION_REPORT.md   # (this file)
└── scripts/
    └── smoke_detector.py      # 379 lines  — 7-test smoke suite
```

## 3. Approach (b) rationale

Per the Phase 3 brief, we copied the detection algorithm verbatim rather
than loading it at runtime via `importlib.util.spec_from_file_location`:

- **Portability.** The wrapper runs on any machine that clones this
  repo, without requiring aloha at a specific path.
- **Stability.** aloha may update; we resync explicitly rather than
  being silently affected.
- **Self-containment.** The package's behavior is visible inside its
  own directory tree.

The single residual use of `spec_from_file_location` is in
`scripts/smoke_detector.py` **for the bitwise equivalence tests only**.
Production code (`__init__.py`, `api.py`, `_detection.py`, `_pool.py`)
contains no reference to the aloha repo.

## 4. Symbols copied — line ranges & verification

All copies are **byte-identical** to their source. Self-check performed
via `diff` of `sed -n 'X,Yp' analyze.py` against the corresponding
range in `_detection.py`. No whitespace differences anywhere — all 10
copied blocks matched line-for-line.

| Symbol | analyze.py | _detection.py | Diff |
|---|---|---|---|
| `T_BLOCK_SHAPE` | 43–52 | 42–51 | identical |
| `T_BLOCK_FILLED_CENTROID` | 55 | 55 | identical |
| `HSV_LOWER_REAL`, `HSV_UPPER_REAL` | 59–60 | 60–61 | identical |
| `HSV_LOWER_WM`, `HSV_UPPER_WM` | 63–64 | 64–65 | identical |
| `sample_contour` | 84–87 | 69–72 | identical |
| `rotate_points` | 74–81 | 76–83 | identical |
| `trimmed_icp` | 103–140 | 87–124 | identical |
| `get_template_contour` | 90–100 | 128–138 | identical |
| `detect_t_block_mask` | 143–153 | 142–152 | identical |
| `estimate_current_pose` | 156–208 | 156–208 | identical |

Top-level ordering in `_detection.py` was rearranged to put leaves
before roots (`sample_contour` before `get_template_contour`,
`trimmed_icp` before `estimate_current_pose`, etc.) per the brief.
**No interior reordering.** Each block carries a
`# Verbatim from analyze.py:LINE-LINE` header citing its source range.

Imports trimmed to the detection path only: `cv2`, `numpy`,
`scipy.spatial.cKDTree`. Dropped from analyze.py at top of file:
`argparse, csv, re, time, concurrent.futures, dataclass, pathlib,
h5py, matplotlib, pandas, scipy.stats, shapely, tqdm`.

## 5. Public API surface

```python
@dataclass(frozen=True)
class TPose:
    x: float           # pixel coords in processing resolution (default 512²)
    y: float
    sin: float         # math.sin(math.radians(angle_deg))
    cos: float         # math.cos(math.radians(angle_deg))
    angle_deg: float   # raw aloha output, unwrapped, roughly in [0, 690]
    error: float       # ICP residual, pixels in processing resolution

def detect(rgb, mode, processing_resolution=512) -> TPose | None: ...
def detect_batch(rgbs, mode, processing_resolution=512, num_workers=None) -> list[TPose | None]: ...

class DetectorPool:
    def __init__(self, num_workers: int): ...
    def __enter__/__exit__: ...
    def start() / shutdown(wait=True): ...
    def detect_batch(rgbs, mode, processing_resolution=512) -> list[TPose | None]: ...
```

The four-step preprocessing block (resize 128→512 INTER_CUBIC, RGB→BGR,
template at scale=processing_resolution/512, run `estimate_current_pose`)
is documented as a header comment in `api.py` with citations to
analyze.py line numbers, despite the actual code now living in
`_detection.py`.

## 6. Smoke-test results — 7/7 PASS

Test environment: Python 3.11.15 (conda env `iws`), cv2 4.10.0, numpy
1.26.4, scipy 1.17.1; aloha reference at SHA above.

Inputs:
- Real frame: `data/mini/pusht/val/episode_0.hdf5` obs/images/camera_1_color[0], preprocessed to 128×128 RGB.
- Decoded frames: 10-frame warmup + 100 imagination steps via
  `WorldModelEnv("pusht_cam1")`, 128×128 RGB.

| # | Test | Result | Detail |
|---|---|---|---|
| 1 | Bitwise vs aloha reference (real frame, mode='real') | **PASS** | (x, y, angle_deg, error) bitwise equal |
| 1b | Bitwise vs aloha reference (decoded frame, mode='wm') | **PASS** | (x, y, angle_deg, error) bitwise equal |
| 2 | 10 decoded frames, no exceptions | **PASS** | 10/10 detected; error range 0.6–0.8 |
| 3 | Batched vs sequential equivalence (N=20, num_workers=4) | **PASS** | all 20 elements equal |
| 4 | Wrong-mode HSV (3 probes) | **PASS** | probe 3 (synthetic magenta) confirms HSV ranges differ — see §7 |
| 5 | Empty-frame failure (zeros image) | **PASS** | both modes return None |
| 6 | Speed: N=100, sequential vs one-shot vs persistent pool | **PASS** | persistent 6.11× sequential — well above 1.5× |

Test 1 (real frame, representative numbers):

```
ours.x       = 223.546357    ref_center[0] = 223.546357
ours.y       = 252.088103    ref_center[1] = 252.088103
ours.angle_deg = -0.435340    ref_angle     = -0.435340
ours.error   = 0.555595    ref_error     = 0.555595
```

Test 1b (decoded frame, representative numbers):

```
ours.x       = 222.351214    ref_center[0] = 222.351214
ours.y       = 252.489326    ref_center[1] = 252.489326
ours.angle_deg = -0.272164    ref_angle     = -0.272164
ours.error   = 0.788051    ref_error     = 0.788051
```

The bitwise tests confirm the verbatim copy has **zero algorithmic
drift**. Any future change to `_detection.py` that breaks them would
be caught immediately.

Test 6 speed:

| Variant | Time for 100 frames | Per-frame | Speedup |
|---|---|---|---|
| (a) sequential | ~4100 ms | ~41 ms | 1.0× |
| (b) one-shot pool n=8 | ~875 ms | ~8.8 ms | 4.7× |
| (c) persistent pool n=8 | ~670 ms | ~6.7 ms | **6.1×** |

The 1.5× threshold is comfortably exceeded. Persistent pool is ~25%
faster than one-shot pool, as expected — one-shot pays the
ProcessPoolExecutor startup cost (~200 ms) on every call.

## 7. Test 4 finding — HSV ranges coincide on natural inputs

The two HSV ranges:

| Range | Lower (H, S, V) | Upper (H, S, V) |
|---|---|---|
| `HSV_LOWER_REAL` / `HSV_UPPER_REAL` | (160, 50, 100) | (179, 200, 244) |
| `HSV_LOWER_WM` / `HSV_UPPER_WM` | (140, 50, 100) | (179, 255, 255) |

WM is a **set-theoretic superset** of REAL: lower H wider (140 vs 160),
upper S wider (255 vs 200), upper V wider (255 vs 244).

Probes 1 and 2 (real episode frame, decoded WM frame) both produced
**identical** TPose for mode='wm' vs mode='real'. This is not a bug —
the codebase's pink T-blocks, in both real captures and WM renders,
have hue in [160, 179], saturation ≤ 200, and value ≤ 244, so they
fall in both ranges. The detection picks up the same blob either way.

Probe 3 (synthetic RGB-magenta T with hue 150) is **outside** REAL but
**inside** WM, and confirms the ranges are doing real work when the
input color exercises the asymmetry: mode='wm' detects, mode='real'
returns None.

**Implication for MPPI.** For this codebase's decoded frames, setting
`mode='wm'` and `mode='real'` will produce the same detection. The
mode parameter is mostly a forward-compatibility hook for cases where
the WM render or the real T color drifts. **Pick `mode='wm'` as the
MPPI default** — the WM range is the more permissive one, so it
remains correct if the decoder's pink ever lands in [140, 160).

## 8. Architectural notes

### Process-pool fork after CUDA init

The smoke test loads `WorldModelEnv`, which puts PyTorch tensors on
GPU. It then forks ProcessPoolExecutor workers (test 3, test 6). The
default Linux fork inherits CUDA state into children that don't use
CUDA — this works in practice because our workers only use CV ops.
Before spawning the pool, the smoke test calls `torch.cuda.empty_cache()`
and `gc.collect()` to reduce inherited memory. **No CUDA errors
observed.** If users encounter issues, the workaround is to construct
`ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn"))`
— not currently exposed on the public `DetectorPool` API; can be added
if needed.

### Template-contour cache

`_pool._cached_template(processing_resolution)` is `@lru_cache(maxsize=8)`,
keyed on the integer resolution. Each ProcessPoolExecutor worker fills
its own cache on first detection at a given resolution. This avoids
rebuilding the 600×600 template mask on every call (~few ms savings).
Caching by `int` instead of by the float `t_scale` keeps hashes stable.
This is the **only** optimization the wrapper adds over aloha's
`estimate_current_pose` — same template, same args, just memoized.

### Lazy TPose import in `_pool.py`

`_pool._worker_task` lazily does `from .api import TPose` to break a
circular import (`api → _pool` for the worker; `_pool → api` for TPose).
Fork inherits `sys.modules`, so the lazy import is a fast sys.modules
lookup in the worker.

## 9. Files not added

- No `tests/` directory; the smoke suite lives in `scripts/` matching
  the `interactive_world_sim_env` convention.
- No `__pycache__` (root `.gitignore` covers it).
- No outputs written to git; the smoke test writes nothing.

## 10. What Phase 4 will produce

Per the Phase 2 brief (deferred to follow-up command):

- `/tmp/cv_overlay_real.png`: 10 real-episode frames with detected
  pose overlay using `mode='real'`.
- `/tmp/cv_overlay_decoded.png`: 10 decoded frames from a fresh
  `imagination_rollout` with detected pose overlay using `mode='wm'`.

Marker: small filled circle at `(x, y)` + ~30-px line in direction
`(cos, sin)`. Overlays drawn at processing resolution (512²) or an
upscaled view, not 128². If decoded detections look qualitatively
worse than real, flag as MPPI-reward viability risk.

---

## Phase 4 — Visualization (follow-up session)

Generated by `scripts/visualize_detection.py`. Each PNG is 1616×5456 px:
a header row + ten 542-px rows, each row = label band + three 512²
cells (SOURCE nearest-upscale / UPSCALED INTER_CUBIC / DETECTED with
marker).

### Numeric report

#### Real overlay — `/tmp/cv_overlay_real.png`

- Frames: t ∈ {0, 20, 40, 60, 80, 100, 120, 140, 160, 180} from
  `episode_0.hdf5` obs/images/camera_1_color, center-cropped 640→480.
- Mode: `'real'`, processing_resolution=512.
- Success: **10/10**. No `None` returns.
- ICP error (pixels in 512² space):
  min 0.3965 / median 0.6457 / mean 0.6008 / max 0.6833.
- `(x, y)` range: x ∈ [222.65, 223.55], y ∈ [252.22, 253.07]. Both
  well within [0, 512).
- Detected centroid is essentially stationary across the whole 200-step
  episode. Consistent with this being a slow-push episode — the gripper
  is approaching but the T-block has not yet been displaced
  significantly by t=180. Angles all in [−0.6°, +0.4°]; the T is nearly
  axis-aligned throughout.

#### Decoded overlay — `/tmp/cv_overlay_decoded.png`

- Frames: t ∈ {10, 20, ..., 100} from a fresh in-script imagination
  rollout (10-frame warmup + 91 imagination steps), captured via
  `env.render()` at the target offsets. Mode: `'wm'`,
  processing_resolution=512.
- Success: **10/10**. No `None` returns.
- ICP error: min 0.6106 / median 0.7282 / mean 0.7193 / max 0.7696.
  Comparable to real but ~20% higher median, attributable to decoder
  edge softness around the pink T's contour.
- `(x, y)` range: x ∈ [214.67, 223.20], y ∈ [252.82, 257.05]. Both
  within [0, 512). The 8-pixel x-drift and 4-pixel y-drift across 90
  imagination steps mirrors the slight latent drift observed earlier
  in `imagination_rollout.py`'s L2 curve (max 0.0858).
- Error trend with t: roughly flat, slope ≈ +0.0005/frame
  (effectively zero). No monotonic degradation — the detector handles
  late-rollout frames as well as early ones.

### Visual interpretation

I inspected both PNGs frame by frame.

**Real overlay**: in every one of the 10 rows, the filled green circle
in column 3 sits squarely on the pink T-block's body, near its
geometric centroid. No marker landed on the gripper, on the background
texture, or on any non-T object. The 30-px orientation line is short
relative to the 512² display and the T's angle is ≈ 0° in all 10
frames, so it is hard to see visually — but the angle_deg values
(±0.6°) are consistent with the T being approximately bar-up-stem-down
across the episode. 10/10 visually correct.

**Decoded overlay**: same outcome. The green markers track the
slightly-drifting decoded T-block centroid in all 10 rows. The marker
movement from row 1 (t=10) to row 10 (t=100) is small but visible — it
shifts a few pixels to the lower-left, consistent with the (x, y) range
numbers above and with the L2 drift observed in earlier imagination
tests. 10/10 visually correct.

**Silent wrong detections**: zero in this sample. In neither overlay
did `detect()` return a TPose for a wrong object. The failure mode
that would be dangerous for MPPI — `TPose` with the marker on the
gripper or on a background blob — did not appear.

### Verdict on MPPI reward viability

**Qualified yes.** The detector is reliable enough to drive an MPPI
reward on `pusht_cam1` decoded rollouts under the conditions sampled
here: T-block approximately centered, gripper approaching from below,
imagination horizon ≤ 100 steps, pink T's hue inside both HSV ranges.
The qualifier is that this episode is low-motion — the T barely
displaces, so the test does not exercise scenarios where the gripper
fully occludes the T-block, where the T is rotated to extreme angles
(45°+), or where the decoder degrades after a longer rollout. Before
trusting the reward in those regimes, sample more episodes (especially
later in the dataset where the T is actively pushed) and re-run this
visualization. The bitwise tests guarantee the detector behaves
identically to aloha's reference; the open question is only whether
aloha's reference is itself robust in those harder regimes.
