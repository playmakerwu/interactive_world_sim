"""Structural tests for scripts/run_multipair_experiment.sh.

We don't execute the launcher in CI (it requires 4 L40S GPUs, ~10 hours,
and a specific conda env); instead we parse the bash source text and
verify the knobs, paths, and control flow we rely on are all present.

Also runs ``bash -n`` to catch syntax regressions.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "scripts" / "run_multipair_experiment.sh"


def _source() -> str:
    if not LAUNCHER.exists():
        pytest.fail(f"launcher missing: {LAUNCHER}")
    return LAUNCHER.read_text()


# ─── 1. bash syntax validates ──────────────────────────────────────────

def test_bash_syntax_valid():
    result = subprocess.run(
        ["bash", "-n", str(LAUNCHER)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"bash -n failed:\n{result.stderr}"
    )


# ─── 2. all 4 pairs referenced ─────────────────────────────────────────

def test_all_four_pairs_listed():
    src = _source()
    m = re.search(r'^PAIRS=\((.*?)\)', src, re.MULTILINE)
    assert m, "PAIRS array declaration not found"
    tokens = re.findall(r'"([^"]+)"', m.group(1))
    assert sorted(tokens) == ["A", "B", "C", "D"], (
        f"PAIRS should be A B C D, got {tokens}"
    )


# ─── 3. all 4 GPUs referenced ──────────────────────────────────────────

def test_all_four_gpus_listed():
    src = _source()
    m = re.search(r'^GPUS=\((.*?)\)', src, re.MULTILINE)
    assert m, "GPUS array declaration not found"
    tokens = m.group(1).split()
    assert sorted(tokens) == ["4", "5", "6", "7"], (
        f"GPUS should be 4 5 6 7, got {tokens}"
    )


# ─── 4. 2 seeds declared ───────────────────────────────────────────────

def test_two_seeds_listed():
    src = _source()
    m = re.search(r'^SEEDS=\((.*?)\)', src, re.MULTILINE)
    assert m, "SEEDS array declaration not found"
    tokens = m.group(1).split()
    assert sorted(tokens) == ["0", "1"], f"SEEDS should be 0 1, got {tokens}"


# ─── 5. references the catalog produced by Task G ──────────────────────

def test_references_pair_catalog():
    src = _source()
    assert "tests/goal_selection/pair_catalog.json" in src, (
        "launcher must read start/goal info from the Task G catalog"
    )


# ─── 6. threads --control_steps and --seed through to run_mppi_v2 ──────

def test_passes_control_steps_and_seed_overrides():
    src = _source()
    assert "--control_steps" in src, "launcher must pass --control_steps"
    assert "--seed" in src, "launcher must pass --seed"


# ─── 7. output directory is unique per pair+seed ───────────────────────

def test_output_dir_unique_per_pair_seed():
    src = _source()
    # The templated path inside heredocs: pair_${PAIR}_seed_${SEED}
    assert "pair_${PAIR}_seed_${SEED}" in src, (
        "launcher must produce output_dir templated as "
        "pair_<PAIR>_seed_<SEED> so runs don't collide"
    )


# ─── 8. launch is detached per pair (parallel, not serial) ─────────────

def test_launches_sessions_detached():
    src = _source()
    # `tmux new-session -d` is what makes the 4 pairs run in parallel.
    assert "tmux new-session -d" in src, (
        "launcher must create tmux sessions with -d so they run in parallel"
    )


# ─── 9. preflight: refuses to overwrite an existing output dir ─────────

def test_refuses_to_overwrite_output_dir():
    src = _source()
    assert 'already exists' in src, (
        "launcher should error out if OUTPUT_ROOT already exists rather "
        "than silently overwriting an in-progress run"
    )


# ─── 10. seeds separated by ';' not '&&' (seed 1 runs even if seed 0 fails) ─

def test_seeds_run_independently():
    src = _source()
    # Seeds are independent variance samples; the launcher should not gate
    # seed 1 on seed 0's success. Check that the per-pair launch script does
    # not chain seeds with '&&'.
    assert 'echo "[pair $PAIR seed $SEED] end' in src, (
        "should echo end-of-seed markers (implies non-chained sequencing)"
    )
    # Heuristic: count the 'python -u scripts/run_mppi_v2.py' invocations in
    # the generated heredoc — each seed should be a distinct invocation, not
    # a single && chain.
    assert src.count("python -u scripts/run_mppi_v2.py") == 1, (
        "run_mppi_v2.py should appear in exactly one heredoc block that "
        "is emitted per-seed by the inner loop; more or fewer suggests the "
        "seed loop structure was changed unexpectedly"
    )
