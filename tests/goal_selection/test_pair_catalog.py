"""Validate the (start, goal) pair catalog produced by build_pair_catalog.py.

These tests do NOT regenerate the catalog. They assume
build_pair_catalog.py has already been run and that the artifacts at
tests/goal_selection/pair_catalog.json + pair_goals/ are present.

If the catalog is missing, every test skips (so CI doesn't fail just
because someone hasn't built it locally).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO_ROOT / "tests" / "goal_selection" / "pair_catalog.json"

REQUIRED_PAIR_IDS = ("pair_A", "pair_B", "pair_C", "pair_D")
REQUIRED_GOAL_KEYS = ("cx", "cy", "sin_theta", "cos_theta", "theta_deg")


def _load_catalog() -> dict:
    if not CATALOG_PATH.exists():
        pytest.skip(f"catalog not built yet: {CATALOG_PATH}")
    return json.loads(CATALOG_PATH.read_text())


# ─── 1. four pairs present ─────────────────────────────────────────────

def test_four_pairs_present():
    catalog = _load_catalog()
    assert set(catalog.keys()) >= set(REQUIRED_PAIR_IDS), (
        f"missing pairs: {set(REQUIRED_PAIR_IDS) - set(catalog.keys())}"
    )


# ─── 2. each pair has required fields ──────────────────────────────────

def test_each_pair_has_required_fields():
    catalog = _load_catalog()
    for pid in REQUIRED_PAIR_IDS:
        p = catalog[pid]
        for top_key in ("start", "goal", "distance_px", "angular_difference_deg"):
            assert top_key in p, f"{pid} missing {top_key}"
        for sub in ("hdf5_path", "frame_idx"):
            assert sub in p["start"], f"{pid}.start missing {sub}"
            assert sub in p["goal"], f"{pid}.goal missing {sub}"
        assert "state_pt_path" in p["goal"], f"{pid}.goal missing state_pt_path"


# ─── 3. distance threshold ─────────────────────────────────────────────

def test_distance_above_40():
    catalog = _load_catalog()
    for pid in REQUIRED_PAIR_IDS:
        d = catalog[pid]["distance_px"]
        assert d > 40.0, f"{pid} distance {d:.1f} <= 40 px"


# ─── 4. each goal .pt file exists ──────────────────────────────────────

def test_goal_pt_files_exist():
    catalog = _load_catalog()
    for pid in REQUIRED_PAIR_IDS:
        rel = catalog[pid]["goal"]["state_pt_path"]
        path = REPO_ROOT / rel
        assert path.exists(), f"{pid} goal .pt missing at {path}"


# ─── 5. start episodes all unique ──────────────────────────────────────

def test_start_episodes_all_different():
    catalog = _load_catalog()
    starts = [catalog[pid]["start"]["hdf5_path"] for pid in REQUIRED_PAIR_IDS]
    assert len(set(starts)) == len(starts), (
        f"start hdf5_paths repeat across pairs: {starts}"
    )


# ─── 6. goal .pt schema matches env.load_goal expectations ─────────────

def test_goal_pt_schema_matches_env_load_goal():
    """env.load_goal reads cx, cy, sin_theta, cos_theta, theta_deg from
    a torch.load'd dict. Verify each goal .pt has those keys (and a
    state tensor) without constructing PushTWMEnv (which loads the WM)."""
    catalog = _load_catalog()
    for pid in REQUIRED_PAIR_IDS:
        rel = catalog[pid]["goal"]["state_pt_path"]
        path = REPO_ROOT / rel
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        for key in REQUIRED_GOAL_KEYS:
            assert key in payload, f"{pid} goal .pt missing {key}: have {list(payload.keys())}"
        assert "state" in payload and isinstance(payload["state"], torch.Tensor)
        assert payload["state"].shape == (4,)


# ─── 7. start ≠ goal episode within each pair ──────────────────────────

def test_start_and_goal_in_different_episodes():
    catalog = _load_catalog()
    for pid in REQUIRED_PAIR_IDS:
        s = catalog[pid]["start"]["hdf5_path"]
        g = catalog[pid]["goal"]["hdf5_path"]
        assert s != g, f"{pid} start and goal share hdf5: {s}"
