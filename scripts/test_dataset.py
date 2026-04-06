#!/usr/bin/env python3
"""Test LatentDataset bootstrapping and data loading.

Usage:
    python scripts/test_dataset.py --dataset_dir data/mini/pusht_latent
    python scripts/test_dataset.py --dataset_dir data/mini/pusht_latent --bootstrap_seed 42
"""

import argparse
import sys

import numpy as np
from collections import Counter
from omegaconf import OmegaConf


def test_bootstrap_distribution(n_episodes: int = 10, seed: int = 42):
    """Test that bootstrap produces expected with-replacement distribution."""
    rng = np.random.default_rng(seed=seed)
    indices = rng.choice(n_episodes, size=n_episodes, replace=True)
    counts = Counter(indices)

    mask = np.zeros(n_episodes, dtype=np.int64)
    for idx in indices:
        mask[idx] += 1

    print(f"\n=== Bootstrap 分布测试 (seed={seed}, n_episodes={n_episodes}) ===")
    print(f"被抽中的 episode 数: {len(counts)}/{n_episodes}")
    print(f"未被抽中的 episode 数: {n_episodes - len(counts)}")
    print(f"出现次数分布: {dict(Counter(mask))}")
    print(f"Integer mask: {mask}")
    # ~37% of episodes should be missing (1 - 1/e)
    expected_missing = n_episodes * (1 - 1 / np.e) ** 1
    print(f"理论缺失比例: ~{1 - 1/np.e:.1%}, 实际: {(n_episodes - len(counts))/n_episodes:.1%}")


def test_dataset_loading(dataset_dir: str, bootstrap_seed=None):
    """Test actual LatentDataset loading."""
    from interactive_world_sim.datasets.latent_dynamics.latent_dataset import LatentDataset

    cfg = OmegaConf.create({
        "dataset_dir": dataset_dir,
        "horizon": 16,
        "val_horizon": 16,
        "skip_frame": 1,
        "pad_before": 1,
        "pad_after": 7,
        "skip_idx": 1,
        "goal_sample": "intermediate",
        "action_mode": "bimanual_push",
        "bootstrap_seed": bootstrap_seed,
        "debug": False,
    })

    print(f"\n=== LatentDataset 加载测试 (bootstrap_seed={bootstrap_seed}) ===")
    ds = LatentDataset(cfg)
    print(f"训练集大小: {len(ds)}")
    print(f"Episode 数: {ds.replay_buffer.n_episodes}")
    print(f"Episode mask: {ds.train_mask}")

    # Test __getitem__
    sample = ds[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"  latent shape: {sample['latent'].shape}")
    print(f"  action shape: {sample['action'].shape}")

    # Test validation dataset
    val_ds = ds.get_validation_dataset()
    print(f"验证集大小: {len(val_ds)}")

    # Compare two different seeds
    if bootstrap_seed is None:
        for s in [42, 43]:
            cfg2 = cfg.copy()
            cfg2.bootstrap_seed = s
            ds2 = LatentDataset(cfg2)
            print(f"\n  Seed {s}: dataset size={len(ds2)}, mask={ds2.train_mask}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="data/mini/pusht_latent")
    parser.add_argument("--bootstrap_seed", type=int, default=None)
    args = parser.parse_args()

    # Pure math test (no data needed)
    test_bootstrap_distribution(n_episodes=10, seed=42)
    test_bootstrap_distribution(n_episodes=600, seed=42)

    # Data loading test (needs pre-encoded data)
    if args.dataset_dir:
        try:
            test_dataset_loading(args.dataset_dir, args.bootstrap_seed)
        except FileNotFoundError as e:
            print(f"\n跳过数据加载测试 (文件未找到): {e}")
            print("请先运行 Phase 1 预编码脚本。")


if __name__ == "__main__":
    main()
