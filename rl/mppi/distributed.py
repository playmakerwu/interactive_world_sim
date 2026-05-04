"""Multi-GPU coordination helpers for MPPI sample-shard parallelism.

When ``torch.distributed`` is initialized (multi-GPU runs spawned via
``mp.spawn`` from ``scripts/run_mppi_v2.py``), these helpers expose the
expected rank/world-size/broadcast/all-gather primitives. When NOT
initialized (single-GPU path), every helper short-circuits to its
single-process equivalent — no behavior change vs the pre-fix code path.

Single-process invariants (asserted by the smoke test below):
  * ``is_dist()`` -> False
  * ``rank()`` -> 0
  * ``world_size()`` -> 1
  * ``broadcast_(t)`` -> no-op, returns ``t`` unchanged
  * ``all_gather_concat(t)`` -> returns ``t`` unchanged
  * ``maybe_barrier()`` -> no-op
"""

from __future__ import annotations

import datetime
import os

import torch
import torch.distributed as dist


def is_dist() -> bool:
    """True only if torch.distributed has been initialized in this process."""
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    """Distributed rank, or 0 in the single-process path."""
    return dist.get_rank() if is_dist() else 0


def world_size() -> int:
    """Distributed world size, or 1 in the single-process path."""
    return dist.get_world_size() if is_dist() else 1


def broadcast_(t: torch.Tensor, src: int = 0) -> torch.Tensor:
    """In-place broadcast from ``src``. No-op when single-process.

    Returns ``t`` for chaining convenience. Tensor must be contiguous
    and on the local CUDA device for NCCL.
    """
    if is_dist():
        dist.broadcast(t.contiguous(), src=src)
    return t


def all_gather_concat(t: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """All-gather ``t`` across ranks and concat along ``dim``.

    Single-process: returns ``t`` unchanged. All ranks must pass tensors
    of identical shape and dtype (NCCL requirement).
    """
    if not is_dist():
        return t
    ws = world_size()
    bufs = [torch.empty_like(t) for _ in range(ws)]
    dist.all_gather(bufs, t.contiguous())
    return torch.cat(bufs, dim=dim)


def maybe_barrier() -> None:
    """``dist.barrier()`` if initialized, else no-op."""
    if is_dist():
        dist.barrier()


def init_dist(
    rank_: int,
    world_size_: int,
    backend: str = "nccl",
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
    timeout_seconds: int = 24 * 3600,
) -> None:
    """Initialize ``torch.distributed`` and pin the current CUDA device.

    Called once per worker process from ``mp.spawn`` in the runner.
    Sets ``MASTER_ADDR`` and ``MASTER_PORT`` env vars (only if not
    already set) before ``init_process_group``.
    """
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", str(master_port))
    # 24h NCCL collective timeout: tolerate transient rank speed asymmetry
    # such as CV pool spawn jitter or brief memory pressure. This does NOT
    # fix persistent contention from multi-tenant GPU sharing — for that,
    # use CUDA_VISIBLE_DEVICES to ensure exclusive GPU access. If a rank is
    # consistently slower (e.g., another user's process on the same GPU),
    # this timeout just delays the eventual abort while wall time balloons.
    dist.init_process_group(
        backend=backend,
        rank=rank_,
        world_size=world_size_,
        timeout=datetime.timedelta(seconds=timeout_seconds),
    )
    torch.cuda.set_device(rank_)


def destroy_dist() -> None:
    """Tear down the process group. Safe to call when uninitialized."""
    if is_dist():
        dist.destroy_process_group()


# ── single-process short-circuit smoke test (S4 in the multi-GPU plan) ─


if __name__ == "__main__":
    print("Single-process short-circuit smoke test:")
    print(f"  is_dist()    = {is_dist()}                  (expect False)")
    print(f"  rank()       = {rank()}                      (expect 0)")
    print(f"  world_size() = {world_size()}                      (expect 1)")

    t = torch.tensor([1.0, 2.0, 3.0])
    out = broadcast_(t.clone(), src=0)
    print(f"  broadcast_   identity={torch.equal(out, t)}  out={out.tolist()}")

    out = all_gather_concat(t.clone())
    print(f"  all_gather   identity={torch.equal(out, t)}  out={out.tolist()}")

    maybe_barrier()
    print("  maybe_barrier()           = returned (no-op)")

    assert not is_dist()
    assert rank() == 0
    assert world_size() == 1
    assert torch.equal(broadcast_(t.clone()), t)
    assert torch.equal(all_gather_concat(t.clone()), t)
    print("\nAll short-circuit invariants hold. ✓")
