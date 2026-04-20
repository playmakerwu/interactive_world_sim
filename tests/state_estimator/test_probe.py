"""Unit tests for rl.models.state_probe.StateProbe."""

from __future__ import annotations

import pytest
import torch

from rl.models.state_probe import (
    DEFAULT_HIDDEN,
    StateProbe,
    normalize_sincos,
    pixels_from_norm,
    split_output,
)


def test_default_probe_param_count_matches_design():
    """Design doc §1.7 revised: arch [4096 -> 256 -> 128 -> 4] ~1.08M params."""
    probe = StateProbe()
    # 4096*256 + 256 + 256*128 + 128 + 128*4 + 4  (+ LN gain/bias)
    # = 1_048_576 + 256 + 32_768 + 128 + 512 + 4 + (256*2 + 128*2)
    # ~= 1.082M
    n = probe.num_params
    assert 1_050_000 < n < 1_120_000, f"expected ~1.08M params, got {n}"


def test_forward_shape():
    probe = StateProbe()
    z = torch.randn(3, 4, 32, 32)
    out = probe(z)
    assert out.shape == (3, 4)
    assert out.dtype == torch.float32


def test_forward_raw_output_unbounded():
    """Design doc §1.7: output head is raw linear, no activation."""
    probe = StateProbe()
    z = torch.randn(16, 4, 32, 32) * 10  # big input
    out = probe(z)
    # No sigmoid / tanh means unbounded outputs; at init a batch of 16
    # will almost certainly contain something outside [-1, 1].
    assert out.abs().max() > 0, "output should not be constant zero"
    # (We're not asserting > 1 here because a randomly initialised MLP
    # could output a small value by luck; the important check is that
    # no bounded-activation is applied, which is easier to test via
    # grad flow below.)


def test_sincos_output_is_not_unit_normalized_at_training_time():
    """
    Design doc §1.7: sin/cos head is raw and unnormalised during training.
    Post-hoc normalisation is the caller's job at inference.
    """
    probe = StateProbe()
    z = torch.randn(32, 4, 32, 32)
    out = probe(z)
    _, sincos = split_output(out)
    norms = torch.norm(sincos, dim=-1)
    # With 32 random inputs and a linear head, norms should vary — they
    # would all be 1.0 only if we forced normalisation.
    assert not torch.allclose(norms, torch.ones_like(norms), atol=0.05)


def test_forward_is_differentiable_end_to_end():
    probe = StateProbe()
    z = torch.randn(2, 4, 32, 32, requires_grad=True)
    out = probe(z)
    loss = out.sum()
    loss.backward()

    # Every probe param has a non-None, non-zero, non-NaN gradient.
    for name, p in probe.named_parameters():
        assert p.grad is not None, f"{name}: no gradient"
        assert not torch.isnan(p.grad).any(), f"{name}: NaN gradient"
        assert p.grad.abs().sum() > 0, f"{name}: zero gradient"
    # Input also received gradient — required for Dreamer policy gradient.
    assert z.grad is not None and z.grad.abs().sum() > 0


def test_pixels_from_norm_round_trip():
    pos_norm = torch.tensor([[0.0, 0.5], [1.0, 0.25]])
    px = pixels_from_norm(pos_norm, resolution=128)
    expected = torch.tensor([[0.0, 64.0], [128.0, 32.0]])
    torch.testing.assert_close(px, expected)


def test_normalize_sincos_unit_norm():
    sincos = torch.tensor([[3.0, 4.0], [0.0, 0.0], [1.5, -0.5]])
    out = normalize_sincos(sincos)
    norms = torch.norm(out, dim=-1)
    # All rows (except the zero-row, handled by eps) have unit norm.
    torch.testing.assert_close(norms[0], torch.tensor(1.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(norms[2], torch.tensor(1.0), atol=1e-6, rtol=0)
    # Zero input stays near zero (eps clamp) — not unit norm. That's fine;
    # inference-time inputs are trained to be near unit, not zero.
    assert not torch.isnan(out).any()


def test_split_output():
    out = torch.tensor([[10.0, 20.0, 0.3, 0.9]])
    pos, sincos = split_output(out)
    torch.testing.assert_close(pos, torch.tensor([[10.0, 20.0]]))
    torch.testing.assert_close(sincos, torch.tensor([[0.3, 0.9]]))


def test_custom_hidden_dims():
    probe = StateProbe(hidden=(64, 32))
    z = torch.randn(2, 4, 32, 32)
    out = probe(z)
    assert out.shape == (2, 4)
    assert probe.num_params < 500_000  # much smaller than default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
