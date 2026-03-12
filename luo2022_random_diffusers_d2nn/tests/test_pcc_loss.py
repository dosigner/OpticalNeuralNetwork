"""Tests for PCC loss and energy penalty."""

import torch
import pytest

from luo2022_d2nn.training.losses import (
    pearson_correlation,
    energy_penalty,
    pcc_energy_loss,
)


class TestPearsonCorrelation:
    def test_pcc_perfect_correlation(self):
        """output == target -> PCC ~ 1.0"""
        x = torch.rand(4, 28, 28)
        pcc = pearson_correlation(x, x)
        assert pcc.item() == pytest.approx(1.0, abs=1e-5)

    def test_pcc_zero_correlation(self):
        """Random uncorrelated tensors -> PCC ~ 0.0"""
        torch.manual_seed(0)
        a = torch.randn(32, 28, 28)
        torch.manual_seed(999)
        b = torch.randn(32, 28, 28)
        pcc = pearson_correlation(a, b)
        assert abs(pcc.item()) < 0.15

    def test_pcc_negative_correlation(self):
        """output = -target -> PCC ~ -1.0"""
        x = torch.rand(4, 28, 28) + 0.1  # ensure nonzero
        pcc = pearson_correlation(-x, x)
        assert pcc.item() == pytest.approx(-1.0, abs=1e-5)

    def test_pcc_scale_invariant(self):
        """PCC(2*x, x) ~ PCC(x, x) = 1.0"""
        x = torch.rand(4, 28, 28) + 0.1
        pcc = pearson_correlation(2.0 * x, x)
        assert pcc.item() == pytest.approx(1.0, abs=1e-5)

    def test_pcc_shift_invariant(self):
        """PCC(x + c, x) ~ 1.0"""
        x = torch.rand(4, 28, 28)
        pcc = pearson_correlation(x + 5.0, x)
        assert pcc.item() == pytest.approx(1.0, abs=1e-5)

    def test_pcc_batch_dimension(self):
        """Works with batch dim > 1 and (B,1,N,N) shape."""
        x = torch.rand(8, 1, 28, 28)
        y = torch.rand(8, 1, 28, 28)
        pcc = pearson_correlation(x, y)
        assert pcc.shape == ()  # scalar
        assert -1.0 <= pcc.item() <= 1.0


class TestEnergyPenalty:
    def test_energy_penalty_all_inside(self):
        """mask=1 everywhere -> penalty is -beta * mean(output)"""
        output = torch.rand(4, 28, 28)
        mask = torch.ones(4, 28, 28)
        beta = 0.5
        penalty = energy_penalty(output, mask, alpha=1.0, beta=beta)
        expected = -beta * output.reshape(4, -1).sum(dim=1).mean() / (28 * 28)
        assert penalty.item() == pytest.approx(expected.item(), abs=1e-5)

    def test_energy_penalty_all_outside(self):
        """mask=0 -> should handle gracefully (avoid /0)"""
        output = torch.rand(4, 28, 28)
        mask = torch.zeros(4, 28, 28)
        # Should not raise; norm is clamped to eps
        penalty = energy_penalty(output, mask, alpha=1.0, beta=0.5)
        assert torch.isfinite(torch.tensor(penalty.item()))


class TestPCCEnergyLoss:
    def test_pcc_energy_loss_gradient_flows(self):
        """Verify gradients exist after backward."""
        output = torch.rand(4, 28, 28, requires_grad=True)
        target = torch.rand(4, 28, 28)
        mask = (target > 0.5).float()
        loss = pcc_energy_loss(output, target, mask)
        loss.backward()
        assert output.grad is not None
        assert output.grad.shape == output.shape
        assert torch.isfinite(output.grad).all()
