from __future__ import annotations

import torch

from d2nn.models.constraints import PhaseConstraint


def test_phase_constraint_range() -> None:
    max_phase = torch.pi
    c = PhaseConstraint(max_phase=max_phase)
    raw = torch.linspace(-20.0, 20.0, steps=256)
    phi = c(raw)

    assert float(phi.min().item()) >= -1e-6
    assert float(phi.max().item()) <= float(max_phase) + 1e-6


def test_phase_constraint_symmetric_tanh_range() -> None:
    max_phase = 2.0 * torch.pi
    c = PhaseConstraint(max_phase=max_phase, mode="symmetric_tanh")
    raw = torch.linspace(-20.0, 20.0, steps=256)
    phi = c(raw)

    assert float(phi.min().item()) >= -float(torch.pi) - 1e-6
    assert float(phi.max().item()) <= float(torch.pi) + 1e-6
