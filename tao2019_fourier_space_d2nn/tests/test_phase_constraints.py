from __future__ import annotations

import torch

from tao2019_fd2nn.models.phase_mask import PhaseConstraint


def test_sigmoid_phase_range() -> None:
    c = PhaseConstraint(phase_max=float(torch.pi), mode="sigmoid")
    raw = torch.linspace(-20.0, 20.0, steps=1024)
    phi = c.apply(raw)
    assert float(phi.min()) >= 0.0
    assert float(phi.max()) <= float(torch.pi) + 1e-6


def test_symmetric_tanh_phase_range() -> None:
    c = PhaseConstraint(phase_max=float(torch.pi), mode="symmetric_tanh")
    raw = torch.linspace(-20.0, 20.0, steps=1024)
    phi = c.apply(raw)
    assert float(phi.min()) >= -float(torch.pi) - 1e-6
    assert float(phi.max()) <= float(torch.pi) + 1e-6
