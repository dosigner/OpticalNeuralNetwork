"""Phase-only trainable modulation layer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PhaseConstraint:
    """Phase range mapping strategy."""

    phase_max: float
    mode: str = "sigmoid"

    def apply(self, raw: torch.Tensor) -> torch.Tensor:
        if self.mode == "sigmoid":
            return float(self.phase_max) * torch.sigmoid(raw)
        if self.mode == "symmetric_tanh":
            return float(self.phase_max) * torch.tanh(raw)
        raise ValueError("phase mode must be 'sigmoid' or 'symmetric_tanh'")


class PhaseMask(nn.Module):
    """Trainable phase-only modulation u -> u * exp(i*phi)."""

    def __init__(
        self,
        N: int,
        phase_max: float,
        *,
        constraint_mode: str = "sigmoid",
        init_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.N = int(N)
        self.constraint = PhaseConstraint(phase_max=float(phase_max), mode=constraint_mode)
        self.raw = nn.Parameter(torch.empty(self.N, self.N))
        init_mode = init_mode.lower()
        if init_mode == "zeros":
            nn.init.zeros_(self.raw)
        elif init_mode == "uniform":
            nn.init.uniform_(self.raw, -1.0, 1.0)
        else:
            raise ValueError("init_mode must be 'zeros' or 'uniform'")

    def phase(self) -> torch.Tensor:
        """Constrained phase map."""

        return self.constraint.apply(self.raw)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        phi = self.phase().to(device=field.device, dtype=field.real.dtype)
        return field * torch.exp(1j * phi)
