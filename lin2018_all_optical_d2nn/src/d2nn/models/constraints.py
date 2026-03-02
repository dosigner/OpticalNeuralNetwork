"""Trainable parameter constraints for D2NN layers."""

from __future__ import annotations

import torch


class PhaseConstraint:
    """Map unconstrained phase parameter to [0, max_phase]."""

    def __init__(self, max_phase: float, mode: str = "sigmoid"):
        self.max_phase = float(max_phase)
        self.mode = str(mode).lower()
        if self.max_phase <= 0.0:
            raise ValueError("max_phase must be > 0")
        if self.mode not in {"sigmoid", "wrap", "clamp", "symmetric_tanh"}:
            raise ValueError("mode must be one of: sigmoid, wrap, clamp, symmetric_tanh")

    def __call__(self, raw_phase: torch.Tensor) -> torch.Tensor:
        """Apply phase constraint according to selected mode."""

        if self.mode == "sigmoid":
            phi = self.max_phase * torch.sigmoid(raw_phase)
            return torch.clamp(phi, min=0.0, max=self.max_phase)
        if self.mode == "symmetric_tanh":
            half = 0.5 * self.max_phase
            phi = half * torch.tanh(raw_phase)
            return torch.clamp(phi, min=-half, max=half)
        elif self.mode == "wrap":
            phi = torch.remainder(raw_phase, self.max_phase)
        else:
            phi = torch.clamp(raw_phase, min=0.0, max=self.max_phase)
        return torch.clamp(phi, min=0.0, max=self.max_phase)


class AmplitudeConstraint:
    """Map unconstrained amplitude parameter to [a_min, a_max]."""

    def __init__(self, amplitude_range: tuple[float, float] = (0.0, 1.0)):
        self.a_min, self.a_max = map(float, amplitude_range)

    def __call__(self, raw_amp: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid-based amplitude constraint."""

        scale = torch.sigmoid(raw_amp)
        return self.a_min + (self.a_max - self.a_min) * scale
