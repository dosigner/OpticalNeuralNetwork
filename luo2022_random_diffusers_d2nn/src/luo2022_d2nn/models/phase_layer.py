"""Single learnable phase-only diffractive layer."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PhaseLayer(nn.Module):
    """A phase-only diffractive layer with a learnable phase mask.

    Parameters
    ----------
    grid_size : int
        Spatial resolution N of the phase mask (N x N).
    init_phase_dist : str
        Initialization distribution. Currently ``"uniform_0_2pi"``.
    """

    def __init__(self, grid_size: int, init_phase_dist: str = "uniform_0_2pi") -> None:
        super().__init__()
        if init_phase_dist == "uniform_0_2pi":
            init_vals = torch.empty(grid_size, grid_size).uniform_(0, 2 * math.pi)
        else:
            raise ValueError(f"Unknown init_phase_dist: {init_phase_dist}")

        self.phase = nn.Parameter(init_vals)  # (N, N), float32

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """Apply phase modulation: out = field * exp(j * phase).

        Parameters
        ----------
        field : Tensor, shape (..., N, N), complex
            Input complex optical field.

        Returns
        -------
        Tensor, same shape and dtype as *field*.
        """
        t_m = torch.exp(1j * self.phase.to(field.dtype).to(field.device))
        return field * t_m
