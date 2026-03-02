"""Material helpers for absorption and phase/height conversions."""

from __future__ import annotations

import numpy as np
import torch


def phase_to_height(phase: np.ndarray, wavelength: float, delta_n: float) -> np.ndarray:
    """Convert phase [rad] to physical height [m].

    Formula:
        dz = (wavelength / (2*pi)) * (phase / delta_n)
    """

    if delta_n == 0:
        raise ValueError("delta_n must be non-zero")
    return (wavelength / (2.0 * np.pi)) * (phase / delta_n)


def apply_absorption(field: torch.Tensor, alpha: float | None) -> torch.Tensor:
    """Apply scalar attenuation exp(-alpha) to complex field."""

    if alpha is None or alpha == 0.0:
        return field
    return field * torch.exp(torch.tensor(-alpha, dtype=field.real.dtype, device=field.device))
