"""Modified von Karman turbulence helpers."""

from __future__ import annotations

import math

import torch


def fried_parameter(*, wavelength_m: float, cn2: float, path_length_m: float) -> float:
    """Compute the Fried coherence diameter."""
    if float(cn2) == 0.0:
        return float("inf")
    k = 2.0 * math.pi / float(wavelength_m)
    return (0.423 * (k**2) * float(cn2) * float(path_length_m)) ** (-3.0 / 5.0)


def phase_psd(
    *,
    fx: torch.Tensor,
    fy: torch.Tensor,
    wavelength_m: float,
    cn2: float,
    path_length_m: float,
    outer_scale_m: float,
    inner_scale_m: float,
) -> torch.Tensor:
    """Return the modified von Karman phase PSD."""
    if float(cn2) == 0.0:
        return torch.zeros_like(fx, dtype=torch.float64)
    r0 = fried_parameter(wavelength_m=wavelength_m, cn2=cn2, path_length_m=path_length_m)
    f_sq = fx.square() + fy.square()
    f0 = 1.0 / float(outer_scale_m)
    fm = 5.92 / (2.0 * math.pi * float(inner_scale_m))
    psd = 0.023 * (r0 ** (-5.0 / 3.0)) * torch.exp(-f_sq / (fm * fm)) / torch.pow(f_sq + f0 * f0, 11.0 / 6.0)
    psd[0, 0] = 0.0
    return psd
