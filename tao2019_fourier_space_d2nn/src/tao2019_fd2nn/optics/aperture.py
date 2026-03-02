"""Aperture masks."""

from __future__ import annotations

import torch

from tao2019_fd2nn.optics.grids import make_frequency_grid, make_spatial_grid


def circular_aperture(
    N: int,
    dx_m: float,
    radius_m: float,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Real-valued circular aperture mask in spatial domain."""

    x, y = make_spatial_grid(N=N, dx_m=dx_m, device=device, dtype=torch.float32)
    return ((x**2 + y**2) <= float(radius_m) ** 2).to(torch.float32)


def na_mask(
    N: int,
    dx_m: float,
    wavelength_m: float,
    na: float,
    *,
    shifted: bool = False,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Frequency-domain mask defined by numerical aperture cutoff."""

    fx, fy = make_frequency_grid(N=N, dx_m=dx_m, shifted=shifted, device=device, dtype=torch.float64)
    cutoff = float(na) / float(wavelength_m)
    return ((fx**2 + fy**2) <= cutoff**2).to(torch.float32)
