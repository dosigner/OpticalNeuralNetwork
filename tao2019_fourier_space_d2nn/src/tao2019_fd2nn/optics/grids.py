"""Spatial and frequency grid helpers."""

from __future__ import annotations

import torch


def make_spatial_grid(
    N: int,
    dx_m: float,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return meshgrid (x, y) in meters with center at zero."""

    dev = torch.device(device) if device is not None else torch.device("cpu")
    idx = torch.arange(N, device=dev, dtype=dtype) - (N // 2)
    x = idx * float(dx_m)
    return torch.meshgrid(x, x, indexing="xy")


def make_frequency_grid(
    N: int,
    dx_m: float,
    *,
    shifted: bool = False,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return frequency-domain meshgrid (fx, fy) in cycles/meter."""

    dev = torch.device(device) if device is not None else torch.device("cpu")
    f = torch.fft.fftfreq(N, d=float(dx_m), device=dev, dtype=dtype)
    if shifted:
        f = torch.fft.fftshift(f)
    return torch.meshgrid(f, f, indexing="xy")
