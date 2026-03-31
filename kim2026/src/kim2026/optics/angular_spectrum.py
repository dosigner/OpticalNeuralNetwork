"""Same-window free-space propagation via a band-limited angular spectrum."""

from __future__ import annotations

import math
from typing import Any

import torch

_TRANSFER_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}


def _build_transfer(
    *,
    n: int,
    wavelength_m: float,
    window_m: float,
    z_m: float,
    device: torch.device,
) -> torch.Tensor:
    key = (int(n), float(wavelength_m), float(window_m), float(z_m), str(device))
    cached = _TRANSFER_CACHE.get(key)
    if cached is not None:
        return cached

    dx = float(window_m) / float(n)
    freq_axis = torch.fft.fftshift(
        torch.fft.fftfreq(n, d=dx, device=device, dtype=torch.float64)
    )
    fy, fx = torch.meshgrid(freq_axis, freq_axis, indexing="ij")
    propagation_term = 1.0 - (float(wavelength_m) * fx).square() - (float(wavelength_m) * fy).square()
    passband = propagation_term >= 0.0
    kz = (2.0 * math.pi / float(wavelength_m)) * torch.sqrt(propagation_term.clamp_min(0.0))

    phase = torch.zeros((n, n), dtype=torch.float64, device=device)
    phase[passband] = float(z_m) * kz[passband]
    transfer = passband.to(torch.complex64) * torch.exp(1j * phase).to(torch.complex64)
    _TRANSFER_CACHE[key] = transfer
    return transfer


def propagate_same_window(
    field: torch.Tensor,
    *,
    wavelength_m: float,
    window_m: float,
    z_m: float,
) -> torch.Tensor:
    """Propagate a square field to an equally sampled destination plane as complex64."""
    if field.ndim < 2 or field.shape[-1] != field.shape[-2]:
        raise ValueError("field must be square in the last two dimensions")
    if wavelength_m <= 0.0:
        raise ValueError("wavelength_m must be > 0")
    if window_m <= 0.0:
        raise ValueError("window_m must be > 0")
    if z_m < 0.0:
        raise ValueError("z_m must be >= 0")

    n = int(field.shape[-1])
    original_shape = field.shape
    flattened = field.reshape(-1, n, n).to(torch.complex64)
    if z_m == 0.0:
        return flattened.reshape(original_shape)

    transfer = _build_transfer(
        n=n,
        wavelength_m=wavelength_m,
        window_m=window_m,
        z_m=z_m,
        device=flattened.device,
    )
    spectrum = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(flattened, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )
    propagated = torch.fft.fftshift(
        torch.fft.ifft2(
            torch.fft.ifftshift(spectrum * transfer.unsqueeze(0), dim=(-2, -1)),
            norm="ortho",
        ),
        dim=(-2, -1),
    )
    return propagated.reshape(original_shape)
