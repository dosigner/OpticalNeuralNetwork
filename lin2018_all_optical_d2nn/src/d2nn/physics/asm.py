"""Angular Spectrum propagation helpers."""

from __future__ import annotations

from typing import Any

import torch

from .grid import make_frequency_grid

_TRANSFER_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}


def _resolve_complex_dtype(dtype: str) -> torch.dtype:
    if dtype == "complex128":
        return torch.complex128
    return torch.complex64


def asm_transfer_function(
    N: int,
    dx: float,
    wavelength: float,
    z: float,
    n: float = 1.0,
    bandlimit: bool = True,
    fftshifted: bool = False,
    dtype: str = "complex64",
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build Angular Spectrum transfer function H(fx, fy).

    Physics:
        H = exp(i*2*pi*z*sqrt((n/lambda)^2 - fx^2 - fy^2))

    Args:
        N: grid size [pixels]
        dx: pixel pitch [m]
        wavelength: vacuum wavelength [m]
        z: propagation distance [m]
        n: refractive index of medium
        bandlimit: if True, suppress evanescent components
        fftshifted: if True, H is returned in shifted frequency ordering
        dtype: complex64 or complex128
        device: torch device

    Returns:
        H: complex tensor, shape (N, N)
    """

    dev = torch.device(device) if device is not None else torch.device("cpu")
    ctype = _resolve_complex_dtype(dtype)
    key = (N, float(dx), float(wavelength), float(z), float(n), bool(bandlimit), bool(fftshifted), str(ctype), str(dev))
    cached = _TRANSFER_CACHE.get(key)
    if cached is not None:
        return cached

    fx, fy = make_frequency_grid(N, dx, fftshift=fftshifted)
    fx_t = torch.as_tensor(fx, dtype=torch.float64, device=dev)
    fy_t = torch.as_tensor(fy, dtype=torch.float64, device=dev)
    FX, FY = torch.meshgrid(fx_t, fy_t, indexing="xy")

    spatial_term = (n / wavelength) ** 2 - FX**2 - FY**2
    spatial_term_c = torch.complex(spatial_term, torch.zeros_like(spatial_term))
    kz = torch.sqrt(spatial_term_c)

    phase = 2.0 * torch.pi * z * kz
    H = torch.exp(1j * phase)

    if bandlimit:
        propagating = spatial_term >= 0
        H = H * propagating.to(H.dtype)

    H = H.to(ctype)
    _TRANSFER_CACHE[key] = H
    return H


def asm_propagate(field: torch.Tensor, H: torch.Tensor, *, fftshifted: bool = False) -> torch.Tensor:
    """Propagate complex field using precomputed transfer function.

    Args:
        field: complex tensor, shape (..., N, N)
        H: complex tensor, shape (N, N)
        fftshifted: if True, apply shifted-FFT convention

    Returns:
        complex tensor, same shape as field
    """

    if fftshifted:
        u = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(field, dim=(-2, -1))), dim=(-2, -1))
        y = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(u * H, dim=(-2, -1))), dim=(-2, -1))
        return y

    u = torch.fft.fft2(field)
    y = torch.fft.ifft2(u * H)
    return y


def clear_transfer_cache() -> None:
    """Clear global transfer-function cache."""

    _TRANSFER_CACHE.clear()
