"""Angular spectrum propagation."""

from __future__ import annotations

from typing import Any

import torch

from tao2019_fd2nn.optics.aperture import na_mask
from tao2019_fd2nn.optics.grids import make_frequency_grid

_TRANSFER_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}


def _complex_dtype(dtype: str) -> torch.dtype:
    if dtype == "complex128":
        return torch.complex128
    return torch.complex64


def asm_transfer_function(
    N: int,
    dx_m: float,
    wavelength_m: float,
    z_m: float,
    *,
    n: float = 1.0,
    evanescent: str = "mask",
    na: float | None = None,
    shifted: bool = False,
    dtype: str = "complex64",
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build transfer function H(fx, fy) for ASM propagation."""

    dev = torch.device(device) if device is not None else torch.device("cpu")
    ctype = _complex_dtype(dtype)
    key = (
        N,
        float(dx_m),
        float(wavelength_m),
        float(z_m),
        float(n),
        str(evanescent),
        None if na is None else float(na),
        bool(shifted),
        str(ctype),
        str(dev),
    )
    cached = _TRANSFER_CACHE.get(key)
    if cached is not None:
        return cached

    fx, fy = make_frequency_grid(N=N, dx_m=dx_m, shifted=shifted, device=dev, dtype=torch.float64)
    k0n = float(n) / float(wavelength_m)
    term = k0n**2 - fx**2 - fy**2
    term_c = torch.complex(term, torch.zeros_like(term))
    kz = torch.sqrt(term_c)
    H = torch.exp(1j * (2.0 * torch.pi * float(z_m) * kz))

    if evanescent == "mask":
        H = H * (term >= 0).to(H.dtype)
    elif evanescent != "keep":
        raise ValueError("evanescent must be 'mask' or 'keep'")

    if na is not None:
        H = H * na_mask(
            N=N,
            dx_m=dx_m,
            wavelength_m=wavelength_m,
            na=float(na),
            shifted=shifted,
            device=dev,
        ).to(H.dtype)

    H = H.to(ctype)
    _TRANSFER_CACHE[key] = H
    return H


def asm_propagate(field: torch.Tensor, H: torch.Tensor, *, shifted: bool = False) -> torch.Tensor:
    """Propagate complex field using precomputed transfer function."""

    if shifted:
        F = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(field, dim=(-2, -1))), dim=(-2, -1))
        Y = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(F * H, dim=(-2, -1))), dim=(-2, -1))
        return Y
    F = torch.fft.fft2(field)
    return torch.fft.ifft2(F * H)


def clear_transfer_cache() -> None:
    """Clear cached transfer functions."""

    _TRANSFER_CACHE.clear()
