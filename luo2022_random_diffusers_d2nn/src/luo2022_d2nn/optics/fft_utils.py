"""Centered FFT utilities."""

from __future__ import annotations

import torch


def fft2c(x: torch.Tensor) -> torch.Tensor:
    """Centered orthonormal 2D FFT."""
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )


def ifft2c(x: torch.Tensor) -> torch.Tensor:
    """Centered orthonormal 2D inverse FFT."""
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )


def gamma_flip2d(x: torch.Tensor) -> torch.Tensor:
    """Coordinate inversion on last two axes."""
    y = torch.flip(x, dims=(-2, -1))
    sy = 1 if (x.shape[-2] % 2 == 0) else 0
    sx = 1 if (x.shape[-1] % 2 == 0) else 0
    if sy != 0 or sx != 0:
        y = torch.roll(y, shifts=(sy, sx), dims=(-2, -1))
    return y
