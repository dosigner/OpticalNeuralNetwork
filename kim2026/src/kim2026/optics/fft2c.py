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
