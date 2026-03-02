"""Aperture and padding helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def center_pad_2d(x: torch.Tensor, target_N: int) -> torch.Tensor:
    """Center-pad a 2D tensor to target_N x target_N with zeros."""

    if x.ndim != 2:
        raise ValueError("x must be 2D")
    h, w = x.shape
    if h > target_N or w > target_N:
        raise ValueError("target_N must be >= input size")
    pad_h = target_N - h
    pad_w = target_N - w
    pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    return F.pad(x, pad, mode="constant", value=0.0)


def circular_aperture(N: int, dx: float, radius: float) -> torch.Tensor:
    """Return circular aperture mask.

    Args:
        N: grid size [pixels]
        dx: pixel pitch [m]
        radius: aperture radius [m]
    Returns:
        mask: float tensor, shape (N, N)
    """

    coords = (torch.arange(N, dtype=torch.float32) - (N // 2)) * dx
    X, Y = torch.meshgrid(coords, coords, indexing="xy")
    return ((X**2 + Y**2) <= radius**2).to(torch.float32)
