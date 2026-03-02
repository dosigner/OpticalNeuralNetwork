"""Spatial and frequency grid builders."""

from __future__ import annotations

import numpy as np


def make_spatial_grid(N: int, dx: float, centered: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return 1D spatial coordinates.

    Args:
        N: grid size [pixels]
        dx: pixel pitch [m]
        centered: when True, center grid around zero

    Returns:
        x, y: shape (N,), unit [m]
    """

    if centered:
        coords = (np.arange(N) - (N // 2)) * dx
    else:
        coords = np.arange(N) * dx
    return coords.astype(np.float64), coords.astype(np.float64)


def make_frequency_grid(N: int, dx: float, fftshift: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return 1D spatial-frequency coordinates.

    Args:
        N: grid size [pixels]
        dx: pixel pitch [m]
        fftshift: if True, return shifted ordering

    Returns:
        fx, fy: shape (N,), unit [cycles/m]
    """

    freqs = np.fft.fftfreq(N, d=dx)
    if fftshift:
        freqs = np.fft.fftshift(freqs)
    return freqs.astype(np.float64), freqs.astype(np.float64)
