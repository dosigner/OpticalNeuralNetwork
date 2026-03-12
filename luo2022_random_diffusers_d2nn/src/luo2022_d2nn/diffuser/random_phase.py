"""Thin random phase diffuser generation.

Height map: D(x,y) = GaussianSmooth(W(x,y), σ), W ~ N(μ, σ₀)
Phase:      φ_D = 2π Δn D(x,y) / λ
Transmittance: t_D = exp(j φ_D)

All length parameters in units of λ (converted internally via wavelength_mm).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


def _gaussian_kernel_1d(sigma_px: float, device: torch.device | str | None = None) -> torch.Tensor:
    """1-D Gaussian kernel (at least 6σ wide, odd size)."""
    radius = max(int(math.ceil(3.0 * sigma_px)), 1)
    size = 2 * radius + 1
    x = torch.arange(size, dtype=torch.float64, device=device) - radius
    k = torch.exp(-0.5 * (x / sigma_px) ** 2)
    k = k / k.sum()
    return k


def _gaussian_smooth_2d(field: torch.Tensor, sigma_px: float) -> torch.Tensor:
    """Separable 2-D Gaussian smoothing (preserves dtype)."""
    if sigma_px <= 0:
        return field
    k1d = _gaussian_kernel_1d(sigma_px, device=field.device).to(field.dtype)
    # Reshape for conv2d: (1, 1, K) and (1, 1, 1, K)
    kx = k1d.view(1, 1, 1, -1)
    ky = k1d.view(1, 1, -1, 1)
    pad_size = k1d.shape[0] // 2

    inp = field.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
    # Smooth along x then y
    out = F.pad(inp, (pad_size, pad_size, 0, 0), mode="reflect")
    out = F.conv2d(out, kx)
    out = F.pad(out, (0, 0, pad_size, pad_size), mode="reflect")
    out = F.conv2d(out, ky)
    return out.squeeze(0).squeeze(0)


def generate_diffuser(
    N: int,
    dx_mm: float,
    wavelength_mm: float,
    *,
    delta_n: float = 0.74,
    height_mean_lambda: float = 25.0,
    height_std_lambda: float = 8.0,
    smoothing_sigma_lambda: float = 4.0,
    seed: Optional[int] = None,
    device: torch.device | str | None = None,
) -> Dict[str, Any]:
    """Generate a thin random-phase diffuser.

    Parameters
    ----------
    N : int
        Grid side length (pixels).
    dx_mm : float
        Pixel pitch in mm.
    wavelength_mm : float
        Operating wavelength in mm.
    delta_n : float
        Refractive index difference (n_diffuser − n_air).
    height_mean_lambda, height_std_lambda : float
        Height map mean / std in wavelength units.
    smoothing_sigma_lambda : float
        Gaussian smoothing σ in wavelength units.
    seed : int or None
        RNG seed for reproducibility.
    device : torch device

    Returns
    -------
    dict with keys:
        height_map, phase_map, transmittance,
        correlation_length_mm, seed
    """
    # Convert λ-units to mm
    mu_mm = height_mean_lambda * wavelength_mm
    sigma0_mm = height_std_lambda * wavelength_mm
    smooth_sigma_mm = smoothing_sigma_lambda * wavelength_mm
    smooth_sigma_px = smooth_sigma_mm / dx_mm  # in pixel units

    # RNG
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    else:
        seed = gen.seed()  # record auto-seed

    # White noise height map W ~ N(μ, σ₀)
    W = torch.randn(N, N, generator=gen, dtype=torch.float64) * sigma0_mm + mu_mm
    if device is not None:
        W = W.to(device)

    # Gaussian smoothing
    height_map = _gaussian_smooth_2d(W, smooth_sigma_px)

    # Phase map: φ = 2π Δn D / λ
    phase_map = (2.0 * math.pi * delta_n / wavelength_mm) * height_map

    # Transmittance
    transmittance = torch.exp(1j * phase_map).to(torch.complex64)

    # Estimate correlation length (lazy import to avoid circular dep)
    from luo2022_d2nn.diffuser.correlation import estimate_correlation_length

    corr_len = estimate_correlation_length(phase_map, dx_mm)

    return {
        "height_map": height_map.to(torch.float32),
        "phase_map": phase_map.to(torch.float32),
        "transmittance": transmittance,
        "correlation_length_mm": float(corr_len),
        "seed": int(seed),
    }
