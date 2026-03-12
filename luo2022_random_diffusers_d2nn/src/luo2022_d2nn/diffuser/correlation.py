"""Correlation length estimation for phase diffusers.

Autocorrelation model: R_d(x,y) = exp(-π (x²+y²) / L²)
Estimate L from the 2-D autocorrelation of the phase map.
"""

from __future__ import annotations

import math

import torch

from luo2022_d2nn.optics.grids import make_spatial_grid


def estimate_correlation_length(phase_map: torch.Tensor, dx_mm: float) -> float:
    """Estimate the 1/e correlation length *L* of a phase map.

    Parameters
    ----------
    phase_map : Tensor, shape (N, N)
    dx_mm : float

    Returns
    -------
    L : float, correlation length in mm.
    """
    phi = phase_map.to(torch.float64)
    phi = phi - phi.mean()

    # 2-D autocorrelation via FFT: R = IFFT(|FFT(φ)|²)
    F_phi = torch.fft.fft2(phi)
    power = F_phi.real ** 2 + F_phi.imag ** 2
    R = torch.fft.ifft2(power).real
    R = torch.fft.fftshift(R, dim=(-2, -1))

    # Normalise so peak = 1
    R = R / R.max()

    # Build radial distance grid
    N = phi.shape[-1]
    x, y = make_spatial_grid(N, dx_mm, device=phi.device, dtype=torch.float64)
    r_sq = x ** 2 + y ** 2

    # Fit: R ≈ exp(-π r² / L²)  →  ln(R) = -π r² / L²
    # Use only points where R > threshold (avoid log of small/negative)
    mask = R > 0.05
    if mask.sum() < 10:
        mask = R > 0.01

    log_R = torch.log(R[mask].clamp(min=1e-12))
    r_sq_vals = r_sq[mask]

    # Weighted least-squares:  log_R = slope * r_sq  where slope = -π/L²
    # slope = Σ(r² * log_R) / Σ(r²²)
    numerator = (r_sq_vals * log_R).sum()
    denominator = (r_sq_vals * r_sq_vals).sum()

    if denominator.abs() < 1e-30:
        return float(dx_mm)  # fallback

    slope = numerator / denominator  # should be negative

    # L² = -π / slope
    L_sq = -math.pi / slope.item()
    if L_sq <= 0:
        return float(dx_mm)  # fallback

    return math.sqrt(L_sq)
