"""Correlation length estimation for phase diffusers.

Paper model (Eq. 5, Luo et al. eLight 2022):
    R_d(x,y) = exp(-π (x²+y²) / L²)

Estimate L from the radially-averaged 2-D autocorrelation of the phase map
by fitting the Gaussian decay to extract L.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from luo2022_d2nn.optics.grids import make_spatial_grid


def estimate_correlation_length(phase_map: torch.Tensor, dx_mm: float) -> float:
    """Estimate correlation length L matching paper's definition.

    Fits the radially-averaged autocorrelation to:
        R(r) = exp(-π r² / L²)

    At r = L, R drops to exp(-π) ≈ 0.043.

    Parameters
    ----------
    phase_map : Tensor, shape (N, N)
    dx_mm : float

    Returns
    -------
    L : float, correlation length in mm (paper's definition).
    """
    phi = phase_map.to(torch.float64)
    phi = phi - phi.mean()

    # 2-D autocorrelation via FFT: R = IFFT(|FFT(φ)|²)
    F_phi = torch.fft.fft2(phi)
    power = F_phi.real ** 2 + F_phi.imag ** 2
    R = torch.fft.ifft2(power).real
    R = torch.fft.fftshift(R, dim=(-2, -1))
    R = R / R.max()

    N = phi.shape[-1]
    mid = N // 2

    # Radially-averaged autocorrelation for robust fitting
    y_idx, x_idx = np.mgrid[:N, :N]
    r_px = np.sqrt((x_idx - mid) ** 2 + (y_idx - mid) ** 2)
    R_np = R.cpu().numpy()

    # Bin by radius (integer pixel bins)
    max_r = min(mid, 60)  # fit within reasonable range
    r_bins = np.arange(0, max_r + 1)
    r_avg = np.zeros(len(r_bins))
    for i, r in enumerate(r_bins):
        mask = (r_px >= r - 0.5) & (r_px < r + 0.5)
        if mask.sum() > 0:
            r_avg[i] = R_np[mask].mean()

    # Fit: ln(R) = -π r² / L²  →  slope = -π / L²
    # Use points where R > 0.05 to avoid noise in the tail
    valid = r_avg > 0.05
    valid[0] = False  # skip r=0 (always 1.0)
    if valid.sum() < 3:
        valid = r_avg > 0.01
        valid[0] = False

    r_mm = r_bins[valid] * dx_mm
    log_R = np.log(np.clip(r_avg[valid], 1e-12, None))
    r_sq = r_mm ** 2

    # Least squares: log_R = slope * r_sq, where slope = -π/L²
    slope = np.sum(r_sq * log_R) / np.sum(r_sq ** 2)

    if slope >= 0:
        return float(dx_mm)  # fallback

    L_sq = -math.pi / slope
    return math.sqrt(L_sq)


def estimate_correlation_fwhm(phase_map: torch.Tensor, dx_mm: float) -> float:
    """Estimate FWHM of autocorrelation (alternative metric).

    Returns the half-width at half-maximum in mm.
    """
    phi = phase_map.to(torch.float64)
    phi = phi - phi.mean()

    F_phi = torch.fft.fft2(phi)
    power = F_phi.real ** 2 + F_phi.imag ** 2
    R = torch.fft.ifft2(power).real
    R = torch.fft.fftshift(R, dim=(-2, -1))
    R = R / R.max()

    N = phi.shape[-1]
    mid = N // 2
    line = R[mid, mid:].cpu().numpy()

    below = np.where(line < 0.5)[0]
    fwhm_px = float(below[0]) if len(below) > 0 else float(N // 2)
    return fwhm_px * dx_mm
