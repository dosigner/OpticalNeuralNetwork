"""Grating period estimation from output intensity patterns."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def estimate_grating_period(intensity: Tensor, dx_mm: float = 0.3) -> float:
    """Estimate grating period from output intensity.

    Steps:
    1. Average output intensity along y-axis to get 1D profile
    2. Fit sum of 3 Gaussians to the profile (use scipy.optimize.curve_fit)
    3. Extract peak positions
    4. Resolved period: p_hat = (max_peak - min_peak) / 2

    intensity: shape (N, N) or (1, N, N), real
    Returns period in mm.
    """
    from scipy.optimize import curve_fit

    # Handle (1, N, N) input
    img = intensity.detach().cpu().numpy()
    if img.ndim == 3:
        img = img.squeeze(0)

    N = img.shape[1]

    # 1. Average along y-axis (axis=0) to get 1D profile of length N
    profile = img.mean(axis=0)

    # x-coordinates in mm
    x = np.arange(N) * dx_mm

    # 2. Fit sum of 3 Gaussians
    def three_gaussians(x, a1, mu1, s1, a2, mu2, s2, a3, mu3, s3):
        return (
            a1 * np.exp(-((x - mu1) ** 2) / (2 * s1 ** 2))
            + a2 * np.exp(-((x - mu2) ** 2) / (2 * s2 ** 2))
            + a3 * np.exp(-((x - mu3) ** 2) / (2 * s3 ** 2))
        )

    # Initial guesses: spread 3 peaks evenly across the domain
    x_range = x[-1] - x[0]
    x_center = x[x.size // 2]
    amp_guess = profile.max()
    sigma_guess = x_range / 20.0

    p0 = [
        amp_guess, x_center - x_range / 4, sigma_guess,
        amp_guess, x_center, sigma_guess,
        amp_guess, x_center + x_range / 4, sigma_guess,
    ]

    bounds_lo = [0, x[0], 0] * 3
    bounds_hi = [np.inf, x[-1], x_range] * 3

    popt, _ = curve_fit(
        three_gaussians, x, profile, p0=p0,
        bounds=(bounds_lo, bounds_hi), maxfev=10000,
    )

    # 3. Extract peak positions
    mu_values = np.array([popt[1], popt[4], popt[7]])

    # 4. Resolved period
    p_hat = (mu_values.max() - mu_values.min()) / 2.0

    return float(p_hat)
