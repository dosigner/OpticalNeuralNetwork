"""Grating period estimation from output intensity patterns."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def estimate_grating_period(intensity: Tensor, dx_mm: float = 0.3) -> float:
    """Estimate grating period from output intensity.

    Steps:
    1. Average output intensity along x-axis to get a 1D profile across the bar spacing
    2. Detect the three dominant bar peaks from the averaged profile
    3. Compute resolved period: p_hat = (max_peak - min_peak) / 2
    4. Fall back to a constrained 3-Gaussian fit if peak detection is ambiguous

    intensity: shape (N, N) or (1, N, N), real
    Returns period in mm.
    """
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks

    # Handle (1, N, N) input
    img = intensity.detach().cpu().numpy()
    if img.ndim == 3:
        img = img.squeeze(0)

    N = img.shape[0]

    # For the paper-style horizontal 3-bar targets, the period varies along y.
    # Average across x so the 1D profile preserves the inter-bar spacing.
    profile = img.mean(axis=1)

    # Coordinates in mm along the varying axis
    x = np.arange(N) * dx_mm

    # First try direct peak detection on the averaged profile. This is robust for
    # the paper-style 3-bar resolution targets and avoids unstable global fits.
    peak_indices, properties = find_peaks(
        profile,
        height=float(profile.max()) * 0.25,
        prominence=float(profile.max()) * 0.10,
    )
    if peak_indices.size >= 3:
        top3 = np.argsort(properties["peak_heights"])[-3:]
        mu_values = np.sort(x[peak_indices[top3]])
        return float((mu_values.max() - mu_values.min()) / 2.0)

    # Fall back to a constrained 3-Gaussian fit when the peaks are not cleanly separated.
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

    # Extract peak positions from the fallback fit
    mu_values = np.array([popt[1], popt[4], popt[7]])

    # Resolved period
    p_hat = (mu_values.max() - mu_values.min()) / 2.0

    return float(p_hat)
