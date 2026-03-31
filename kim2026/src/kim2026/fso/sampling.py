"""Sampling-constraint analysis and automatic grid-parameter selection.

Implements the sampling criteria from Schmidt, *Numerical Simulation of
Optical Wave Propagation* (SPIE Press), Eqs 9.84--9.90 and Listing 9.6.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.special import beta as beta_func

from kim2026.fso.atmosphere import compute_atmospheric_params
from kim2026.fso.config import SimulationConfig


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SamplingResult:
    """Resolved grid and propagation parameters."""

    delta1: float  # source-plane grid spacing [m]
    delta_n: float  # observation-plane grid spacing [m]
    N: int  # grid points (power of 2)
    n_scr: int  # number of phase screens
    D1_prime: float  # turbulence-corrected source aperture [m]
    D2_prime: float  # turbulence-corrected observation aperture [m]
    dz_max: float  # max partial propagation distance [m]
    z_planes: List[float] = field(default_factory=list)  # screen positions [m]
    delta_values: List[float] = field(default_factory=list)  # grid spacing per plane [m]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_pow2(n: float) -> int:
    """Return the smallest power of 2 >= ceil(n)."""
    n_ceil = math.ceil(n)
    if n_ceil <= 1:
        return 1
    return 1 << (n_ceil - 1).bit_length()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_sampling(config: SimulationConfig) -> SamplingResult:
    """Determine grid parameters that satisfy all sampling constraints.

    Parameters
    ----------
    config : SimulationConfig
        Fully-specified simulation configuration.

    Returns
    -------
    SamplingResult
        Resolved grid and propagation layout.

    Raises
    ------
    ValueError
        If the constraints yield an empty feasible region for *delta1*.
    """

    wvl = config.wvl
    k = config.k
    Dz = config.Dz
    Cn2 = config.Cn2
    D1 = config.D1
    D_ROI = config.D_roi
    delta_n = config.delta_n

    # ------------------------------------------------------------------
    # 1. Turbulence beam-spreading correction (Eqs 9.84--9.85)
    # ------------------------------------------------------------------
    atm = compute_atmospheric_params(k, Cn2, Dz)
    r0_sw = atm["r0_sw"]

    c = 2.0  # 97% energy containment (Listing 9.6 line 2)
    D1_prime = D1 + c * wvl * Dz / r0_sw  # Eq 9.84
    D2_prime = D_ROI + c * wvl * Dz / r0_sw  # Eq 9.85

    # ------------------------------------------------------------------
    # 2. Feasible delta1 (Constraints 1 and 3 intersection)
    # ------------------------------------------------------------------
    # Constraint 1 (Eq 9.86) — upper bound on delta1
    delta1_upper_c1 = (wvl * Dz - D1_prime * delta_n) / D2_prime

    # Constraint 3 (Eq 9.88, R = inf for collimated beam)
    delta1_lower_c3 = delta_n - wvl * Dz / D1
    delta1_upper_c3 = delta_n + wvl * Dz / D1

    # Minimum samples across source aperture
    delta1_upper_samples = D1 / 5.0

    # Intersection of all bounds
    delta1_lo = max(delta1_lower_c3, 1e-30)  # must be positive
    delta1_hi = min(delta1_upper_c1, delta1_upper_c3, delta1_upper_samples)

    if delta1_lo > delta1_hi:
        # Recommend valid delta_n range from Constraint 1 rearranged
        dn_max = wvl * Dz / (D1_prime + D2_prime)
        raise ValueError(
            f"No feasible delta1 exists for delta_n={delta_n:.4e} m. "
            f"Try delta_n <= {dn_max:.4e} m."
        )

    # Pick the maximum delta1 in the valid range (minimises grid size)
    delta1 = delta1_hi

    # ------------------------------------------------------------------
    # 3. Grid size N (Constraint 2, Eq 9.87)
    # ------------------------------------------------------------------
    N_min = (
        D1_prime / (2.0 * delta1)
        + D2_prime / (2.0 * delta_n)
        + wvl * Dz / (2.0 * delta1 * delta_n)
    )
    N_auto = _next_pow2(N_min)

    if config.N is not None:
        N = config.N
        if N < math.ceil(N_min):
            warnings.warn(
                f"User-specified N={N} is below the minimum N_min={math.ceil(N_min)} "
                f"required by Constraint 2 (Eq 9.87). Results may have aliasing.",
                stacklevel=2,
            )
    else:
        N = N_auto

    # ------------------------------------------------------------------
    # 4. Number of partial propagations (Constraint 4, Eqs 9.89--9.90)
    # ------------------------------------------------------------------
    delta_min = min(delta1, delta_n)
    dz_max = delta_min ** 2 * N / wvl  # Eq 9.89
    n_min = math.ceil(Dz / dz_max) + 1  # Eq 9.90
    # At least 6 screens: fewer gives the constrained r0 optimiser too
    # few degrees of freedom to match both r0_sw and sigma2_chi targets.
    n_min = max(n_min, 6)

    # Rytov constraint: partial Rytov number <= 0.1
    # sigma2_chi_partial = 0.563 * k^(7/6) * Cn2 * (Dz/n_scr)^(11/6) * B(11/6,11/6)
    B_val = beta_func(11.0 / 6.0, 11.0 / 6.0)
    n_scr_rytov = n_min
    while True:
        dz_partial = Dz / n_scr_rytov
        partial_rytov = (
            0.563 * k ** (7.0 / 6.0) * Cn2
            * dz_partial ** (11.0 / 6.0) * B_val
        )
        if partial_rytov <= 0.1:
            break
        n_scr_rytov += 1

    n_scr = max(n_min, n_scr_rytov)

    # ------------------------------------------------------------------
    # 5. Plane positions and per-plane grid spacings
    # ------------------------------------------------------------------
    z_planes_arr = np.linspace(0.0, Dz, n_scr)
    alpha = z_planes_arr / Dz if Dz > 0 else np.zeros_like(z_planes_arr)
    delta_values_arr = (1.0 - alpha) * delta1 + alpha * delta_n

    return SamplingResult(
        delta1=delta1,
        delta_n=delta_n,
        N=N,
        n_scr=n_scr,
        D1_prime=D1_prime,
        D2_prime=D2_prime,
        dz_max=dz_max,
        z_planes=z_planes_arr.tolist(),
        delta_values=delta_values_arr.tolist(),
    )
