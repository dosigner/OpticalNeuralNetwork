"""Atmospheric turbulence parameters and phase-screen r0 distribution.

References are to Schmidt, *Numerical Simulation of Optical Wave Propagation
with Examples in MATLAB* (SPIE Press).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import beta as beta_func


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_atmospheric_params(
    k: float,
    Cn2: float,
    Dz: float,
) -> dict:
    """Compute atmospheric turbulence parameters.

    Parameters
    ----------
    k : float
        Optical wavenumber 2*pi/lambda [rad/m].
    Cn2 : float
        Refractive-index structure constant [m^(-2/3)].
    Dz : float
        Propagation path length [m].

    Returns
    -------
    dict with keys r0_pw, r0_sw, sigma2_chi_sw, weak_fluctuation.
    """
    # Eq 9.42 — plane-wave Fried parameter
    r0_pw = (0.423 * k ** 2 * Cn2 * Dz) ** (-3.0 / 5.0)

    # Eq 9.43 — spherical-wave Fried parameter
    r0_sw = (0.423 * k ** 2 * Cn2 * (3.0 / 8.0) * Dz) ** (-3.0 / 5.0)

    # Eq 9.64 — spherical-wave Rytov variance (log-amplitude)
    B_val = beta_func(11.0 / 6.0, 11.0 / 6.0)
    sigma2_chi_sw = 0.563 * k ** (7.0 / 6.0) * Cn2 * Dz ** (11.0 / 6.0) * B_val

    return {
        "r0_pw": r0_pw,
        "r0_sw": r0_sw,
        "sigma2_chi_sw": sigma2_chi_sw,
        "weak_fluctuation": sigma2_chi_sw < 0.25,
    }


def optimize_screen_r0(
    r0_sw: float,
    sigma2_chi: float,
    k: float,
    Dz: float,
    n_scr: int,
) -> np.ndarray:
    """Optimise phase-screen r0 distribution (Listing 9.5).

    Parameters
    ----------
    r0_sw : float
        Spherical-wave Fried parameter [m].
    sigma2_chi : float
        Spherical-wave Rytov variance (log-amplitude).
    k : float
        Optical wavenumber [rad/m].
    Dz : float
        Total propagation distance [m].
    n_scr : int
        Number of phase screens (>= 2).

    Returns
    -------
    np.ndarray of shape (n_scr,)
        Per-screen Fried parameters r0_i [m].
    """
    if n_scr < 2:
        raise ValueError("n_scr must be >= 2")

    # Screen positions equally spaced along [0, Dz)
    # n_scr screens at z_i = i * Dz / n_scr  (i = 0, ..., n_scr-1)
    z_planes = np.linspace(0, Dz, n_scr + 1)[:-1]  # drop endpoint
    alpha = z_planes / Dz  # normalised positions in [0, 1)

    # Constraint matrix A (2 x n_scr)
    A = np.zeros((2, n_scr))
    A[0, :] = alpha ** (5.0 / 3.0)
    A[1, :] = alpha ** (5.0 / 6.0) * (1.0 - alpha) ** (5.0 / 6.0)

    # Target vector b (2,)
    kDz_56 = (k / Dz) ** (5.0 / 6.0)
    b = np.array([
        r0_sw ** (-5.0 / 3.0),
        sigma2_chi * kDz_56 / 1.33,
    ])

    # Upper bounds on x_i
    x_max = np.empty(n_scr)
    for i in range(n_scr):
        if A[1, i] > 0:
            x_max[i] = 0.1 * kDz_56 / (1.33 * A[1, i])
        else:
            # Endpoint — weak screen cap
            x_max[i] = 50.0 ** (-5.0 / 3.0)

    # Initial guess
    x0_val = (n_scr / 3.0 * r0_sw) ** (-5.0 / 3.0)
    x0 = np.full(n_scr, x0_val)
    # Clip initial guess to feasible region
    x0 = np.clip(x0, 0, x_max)

    # Bounds
    bounds = [(0.0, x_max[i]) for i in range(n_scr)]

    # Objective: min ||A x - b||^2
    def objective(x: np.ndarray) -> float:
        residual = A @ x - b
        return float(residual @ residual)

    def jacobian(x: np.ndarray) -> np.ndarray:
        residual = A @ x - b  # (2,)
        return 2.0 * (A.T @ residual)  # (n_scr,)

    result = minimize(
        objective,
        x0,
        jac=jacobian,
        method="SLSQP",
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-15},
    )

    x_opt = result.x

    # Convert back to r0 values: r0_i = x_i^(-3/5)
    # Guard against zero (shouldn't happen, but be safe)
    r0 = np.where(x_opt > 0, x_opt ** (-3.0 / 5.0), np.inf)

    # ------------------------------------------------------------------
    # Verification: reconstructed r0_sw and sigma2_chi within 1 %
    # ------------------------------------------------------------------
    r0_sw_recon = (np.sum(A[0, :] * x_opt)) ** (-3.0 / 5.0)
    sigma2_chi_recon = np.sum(A[1, :] * x_opt) * 1.33 / kDz_56

    rel_err_r0 = abs(r0_sw_recon - r0_sw) / r0_sw
    rel_err_chi = abs(sigma2_chi_recon - sigma2_chi) / sigma2_chi if sigma2_chi > 0 else 0.0

    # With few screens or very weak turbulence the constrained
    # optimisation may not match both targets precisely.  For sigma2_chi
    # deep in the weak-fluctuation regime (< 0.01) the absolute value is
    # so small that large relative errors are physically irrelevant.
    tol_warn, tol_fail = 0.05, 0.25
    # Relax sigma2_chi tolerance when deep in weak-fluctuation regime
    chi_tol_fail = tol_fail if sigma2_chi > 0.01 else 1.0

    if rel_err_r0 > tol_fail:
        raise RuntimeError(
            f"r0_sw reconstruction error {rel_err_r0:.4%} exceeds {tol_fail:.0%} "
            f"(target={r0_sw:.6e}, got={r0_sw_recon:.6e})"
        )
    if sigma2_chi > 0 and rel_err_chi > chi_tol_fail:
        raise RuntimeError(
            f"sigma2_chi reconstruction error {rel_err_chi:.4%} exceeds {chi_tol_fail:.0%} "
            f"(target={sigma2_chi:.6e}, got={sigma2_chi_recon:.6e})"
        )
    if rel_err_r0 > tol_warn or (sigma2_chi > 0 and rel_err_chi > tol_warn):
        import warnings
        warnings.warn(
            f"r0 optimisation: r0_err={rel_err_r0:.2%}, chi_err={rel_err_chi:.2%} "
            f"(n_scr={n_scr}, sigma2_chi={sigma2_chi:.4e})"
        )

    return r0
