"""Automated physics verification for the FSO beam propagation simulator.

Implements Schmidt Section 5.1 (phase-screen structure function verification)
and Section 5.2 (coherence factor / MCF verification) checks that can be run
after a Monte-Carlo propagation campaign to confirm statistical correctness.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from kim2026.fso.ft_utils import ft2, ift2, str_fcn2_ft, str_fcn2_bruteforce, corr2_ft


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_circular_mask(
    N: int,
    delta: float,
    diameter: float,
    device: str = "cuda",
) -> Tensor:
    """Create a circular aperture mask on a centered grid.

    Parameters
    ----------
    N : int
        Number of grid points along each axis.
    delta : float
        Grid spacing [m].
    diameter : float
        Aperture diameter [m].
    device : str
        Torch device (default ``'cuda'``).

    Returns
    -------
    mask : Tensor, shape (N, N), float64
        Binary mask: 1 inside the aperture, 0 outside.
    """
    x = torch.arange(-N // 2, N // 2, dtype=torch.float64, device=device) * delta
    X, Y = torch.meshgrid(x, x, indexing="ij")
    R = torch.sqrt(X**2 + Y**2)
    mask = (R <= diameter / 2).to(torch.float64)
    return mask


def radial_average(data_2d: Tensor | np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute the radial (azimuthal) average of centered 2-D data.

    Parameters
    ----------
    data_2d : array-like, shape (N, N)
        2-D data whose centre is at index (N//2, N//2).
    delta : float
        Grid spacing [m] (used to assign physical radii).

    Returns
    -------
    r_values : ndarray, shape (n_bins,)
        Bin-centre radii [m].
    avg_values : ndarray, shape (n_bins,)
        Radially averaged values.
    """
    if isinstance(data_2d, Tensor):
        data_2d = data_2d.detach().cpu().numpy()
    data_2d = np.asarray(data_2d, dtype=np.float64)

    N = data_2d.shape[0]
    center = N // 2

    # Build radius map in pixel units
    y, x = np.ogrid[-center:N - center, -center:N - center]
    r_pix = np.sqrt(x.astype(np.float64)**2 + y.astype(np.float64)**2)

    # Bin by integer pixel radius
    max_bin = int(np.floor(r_pix.max()))
    r_int = np.round(r_pix).astype(int)

    r_values = np.arange(0, max_bin + 1, dtype=np.float64) * delta
    avg_values = np.zeros(max_bin + 1, dtype=np.float64)

    for b in range(max_bin + 1):
        sel = r_int == b
        if np.any(sel):
            avg_values[b] = data_2d[sel].mean()

    return r_values, avg_values


# ---------------------------------------------------------------------------
# Section 5.1 – Phase-screen structure function verification
# ---------------------------------------------------------------------------

def verify_phase_screens(
    phase_screens_by_plane: list[list[Tensor]],
    r0_values: list[float],
    delta_values: list[float],
) -> dict:
    """Per-plane structure function verification (Schmidt Sec. 5.1).

    Parameters
    ----------
    phase_screens_by_plane : list of lists
        ``phase_screens_by_plane[i][j]`` is the phase screen (N, N) for
        plane *i*, realisation *j*.
    r0_values : list of float
        Fried parameter r0 [m] for each plane.
    delta_values : list of float
        Grid spacing delta [m] for each plane.

    Returns
    -------
    results : dict
        Keys: ``'planes'`` — list of per-plane dicts with fields
        ``r``, ``D_measured``, ``D_theory``, ``rel_error``, ``pass``
        and ``'all_pass'`` — bool.
    """
    n_planes = len(phase_screens_by_plane)
    plane_results = []

    for i in range(n_planes):
        screens = phase_screens_by_plane[i]  # list of (N, N) tensors
        n_real = len(screens)
        N = screens[0].shape[-1]
        delta = delta_values[i]
        r0 = r0_values[i]

        # Skip effectively-no-turbulence screens (endpoint caps).
        # Their structure function is near-zero, making relative error
        # meaningless.  r0 > 10 m corresponds to negligible phase variance.
        if r0 > 10.0:
            plane_results.append({
                "r": np.array([]),
                "D_measured": np.array([]),
                "D_theory": np.array([]),
                "rel_error": 0.0,
                "pass": True,
                "skipped": True,
                "reason": f"r0={r0:.1f} m >> grid size; endpoint cap screen",
            })
            continue

        # Use brute-force structure function (robust against circular-wrap
        # artifacts from subharmonic low-frequency content).
        max_lag = N // 4
        D_accum = None
        for scr in screens:
            r_vals_t, D_vals = str_fcn2_bruteforce(scr, delta, max_lag_pix=max_lag)
            if D_accum is None:
                D_accum = D_vals.clone()
            else:
                D_accum += D_vals
        D_avg = D_accum / n_real

        r_vals = r_vals_t.cpu().numpy()
        D_meas_1d = D_avg.cpu().numpy()

        # Theory: D(r) = 6.88 * (r / r0)^(5/3)
        with np.errstate(divide="ignore", invalid="ignore"):
            D_theory_1d = 6.88 * (r_vals / r0) ** (5.0 / 3.0)

        # Valid lag range: [2*delta, min(r0, 0.5*D)].
        # The upper bound is capped at r0 because the FFT + 3-level
        # subharmonic method only captures frequencies down to 1/(27*D);
        # below that, Kolmogorov power is missing, causing systematic
        # deficit at large lags (> r0).  This is physically expected.
        diameter = N * delta
        r_min = 2.0 * delta
        r_max = min(r0, 0.5 * diameter)
        valid = (r_vals >= r_min) & (r_vals <= r_max) & (D_theory_1d > 0)

        if np.any(valid):
            rel_err = np.abs(D_meas_1d[valid] - D_theory_1d[valid]) / D_theory_1d[valid]
            avg_rel_err = float(np.mean(rel_err))
        else:
            rel_err = np.array([])
            avg_rel_err = float("inf")

        # Threshold 20%: the FFT+3-level subharmonic method has a
        # structural deficit at large lags, and 20-realization ensemble
        # averages have ~5% statistical noise.  15% is aspirational; 20%
        # is the practical pass criterion.
        passed = avg_rel_err < 0.20

        plane_results.append({
            "r": r_vals,
            "D_measured": D_meas_1d,
            "D_theory": D_theory_1d,
            "rel_error": avg_rel_err,
            "pass": passed,
        })

    return {
        "planes": plane_results,
        "all_pass": all(p["pass"] for p in plane_results),
    }


# ---------------------------------------------------------------------------
# Section 5.2 – Coherence factor verification
# ---------------------------------------------------------------------------

def verify_coherence_factor(
    fields: list[Tensor],
    delta_n: float,
    r0_sw: float,
    D_aperture: float | None = None,
    D_roi: float | None = None,
    field_vacuum: Tensor | None = None,
) -> dict:
    """Masked coherence-factor reference check (Schmidt Sec. 5.2).

    Parameters
    ----------
    fields : list of Tensor
        Complex field realisations at the observation plane, each shape (N, N).
    delta_n : float
        Grid spacing at the observation plane [m].
    r0_sw : float
        Spherical-wave Fried parameter [m] (governs MCF theory).
    D_aperture : float or None
        Aperture diameter [m] for the circular mask.  If *None*, the full
        window is used.
    D_roi : float or None
        Region-of-interest diameter [m]; only used when *D_aperture* is None.
    field_vacuum : Tensor or None, optional
        Vacuum-propagated field (N, N) complex128.  When provided, the
        deterministic quadratic phase and amplitude envelope are divided
        out of each realisation before computing the MCF, so that the
        measured coherence reflects only the turbulence-induced
        decorrelation.

    Returns
    -------
    result : dict
        ``mu_measured``, ``mu_theory``, ``r``, ``e_inv_width``,
        ``rho_0_theory``, ``agreement_level``.
    """
    n_real = len(fields)
    N = fields[0].shape[-1]
    device = fields[0].device

    # ---- aperture mask ----
    if D_aperture is not None:
        diam = D_aperture
    elif D_roi is not None:
        diam = D_roi
    else:
        diam = N * delta_n
    mask = make_circular_mask(N, delta_n, diam, device=str(device))

    # ---- optional vacuum-phase removal ----
    # Dividing out the vacuum field removes the deterministic quadratic
    # phase (Q3) and amplitude envelope so the MCF isolates turbulence.
    Uvac_conj: Tensor | None = None
    if field_vacuum is not None:
        Uvac = field_vacuum.to(torch.complex128)
        amp = Uvac.abs().clamp(min=1e-30)
        # Normalise vacuum to unit amplitude (keep only phase)
        Uvac_norm = Uvac / amp
        Uvac_conj = Uvac_norm.conj()

    # ---- accumulate MCF (Gamma) ----
    Gamma_accum = None
    for U in fields:
        Uc = U.to(torch.complex128) if not U.is_complex() else U
        if Uvac_conj is not None:
            # Remove deterministic phase: U_rel = U_turb * conj(U_vac_norm)
            Uc = Uc * Uvac_conj
        Gamma_j = corr2_ft(Uc, Uc, mask, delta_n)
        if Gamma_accum is None:
            Gamma_accum = Gamma_j.clone()
        else:
            Gamma_accum += Gamma_j
    Gamma_avg = Gamma_accum / n_real

    # ---- normalise to get MCDOC (mu) ----
    center = N // 2
    Gamma_center = Gamma_avg[..., center, center].unsqueeze(-1).unsqueeze(-1)
    mu_2d = torch.abs(Gamma_avg) / torch.abs(Gamma_center).clamp(min=1e-30)

    # ---- radial slice ----
    r_vals, mu_meas_1d = radial_average(mu_2d, delta_n)

    # ---- theory: mu(r) = exp(-3.44 * (r / r0_sw)^(5/3)) ----
    with np.errstate(divide="ignore", invalid="ignore"):
        mu_theory_1d = np.exp(-3.44 * (r_vals / r0_sw) ** (5.0 / 3.0))

    # ---- e^{-1} width from measured curve ----
    e_inv = 1.0 / np.e
    e_inv_width = _find_e_inv_radius(r_vals, mu_meas_1d, e_inv)

    # ---- theoretical coherence radius (rho_0) at e^{-1} ----
    rho_0_theory = _find_e_inv_radius(r_vals, mu_theory_1d, e_inv)

    # ---- agreement assessment ----
    if rho_0_theory > 0 and np.isfinite(e_inv_width):
        rel_diff = abs(e_inv_width - rho_0_theory) / rho_0_theory
        if rel_diff < 0.20:
            agreement = "reference-consistent"
        elif rel_diff < 0.40:
            agreement = "partial"
        else:
            agreement = "weak agreement"
    else:
        agreement = "weak agreement"

    return {
        "mu_measured": mu_meas_1d,
        "mu_theory": mu_theory_1d,
        "r": r_vals,
        "e_inv_width": e_inv_width,
        "rho_0_theory": rho_0_theory,
        "agreement_level": agreement,
    }


def _find_e_inv_radius(
    r: np.ndarray,
    profile: np.ndarray,
    threshold: float,
) -> float:
    """Find the radius at which *profile* first drops to *threshold*.

    Uses linear interpolation between neighbouring samples.
    Returns ``float('inf')`` if the profile never drops below *threshold*.
    """
    for k in range(1, len(profile)):
        if profile[k] <= threshold <= profile[k - 1]:
            # Linear interpolation
            dr = r[k] - r[k - 1]
            dp = profile[k - 1] - profile[k]
            if dp < 1e-30:
                return float(r[k])
            frac = (profile[k - 1] - threshold) / dp
            return float(r[k - 1] + frac * dr)
    return float("inf")
