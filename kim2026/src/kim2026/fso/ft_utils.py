"""Fourier-transform utilities for beam propagation.

Implements the centered-FT convention from Schmidt, *Numerical Simulation of
Optical Wave Propagation with Examples in MATLAB* (SPIE Press, 2010):

    ft2(g, delta)     = fftshift(fft2(ifftshift(g))) * delta**2
    ift2(G, delta_f)  = ifftshift(ifft2(fftshift(G))) * (N * delta_f)**2

where delta_f = 1 / (N * delta).

All functions operate on torch tensors and are CUDA-compatible.
"""

from __future__ import annotations

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Core FT pair
# ---------------------------------------------------------------------------

def ft2(g: Tensor, delta: float) -> Tensor:
    """Centered 2-D Fourier transform with physical scaling.

    Parameters
    ----------
    g : Tensor, shape (..., N, N)
        Spatial-domain field (complex128).
    delta : float
        Grid spacing in the spatial domain [m].

    Returns
    -------
    G : Tensor, same shape as *g*
        Frequency-domain field, scaled so that the continuous-FT
        amplitude convention is preserved.
    """
    G = torch.fft.fftshift(
        torch.fft.fft2(
            torch.fft.ifftshift(g, dim=(-2, -1)),
        ),
        dim=(-2, -1),
    )
    return G * delta**2


def ift2(G: Tensor, delta_f: float) -> Tensor:
    """Centered 2-D inverse Fourier transform with physical scaling.

    Parameters
    ----------
    G : Tensor, shape (..., N, N)
        Frequency-domain field (complex128).
    delta_f : float
        Grid spacing in the frequency domain [1/m].

    Returns
    -------
    g : Tensor, same shape as *G*
        Spatial-domain field.
    """
    N = G.shape[-1]
    g = torch.fft.ifftshift(
        torch.fft.ifft2(
            torch.fft.fftshift(G, dim=(-2, -1)),
        ),
        dim=(-2, -1),
    )
    return g * (N * delta_f) ** 2


# ---------------------------------------------------------------------------
# Statistical helpers (Schmidt Listings 3.6 / 3.7)
# ---------------------------------------------------------------------------

def corr2_ft(
    u1: Tensor,
    u2: Tensor,
    mask: Tensor,
    delta: float,
) -> Tensor:
    """Cross-correlation of two fields via FFT (Schmidt Listing 3.6).

    Computes the masked, normalised cross-correlation::

        Gamma = ift2( ft2(u1*mask) * conj(ft2(u2*mask)) )
              / ift2( |ft2(mask)|^2 )

    Parameters
    ----------
    u1, u2 : Tensor, shape (..., N, N)
        Complex field realisations.
    mask : Tensor, shape (N, N)
        Binary / smooth aperture mask (real-valued).
    delta : float
        Grid spacing [m].

    Returns
    -------
    Gamma : Tensor, same shape as u1
        Normalised cross-correlation (mutual coherence function estimate).
    """
    N = u1.shape[-1]
    delta_f = 1.0 / (N * delta)

    U1 = ft2(u1 * mask, delta)
    U2 = ft2(u2 * mask, delta)
    M = ft2(mask, delta)

    numerator = ift2(U1 * U2.conj(), delta_f)
    denominator = ift2(torch.abs(M) ** 2, delta_f)

    # Avoid division by zero at edges where mask support vanishes.
    denominator_safe = denominator.clone()
    denominator_safe[denominator_safe.abs() < 1e-30] = 1.0

    return numerator / denominator_safe


def str_fcn2_ft(
    phase: Tensor,
    mask: Tensor,
    delta: float,
) -> Tensor:
    """Phase structure function via FFT (Schmidt Listing 3.7).

    Uses the exact Listing 3.7 formula which correctly handles masked
    data by computing::

        D(r) = 2 * [ift2(S·W*) · w2* − |ift2(P·W*)|²] / w2*²

    where P = ft2(φ·m), S = ft2((φ·m)²), W = ft2(m), w2 = ift2(|W|²).
    This is equivalent to computing the mean-subtracted second moment
    (variance) in the overlap region at each lag, which equals the
    structure function for stationary fields.

    Parameters
    ----------
    phase : Tensor, shape (..., N, N)
        Real-valued phase screen [rad].
    mask : Tensor, shape (N, N)
        Binary / smooth aperture mask (real-valued).
    delta : float
        Grid spacing [m].

    Returns
    -------
    D_phi : Tensor, shape (..., N, N)
        Phase structure function estimate.
    """
    N = phase.shape[-1]
    delta_f = 1.0 / (N * delta)

    # Ensure complex type for FFT
    phase_c = phase.to(torch.complex128) if not phase.is_complex() else phase
    mask_c = mask.to(torch.complex128) if not mask.is_complex() else mask

    # Masked phase and its square
    ph = phase_c * mask_c
    P = ft2(ph, delta)             # FT of φ·m
    S = ft2(ph * ph, delta)        # FT of (φ·m)²
    W = ft2(mask_c, delta)         # FT of mask

    w2 = ift2(torch.abs(W) ** 2, delta_f)   # autocorrelation of mask

    # Listing 3.7 formula
    SW = ift2(S * W.conj(), delta_f)         # cross-corr of (φ·m)² with mask
    PW = ift2(P * W.conj(), delta_f)         # cross-corr of φ·m with mask

    w2_safe = w2.clone()
    w2_safe[w2_safe.abs() < 1e-30] = 1.0

    D_phi = 2.0 * (SW * w2_safe.conj() - torch.abs(PW) ** 2) / (w2_safe.conj() ** 2)

    return D_phi.real


def str_fcn2_bruteforce(
    phase: Tensor,
    delta: float,
    max_lag_pix: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Brute-force 1-D radial structure function (no circular-wrap artifacts).

    Computes D(r) = <|φ(x) − φ(x+r)|²> by direct averaging along x and y
    axes, then averaging the two orientations.  Robust for screens containing
    sub-grid-frequency (subharmonic) content.

    Parameters
    ----------
    phase : Tensor, shape (N, N), float64
        Phase screen [rad].
    delta : float
        Grid spacing [m].
    max_lag_pix : int or None
        Maximum lag in pixels.  Default: N // 4.

    Returns
    -------
    r : Tensor, shape (n_lags,)
        Lag distances [m].
    D : Tensor, shape (n_lags,)
        Structure-function values [rad²].
    """
    N = phase.shape[-1]
    if max_lag_pix is None:
        max_lag_pix = N // 4

    device = phase.device
    r_vals = torch.arange(1, max_lag_pix + 1, dtype=torch.float64, device=device) * delta
    D_vals = torch.empty(max_lag_pix, dtype=torch.float64, device=device)

    for lag in range(1, max_lag_pix + 1):
        # Average along x-axis and y-axis shifts
        dx = (phase[:, lag:] - phase[:, :-lag]) ** 2
        dy = (phase[lag:, :] - phase[:-lag, :]) ** 2
        D_vals[lag - 1] = 0.5 * (dx.mean() + dy.mean())

    return r_vals, D_vals
