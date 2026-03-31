"""Kolmogorov turbulence phase-screen generation via FFT + subharmonics.

Implements Listings 9.2 and 9.3 from Schmidt, *Numerical Simulation of
Optical Wave Propagation with Examples in MATLAB* (SPIE Press, 2010).

The Kolmogorov phase power spectral density (Eq. 9.52, ordinary frequency) is

    Phi_phi(f) = 0.023 * r0^{-5/3} * f^{-11/3}

where *r0* is the Fried parameter and *f* is spatial frequency magnitude.

Phase screens are generated in float64 for numerical accuracy and are
fully CUDA-compatible.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# High-frequency phase screen (Listing 9.2)
# ---------------------------------------------------------------------------

def ft_phase_screen(
    r0: float,
    N: int,
    delta: float,
    device: str = "cuda",
) -> Tensor:
    """FFT-based Kolmogorov phase screen (high-frequency component).

    Parameters
    ----------
    r0 : float
        Fried parameter [m].
    N : int
        Grid side length (pixels).  Should be even.
    delta : float
        Grid spacing [m].
    device : str
        Torch device (default ``'cuda'``).

    Returns
    -------
    phz_hi : Tensor, shape (N, N), dtype float64
        High-frequency phase screen [rad].
    """
    del_f = 1.0 / (N * delta)

    # Centred frequency grid
    fx = torch.arange(-N // 2, N // 2, dtype=torch.float64, device=device) * del_f
    fy = fx.clone()
    FX, FY = torch.meshgrid(fx, fy, indexing="ij")
    f = torch.sqrt(FX ** 2 + FY ** 2)

    # Kolmogorov PSD
    PSD_phi = 0.023 * r0 ** (-5.0 / 3.0) * f ** (-11.0 / 3.0)
    PSD_phi[N // 2, N // 2] = 0.0  # zero DC to avoid inf

    # Random complex Fourier coefficients
    cn = (
        torch.randn(N, N, dtype=torch.float64, device=device)
        + 1j * torch.randn(N, N, dtype=torch.float64, device=device)
    ) * torch.sqrt(PSD_phi) * del_f

    # Inverse FT to spatial domain  –  ift2(cn, 1) = ifftshift(ifft2(fftshift(cn))) * N^2
    phz_hi = torch.fft.ifftshift(
        torch.fft.ifft2(torch.fft.fftshift(cn))
    ).real * (N ** 2)

    return phz_hi


# ---------------------------------------------------------------------------
# FT + Subharmonic phase screen (Listing 9.3)
# ---------------------------------------------------------------------------

def ft_sh_phase_screen(
    r0: float,
    N: int,
    delta: float,
    device: str = "cuda",
) -> Tensor:
    """FFT + subharmonic Kolmogorov phase screen.

    Combines the high-frequency FFT screen with three levels of
    subharmonic correction to fill in the low-frequency content that
    the FFT grid misses.

    Parameters
    ----------
    r0 : float
        Fried parameter [m].
    N : int
        Grid side length (pixels).  Should be even.
    delta : float
        Grid spacing [m].
    device : str
        Torch device (default ``'cuda'``).

    Returns
    -------
    phz : Tensor, shape (N, N), dtype float64
        Full phase screen [rad].
    """
    # High-frequency component
    phz_hi = ft_phase_screen(r0, N, delta, device=device)

    D = N * delta  # physical grid extent

    phz_lo = torch.zeros(N, N, dtype=torch.float64, device=device)

    # Spatial coordinate grids
    x = torch.arange(-N // 2, N // 2, dtype=torch.float64, device=device) * delta
    y = x.clone()
    X, Y = torch.meshgrid(x, y, indexing="ij")

    for p in range(1, 4):  # subharmonic levels 1, 2, 3
        del_f_p = 1.0 / (3 ** p * D)

        fx_p = torch.tensor([-1, 0, 1], dtype=torch.float64, device=device) * del_f_p
        fy_p = fx_p.clone()
        FX_p, FY_p = torch.meshgrid(fx_p, fy_p, indexing="ij")
        f_p = torch.sqrt(FX_p ** 2 + FY_p ** 2)

        # Kolmogorov PSD at subharmonic frequencies
        PSD_p = 0.023 * r0 ** (-5.0 / 3.0) * f_p ** (-11.0 / 3.0)
        PSD_p[1, 1] = 0.0  # zero DC

        # Random complex coefficients for this level
        cn_p = (
            torch.randn(3, 3, dtype=torch.float64, device=device)
            + 1j * torch.randn(3, 3, dtype=torch.float64, device=device)
        ) * torch.sqrt(PSD_p) * del_f_p

        # Accumulate subharmonic contribution via direct DFT synthesis
        SH = torch.zeros(N, N, dtype=torch.complex128, device=device)
        for ii in range(3):
            for jj in range(3):
                SH = SH + cn_p[ii, jj] * torch.exp(
                    1j * 2.0 * math.pi * (FX_p[ii, jj] * X + FY_p[ii, jj] * Y)
                )
        phz_lo = phz_lo + SH.real

    # Remove residual mean from low-frequency part
    phz_lo = phz_lo - phz_lo.mean()

    return phz_lo + phz_hi


# ---------------------------------------------------------------------------
# Batch generation helper
# ---------------------------------------------------------------------------

def generate_phase_screens(
    r0_values: list[float] | Tensor,
    N: int,
    delta_values: list[float] | Tensor,
    device: str = "cuda",
) -> list[Tensor]:
    """Generate one Kolmogorov phase screen per propagation plane.

    Parameters
    ----------
    r0_values : sequence of float
        Fried parameter for each plane [m].
    N : int
        Grid side length (pixels).
    delta_values : sequence of float
        Grid spacing for each plane [m].
    device : str
        Torch device (default ``'cuda'``).

    Returns
    -------
    screens : list[Tensor]
        List of phase screens, each shape (N, N), dtype float64.
    """
    screens: list[Tensor] = []
    for r0, delta in zip(r0_values, delta_values):
        r0_val = float(r0)
        delta_val = float(delta)
        phz = ft_sh_phase_screen(r0_val, N, delta_val, device=device)
        screens.append(phz)
    return screens
