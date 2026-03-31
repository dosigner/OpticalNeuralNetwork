"""Split-step angular-spectrum beam propagation.

Implements the split-step angular-spectrum method from Schmidt,
*Numerical Simulation of Optical Wave Propagation with Examples in MATLAB*
(SPIE Press, 2010), Listing 9.1.

All computation uses complex128 on CUDA for numerical precision.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from kim2026.fso.ft_utils import ft2, ift2


# ---------------------------------------------------------------------------
# Gaussian source
# ---------------------------------------------------------------------------

def make_gaussian_source(
    N: int,
    delta1: float,
    w0: float,
    device: str = "cuda",
) -> Tensor:
    """Create a collimated Gaussian beam at the source plane.

    Parameters
    ----------
    N : int
        Grid size (N x N pixels).
    delta1 : float
        Source-plane grid spacing [m].
    w0 : float
        Beam waist radius (1/e amplitude) [m].
    device : str
        Torch device (default ``'cuda'``).

    Returns
    -------
    U : Tensor, shape (N, N), complex128
        Source field with unit peak amplitude and flat phase
        (collimated beam, radius of curvature R = infinity).
    """
    x1 = torch.arange(-N // 2, N // 2, dtype=torch.float64, device=device) * delta1
    y1 = x1.clone()
    X1, Y1 = torch.meshgrid(x1, y1, indexing="ij")
    r1sq = X1**2 + Y1**2
    U = torch.exp(-r1sq / w0**2).to(torch.complex128)
    return U


# ---------------------------------------------------------------------------
# Split-step angular-spectrum propagation (Listing 9.1)
# ---------------------------------------------------------------------------

def ang_spec_multi_prop(
    Uin: Tensor,
    wvl: float,
    delta1: float,
    deltan: float,
    z_planes: Union[Sequence[float], Tensor],
    phase_screens: Optional[List[Optional[Tensor]]] = None,
    device: str = "cuda",
) -> Tuple[Tensor, Tensor, Tensor]:
    """Split-step angular-spectrum propagation through multiple planes.

    Parameters
    ----------
    Uin : Tensor, shape (N, N), complex128
        Source-plane field.
    wvl : float
        Wavelength [m].
    delta1 : float
        Source-plane grid spacing [m].
    deltan : float
        Observation-plane grid spacing [m].
    z_planes : sequence of float or Tensor, length *n*
        Axial positions of each phase-screen plane [m].
        The first entry is the source plane; the last is the observation plane.
    phase_screens : list of Tensor or None, optional
        Phase screens at each plane.  Each element is either an (N, N)
        float64 tensor giving the phase [rad] or ``None`` (vacuum / identity).
        Pass ``None`` for the entire list to propagate through vacuum.
    device : str
        Torch device (default ``'cuda'``).

    Returns
    -------
    xn : Tensor, shape (N,)
        Observation-plane x coordinates [m].
    yn : Tensor, shape (N,)
        Observation-plane y coordinates [m].
    Uout : Tensor, shape (N, N), complex128
        Field at the observation plane.
    """
    N = Uin.shape[0]
    k = 2.0 * math.pi / wvl

    # Ensure z_planes is a tensor on the correct device
    if not isinstance(z_planes, Tensor):
        z_planes = torch.tensor(z_planes, dtype=torch.float64, device=device)
    else:
        z_planes = z_planes.to(dtype=torch.float64, device=device)

    n = len(z_planes)  # number of planes

    # Partial propagation distances between consecutive planes
    Delta_z = z_planes[1:] - z_planes[:-1]  # [n-1]

    # Fractional positions along the total propagation path
    alpha = z_planes / z_planes[-1]  # [n]

    # Grid spacing at each plane (linear interpolation, Eq. 8.8)
    delta = (1.0 - alpha) * delta1 + alpha * deltan  # [n]

    # Magnification factors between consecutive planes
    m = delta[1:] / delta[:-1]  # [n-1]

    # Integer coordinate grid
    nx = torch.arange(-N // 2, N // 2, dtype=torch.float64, device=device)
    ny = nx.clone()
    NX, NY = torch.meshgrid(nx, ny, indexing="ij")
    nsq = NX**2 + NY**2

    # Super-Gaussian absorbing boundary (Listing 9.1 lines 10-12)
    w = 0.47 * N
    sg = torch.exp(-nsq**8 / w**16).to(torch.complex128)

    # Build transmittance screens ------------------------------------------------
    T: List[Tensor] = []
    if phase_screens is None:
        T = [torch.ones(N, N, dtype=torch.complex128, device=device)] * n
    else:
        for phz in phase_screens:
            if phz is None:
                T.append(torch.ones(N, N, dtype=torch.complex128, device=device))
            else:
                T.append(torch.exp(1j * phz.to(torch.complex128)))

    # Source-plane spatial coordinates
    x1 = nx * delta[0]
    y1 = ny * delta[0]
    X1, Y1 = torch.meshgrid(x1, y1, indexing="ij")
    r1sq = X1**2 + Y1**2

    # Initial quadratic phase Q1 (Listing 9.1 lines 25-26)
    Q1 = torch.exp(1j * k / 2.0 * (1.0 - m[0]) / Delta_z[0] * r1sq)

    # Apply first phase screen and initial quadratic phase
    U = Uin * Q1 * T[0]

    # Iterative propagation (Listing 9.1 lines 27-39) ---------------------------
    for i in range(n - 1):
        # Frequency-domain grid at plane i
        deltaf = 1.0 / (N * delta[i].item())
        fX = nx * deltaf
        fY = ny * deltaf
        FX, FY = torch.meshgrid(fX, fY, indexing="ij")
        fsq = FX**2 + FY**2

        # Frequency-domain quadratic phase Q2
        Q2 = torch.exp(
            -1j * math.pi**2 * 2.0 * Delta_z[i] / (m[i] * k) * fsq
        )

        # Forward FT, multiply Q2, inverse FT
        G = ft2(U / m[i], delta[i].item())
        G = Q2 * G
        U_prop = ift2(G, deltaf)

        # Apply absorbing boundary and next phase screen
        if i < n - 2:
            U = sg * T[i + 1] * U_prop
        else:
            # Last propagation step
            U = sg * T[i + 1] * U_prop

    # Final quadratic phase Q3 (Listing 9.1 lines 44-46) ------------------------
    xn = nx * delta[-1]
    yn = ny * delta[-1]
    Xn, Yn = torch.meshgrid(xn, yn, indexing="ij")
    rnsq = Xn**2 + Yn**2
    Q3 = torch.exp(1j * k / 2.0 * (m[-1] - 1.0) / (m[-1] * Delta_z[-1]) * rnsq)
    Uout = Q3 * U

    return xn, yn, Uout


# ---------------------------------------------------------------------------
# Irradiance
# ---------------------------------------------------------------------------

def compute_irradiance(U: Tensor) -> Tensor:
    """Compute optical irradiance from a complex field.

    Parameters
    ----------
    U : Tensor, complex
        Complex field amplitude.

    Returns
    -------
    I : Tensor, real (float64)
        Irradiance :math:`I = |U|^2`.
    """
    return (U * U.conj()).real
