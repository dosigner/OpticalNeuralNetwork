"""Band-Limited Angular Spectrum Method propagator.

Implements H(fx,fy;z) = exp(j 2π z sqrt(1/λ² - fx² - fy²)) with
evanescent-wave masking for physically correct free-space propagation.
All spatial units in **mm**.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

from luo2022_d2nn.optics.grids import make_frequency_grid

# ---------------------------------------------------------------------------
# Transfer-function cache  (key → H tensor)
# ---------------------------------------------------------------------------
_tf_cache: Dict[Tuple, torch.Tensor] = {}


def clear_transfer_cache() -> None:
    """Clear the cached transfer functions."""
    _tf_cache.clear()


# ---------------------------------------------------------------------------
# Transfer function
# ---------------------------------------------------------------------------

def bl_asm_transfer_function(
    N: int,
    dx_mm: float,
    wavelength_mm: float,
    z_mm: float,
    *,
    pad_factor: int = 2,
    evanescent: str = "mask",
    device: torch.device | str | None = None,
    dtype: str = "complex64",
) -> torch.Tensor:
    """Compute (and cache) the BL-ASM transfer function.

    Parameters
    ----------
    N : int
        Side length of the *unpadded* field.
    dx_mm : float
        Pixel pitch in mm.
    wavelength_mm : float
        Wavelength in mm.
    z_mm : float
        Propagation distance in mm.
    pad_factor : int
        Padding multiplier (padded size = N * pad_factor).
    evanescent : str
        How to handle evanescent waves. ``"mask"`` zeros them out.
    device, dtype
        Torch device / complex dtype string.

    Returns
    -------
    H : Tensor of shape (N*pad_factor, N*pad_factor), complex.
    """
    dev_str = str(device) if device is not None else "cpu"
    key = (N, dx_mm, wavelength_mm, z_mm, pad_factor, evanescent, dev_str, dtype)
    if key in _tf_cache:
        return _tf_cache[key]

    N_pad = N * pad_factor
    cdtype = getattr(torch, dtype)

    # Frequency grid on padded size
    fx, fy = make_frequency_grid(N_pad, dx_mm, device=device, dtype=torch.float64)
    f_sq = fx ** 2 + fy ** 2  # (cycles/mm)²
    cutoff_sq = 1.0 / (wavelength_mm ** 2)

    propagating = f_sq < cutoff_sq  # strict < to avoid sqrt(0) edge

    # Phase argument: 2π z sqrt(1/λ² - fx² - fy²)
    arg = torch.zeros_like(f_sq)
    arg[propagating] = (2.0 * math.pi * z_mm) * torch.sqrt(
        cutoff_sq - f_sq[propagating]
    )

    H = torch.zeros(N_pad, N_pad, dtype=cdtype, device=device)
    H[propagating] = torch.exp(1j * arg[propagating]).to(cdtype)

    _tf_cache[key] = H
    return H


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------

def bl_asm_propagate(
    field: torch.Tensor,
    H: torch.Tensor,
    *,
    pad_factor: int = 2,
) -> torch.Tensor:
    """Propagate *field* using a precomputed transfer function *H*.

    Steps: zero-pad → FFT → multiply H → IFFT → center-crop.

    Parameters
    ----------
    field : Tensor, shape (..., N, N)
    H : Tensor, shape (N_pad, N_pad)
    pad_factor : int

    Returns
    -------
    Tensor, shape (..., N, N) — propagated field.
    """
    N = field.shape[-1]
    N_pad = N * pad_factor

    # Zero-pad
    pad_lo = (N_pad - N) // 2
    pad_hi = N_pad - N - pad_lo
    padded = torch.nn.functional.pad(
        field, (pad_lo, pad_hi, pad_lo, pad_hi), mode="constant", value=0
    )

    # FFT → multiply → IFFT  (non-centered; H lives in fftfreq order)
    spectrum = torch.fft.fft2(padded)
    spectrum = spectrum * H
    propagated = torch.fft.ifft2(spectrum)

    # Center-crop
    out = propagated[..., pad_lo: pad_lo + N, pad_lo: pad_lo + N]
    return out
