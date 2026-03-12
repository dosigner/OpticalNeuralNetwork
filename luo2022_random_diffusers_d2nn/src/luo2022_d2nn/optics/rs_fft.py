"""Rayleigh-Sommerfeld FFT propagator (oracle / reference).

Implements the RS kernel via FFT convolution:
    w(x,y,z) = (z/r²) * (1/(2πr) + 1/(jλ)) * exp(j 2π r / λ)
    where r = sqrt(x² + y² + z²)

All spatial units in **mm**.
"""

from __future__ import annotations

import math

import torch

from luo2022_d2nn.optics.grids import make_spatial_grid


def rs_kernel(
    N: int,
    dx_mm: float,
    wavelength_mm: float,
    z_mm: float,
    *,
    pad_factor: int = 2,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build the Rayleigh-Sommerfeld impulse-response kernel.

    Parameters
    ----------
    N : int
        Side length of the *unpadded* field.
    dx_mm, wavelength_mm, z_mm : float
        Physical parameters in mm.
    pad_factor : int
        Kernel lives on an (N*pad_factor × N*pad_factor) grid.

    Returns
    -------
    kernel : complex64 tensor, shape (N_pad, N_pad).
    """
    N_pad = N * pad_factor
    x, y = make_spatial_grid(N_pad, dx_mm, device=device, dtype=torch.float64)

    r_sq = x ** 2 + y ** 2 + z_mm ** 2
    r = torch.sqrt(r_sq)

    # Avoid r=0 singularity (center pixel) — clamp to tiny value
    r_safe = r.clamp(min=1e-12)
    r_sq_safe = r_sq.clamp(min=1e-24)

    k = 2.0 * math.pi / wavelength_mm

    # w = (z/r²) * (1/(2πr) + 1/(jλ)) * exp(j k r)
    # Note: 1/(jλ) = -j/λ
    factor1 = z_mm / r_sq_safe  # z / r²
    factor2_real = 1.0 / (2.0 * math.pi * r_safe)  # 1/(2πr)
    factor2_imag = -1.0 / wavelength_mm  # Im part from 1/(jλ) = -j/λ

    phase = torch.exp(1j * k * r_safe)

    # Combine: (factor2_real + j*factor2_imag)
    factor2 = torch.complex(factor2_real, torch.full_like(factor2_real, factor2_imag))

    kernel = (factor1.to(torch.complex128) * factor2 * phase).to(torch.complex64)
    return kernel


def rs_propagate(
    field: torch.Tensor,
    wavelength_mm: float,
    dx_mm: float,
    z_mm: float,
    *,
    pad_factor: int = 2,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Propagate *field* using the RS FFT convolution.

    Parameters
    ----------
    field : Tensor, shape (..., N, N)
    wavelength_mm, dx_mm, z_mm : float
    pad_factor : int
    device : optional device override

    Returns
    -------
    Tensor, shape (..., N, N) — propagated field.
    """
    dev = device or field.device
    N = field.shape[-1]
    N_pad = N * pad_factor

    kernel = rs_kernel(N, dx_mm, wavelength_mm, z_mm,
                       pad_factor=pad_factor, device=dev)

    # Zero-pad the input field
    pad_lo = (N_pad - N) // 2
    pad_hi = N_pad - N - pad_lo
    padded = torch.nn.functional.pad(
        field, (pad_lo, pad_hi, pad_lo, pad_hi), mode="constant", value=0
    )

    # FFT convolution
    F_field = torch.fft.fft2(padded)
    F_kernel = torch.fft.fft2(torch.fft.ifftshift(kernel, dim=(-2, -1)))
    conv = torch.fft.ifft2(F_field * F_kernel)

    # Center-crop and multiply by dx² for integral approximation
    out = conv[..., pad_lo: pad_lo + N, pad_lo: pad_lo + N]
    out = out * (dx_mm ** 2)
    return out
