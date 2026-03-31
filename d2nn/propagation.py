"""
propagation.py
==============
Optical wave-propagation kernels used inside diffractive layers.

Two propagation models are provided:

1. **Angular Spectrum Method (ASM)** – exact free-space propagation for a
   monochromatic field over a distance *z* using the transfer function

       H(fx, fy) = exp(j 2π z √(1/λ² − fx² − fy²))

   Evanescent components (fx² + fy² > 1/λ²) are suppressed.

2. **Fourier-lens propagation** – models a thin ideal lens placed at the
   input plane, which maps the field to its 2-D Fourier transform at the
   back focal plane.  A pair of such operations forms the standard 4-f
   Fourier-plane processing architecture used in Fourier D²NN.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def angular_spectrum_propagation(
    field: torch.Tensor,
    wavelength: float,
    z: float,
    dx: float,
    dy: float | None = None,
) -> torch.Tensor:
    """Propagate a complex field using the Angular Spectrum Method.

    Parameters
    ----------
    field:
        Complex-valued tensor of shape ``(..., H, W)``.
    wavelength:
        Wavelength of the monochromatic light source (metres).
    z:
        Propagation distance (metres).  Use negative values to propagate
        backwards.
    dx:
        Spatial sampling interval in the *x* (column) direction (metres).
    dy:
        Spatial sampling interval in the *y* (row) direction (metres).
        Defaults to *dx* when ``None``.

    Returns
    -------
    torch.Tensor
        Propagated field, same shape and dtype as *field*.
    """
    if dy is None:
        dy = dx

    *batch, H, W = field.shape

    # Spatial-frequency coordinates (cycles per metre)
    fx = torch.fft.fftfreq(W, d=dx, device=field.device, dtype=field.real.dtype)
    fy = torch.fft.fftfreq(H, d=dy, device=field.device, dtype=field.real.dtype)
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")  # (H, W)

    # Transfer function H(fx, fy)
    k = 1.0 / wavelength  # wavenumber magnitude
    sq = k**2 - FX**2 - FY**2
    # Propagating components only (evanescent waves are zeroed out)
    propagating = sq >= 0.0
    phase = torch.where(
        propagating,
        2.0 * torch.pi * z * torch.sqrt(sq.clamp(min=0.0)),
        torch.zeros_like(sq),
    )
    H_tf = torch.polar(propagating.to(field.real.dtype), phase)  # (H, W) complex

    # Apply transfer function in frequency domain
    spectrum = torch.fft.fft2(field)
    propagated = torch.fft.ifft2(spectrum * H_tf)
    return propagated


def fourier_lens_propagation(
    field: torch.Tensor,
    forward: bool = True,
) -> torch.Tensor:
    """Apply a 2-D Fourier lens transform (ideal thin-lens).

    A thin ideal lens placed at the input plane converts the incoming field
    into its 2-D Fourier transform (up to a quadratic phase factor that is
    omitted here for the neural-network use case where only the field
    magnitude / intensity at a detector matters).

    This is the building block of the Fourier D²NN 4-f architecture.

    Parameters
    ----------
    field:
        Complex-valued tensor of shape ``(..., H, W)``.
    forward:
        When ``True`` (default) performs ``fft2``; when ``False`` performs
        ``ifft2``.

    Returns
    -------
    torch.Tensor
        Transformed field, same shape as *field*.
    """
    if forward:
        return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(field, dim=(-2, -1))), dim=(-2, -1))
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(field, dim=(-2, -1))), dim=(-2, -1))
