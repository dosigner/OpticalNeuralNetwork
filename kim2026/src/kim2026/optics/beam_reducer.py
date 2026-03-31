"""Ideal afocal beam reducer (telescope) for spatial field compression.

Models an ideal afocal telescope that compresses a large-aperture field
onto a smaller metalens/D2NN input plane. The telescope preserves phase
structure and conserves energy:

    E_out(x') = M * E_in(M * x')

where M = D_in / D_out (magnification factor).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from kim2026.optics.aperture import circular_aperture


def apply_beam_reducer(
    field: torch.Tensor,
    *,
    input_window_m: float,
    aperture_diameter_m: float,
    output_window_m: float,
) -> torch.Tensor:
    """Apply an ideal beam reducer to a complex field.

    Parameters
    ----------
    field : torch.Tensor
        Complex field on the input grid, shape (..., N, N).
    input_window_m : float
        Physical size of the input grid [m].
    aperture_diameter_m : float
        Diameter of the telescope entrance aperture [m].
    output_window_m : float
        Physical size of the output (metalens) grid [m].

    Returns
    -------
    torch.Tensor
        Beam-reduced complex field on the output grid, same shape as input.
    """
    if field.ndim < 2 or field.shape[-1] != field.shape[-2]:
        raise ValueError("field must be square in the last two dimensions")

    n = field.shape[-1]
    magnification = aperture_diameter_m / output_window_m

    # Step 1: Apply circular aperture on the input plane
    mask = circular_aperture(
        n=n,
        window_m=input_window_m,
        diameter_m=aperture_diameter_m,
        device=field.device,
    )
    apertured = field * mask

    # Step 2: Extract and resample the aperture region to the output grid
    # The aperture spans aperture_diameter_m within input_window_m.
    # We need to crop the central region and resample to N×N.
    aperture_fraction = aperture_diameter_m / input_window_m
    # Number of pixels covering the aperture diameter
    aperture_pixels = int(round(aperture_fraction * n))
    # Ensure even for symmetric cropping
    if aperture_pixels % 2 != 0:
        aperture_pixels += 1
    aperture_pixels = min(aperture_pixels, n)

    half = aperture_pixels // 2
    center = n // 2
    start = center - half
    end = start + aperture_pixels

    cropped = apertured[..., start:end, start:end]

    # Step 3: Resample to output grid (N×N) using bilinear interpolation
    if cropped.shape[-1] != n:
        # F.interpolate needs (B, C, H, W) real tensors
        original_shape = cropped.shape[:-2]
        flat_real = cropped.real.reshape(-1, 1, aperture_pixels, aperture_pixels)
        flat_imag = cropped.imag.reshape(-1, 1, aperture_pixels, aperture_pixels)
        resampled_real = F.interpolate(flat_real, size=(n, n), mode="bilinear", align_corners=True)
        resampled_imag = F.interpolate(flat_imag, size=(n, n), mode="bilinear", align_corners=True)
        resampled = torch.complex(resampled_real, resampled_imag)
        resampled = resampled.reshape(*original_shape, n, n)
    else:
        resampled = cropped

    # Step 4: Scale amplitude for energy conservation (2D)
    resampled = resampled * magnification

    return resampled
