"""Fresnel thin-lens transmission function."""

from __future__ import annotations

import math

import torch

from luo2022_d2nn.optics.grids import make_spatial_grid


def fresnel_lens_transmission(
    N: int,
    dx_mm: float,
    wavelength_mm: float,
    focal_length_mm: float,
    pupil_radius_mm: float,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Fresnel thin-lens transmission function.

    t_L(x,y) = A(x,y) * exp(-j pi (x^2 + y^2) / (lambda * f))

    where A(x,y) = 1 inside the pupil (circular aperture of given radius),
    and A(x,y) = 0 outside.

    Parameters
    ----------
    N : int
        Grid side length (pixels).
    dx_mm : float
        Pixel pitch in mm.
    wavelength_mm : float
        Operating wavelength in mm.
    focal_length_mm : float
        Focal length in mm.
    pupil_radius_mm : float
        Pupil radius in mm.
    device : torch device, optional

    Returns
    -------
    Tensor, shape (N, N), complex64 — lens transmittance.
    """
    x, y = make_spatial_grid(N, dx_mm, device=device, dtype=torch.float64)
    r_sq = x ** 2 + y ** 2

    # Circular pupil mask
    pupil = (r_sq <= pupil_radius_mm ** 2)

    # Quadratic phase: -pi * r^2 / (lambda * f)
    phase = -math.pi * r_sq / (wavelength_mm * focal_length_mm)

    t = torch.zeros(N, N, dtype=torch.complex64, device=device)
    t[pupil] = torch.exp(1j * phase[pupil]).to(torch.complex64)

    return t
