"""Ideal 2f lens operations."""

from __future__ import annotations

import torch

from tao2019_fd2nn.optics.aperture import na_mask
from tao2019_fd2nn.optics.fft2c import fft2c, ifft2c
from tao2019_fd2nn.optics.scaling import fourier_plane_pitch, image_plane_pitch_from_fourier


def _apply_na(
    field: torch.Tensor,
    *,
    dx_m: float,
    wavelength_m: float,
    na: float | None,
) -> torch.Tensor:
    if na is None:
        return field
    N = int(field.shape[-1])
    mask = na_mask(
        N=N,
        dx_m=float(dx_m),
        wavelength_m=float(wavelength_m),
        na=float(na),
        shifted=True,
        device=field.device,
    ).to(field.dtype)
    return field * mask


def lens_2f_forward(
    field: torch.Tensor,
    *,
    dx_in_m: float,
    wavelength_m: float,
    f_m: float,
    na: float | None,
    apply_scaling: bool,
) -> tuple[torch.Tensor, float]:
    """Ideal 2f Fourier transform with physical sampling update."""

    out = fft2c(field)
    out = _apply_na(out, dx_m=float(dx_in_m), wavelength_m=float(wavelength_m), na=na)
    dx_fourier_m = fourier_plane_pitch(
        dx_in_m=float(dx_in_m),
        wavelength_m=float(wavelength_m),
        f_m=float(f_m),
        N=int(field.shape[-1]),
    )
    if apply_scaling:
        scale = float(dx_in_m) / max(float(dx_fourier_m), 1e-30)
        out = out * scale
    return out, dx_fourier_m


def lens_2f_inverse(
    field: torch.Tensor,
    *,
    dx_fourier_m: float,
    wavelength_m: float,
    f_m: float,
    na: float | None,
    apply_scaling: bool,
) -> tuple[torch.Tensor, float]:
    """Ideal 2f inverse Fourier transform with physical sampling update."""

    inp = _apply_na(field, dx_m=float(dx_fourier_m), wavelength_m=float(wavelength_m), na=na)
    out = ifft2c(inp)
    dx_out_m = image_plane_pitch_from_fourier(
        dx_fourier_m=float(dx_fourier_m),
        wavelength_m=float(wavelength_m),
        f_m=float(f_m),
        N=int(field.shape[-1]),
    )
    if apply_scaling:
        scale = float(dx_fourier_m) / max(float(dx_out_m), 1e-30)
        out = out * scale
    return out, dx_out_m
