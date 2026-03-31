"""Ideal dual-2f lens operations for Fourier-space D2NNs."""

from __future__ import annotations

import torch

from kim2026.optics.fft2c import fft2c, ifft2c


def fourier_plane_pitch(*, dx_in_m: float, wavelength_m: float, f_m: float, n: int) -> float:
    """Return the sampling pitch at the Fourier plane of an ideal 2f transform."""
    return float(wavelength_m) * float(f_m) / (float(n) * float(dx_in_m))


def image_plane_pitch_from_fourier(*, dx_fourier_m: float, wavelength_m: float, f_m: float, n: int) -> float:
    """Return the image-plane pitch recovered from an ideal inverse 2f transform."""
    return float(wavelength_m) * float(f_m) / (float(n) * float(dx_fourier_m))


def _na_mask(
    *,
    n: int,
    dx_m: float,
    wavelength_m: float,
    na: float | None,
    device: torch.device,
) -> torch.Tensor | None:
    if na is None:
        return None
    freq_axis = torch.fft.fftshift(
        torch.fft.fftfreq(n, d=float(dx_m), device=device, dtype=torch.float64)
    )
    fy, fx = torch.meshgrid(freq_axis, freq_axis, indexing="ij")
    cutoff = float(na) / float(wavelength_m)
    return ((fx.square() + fy.square()) <= cutoff**2).to(torch.float32)


def _validate_common(*, field: torch.Tensor, wavelength_m: float, f_m: float) -> int:
    if field.ndim < 2 or field.shape[-1] != field.shape[-2]:
        raise ValueError("field must be square in the last two dimensions")
    if float(wavelength_m) <= 0.0:
        raise ValueError("wavelength_m must be > 0")
    if float(f_m) <= 0.0:
        raise ValueError("f_m must be > 0")
    return int(field.shape[-1])


def lens_2f_forward(
    field: torch.Tensor,
    *,
    dx_in_m: float,
    wavelength_m: float,
    f_m: float,
    na: float | None,
    apply_scaling: bool,
) -> tuple[torch.Tensor, float]:
    """Ideal 2f forward transform into the Fourier plane."""
    n = _validate_common(field=field, wavelength_m=wavelength_m, f_m=f_m)
    if float(dx_in_m) <= 0.0:
        raise ValueError("dx_in_m must be > 0")

    out = fft2c(field.to(torch.complex64))
    mask = _na_mask(
        n=n,
        dx_m=float(dx_in_m),
        wavelength_m=float(wavelength_m),
        na=na,
        device=out.device,
    )
    if mask is not None:
        out = out * mask.to(out.dtype)

    dx_fourier_m = fourier_plane_pitch(
        dx_in_m=float(dx_in_m),
        wavelength_m=float(wavelength_m),
        f_m=float(f_m),
        n=n,
    )
    if apply_scaling:
        out = out * (float(dx_in_m) / max(float(dx_fourier_m), 1e-30))
    return out, float(dx_fourier_m)


def lens_2f_inverse(
    field: torch.Tensor,
    *,
    dx_fourier_m: float,
    wavelength_m: float,
    f_m: float,
    na: float | None,
    apply_scaling: bool,
) -> tuple[torch.Tensor, float]:
    """Ideal 2f inverse transform back to the image plane."""
    n = _validate_common(field=field, wavelength_m=wavelength_m, f_m=f_m)
    if float(dx_fourier_m) <= 0.0:
        raise ValueError("dx_fourier_m must be > 0")

    inp = field.to(torch.complex64)
    mask = _na_mask(
        n=n,
        dx_m=float(dx_fourier_m),
        wavelength_m=float(wavelength_m),
        na=na,
        device=inp.device,
    )
    if mask is not None:
        inp = inp * mask.to(inp.dtype)

    out = ifft2c(inp)
    dx_out_m = image_plane_pitch_from_fourier(
        dx_fourier_m=float(dx_fourier_m),
        wavelength_m=float(wavelength_m),
        f_m=float(f_m),
        n=n,
    )
    if apply_scaling:
        out = out * (float(dx_fourier_m) / max(float(dx_out_m), 1e-30))
    return out, float(dx_out_m)
