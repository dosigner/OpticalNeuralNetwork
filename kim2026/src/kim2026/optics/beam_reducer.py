"""Ideal and reference beam-reducer operators."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from kim2026.optics.aperture import circular_aperture
from kim2026.optics.scaled_fresnel import scaled_fresnel_propagate

_LANCZOS_CACHE: dict[tuple[int, int, int, str], torch.Tensor] = {}


@dataclass(frozen=True)
class BeamReducerPlane:
    """Physical geometry for a beam reducer plane."""

    window_m: float
    n: int
    aperture_diameter_m: float

    @property
    def dx_m(self) -> float:
        return float(self.window_m) / float(self.n)


def _validate_square(field: torch.Tensor) -> int:
    if field.ndim < 2 or field.shape[-1] != field.shape[-2]:
        raise ValueError("field must be square in the last two dimensions")
    return int(field.shape[-1])


def _crop_aperture(
    field: torch.Tensor,
    *,
    input_plane: BeamReducerPlane,
) -> torch.Tensor:
    n = _validate_square(field)
    if n != int(input_plane.n):
        raise ValueError(f"field grid {n} does not match input_plane.n={input_plane.n}")
    mask = circular_aperture(
        n=n,
        window_m=input_plane.window_m,
        diameter_m=input_plane.aperture_diameter_m,
        device=field.device,
    )
    apertured = field.to(torch.complex64) * mask.to(field.dtype if torch.is_complex(field) else torch.complex64)
    crop_n = int(round(float(input_plane.aperture_diameter_m) / float(input_plane.window_m) * n))
    crop_n = max(2, min(crop_n, n))
    if crop_n % 2 != 0:
        crop_n += 1 if crop_n < n else -1
    center = n // 2
    half = crop_n // 2
    return apertured[..., center - half : center + half, center - half : center + half]


def _sinc(x: torch.Tensor) -> torch.Tensor:
    pix = math.pi * x
    return torch.where(torch.abs(x) < 1.0e-8, torch.ones_like(x), torch.sin(pix) / pix)


def _lanczos_matrix(*, positions: torch.Tensor, in_n: int, a: int) -> torch.Tensor:
    source_idx = torch.arange(in_n, device=positions.device, dtype=torch.float32)
    delta = positions.unsqueeze(1) - source_idx.unsqueeze(0)
    weights = torch.where(
        torch.abs(delta) < float(a),
        _sinc(delta) * _sinc(delta / float(a)),
        torch.zeros_like(delta),
    )
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-12)


def _uniform_lanczos_matrix(*, in_n: int, out_n: int, a: int, device: torch.device) -> torch.Tensor:
    key = (int(in_n), int(out_n), int(a), str(device))
    cached = _LANCZOS_CACHE.get(key)
    if cached is not None:
        return cached
    if in_n == out_n:
        matrix = torch.eye(in_n, device=device, dtype=torch.float32)
        _LANCZOS_CACHE[key] = matrix
        return matrix

    scale = float(in_n) / float(out_n)
    positions = (torch.arange(out_n, device=device, dtype=torch.float32) + 0.5) * scale - 0.5
    weights = _lanczos_matrix(positions=positions, in_n=in_n, a=a)
    _LANCZOS_CACHE[key] = weights
    return weights


def _lanczos_resample_square(field: torch.Tensor, *, out_n: int, support: int) -> torch.Tensor:
    in_n = int(field.shape[-1])
    weights = _uniform_lanczos_matrix(in_n=in_n, out_n=out_n, a=support, device=field.device)
    work = field.to(torch.complex64)
    tmp = torch.matmul(weights.to(work.dtype), work)
    return torch.matmul(tmp, weights.t().to(work.dtype))


def _plane_resample_matrix(
    *,
    input_plane: BeamReducerPlane,
    output_plane: BeamReducerPlane,
    magnification_inverse: float,
    support: int,
    device: torch.device,
) -> torch.Tensor:
    n_in = int(input_plane.n)
    n_out = int(output_plane.n)
    x_out = (torch.arange(n_out, device=device, dtype=torch.float32) - n_out / 2 + 0.5) * float(output_plane.dx_m)
    x_in = x_out * float(magnification_inverse)
    positions = x_in / float(input_plane.dx_m) + n_in / 2 - 0.5
    return _lanczos_matrix(positions=positions, in_n=n_in, a=support)


def _lens_phase(
    *,
    n: int,
    window_m: float,
    wavelength_m: float,
    f_m: float,
    device: torch.device,
) -> torch.Tensor:
    dx = float(window_m) / float(n)
    axis = (torch.arange(n, device=device, dtype=torch.float32) - n / 2 + 0.5) * dx
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    k = 2.0 * math.pi / float(wavelength_m)
    phase = -(k / (2.0 * float(f_m))) * (xx.square() + yy.square())
    return torch.exp(1j * phase).to(torch.complex64)


def apply_beam_reducer_bilinear_legacy(
    field: torch.Tensor,
    *,
    input_window_m: float,
    aperture_diameter_m: float,
    output_window_m: float,
) -> torch.Tensor:
    """Legacy real/imag bilinear reducer kept for regression comparisons."""
    n = _validate_square(field)
    magnification = aperture_diameter_m / output_window_m

    mask = circular_aperture(
        n=n,
        window_m=input_window_m,
        diameter_m=aperture_diameter_m,
        device=field.device,
    )
    apertured = field * mask
    aperture_fraction = aperture_diameter_m / input_window_m
    aperture_pixels = int(round(aperture_fraction * n))
    if aperture_pixels % 2 != 0:
        aperture_pixels += 1
    aperture_pixels = min(aperture_pixels, n)

    half = aperture_pixels // 2
    center = n // 2
    start = center - half
    end = start + aperture_pixels
    cropped = apertured[..., start:end, start:end]
    if cropped.shape[-1] != n:
        original_shape = cropped.shape[:-2]
        flat_real = cropped.real.reshape(-1, 1, aperture_pixels, aperture_pixels)
        flat_imag = cropped.imag.reshape(-1, 1, aperture_pixels, aperture_pixels)
        resampled_real = F.interpolate(flat_real, size=(n, n), mode="bilinear", align_corners=True)
        resampled_imag = F.interpolate(flat_imag, size=(n, n), mode="bilinear", align_corners=True)
        resampled = torch.complex(resampled_real, resampled_imag).reshape(*original_shape, n, n)
    else:
        resampled = cropped
    return resampled * magnification


def apply_beam_reducer(
    field: torch.Tensor,
    *,
    input_plane: BeamReducerPlane,
    output_plane: BeamReducerPlane,
    pad_factor: int = 2,
) -> torch.Tensor:
    """Apply the alias-safe ideal reducer by complex windowed-sinc resampling."""
    n = _validate_square(field)
    if n != int(input_plane.n):
        raise ValueError(f"field grid {n} does not match input_plane.n={input_plane.n}")
    mask = circular_aperture(
        n=n,
        window_m=input_plane.window_m,
        diameter_m=input_plane.aperture_diameter_m,
        device=field.device,
    ).to(torch.complex64)
    apertured = field.to(torch.complex64) * mask
    support = max(4, 2 * int(pad_factor))
    m = float(output_plane.window_m) / float(input_plane.aperture_diameter_m)
    weights = _plane_resample_matrix(
        input_plane=input_plane,
        output_plane=output_plane,
        magnification_inverse=1.0 / m,
        support=support,
        device=field.device,
    )
    tmp = torch.matmul(weights.to(apertured.dtype), apertured)
    resampled = torch.matmul(tmp, weights.t().to(apertured.dtype))
    magnification = float(input_plane.aperture_diameter_m) / float(output_plane.window_m)
    return resampled * magnification


def apply_physical_beam_reducer_reference(
    field: torch.Tensor,
    *,
    input_plane: BeamReducerPlane,
    output_plane: BeamReducerPlane,
    wavelength_m: float = 1.55e-6,
    f1_m: float = 75.0e-3,
    f2_m: float = 1.0e-3,
) -> torch.Tensor:
    """Reference thin-lens afocal relay using scaled Fresnel propagation."""
    n = _validate_square(field)
    if n != int(input_plane.n):
        raise ValueError(f"field grid {n} does not match input_plane.n={input_plane.n}")
    entrance = field.to(torch.complex64) * circular_aperture(
        n=n,
        window_m=input_plane.window_m,
        diameter_m=input_plane.aperture_diameter_m,
        device=field.device,
    ).to(torch.complex64)
    fourier_window_m = float(wavelength_m) * float(f1_m) * float(n) / float(input_plane.aperture_diameter_m)
    lens1 = entrance * _lens_phase(
        n=n,
        window_m=input_plane.window_m,
        wavelength_m=wavelength_m,
        f_m=f1_m,
        device=field.device,
    )
    fourier = scaled_fresnel_propagate(
        lens1,
        wavelength_m=wavelength_m,
        source_window_m=input_plane.window_m,
        destination_window_m=fourier_window_m,
        z_m=f1_m,
    )
    lens2_plane = scaled_fresnel_propagate(
        fourier,
        wavelength_m=wavelength_m,
        source_window_m=fourier_window_m,
        destination_window_m=output_plane.window_m,
        z_m=f2_m,
    )
    lens2_plane = lens2_plane * circular_aperture(
        n=n,
        window_m=output_plane.window_m,
        diameter_m=output_plane.aperture_diameter_m,
        device=field.device,
    ).to(torch.complex64)
    lens2 = lens2_plane * _lens_phase(
        n=n,
        window_m=output_plane.window_m,
        wavelength_m=wavelength_m,
        f_m=f2_m,
        device=field.device,
    )
    return scaled_fresnel_propagate(
        lens2,
        wavelength_m=wavelength_m,
        source_window_m=output_plane.window_m,
        destination_window_m=output_plane.window_m,
        z_m=f2_m,
    )
