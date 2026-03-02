"""Unit conversion helpers."""

from __future__ import annotations


def um_to_m(x_um: float) -> float:
    """Micrometer to meter."""

    return float(x_um) * 1e-6


def mm_to_m(x_mm: float) -> float:
    """Millimeter to meter."""

    return float(x_mm) * 1e-3


def fourier_plane_pitch(
    *,
    dx_in_m: float,
    wavelength_m: float,
    f_m: float,
    N: int,
) -> float:
    """Return sampling pitch at the Fourier plane of an ideal 2f transform."""

    return float(wavelength_m) * float(f_m) / (float(N) * float(dx_in_m))


def image_plane_pitch_from_fourier(
    *,
    dx_fourier_m: float,
    wavelength_m: float,
    f_m: float,
    N: int,
) -> float:
    """Return sampling pitch at the image plane from Fourier-plane sampling."""

    return float(wavelength_m) * float(f_m) / (float(N) * float(dx_fourier_m))
