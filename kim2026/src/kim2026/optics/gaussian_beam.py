"""Gaussian beam source helpers."""

from __future__ import annotations

import math

import torch


def coordinate_axis(n: int, window_m: float, *, device: torch.device | str | None = None) -> torch.Tensor:
    """Return a centered 1-D coordinate axis in meters."""
    dx = float(window_m) / float(n)
    return (torch.arange(n, dtype=torch.float64, device=device) - (n // 2)) * dx


def gaussian_waist_from_half_angle(wavelength_m: float, half_angle_rad: float, m2: float = 1.0) -> float:
    """Return the 1/e^2 intensity waist from far-field half-angle divergence."""
    return float(m2) * float(wavelength_m) / (math.pi * float(half_angle_rad))


def gaussian_rayleigh_range(wavelength_m: float, half_angle_rad: float, m2: float = 1.0) -> float:
    """Return Rayleigh range for the derived Gaussian waist."""
    waist = gaussian_waist_from_half_angle(wavelength_m, half_angle_rad, m2=m2)
    return math.pi * waist * waist / (float(m2) * float(wavelength_m))


def gaussian_radius_at_distance(wavelength_m: float, half_angle_rad: float, z_m: float, m2: float = 1.0) -> float:
    """Return the 1/e^2 intensity radius after vacuum propagation."""
    waist = gaussian_waist_from_half_angle(wavelength_m, half_angle_rad, m2=m2)
    z_r = gaussian_rayleigh_range(wavelength_m, half_angle_rad, m2=m2)
    return waist * math.sqrt(1.0 + (float(z_m) / z_r) ** 2)


def make_collimated_gaussian_field(
    *,
    n: int,
    window_m: float,
    wavelength_m: float,
    half_angle_rad: float,
    m2: float = 1.0,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a centered complex Gaussian field with flat phase."""
    axis = coordinate_axis(n, window_m, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    waist = gaussian_waist_from_half_angle(wavelength_m, half_angle_rad, m2=m2)
    amplitude = torch.exp(-(xx.square() + yy.square()) / (waist * waist))
    field = torch.complex(amplitude.to(torch.float32), torch.zeros_like(amplitude, dtype=torch.float32))
    return field, axis, axis
