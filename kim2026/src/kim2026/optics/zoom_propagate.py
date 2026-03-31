"""Zoom propagation wrapper for arbitrary-window Fresnel propagation."""

from __future__ import annotations

import math

import torch

from kim2026.optics.scaled_fresnel import scaled_fresnel_propagate


def zoom_propagate(
    field: torch.Tensor,
    *,
    wavelength_m: float,
    source_window_m: float,
    destination_window_m: float,
    z_m: float,
) -> torch.Tensor:
    """Propagate between unequal windows using the scaled Fresnel kernel."""
    if field.ndim < 2 or field.shape[-1] != field.shape[-2]:
        raise ValueError("field must be square in the last two dimensions")
    if wavelength_m <= 0.0:
        raise ValueError("wavelength_m must be > 0")
    if source_window_m <= 0.0:
        raise ValueError("source_window_m must be > 0")
    if destination_window_m <= 0.0:
        raise ValueError("destination_window_m must be > 0")
    if z_m <= 0.0:
        raise ValueError("z_m must be > 0")
    if math.isclose(
        float(source_window_m),
        float(destination_window_m),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "source_window_m and destination_window_m are effectively equal; "
            "use propagate_same_window instead"
        )
    return scaled_fresnel_propagate(
        field,
        wavelength_m=wavelength_m,
        source_window_m=source_window_m,
        destination_window_m=destination_window_m,
        z_m=z_m,
    )
