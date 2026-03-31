"""Target generation helpers."""

from __future__ import annotations

import torch

from kim2026.optics import MIN_PAD_FACTOR, propagate_padded_same_window
from kim2026.optics.aperture import circular_aperture


def apply_receiver_aperture(field: torch.Tensor, *, receiver_window_m: float, aperture_diameter_m: float) -> torch.Tensor:
    """Apply the receiver aperture mask while preserving the input rank."""
    n = field.shape[-1]
    mask = circular_aperture(
        n=n,
        window_m=receiver_window_m,
        diameter_m=aperture_diameter_m,
        device=field.device,
    ).to(field.dtype)
    mask_shape = (1,) * max(field.ndim - 2, 0) + tuple(mask.shape)
    return field * mask.reshape(mask_shape)


def center_crop_field(field: torch.Tensor, *, crop_n: int) -> torch.Tensor:
    """Center-crop square complex fields to the requested spatial size."""
    n = int(field.shape[-1])
    crop_n = int(crop_n)
    if field.shape[-2] != field.shape[-1]:
        raise ValueError("field must be square in the last two dimensions")
    if crop_n <= 0 or crop_n > n:
        raise ValueError("crop_n must be in the range [1, n]")
    if crop_n == n:
        return field
    start = (n - crop_n) // 2
    end = start + crop_n
    return field[..., start:end, start:end]


def make_detector_plane_target(
    vacuum_field: torch.Tensor,
    *,
    wavelength_m: float,
    receiver_window_m: float,
    aperture_diameter_m: float,
    total_distance_m: float,
    complex_mode: bool = False,
    propagation_pad_factor: int = MIN_PAD_FACTOR,
) -> torch.Tensor:
    """Propagate the aperture-limited vacuum field to the detector plane.

    If complex_mode=True, returns the complex field directly.
    If complex_mode=False (default), returns intensity |field|^2.
    """
    apertured = apply_receiver_aperture(
        vacuum_field,
        receiver_window_m=receiver_window_m,
        aperture_diameter_m=aperture_diameter_m,
    )
    propagated = propagate_padded_same_window(
        apertured,
        wavelength_m=wavelength_m,
        window_m=receiver_window_m,
        z_m=total_distance_m,
        pad_factor=propagation_pad_factor,
        max_distance_m=None,
    )
    if complex_mode:
        return propagated
    return propagated.abs().square()
