"""Alias-safe same-window propagation for kim2026 D2NN paths."""

from __future__ import annotations

import torch

from kim2026.optics.angular_spectrum import propagate_same_window

MIN_PAD_FACTOR = 2
MAX_ALIAS_SAFE_DISTANCE_M = 0.05


def _validate_padded_same_window(
    field: torch.Tensor,
    *,
    wavelength_m: float,
    window_m: float,
    z_m: float,
    pad_factor: int,
    max_distance_m: float | None,
) -> tuple[int, int]:
    if field.ndim < 2 or field.shape[-1] != field.shape[-2]:
        raise ValueError("field must be square in the last two dimensions")
    if wavelength_m <= 0.0:
        raise ValueError("wavelength_m must be > 0")
    if window_m <= 0.0:
        raise ValueError("window_m must be > 0")
    if z_m < 0.0:
        raise ValueError("z_m must be >= 0")
    if int(pad_factor) < MIN_PAD_FACTOR:
        raise ValueError(f"pad_factor must be >= {MIN_PAD_FACTOR}")
    if max_distance_m is not None and z_m > float(max_distance_m):
        raise ValueError(
            f"z_m={z_m:.6f} exceeds kim2026 alias-safe guard "
            f"({float(max_distance_m):.6f} m); reduce spacing or increase physical window"
        )

    n = int(field.shape[-1])
    n_pad = n * int(pad_factor)
    return n, n_pad


def propagate_padded_same_window(
    field: torch.Tensor,
    *,
    wavelength_m: float,
    window_m: float,
    z_m: float,
    pad_factor: int = MIN_PAD_FACTOR,
    max_distance_m: float | None = MAX_ALIAS_SAFE_DISTANCE_M,
) -> torch.Tensor:
    """Approximate linear-convolution free-space propagation by zero-padding and crop."""
    n, n_pad = _validate_padded_same_window(
        field,
        wavelength_m=wavelength_m,
        window_m=window_m,
        z_m=z_m,
        pad_factor=pad_factor,
        max_distance_m=max_distance_m,
    )
    if z_m == 0.0:
        return field.to(torch.complex64)

    pad_total = n_pad - n
    pad_lo = pad_total // 2
    pad_hi = pad_total - pad_lo
    padded = torch.nn.functional.pad(field, (pad_lo, pad_hi, pad_lo, pad_hi))
    propagated = propagate_same_window(
        padded,
        wavelength_m=wavelength_m,
        window_m=float(window_m) * int(pad_factor),
        z_m=z_m,
    )
    return propagated[..., pad_lo : pad_lo + n, pad_lo : pad_lo + n].to(torch.complex64)
