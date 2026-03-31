"""Optics helpers for kim2026."""

from kim2026.optics.angular_spectrum import propagate_same_window
from kim2026.optics.aperture import circular_aperture
from kim2026.optics.gaussian_beam import (
    gaussian_radius_at_distance,
    gaussian_waist_from_half_angle,
    make_collimated_gaussian_field,
)
from kim2026.optics.propagation_schedule import (
    AdaptiveSchedule,
    PropagationInterval,
    ScreenCell,
    build_adaptive_schedule,
    build_screen_cells,
)
from kim2026.optics.scaled_fresnel import scaled_fresnel_propagate
from kim2026.optics.fft2c import fft2c, ifft2c
from kim2026.optics.lens_2f import (
    fourier_plane_pitch,
    image_plane_pitch_from_fourier,
    lens_2f_forward,
    lens_2f_inverse,
)
from kim2026.optics.padded_angular_spectrum import (
    MAX_ALIAS_SAFE_DISTANCE_M,
    MIN_PAD_FACTOR,
    propagate_padded_same_window,
)
from kim2026.optics.zoom_propagate import zoom_propagate

__all__ = [
    "AdaptiveSchedule",
    "MAX_ALIAS_SAFE_DISTANCE_M",
    "MIN_PAD_FACTOR",
    "PropagationInterval",
    "ScreenCell",
    "build_adaptive_schedule",
    "build_screen_cells",
    "fft2c",
    "ifft2c",
    "fourier_plane_pitch",
    "image_plane_pitch_from_fourier",
    "lens_2f_forward",
    "lens_2f_inverse",
    "circular_aperture",
    "gaussian_radius_at_distance",
    "gaussian_waist_from_half_angle",
    "make_collimated_gaussian_field",
    "propagate_padded_same_window",
    "propagate_same_window",
    "scaled_fresnel_propagate",
    "zoom_propagate",
]
