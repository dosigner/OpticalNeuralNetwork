from __future__ import annotations

import math

import pytest
import torch

from kim2026.optics.angular_spectrum import propagate_same_window
from kim2026.optics.padded_angular_spectrum import propagate_padded_same_window
from kim2026.optics.aperture import circular_aperture
from kim2026.optics.gaussian_beam import (
    gaussian_radius_at_distance,
    gaussian_waist_from_half_angle,
    make_collimated_gaussian_field,
)
from kim2026.optics.lens_2f import lens_2f_forward, lens_2f_inverse
from kim2026.optics.scaled_fresnel import scaled_fresnel_propagate
from kim2026.optics.zoom_propagate import zoom_propagate


def _second_moment_radius(intensity: torch.Tensor, window_m: float) -> float:
    n = intensity.shape[-1]
    dx = window_m / n
    coords = (torch.arange(n, dtype=torch.float64) - (n // 2)) * dx
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    total = intensity.sum().clamp_min(1e-12)
    r2 = (intensity.to(torch.float64) * (xx.square() + yy.square())).sum() / total
    return float(torch.sqrt(2.0 * r2))


def _analytic_gaussian_intensity(*, n: int, window_m: float, radius_m: float) -> torch.Tensor:
    coords = (torch.arange(n, dtype=torch.float64) - (n // 2)) * (window_m / n)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    intensity = torch.exp(-2.0 * (xx.square() + yy.square()) / (radius_m * radius_m))
    return intensity / intensity.max()


def test_gaussian_waist_matches_half_angle_formula() -> None:
    wavelength_m = 1.55e-6
    half_angle = 3.0e-4

    waist = gaussian_waist_from_half_angle(wavelength_m, half_angle)

    assert math.isclose(waist, wavelength_m / (math.pi * half_angle), rel_tol=1e-12)


def test_make_collimated_gaussian_field_has_expected_central_amplitude() -> None:
    field, x_m, y_m = make_collimated_gaussian_field(
        n=128,
        window_m=0.03,
        wavelength_m=1.55e-6,
        half_angle_rad=3.0e-4,
    )

    assert field.dtype == torch.complex64
    assert x_m.shape == (128,)
    assert y_m.shape == (128,)
    assert torch.isclose(field.abs().amax(), torch.tensor(1.0), atol=1e-5)


def test_circular_aperture_masks_outside_radius() -> None:
    mask = circular_aperture(n=64, window_m=0.64, diameter_m=0.2)

    assert mask[32, 32] == 1.0
    assert mask[0, 0] == 0.0


def test_zoom_propagate_tracks_vacuum_gaussian_radius() -> None:
    wavelength_m = 1.55e-6
    half_angle = 3.0e-4
    source_window_m = 0.03
    receiver_window_m = 0.72
    z_m = 1000.0

    field, _, _ = make_collimated_gaussian_field(
        n=256,
        window_m=source_window_m,
        wavelength_m=wavelength_m,
        half_angle_rad=half_angle,
    )
    propagated = zoom_propagate(
        field.unsqueeze(0),
        wavelength_m=wavelength_m,
        source_window_m=source_window_m,
        destination_window_m=receiver_window_m,
        z_m=z_m,
    ).squeeze(0)

    measured_radius = _second_moment_radius(propagated.abs().square(), receiver_window_m)
    expected_radius = gaussian_radius_at_distance(
        wavelength_m=wavelength_m,
        half_angle_rad=half_angle,
        z_m=z_m,
    )

    assert math.isclose(measured_radius, expected_radius, rel_tol=0.08)


def test_same_window_propagator_preserves_vacuum_energy() -> None:
    field, _, _ = make_collimated_gaussian_field(
        n=256,
        window_m=0.24,
        wavelength_m=1.55e-6,
        half_angle_rad=3.0e-4,
    )

    propagated = propagate_same_window(
        field.unsqueeze(0),
        wavelength_m=1.55e-6,
        window_m=0.24,
        z_m=50.0,
    ).squeeze(0)

    assert math.isclose(
        float(propagated.abs().square().sum().item()),
        float(field.abs().square().sum().item()),
        rel_tol=1e-3,
    )


def test_same_window_propagator_tracks_vacuum_gaussian_radius() -> None:
    wavelength_m = 1.55e-6
    half_angle = 3.0e-4
    window_m = 0.24
    z_m = 50.0

    field, _, _ = make_collimated_gaussian_field(
        n=256,
        window_m=window_m,
        wavelength_m=wavelength_m,
        half_angle_rad=half_angle,
    )

    propagated = propagate_same_window(
        field.unsqueeze(0),
        wavelength_m=wavelength_m,
        window_m=window_m,
        z_m=z_m,
    ).squeeze(0)

    measured_radius = _second_moment_radius(propagated.abs().square(), window_m)
    expected_radius = gaussian_radius_at_distance(
        wavelength_m=wavelength_m,
        half_angle_rad=half_angle,
        z_m=z_m,
    )

    assert math.isclose(measured_radius, expected_radius, rel_tol=0.05)


def test_padded_same_window_suppresses_wraparound_for_edge_gaussian() -> None:
    n = 256
    window_m = n * 2.0e-6
    wavelength_m = 1.55e-6
    z_m = 0.01
    dx = window_m / n
    coords = (torch.arange(n, dtype=torch.float32) - (n // 2)) * dx
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    sigma = 6.0 * dx
    center_x = coords[-12]
    field = torch.exp(-((xx - center_x).square() + yy.square()) / (2.0 * sigma * sigma)).to(torch.complex64)

    periodic = propagate_same_window(
        field.unsqueeze(0),
        wavelength_m=wavelength_m,
        window_m=window_m,
        z_m=z_m,
    ).squeeze(0)
    padded = propagate_padded_same_window(
        field.unsqueeze(0),
        wavelength_m=wavelength_m,
        window_m=window_m,
        z_m=z_m,
        pad_factor=2,
    ).squeeze(0)

    periodic_left = float(periodic[:, :16].abs().square().sum() / periodic.abs().square().sum())
    padded_left = float(padded[:, :16].abs().square().sum() / padded.abs().square().sum())

    assert periodic_left > 1.0e-2
    assert padded_left < periodic_left * 0.05


def test_padded_same_window_rejects_alias_prone_distance() -> None:
    field = torch.ones((1, 32, 32), dtype=torch.complex64)

    with pytest.raises(ValueError, match="alias-safe guard"):
        propagate_padded_same_window(
            field,
            wavelength_m=1.55e-6,
            window_m=32 * 2.0e-6,
            z_m=0.051,
            pad_factor=2,
        )


def test_lens_2f_pitch_depends_on_focal_length() -> None:
    field = torch.randn(1, 64, 64, dtype=torch.complex64)

    _, dx_f1 = lens_2f_forward(
        field,
        dx_in_m=1.0e-6,
        wavelength_m=5.32e-7,
        f_m=1.0e-3,
        na=None,
        apply_scaling=False,
    )
    _, dx_f4 = lens_2f_forward(
        field,
        dx_in_m=1.0e-6,
        wavelength_m=5.32e-7,
        f_m=4.0e-3,
        na=None,
        apply_scaling=False,
    )

    assert dx_f4 / dx_f1 == pytest.approx(4.0, rel=1e-6, abs=1e-6)


def test_lens_2f_inverse_recovers_input_pitch_when_f1_equals_f2() -> None:
    field = torch.randn(1, 64, 64, dtype=torch.complex64)
    dx_in = 1.0e-6

    fourier, dx_fourier = lens_2f_forward(
        field,
        dx_in_m=dx_in,
        wavelength_m=5.32e-7,
        f_m=1.0e-3,
        na=None,
        apply_scaling=False,
    )
    _, dx_out = lens_2f_inverse(
        fourier,
        dx_fourier_m=dx_fourier,
        wavelength_m=5.32e-7,
        f_m=1.0e-3,
        na=None,
        apply_scaling=False,
    )

    assert dx_out == pytest.approx(dx_in, rel=1e-6, abs=1e-12)


def test_lens_2f_na_filter_reduces_energy() -> None:
    field = torch.randn(1, 64, 64, dtype=torch.complex64)

    out_lo, _ = lens_2f_forward(
        field,
        dx_in_m=1.0e-6,
        wavelength_m=5.32e-7,
        f_m=1.0e-3,
        na=0.05,
        apply_scaling=False,
    )
    out_hi, _ = lens_2f_forward(
        field,
        dx_in_m=1.0e-6,
        wavelength_m=5.32e-7,
        f_m=1.0e-3,
        na=0.16,
        apply_scaling=False,
    )

    assert float((out_lo.abs() ** 2).sum().item()) <= float((out_hi.abs() ** 2).sum().item()) + 1e-7


@pytest.mark.parametrize(
    ("wavelength_m", "source_window_m", "destination_window_m", "error_message"),
    [
        (0.0, 0.03, 0.72, "wavelength_m must be > 0"),
        (-1.55e-6, 0.03, 0.72, "wavelength_m must be > 0"),
        (1.55e-6, 0.0, 0.72, "source_window_m must be > 0"),
        (1.55e-6, -0.03, 0.72, "source_window_m must be > 0"),
        (1.55e-6, 0.03, 0.0, "destination_window_m must be > 0"),
        (1.55e-6, 0.03, -0.72, "destination_window_m must be > 0"),
    ],
)
def test_scaled_fresnel_rejects_nonpositive_physical_parameters(
    wavelength_m: float,
    source_window_m: float,
    destination_window_m: float,
    error_message: str,
) -> None:
    field = torch.zeros(1, 16, 16, dtype=torch.complex64)

    with pytest.raises(ValueError, match=error_message):
        scaled_fresnel_propagate(
            field,
            wavelength_m=wavelength_m,
            source_window_m=source_window_m,
            destination_window_m=destination_window_m,
            z_m=1.0,
        )


def test_zoom_propagate_rejects_equal_window_calls() -> None:
    field = torch.zeros(1, 16, 16, dtype=torch.complex64)

    with pytest.raises(ValueError, match="propagate_same_window"):
        zoom_propagate(
            field,
            wavelength_m=1.55e-6,
            source_window_m=0.24,
            destination_window_m=0.24,
            z_m=1.0,
        )


def test_zoom_propagate_surfaces_square_field_validation_before_equal_window_guidance() -> None:
    field = torch.zeros(1, 16, 8, dtype=torch.complex64)

    with pytest.raises(ValueError, match="field must be square in the last two dimensions"):
        zoom_propagate(
            field,
            wavelength_m=1.55e-6,
            source_window_m=0.24,
            destination_window_m=0.24,
            z_m=1.0,
        )


@pytest.mark.parametrize(
    ("wavelength_m", "source_window_m", "destination_window_m", "z_m", "error_message"),
    [
        (0.0, 0.24, 0.24, 1.0, "wavelength_m must be > 0"),
        (1.55e-6, 0.0, 0.0, 1.0, "source_window_m must be > 0"),
        (1.55e-6, -0.24, -0.24, 1.0, "source_window_m must be > 0"),
        (1.55e-6, 0.24, 0.24, 0.0, "z_m must be > 0"),
    ],
)
def test_zoom_propagate_surfaces_validation_errors_before_equal_window_guidance(
    wavelength_m: float,
    source_window_m: float,
    destination_window_m: float,
    z_m: float,
    error_message: str,
) -> None:
    field = torch.zeros(1, 16, 16, dtype=torch.complex64)

    with pytest.raises(ValueError, match=error_message):
        zoom_propagate(
            field,
            wavelength_m=wavelength_m,
            source_window_m=source_window_m,
            destination_window_m=destination_window_m,
            z_m=z_m,
        )


@pytest.mark.parametrize(
    "field",
    [
        torch.zeros((), dtype=torch.complex64),
        torch.zeros(16, dtype=torch.complex64),
    ],
)
def test_propagate_same_window_rejects_low_rank_inputs(field: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="field must be square in the last two dimensions"):
        propagate_same_window(
            field,
            wavelength_m=1.55e-6,
            window_m=0.24,
            z_m=50.0,
        )


@pytest.mark.parametrize(
    "field",
    [
        torch.zeros((), dtype=torch.complex64),
        torch.zeros(16, dtype=torch.complex64),
    ],
)
def test_scaled_fresnel_rejects_low_rank_inputs(field: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="field must be square in the last two dimensions"):
        scaled_fresnel_propagate(
            field,
            wavelength_m=1.55e-6,
            source_window_m=0.03,
            destination_window_m=0.72,
            z_m=1.0,
        )


@pytest.mark.parametrize(
    "field",
    [
        torch.zeros((), dtype=torch.complex64),
        torch.zeros(16, dtype=torch.complex64),
    ],
)
def test_zoom_propagate_rejects_low_rank_inputs(field: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="field must be square in the last two dimensions"):
        zoom_propagate(
            field,
            wavelength_m=1.55e-6,
            source_window_m=0.03,
            destination_window_m=0.72,
            z_m=1.0,
        )


def test_zoom_propagate_preserves_vacuum_gaussian_shape() -> None:
    wavelength_m = 1.55e-6
    half_angle = 3.0e-4
    source_window_m = 0.03
    receiver_window_m = 0.72
    z_m = 1000.0
    n = 256

    field, _, _ = make_collimated_gaussian_field(
        n=n,
        window_m=source_window_m,
        wavelength_m=wavelength_m,
        half_angle_rad=half_angle,
    )
    propagated = zoom_propagate(
        field.unsqueeze(0),
        wavelength_m=wavelength_m,
        source_window_m=source_window_m,
        destination_window_m=receiver_window_m,
        z_m=z_m,
    ).squeeze(0)

    measured = propagated.abs().square().to(torch.float64)
    measured = measured / measured.max()
    expected = _analytic_gaussian_intensity(
        n=n,
        window_m=receiver_window_m,
        radius_m=gaussian_radius_at_distance(
            wavelength_m=wavelength_m,
            half_angle_rad=half_angle,
            z_m=z_m,
        ),
    )

    max_abs_error = torch.max(torch.abs(measured - expected)).item()

    assert max_abs_error < 0.1
