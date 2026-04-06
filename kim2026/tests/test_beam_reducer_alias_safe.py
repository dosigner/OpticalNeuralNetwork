from __future__ import annotations

import pytest
import torch

from kim2026.optics.beam_reducer import (
    BeamReducerPlane,
    apply_beam_reducer,
    apply_beam_reducer_bilinear_legacy,
)


def _energy(field: torch.Tensor, *, window_m: float) -> torch.Tensor:
    dx = float(window_m) / field.shape[-1]
    return field.abs().square().sum(dim=(-2, -1)) * (dx * dx)


def _coordinates(n: int, window_m: float) -> torch.Tensor:
    dx = window_m / n
    return (torch.arange(n, dtype=torch.float32) - n / 2 + 0.5) * dx


def _analytic_field(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    amplitude = torch.exp(-((x / 0.025) ** 2 + (y / 0.020) ** 2))
    phase = 12.0 * x + 7.0 * y
    return amplitude.to(torch.complex64) * torch.exp(1j * phase)


def _pupil_mask(n: int, window_m: float, aperture_diameter_m: float) -> torch.Tensor:
    axis = _coordinates(n, window_m)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    return (radius <= (aperture_diameter_m / 2.0)).to(torch.float32)


def _planes(n: int) -> tuple[BeamReducerPlane, BeamReducerPlane]:
    input_window_m = float(n) * 150e-6
    aperture_diameter_m = input_window_m * (1000.0 / 1024.0)
    output_window_m = float(n) * 2e-6
    return (
        BeamReducerPlane(window_m=input_window_m, n=n, aperture_diameter_m=aperture_diameter_m),
        BeamReducerPlane(window_m=output_window_m, n=n, aperture_diameter_m=output_window_m),
    )


def test_alias_safe_reducer_preserves_energy_within_one_percent() -> None:
    n = 128
    input_plane, output_plane = _planes(n)
    amplitude = torch.exp(
        -(
            torch.linspace(-1.0, 1.0, n, dtype=torch.float32).unsqueeze(0).square()
            + torch.linspace(-1.0, 1.0, n, dtype=torch.float32).unsqueeze(1).square()
        )
    )
    phase = torch.linspace(-0.4, 0.4, n * n, dtype=torch.float32).reshape(n, n)
    field = amplitude.to(torch.complex64) * torch.exp(1j * phase) * _pupil_mask(n, input_plane.window_m, input_plane.aperture_diameter_m)

    reduced = apply_beam_reducer(field, input_plane=input_plane, output_plane=output_plane, pad_factor=2)
    ratio = (_energy(reduced.unsqueeze(0), window_m=output_plane.window_m) / _energy(field.unsqueeze(0), window_m=input_plane.window_m)).item()

    assert ratio == pytest.approx(1.0, abs=1.0e-2)


def test_alias_safe_reducer_keeps_zero_phase_continuous() -> None:
    n = 128
    input_plane, output_plane = _planes(n)
    field = torch.ones((n, n), dtype=torch.complex64)

    reduced = apply_beam_reducer(field, input_plane=input_plane, output_plane=output_plane, pad_factor=2)
    support = reduced.abs() > (0.5 * reduced.abs().amax())

    assert torch.max(torch.abs(torch.angle(reduced[support]))).item() < 1.0e-5


def test_alias_safe_reducer_is_stable_across_pad_factors() -> None:
    n = 128
    input_plane, output_plane = _planes(n)
    phase = torch.linspace(-1.0, 1.0, n * n, dtype=torch.float32).reshape(n, n)
    field = torch.exp(1j * phase).to(torch.complex64) * _pupil_mask(n, input_plane.window_m, input_plane.aperture_diameter_m)

    reduced_pf2 = apply_beam_reducer(field, input_plane=input_plane, output_plane=output_plane, pad_factor=2)
    reduced_pf4 = apply_beam_reducer(field, input_plane=input_plane, output_plane=output_plane, pad_factor=4)
    rel_diff = torch.linalg.vector_norm((reduced_pf2 - reduced_pf4).reshape(-1)) / torch.linalg.vector_norm(reduced_pf4.reshape(-1))

    assert float(rel_diff.item()) < 1.5e-2


def test_alias_safe_reducer_has_lower_error_than_legacy_bilinear_path() -> None:
    n = 128
    input_plane, output_plane = _planes(n)
    x_in = _coordinates(n, input_plane.window_m)
    yy_in, xx_in = torch.meshgrid(x_in, x_in, indexing="ij")
    field = _analytic_field(xx_in, yy_in) * _pupil_mask(n, input_plane.window_m, input_plane.aperture_diameter_m)

    x_out = _coordinates(n, output_plane.window_m)
    yy_out, xx_out = torch.meshgrid(x_out, x_out, indexing="ij")
    magnification = input_plane.aperture_diameter_m / output_plane.window_m
    m = output_plane.window_m / input_plane.aperture_diameter_m
    expected = magnification * _analytic_field(xx_out / m, yy_out / m)

    alias_safe = apply_beam_reducer(field, input_plane=input_plane, output_plane=output_plane, pad_factor=2)
    bilinear = apply_beam_reducer_bilinear_legacy(
        field,
        input_window_m=input_plane.window_m,
        aperture_diameter_m=input_plane.aperture_diameter_m,
        output_window_m=output_plane.window_m,
    )

    alias_err = torch.linalg.vector_norm((alias_safe - expected).reshape(-1)).item()
    bilinear_err = torch.linalg.vector_norm((bilinear - expected).reshape(-1)).item()

    assert alias_err < bilinear_err
