from __future__ import annotations

import math

import torch

from kim2026.optics.gaussian_beam import make_collimated_gaussian_field
from kim2026.optics.zoom_propagate import zoom_propagate
from kim2026.turbulence.channel import (
    _build_channel_schedule,
    _propagate_schedule_to_receiver,
    propagate_split_step,
)
from kim2026.turbulence.frozen_flow import extract_frozen_flow_window
from kim2026.turbulence.phase_screens import generate_phase_screen


def test_generate_phase_screen_is_deterministic_for_seed() -> None:
    kwargs = {
        "n": 64,
        "window_m": 0.2,
        "wavelength_m": 1.55e-6,
        "cn2": 2.0e-14,
        "path_segment_m": 125.0,
        "outer_scale_m": 30.0,
        "inner_scale_m": 5.0e-3,
        "seed": 7,
    }

    first = generate_phase_screen(**kwargs)
    second = generate_phase_screen(**kwargs)

    assert torch.equal(first, second)


def test_extract_frozen_flow_window_shifts_periodically() -> None:
    canvas = torch.arange(0, 16, dtype=torch.float32).reshape(4, 4)
    window = extract_frozen_flow_window(
        canvas,
        output_n=2,
        frame_index=1,
        dt_s=1.0,
        dx_m=1.0,
        wind_speed_mps=1.0,
        wind_dir_rad=0.0,
    )

    expected = torch.tensor([[7.0, 4.0], [11.0, 8.0]])
    assert torch.equal(window, expected)


def test_split_step_channel_matches_vacuum_for_zero_screens() -> None:
    field, _, _ = make_collimated_gaussian_field(
        n=64,
        window_m=0.03,
        wavelength_m=1.55e-6,
        half_angle_rad=3.0e-4,
    )
    plane_windows = [0.03, 0.08, 0.14, 0.2]
    segment_lengths = [50.0, 50.0, 50.0]
    zero_screens = [torch.zeros(64, 64), torch.zeros(64, 64)]

    vacuum = propagate_split_step(
        field.unsqueeze(0),
        wavelength_m=1.55e-6,
        plane_windows_m=plane_windows,
        segment_lengths_m=segment_lengths,
        phase_screens=None,
    )
    turbulent = propagate_split_step(
        field.unsqueeze(0),
        wavelength_m=1.55e-6,
        plane_windows_m=plane_windows,
        segment_lengths_m=segment_lengths,
        phase_screens=zero_screens,
    )

    assert torch.allclose(vacuum, turbulent, atol=1e-5, rtol=1e-4)


def test_split_step_channel_changes_field_for_nonzero_screen() -> None:
    field, _, _ = make_collimated_gaussian_field(
        n=64,
        window_m=0.03,
        wavelength_m=1.55e-6,
        half_angle_rad=3.0e-4,
    )
    plane_windows = [0.03, 0.08, 0.14, 0.2]
    segment_lengths = [50.0, 50.0, 50.0]
    screen = generate_phase_screen(
        n=64,
        window_m=0.08,
        wavelength_m=1.55e-6,
        cn2=2.0e-14,
        path_segment_m=50.0,
        outer_scale_m=30.0,
        inner_scale_m=5.0e-3,
        seed=11,
    )

    vacuum = propagate_split_step(
        field.unsqueeze(0),
        wavelength_m=1.55e-6,
        plane_windows_m=plane_windows,
        segment_lengths_m=segment_lengths,
        phase_screens=None,
    )
    turbulent = propagate_split_step(
        field.unsqueeze(0),
        wavelength_m=1.55e-6,
        plane_windows_m=plane_windows,
        segment_lengths_m=segment_lengths,
        phase_screens=[screen, screen],
    )

    assert not math.isclose(torch.linalg.vector_norm(vacuum - turbulent).item(), 0.0, abs_tol=1e-6)


def test_equivalent_r0_matches_cn2_for_schedule_screen_cells() -> None:
    from kim2026.optics.propagation_schedule import build_adaptive_schedule
    from kim2026.turbulence.channel import equivalent_r0_for_cell

    schedule = build_adaptive_schedule(
        wavelength_m=1.55e-6,
        half_angle_rad=3.0e-4,
        path_length_m=1000.0,
        receiver_window_m=0.72,
        internal_receiver_window_m=0.96,
        source_window_m=0.03,
        window_ladder_m=[0.03, 0.06, 0.12, 0.24, 0.48, 0.96],
        screen_positions_m=[125.0, 250.0, 375.0, 500.0, 625.0, 750.0, 875.0],
        beam_diameter_fill_fraction=0.625,
    )

    r0_values = [
        equivalent_r0_for_cell(
            wavelength_m=1.55e-6,
            cn2=2.0e-14,
            cell_length_m=cell.length_m,
        )
        for cell in schedule.screen_cells
    ]

    assert math.isclose(r0_values[0], 0.14137063694418073, rel_tol=1e-6)
    assert math.isclose(r0_values[1], 0.18030757402768166, rel_tol=1e-6)
    assert math.isclose(r0_values[-1], 0.14137063694418073, rel_tol=1e-6)


def test_schedule_mode_default_regrid_distance_keeps_vacuum_field_finite() -> None:
    schedule_cfg = {
        "optics": {"lambda_m": 1.55e-6, "half_angle_rad": 3.0e-4},
        "grid": {"n": 128, "source_window_m": 0.03, "receiver_window_m": 0.72},
        "channel": {"path_length_m": 1000.0, "num_screens": 8},
    }
    schedule, regrid_distance_m = _build_channel_schedule(schedule_cfg)
    field, _, _ = make_collimated_gaussian_field(
        n=128,
        window_m=0.03,
        wavelength_m=1.55e-6,
        half_angle_rad=3.0e-4,
    )

    propagated = _propagate_schedule_to_receiver(
        field,
        wavelength_m=1.55e-6,
        schedule=schedule,
        phase_screens=None,
        receiver_window_m=0.72,
        regrid_distance_m=regrid_distance_m,
    )

    assert torch.isfinite(propagated.real).all()
    assert torch.isfinite(propagated.imag).all()


def test_schedule_mode_matches_direct_vacuum_reference() -> None:
    schedule_cfg = {
        "optics": {"lambda_m": 1.55e-6, "half_angle_rad": 3.0e-4},
        "grid": {"n": 128, "source_window_m": 0.03, "receiver_window_m": 0.72},
        "channel": {"path_length_m": 1000.0, "num_screens": 8},
    }
    schedule, regrid_distance_m = _build_channel_schedule(schedule_cfg)
    field, _, _ = make_collimated_gaussian_field(
        n=128,
        window_m=0.03,
        wavelength_m=1.55e-6,
        half_angle_rad=3.0e-4,
    )

    propagated = _propagate_schedule_to_receiver(
        field,
        wavelength_m=1.55e-6,
        schedule=schedule,
        phase_screens=None,
        receiver_window_m=0.72,
        regrid_distance_m=regrid_distance_m,
    )
    reference = zoom_propagate(
        field,
        wavelength_m=1.55e-6,
        source_window_m=0.03,
        destination_window_m=0.72,
        z_m=1000.0,
    )

    measured = propagated.abs().square().to(torch.float64)
    measured = measured / measured.max()
    expected = reference.abs().square().to(torch.float64)
    expected = expected / expected.max()

    max_abs_error = torch.max(torch.abs(measured - expected)).item()

    assert max_abs_error < 0.1
