from __future__ import annotations

import pytest

from kim2026.optics.propagation_schedule import build_adaptive_schedule, build_screen_cells


def _valid_schedule_kwargs() -> dict[str, object]:
    return {
        "wavelength_m": 1.55e-6,
        "half_angle_rad": 3.0e-4,
        "path_length_m": 1000.0,
        "receiver_window_m": 0.72,
        "internal_receiver_window_m": 0.96,
        "source_window_m": 0.03,
        "window_ladder_m": [0.03, 0.06, 0.12, 0.24, 0.48, 0.96],
        "screen_positions_m": [125.0, 250.0, 375.0, 500.0, 625.0, 750.0, 875.0],
        "beam_diameter_fill_fraction": 0.625,
    }


def test_adaptive_schedule_matches_expected_window_ladder() -> None:
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

    assert [round(interval.start_z_m, 2) for interval in schedule.intervals] == [
        0.0,
        31.25,
        62.5,
        125.0,
        250.0,
        375.0,
        500.0,
        625.0,
        750.0,
        875.0,
    ]
    assert [round(interval.end_z_m, 2) for interval in schedule.intervals] == [
        31.25,
        62.5,
        125.0,
        250.0,
        375.0,
        500.0,
        625.0,
        750.0,
        875.0,
        1000.0,
    ]
    assert [interval.window_m for interval in schedule.intervals] == [
        0.03,
        0.06,
        0.12,
        0.24,
        0.48,
        0.48,
        0.96,
        0.96,
        0.96,
        0.96,
    ]


def test_screen_cells_use_midpoint_rule_for_equivalent_r0() -> None:
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

    assert [cell.length_m for cell in schedule.screen_cells] == [
        187.5,
        125.0,
        125.0,
        125.0,
        125.0,
        125.0,
        187.5,
    ]


def test_shared_switch_and_screen_planes_keep_both_interval_semantics() -> None:
    schedule = build_adaptive_schedule(**_valid_schedule_kwargs())

    expected_by_plane = {
        125.0: ((0,), 0.24),
        250.0: ((1,), 0.48),
        500.0: ((3,), 0.96),
    }

    for plane_m, (screen_indices_at_end, zoom_to_window_m) in expected_by_plane.items():
        interval = next(interval for interval in schedule.intervals if interval.end_z_m == plane_m)
        assert interval.screen_indices_at_end == screen_indices_at_end
        assert interval.zoom_to_window_m == zoom_to_window_m


@pytest.mark.parametrize("wavelength_m", [0.0, -1.55e-6])
def test_build_adaptive_schedule_rejects_nonpositive_wavelength(wavelength_m: float) -> None:
    kwargs = _valid_schedule_kwargs()
    kwargs["wavelength_m"] = wavelength_m

    with pytest.raises(ValueError, match="wavelength_m must be positive"):
        build_adaptive_schedule(**kwargs)


def test_build_adaptive_schedule_rejects_receiver_window_larger_than_internal_window() -> None:
    kwargs = _valid_schedule_kwargs()
    kwargs["receiver_window_m"] = 1.2

    with pytest.raises(
        ValueError,
        match="receiver_window_m must be less than or equal to internal_receiver_window_m",
    ):
        build_adaptive_schedule(**kwargs)


@pytest.mark.parametrize(
    "window_ladder_m",
    [
        [0.0, 0.03, 0.06, 0.12, 0.24, 0.48, 0.96],
        [-0.03, 0.03, 0.06, 0.12, 0.24, 0.48, 0.96],
    ],
)
def test_build_adaptive_schedule_rejects_nonpositive_window_ladder_values(window_ladder_m: list[float]) -> None:
    kwargs = _valid_schedule_kwargs()
    kwargs["window_ladder_m"] = window_ladder_m

    with pytest.raises(ValueError, match="window_ladder_m must contain only positive values"):
        build_adaptive_schedule(**kwargs)


@pytest.mark.parametrize("builder_name", ["screen_cells", "adaptive_schedule"])
def test_unsorted_screen_positions_are_rejected(builder_name: str) -> None:
    screen_positions_m = [250.0, 125.0]

    with pytest.raises(ValueError, match="screen_positions_m must be strictly increasing"):
        if builder_name == "screen_cells":
            build_screen_cells(path_length_m=1000.0, screen_positions_m=screen_positions_m)
        else:
            kwargs = _valid_schedule_kwargs()
            kwargs["screen_positions_m"] = screen_positions_m
            build_adaptive_schedule(**kwargs)
