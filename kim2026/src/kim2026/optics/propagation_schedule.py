"""Adaptive propagation schedule builders for the pilot design."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ScreenCell:
    screen_index: int
    z_m: float
    start_z_m: float
    end_z_m: float
    length_m: float


@dataclass(frozen=True)
class PropagationInterval:
    """Single propagation segment ending at an event plane.

    If an interval end carries both `screen_indices_at_end` and `zoom_to_window_m`,
    downstream code must apply the screen(s) first and then the zoom at that plane.
    """

    index: int
    start_z_m: float
    end_z_m: float
    dz_m: float
    window_m: float
    zoom_to_window_m: float | None
    screen_indices_at_end: tuple[int, ...]


@dataclass(frozen=True)
class AdaptiveSchedule:
    intervals: tuple[PropagationInterval, ...]
    screen_cells: tuple[ScreenCell, ...]


def build_screen_cells(path_length_m: float, screen_positions_m: list[float]) -> tuple[ScreenCell, ...]:
    """Build midpoint-partitioned screen cells after validating the screen geometry."""
    if path_length_m <= 0.0:
        raise ValueError("path_length_m must be positive")

    validated_positions = _validate_screen_positions(
        path_length_m=path_length_m,
        screen_positions_m=screen_positions_m,
    )
    cells = []
    for index, z_m in enumerate(validated_positions):
        start_z_m = 0.0 if index == 0 else 0.5 * (validated_positions[index - 1] + z_m)
        end_z_m = path_length_m if index == len(validated_positions) - 1 else 0.5 * (z_m + validated_positions[index + 1])
        cells.append(
            ScreenCell(
                screen_index=index,
                z_m=float(z_m),
                start_z_m=float(start_z_m),
                end_z_m=float(end_z_m),
                length_m=float(end_z_m - start_z_m),
            )
        )
    return tuple(cells)


def build_adaptive_schedule(
    *,
    wavelength_m: float,
    half_angle_rad: float,
    path_length_m: float,
    receiver_window_m: float,
    internal_receiver_window_m: float,
    source_window_m: float,
    window_ladder_m: list[float],
    screen_positions_m: list[float],
    beam_diameter_fill_fraction: float,
) -> AdaptiveSchedule:
    """Build the pilot propagation schedule for the internal computational window ladder.

    Switch planes currently come from the half-angle fill-threshold relation used by the
    pilot design. `receiver_window_m` is the physical receiver-side aperture/window and is
    validated against `internal_receiver_window_m`, which remains the top of the internal
    propagation ladder used to build the intervals. At a shared interval-end plane,
    downstream code applies any `screen_indices_at_end` first and then `zoom_to_window_m`.
    """

    if wavelength_m <= 0.0:
        raise ValueError("wavelength_m must be positive")
    if path_length_m <= 0.0:
        raise ValueError("path_length_m must be positive")
    if half_angle_rad <= 0.0:
        raise ValueError("half_angle_rad must be positive")
    if receiver_window_m <= 0.0:
        raise ValueError("receiver_window_m must be positive")
    if receiver_window_m > internal_receiver_window_m:
        raise ValueError("receiver_window_m must be less than or equal to internal_receiver_window_m")
    if beam_diameter_fill_fraction <= 0.0:
        raise ValueError("beam_diameter_fill_fraction must be positive")

    validated_screen_positions = _validate_screen_positions(
        path_length_m=path_length_m,
        screen_positions_m=screen_positions_m,
    )
    active_window_ladder = _select_active_window_ladder(
        window_ladder_m=window_ladder_m,
        source_window_m=source_window_m,
        internal_receiver_window_m=internal_receiver_window_m,
    )
    switch_planes_m = _build_switch_planes(
        half_angle_rad=half_angle_rad,
        path_length_m=path_length_m,
        beam_diameter_fill_fraction=beam_diameter_fill_fraction,
        active_window_ladder_m=active_window_ladder,
    )
    event_planes_m = _merge_event_planes(
        switch_planes_m=switch_planes_m,
        screen_positions_m=validated_screen_positions,
        path_length_m=path_length_m,
    )
    screen_indices_by_plane = _group_screen_indices_by_plane(list(validated_screen_positions))
    transitions_by_plane = {
        _plane_key(z_m): next_window_index for next_window_index, z_m in enumerate(switch_planes_m, start=1)
    }

    interval_specs: list[tuple[float, float, float, tuple[int, ...]]] = []
    current_window_index = 0
    start_z_m = 0.0
    for end_z_m in event_planes_m:
        interval_specs.append(
            (
                start_z_m,
                end_z_m,
                active_window_ladder[current_window_index],
                screen_indices_by_plane.get(_plane_key(end_z_m), ()),
            )
        )
        current_window_index = transitions_by_plane.get(_plane_key(end_z_m), current_window_index)
        start_z_m = end_z_m

    intervals = []
    for index, (start_z_m, end_z_m, window_m, screen_indices_at_end) in enumerate(interval_specs):
        next_window_m = interval_specs[index + 1][2] if index + 1 < len(interval_specs) else None
        intervals.append(
            PropagationInterval(
                index=index,
                start_z_m=float(start_z_m),
                end_z_m=float(end_z_m),
                dz_m=float(end_z_m - start_z_m),
                window_m=float(window_m),
                zoom_to_window_m=float(next_window_m) if next_window_m is not None and next_window_m > window_m else None,
                screen_indices_at_end=screen_indices_at_end,
            )
        )

    return AdaptiveSchedule(
        intervals=tuple(intervals),
        screen_cells=build_screen_cells(path_length_m=path_length_m, screen_positions_m=list(validated_screen_positions)),
    )


def _build_switch_planes(
    *,
    half_angle_rad: float,
    path_length_m: float,
    beam_diameter_fill_fraction: float,
    active_window_ladder_m: tuple[float, ...],
) -> tuple[float, ...]:
    switch_planes_m = []
    for window_m in active_window_ladder_m[:-1]:
        z_m = beam_diameter_fill_fraction * window_m / (2.0 * half_angle_rad)
        if 0.0 < z_m < path_length_m:
            switch_planes_m.append(float(z_m))
    return _sorted_unique(switch_planes_m)


def _group_screen_indices_by_plane(screen_positions_m: list[float]) -> dict[float, tuple[int, ...]]:
    indices_by_plane: dict[float, list[int]] = {}
    for index, z_m in enumerate(screen_positions_m):
        indices_by_plane.setdefault(_plane_key(z_m), []).append(index)
    return {plane: tuple(indices) for plane, indices in indices_by_plane.items()}


def _merge_event_planes(
    *,
    switch_planes_m: tuple[float, ...],
    screen_positions_m: list[float],
    path_length_m: float,
) -> tuple[float, ...]:
    return _sorted_unique([*switch_planes_m, *screen_positions_m, float(path_length_m)])


def _plane_key(z_m: float) -> float:
    return round(float(z_m), 12)


def _select_active_window_ladder(
    *,
    window_ladder_m: list[float],
    source_window_m: float,
    internal_receiver_window_m: float,
) -> tuple[float, ...]:
    ladder = tuple(float(window_m) for window_m in window_ladder_m)
    if not ladder:
        raise ValueError("window_ladder_m must not be empty")
    if any(window_m <= 0.0 for window_m in ladder):
        raise ValueError("window_ladder_m must contain only positive values")

    _validate_strictly_increasing(name="window_ladder_m", values=ladder)
    start_index = _find_matching_index(values=ladder, target=source_window_m, name="source_window_m")
    end_index = _find_matching_index(
        values=ladder,
        target=internal_receiver_window_m,
        name="internal_receiver_window_m",
    )
    if start_index > end_index:
        raise ValueError("source_window_m must not exceed internal_receiver_window_m within window_ladder_m")
    return ladder[start_index : end_index + 1]


def _validate_screen_positions(*, path_length_m: float, screen_positions_m: list[float]) -> tuple[float, ...]:
    positions = tuple(float(z_m) for z_m in screen_positions_m)
    if any(z_m <= 0.0 or z_m > path_length_m for z_m in positions):
        raise ValueError("screen_positions_m must lie in (0, path_length_m]")
    _validate_strictly_increasing(name="screen_positions_m", values=positions)
    return positions


def _validate_strictly_increasing(*, name: str, values: tuple[float, ...]) -> None:
    for left, right in zip(values, values[1:]):
        if not right > left:
            raise ValueError(f"{name} must be strictly increasing")


def _find_matching_index(*, values: tuple[float, ...], target: float, name: str) -> int:
    for index, value in enumerate(values):
        if math.isclose(value, float(target), rel_tol=0.0, abs_tol=1e-12):
            return index
    raise ValueError(f"{name}={target!r} is not present in window_ladder_m")


def _sorted_unique(values: list[float]) -> tuple[float, ...]:
    unique_by_plane = {_plane_key(value): float(value) for value in values}
    return tuple(unique_by_plane[key] for key in sorted(unique_by_plane))


__all__ = [
    "AdaptiveSchedule",
    "PropagationInterval",
    "ScreenCell",
    "build_adaptive_schedule",
    "build_screen_cells",
]
