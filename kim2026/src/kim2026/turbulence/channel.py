"""Split-step turbulence propagation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kim2026.data.manifest import build_split_manifest
from kim2026.data.npz_pairs import write_pair_npz
from kim2026.optics import (
    AdaptiveSchedule,
    PropagationInterval,
    build_adaptive_schedule,
    propagate_same_window,
    zoom_propagate,
)
from kim2026.optics.gaussian_beam import coordinate_axis, make_collimated_gaussian_field
from kim2026.turbulence.frozen_flow import extract_frozen_flow_window
from kim2026.turbulence.phase_screens import generate_phase_screen


def propagate_split_step(
    field: torch.Tensor,
    *,
    wavelength_m: float,
    plane_windows_m: list[float] | None = None,
    segment_lengths_m: list[float] | None = None,
    phase_screens: list[torch.Tensor] | None,
    schedule: AdaptiveSchedule | None = None,
    regrid_distance_m: float | None = None,
) -> torch.Tensor:
    """Propagate through free-space intervals and optional phase screens.

    Legacy compatibility mode accepts `plane_windows_m` and `segment_lengths_m`, where
    each phase screen is applied at the end of a non-terminal segment. Schedule mode
    currently uses a direct source-to-destination fallback with any provided schedule
    screens bundled at the source plane. The config-driven channel builder interprets
    `channel.num_screens` as a physical path cell count, so the actual number of
    generated/applied phase screens is `len(schedule.screen_cells)`, not necessarily
    the raw config value.
    """
    if schedule is not None:
        if plane_windows_m is not None or segment_lengths_m is not None:
            raise ValueError("pass either schedule or plane_windows_m/segment_lengths_m, not both")
        _validate_schedule_phase_screens(schedule=schedule, phase_screens=phase_screens)
        return _propagate_schedule_direct(
            field,
            wavelength_m=wavelength_m,
            schedule=schedule,
            phase_screens=phase_screens,
            destination_window_m=float(schedule.intervals[-1].window_m),
        )

    intervals = _resolve_propagation_intervals(
        plane_windows_m=plane_windows_m,
        segment_lengths_m=segment_lengths_m,
        phase_screens=phase_screens,
        schedule=None,
    )
    regrid_distance_m = _resolve_regrid_distance_m(
        schedule=None,
        segment_lengths_m=segment_lengths_m,
        regrid_distance_m=regrid_distance_m,
    )

    output = field
    for interval in intervals:
        output = propagate_same_window(
            output,
            wavelength_m=wavelength_m,
            window_m=interval.window_m,
            z_m=interval.dz_m,
        )
        for screen_index in interval.screen_indices_at_end:
            if phase_screens is None:
                break
            screen = torch.exp(1j * phase_screens[screen_index].to(output.device)).to(output.dtype)
            screen_shape = (1,) * max(output.ndim - 2, 0) + tuple(screen.shape)
            output = output * screen.reshape(screen_shape)
        if interval.zoom_to_window_m is not None:
            output = zoom_propagate(
                output,
                wavelength_m=wavelength_m,
                source_window_m=interval.window_m,
                destination_window_m=interval.zoom_to_window_m,
                z_m=regrid_distance_m,
            )
    return output


def _apply_phase_screen_bundle(
    field: torch.Tensor,
    *,
    phase_screens: list[torch.Tensor] | None,
) -> torch.Tensor:
    output = field
    if phase_screens is None:
        return output
    for phase_screen in phase_screens:
        screen = torch.exp(1j * phase_screen.to(output.device)).to(output.dtype)
        screen_shape = (1,) * max(output.ndim - 2, 0) + tuple(screen.shape)
        output = output * screen.reshape(screen_shape)
    return output


def _propagate_direct_window(
    field: torch.Tensor,
    *,
    wavelength_m: float,
    source_window_m: float,
    destination_window_m: float,
    z_m: float,
) -> torch.Tensor:
    if math.isclose(source_window_m, destination_window_m, rel_tol=0.0, abs_tol=1e-12):
        return propagate_same_window(
            field,
            wavelength_m=wavelength_m,
            window_m=source_window_m,
            z_m=z_m,
        )
    return zoom_propagate(
        field,
        wavelength_m=wavelength_m,
        source_window_m=source_window_m,
        destination_window_m=destination_window_m,
        z_m=z_m,
    )


def _propagate_schedule_direct(
    field: torch.Tensor,
    *,
    wavelength_m: float,
    schedule: AdaptiveSchedule,
    phase_screens: list[torch.Tensor] | None,
    destination_window_m: float,
) -> torch.Tensor:
    source_window_m = float(schedule.intervals[0].window_m)
    total_distance_m = float(schedule.intervals[-1].end_z_m)
    bundled = _apply_phase_screen_bundle(field, phase_screens=phase_screens)
    return _propagate_direct_window(
        bundled,
        wavelength_m=wavelength_m,
        source_window_m=source_window_m,
        destination_window_m=destination_window_m,
        z_m=total_distance_m,
    )


def _resolve_regrid_distance_m(
    *,
    schedule: AdaptiveSchedule | None,
    segment_lengths_m: list[float] | None,
    regrid_distance_m: float | None,
) -> float:
    if regrid_distance_m is not None:
        if regrid_distance_m <= 0.0:
            raise ValueError("regrid_distance_m must be > 0")
        return float(regrid_distance_m)
    if schedule is not None:
        return float(schedule.intervals[-1].end_z_m)
    if segment_lengths_m is not None:
        total_distance_m = sum(float(dz_m) for dz_m in segment_lengths_m)
        if total_distance_m <= 0.0:
            raise ValueError("sum(segment_lengths_m) must be > 0")
        return float(total_distance_m)
    raise ValueError("regrid_distance_m could not be derived")


def equivalent_r0_for_cell(*, wavelength_m: float, cn2: float, cell_length_m: float) -> float:
    """Return the Fried parameter for a screen cell with the given path-integrated strength."""
    k = 2.0 * math.pi / wavelength_m
    return float((0.423 * (k**2) * cn2 * cell_length_m) ** (-3.0 / 5.0))


def _resolve_propagation_intervals(
    *,
    plane_windows_m: list[float] | None,
    segment_lengths_m: list[float] | None,
    phase_screens: list[torch.Tensor] | None,
    schedule: AdaptiveSchedule | None,
) -> tuple[PropagationInterval, ...]:
    if schedule is not None:
        if plane_windows_m is not None or segment_lengths_m is not None:
            raise ValueError("pass either schedule or plane_windows_m/segment_lengths_m, not both")
        _validate_schedule_phase_screens(schedule=schedule, phase_screens=phase_screens)
        return schedule.intervals

    if plane_windows_m is None or segment_lengths_m is None:
        raise ValueError("plane_windows_m and segment_lengths_m are required when schedule is not provided")
    if len(plane_windows_m) != len(segment_lengths_m) + 1:
        raise ValueError("plane_windows_m must be one element longer than segment_lengths_m")
    if phase_screens is not None and len(phase_screens) != max(len(segment_lengths_m) - 1, 0):
        raise ValueError("phase_screens must have len(segment_lengths_m) - 1 elements")

    intervals = []
    start_z_m = 0.0
    screen_count = 0 if phase_screens is None else len(phase_screens)
    for index, dz_m in enumerate(segment_lengths_m):
        window_m = float(plane_windows_m[index])
        next_window_m = float(plane_windows_m[index + 1])
        zoom_to_window_m = None
        if not math.isclose(window_m, next_window_m, rel_tol=0.0, abs_tol=1e-12):
            zoom_to_window_m = next_window_m
        screen_indices_at_end = (index,) if index < screen_count else ()
        intervals.append(
            PropagationInterval(
                index=index,
                start_z_m=float(start_z_m),
                end_z_m=float(start_z_m + dz_m),
                dz_m=float(dz_m),
                window_m=window_m,
                zoom_to_window_m=zoom_to_window_m,
                screen_indices_at_end=screen_indices_at_end,
            )
        )
        start_z_m += float(dz_m)
    return tuple(intervals)


def _validate_schedule_phase_screens(*, schedule: AdaptiveSchedule, phase_screens: list[torch.Tensor] | None) -> None:
    required_screen_count = max(
        (screen_index for interval in schedule.intervals for screen_index in interval.screen_indices_at_end),
        default=-1,
    ) + 1
    if phase_screens is None:
        return
    if len(phase_screens) != required_screen_count:
        raise ValueError(f"phase_screens must have {required_screen_count} elements for the provided schedule")


def _build_window_ladder(*, source_window_m: float, receiver_window_m: float) -> list[float]:
    ladder = [float(source_window_m)]
    while ladder[-1] < float(receiver_window_m):
        ladder.append(2.0 * ladder[-1])
    return ladder


def _build_channel_schedule(cfg: dict[str, Any]) -> tuple[AdaptiveSchedule, float]:
    """Build the adaptive channel schedule.

    `channel.num_screens` is interpreted as the number of physical path cells. That
    produces one interior phase screen at each cell boundary, so the actual phase-screen
    count is `max(channel.num_screens - 1, 0)` and is exposed downstream via
    `len(schedule.screen_cells)`.
    """
    source_window_m = float(cfg["grid"]["source_window_m"])
    receiver_window_m = float(cfg["grid"]["receiver_window_m"])
    path_length_m = float(cfg["channel"]["path_length_m"])
    path_cell_count = int(cfg["channel"]["num_screens"])
    window_ladder_m = _build_window_ladder(
        source_window_m=source_window_m,
        receiver_window_m=receiver_window_m,
    )
    screen_positions_m = [
        float(path_length_m) * float(index) / float(path_cell_count)
        for index in range(1, path_cell_count)
    ]
    schedule = build_adaptive_schedule(
        wavelength_m=float(cfg["optics"]["lambda_m"]),
        half_angle_rad=float(cfg["optics"]["half_angle_rad"]),
        path_length_m=path_length_m,
        receiver_window_m=receiver_window_m,
        internal_receiver_window_m=window_ladder_m[-1],
        source_window_m=source_window_m,
        window_ladder_m=window_ladder_m,
        screen_positions_m=screen_positions_m,
        beam_diameter_fill_fraction=0.625,
    )
    return schedule, float(cfg["channel"].get("regrid_distance_m", path_length_m))


def _warmup_schedule_propagation(
    *,
    field_shape: tuple[int, int, int],
    device: torch.device,
    wavelength_m: float,
    schedule: AdaptiveSchedule,
    receiver_window_m: float,
    regrid_distance_m: float,
    iterations: int,
) -> None:
    if int(iterations) <= 0:
        return
    field = torch.zeros(field_shape, dtype=torch.complex64, device=device)
    for _ in range(int(iterations)):
        _ = _propagate_schedule_to_receiver(
            field,
            wavelength_m=wavelength_m,
            schedule=schedule,
            phase_screens=None,
            receiver_window_m=receiver_window_m,
            regrid_distance_m=regrid_distance_m,
        )


def _propagate_schedule_to_receiver(
    field: torch.Tensor,
    *,
    wavelength_m: float,
    schedule: AdaptiveSchedule,
    phase_screens: list[torch.Tensor] | None,
    receiver_window_m: float,
    regrid_distance_m: float,
) -> torch.Tensor:
    """Run the stable direct fallback and land on the physical receiver window."""
    return _propagate_schedule_direct(
        field,
        wavelength_m=wavelength_m,
        schedule=schedule,
        phase_screens=phase_screens,
        destination_window_m=receiver_window_m,
    )


def _episode_seed(global_seed: int, episode_id: int) -> int:
    return int(global_seed) + 1009 * int(episode_id)


def generate_pair_cache(cfg: dict[str, Any]) -> dict[str, Any]:
    """Generate deterministic vacuum/turbulence NPZ pairs from config."""
    cache_dir = Path(cfg["data"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    global_seed = int(cfg["runtime"]["seed"])
    n = int(cfg["grid"]["n"])
    source_window_m = float(cfg["grid"]["source_window_m"])
    receiver_window_m = float(cfg["grid"]["receiver_window_m"])
    wavelength_m = float(cfg["optics"]["lambda_m"])
    half_angle_rad = float(cfg["optics"]["half_angle_rad"])
    path_length_m = float(cfg["channel"]["path_length_m"])
    cn2 = float(cfg["channel"]["cn2"])
    outer_scale_m = float(cfg["channel"]["outer_scale_m"])
    inner_scale_m = float(cfg["channel"]["inner_scale_m"])
    path_cell_count = int(cfg["channel"]["num_screens"])
    frozen = cfg["channel"]["frozen_flow"]
    dt_s = float(frozen["dt_s"])
    frames_per_episode = int(frozen["frames_per_episode"])
    wind_speed_mps = float(frozen["wind_speed_mps"])
    screen_canvas_scale = float(frozen["screen_canvas_scale"])
    aperture_diameter_m = float(cfg["receiver"]["aperture_diameter_m"])
    split_counts = dict(cfg["data"]["split_episode_counts"])

    source_field, _, _ = make_collimated_gaussian_field(
        n=n,
        window_m=source_window_m,
        wavelength_m=wavelength_m,
        half_angle_rad=half_angle_rad,
    )
    schedule, regrid_distance_m = _build_channel_schedule(cfg)
    phase_screen_count = len(schedule.screen_cells)
    _warmup_schedule_propagation(
        field_shape=(1, n, n),
        device=source_field.device,
        wavelength_m=wavelength_m,
        schedule=schedule,
        receiver_window_m=receiver_window_m,
        regrid_distance_m=regrid_distance_m,
        iterations=int(cfg["runtime"].get("fft_warmup_iters", 0)),
    )
    vacuum_field = _propagate_schedule_to_receiver(
        source_field.unsqueeze(0),
        wavelength_m=wavelength_m,
        schedule=schedule,
        phase_screens=None,
        receiver_window_m=receiver_window_m,
        regrid_distance_m=regrid_distance_m,
    ).squeeze(0)
    x_m = coordinate_axis(n, receiver_window_m).cpu().numpy()
    y_m = coordinate_axis(n, receiver_window_m).cpu().numpy()

    total_episodes = sum(int(v) for v in split_counts.values())
    episode_manifest: list[dict[str, Any]] = []
    split_filenames: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    split_episodes = build_split_manifest(
        episode_ids=list(range(total_episodes)),
        split_counts={key: int(value) for key, value in split_counts.items()},
    )
    episode_to_split = {episode_id: split for split, ids in split_episodes.items() for episode_id in ids}

    for episode_id in range(total_episodes):
        episode_seed = _episode_seed(global_seed, episode_id)
        rng = np.random.default_rng(episode_seed)
        wind_dir_rad = float(rng.uniform(0.0, 2.0 * math.pi))

        canvases = []
        screen_seeds = []
        for cell in schedule.screen_cells:
            screen_idx = int(cell.screen_index)
            screen_seed = episode_seed * 100 + screen_idx
            screen_seeds.append(screen_seed)
            canvas_n = max(int(math.ceil(n * screen_canvas_scale)), n)
            canvas_window_m = source_window_m * screen_canvas_scale
            canvases.append(
                generate_phase_screen(
                    n=canvas_n,
                    window_m=canvas_window_m,
                    wavelength_m=wavelength_m,
                    cn2=cn2,
                    path_segment_m=cell.length_m,
                    outer_scale_m=outer_scale_m,
                    inner_scale_m=inner_scale_m,
                    seed=screen_seed,
                )
            )

        episode_record = {
            "episode_id": episode_id,
            "episode_seed": episode_seed,
            "path_cell_count": path_cell_count,
            "screen_count": phase_screen_count,
            "screen_seeds": screen_seeds,
            "wind_dir_rad": wind_dir_rad,
        }
        episode_manifest.append(episode_record)

        for frame_index in range(frames_per_episode):
            phase_screens = []
            for canvas in canvases:
                canvas_window_m = source_window_m * screen_canvas_scale
                dx_m = canvas_window_m / canvas.shape[-1]
                phase_screens.append(
                    extract_frozen_flow_window(
                        canvas,
                        output_n=n,
                        frame_index=frame_index,
                        dt_s=dt_s,
                        dx_m=dx_m,
                        wind_speed_mps=wind_speed_mps,
                        wind_dir_rad=wind_dir_rad,
                    )
                )

            turbulent_field = _propagate_schedule_to_receiver(
                source_field.unsqueeze(0),
                wavelength_m=wavelength_m,
                schedule=schedule,
                phase_screens=phase_screens,
                receiver_window_m=receiver_window_m,
                regrid_distance_m=regrid_distance_m,
            ).squeeze(0)
            filename = f"episode_{episode_id:05d}_frame_{frame_index:03d}.npz"
            write_pair_npz(
                cache_dir / filename,
                u_vacuum=vacuum_field,
                u_turb=turbulent_field,
                x_m=x_m,
                y_m=y_m,
                metadata={
                    "episode_id": episode_id,
                    "frame_index": frame_index,
                    "global_seed": global_seed,
                    "episode_seed": episode_seed,
                    "screen_seeds": screen_seeds,
                    "wind_dir_rad": wind_dir_rad,
                    "dt_s": dt_s,
                    "lambda_m": wavelength_m,
                    "path_length_m": path_length_m,
                    "cn2": cn2,
                    "half_angle_rad": half_angle_rad,
                    "aperture_diameter_m": aperture_diameter_m,
                    "receiver_window_m": receiver_window_m,
                    "L0_m": outer_scale_m,
                    "l0_m": inner_scale_m,
                    "path_cell_count": path_cell_count,
                    "screen_count": phase_screen_count,
                },
            )
            split_filenames[episode_to_split[episode_id]].append(filename)

    split_manifest_path = Path(cfg["data"]["split_manifest_path"])
    split_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    split_manifest_path.write_text(json.dumps(split_filenames, indent=2), encoding="utf-8")

    episode_manifest_path = Path(cfg["data"]["episode_manifest_path"])
    episode_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    episode_manifest_path.write_text(json.dumps(episode_manifest, indent=2), encoding="utf-8")
    return {"split_manifest_path": split_manifest_path, "episode_manifest_path": episode_manifest_path}
