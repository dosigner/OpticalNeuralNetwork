"""Preview the adaptive vacuum split-step schedule."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from kim2026.cli.common import (
    apply_runtime_environment,
    choose_device,
    dump_json,
    load_config,
    resolve_run_subdir,
)
from kim2026.optics.gaussian_beam import make_collimated_gaussian_field
from kim2026.turbulence.channel import _build_channel_schedule, _propagate_direct_window
from kim2026.utils.seed import set_global_seed


def _field_to_image(field: torch.Tensor) -> torch.Tensor:
    image = field.detach().abs().square().cpu()
    if image.ndim != 2:
        raise ValueError("expected a 2D complex field")
    return image


def _save_intensity_png(path: Path, *, field: torch.Tensor, title: str) -> None:
    import matplotlib.pyplot as plt

    image = _field_to_image(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image.numpy(), cmap="magma")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _append_event(
    events: list[dict[str, Any]],
    *,
    output_dir: Path,
    field: torch.Tensor,
    z_m: float,
    window_m: float,
    event_type: str,
) -> None:
    event_index = len(events)
    filename = f"{event_index:02d}_{event_type}.png"
    _save_intensity_png(
        output_dir / filename,
        field=field,
        title=f"{event_type} | z={z_m:.2f} m | W={window_m:.3f} m",
    )
    events.append(
        {
            "index": event_index,
            "z_m": float(z_m),
            "window_m": float(window_m),
            "event_type": event_type,
            "path": filename,
        }
    )


def _trace_vacuum_schedule(cfg: dict[str, Any], *, device: torch.device) -> list[dict[str, Any]]:
    wavelength_m = float(cfg["optics"]["lambda_m"])
    source_window_m = float(cfg["grid"]["source_window_m"])
    receiver_window_m = float(cfg["grid"]["receiver_window_m"])
    source_field, _, _ = make_collimated_gaussian_field(
        n=int(cfg["grid"]["n"]),
        window_m=source_window_m,
        wavelength_m=wavelength_m,
        half_angle_rad=float(cfg["optics"]["half_angle_rad"]),
    )
    schedule, _ = _build_channel_schedule(cfg)
    output_dir = resolve_run_subdir(cfg, "vacuum_split_step")

    source = source_field.to(device)
    events: list[dict[str, Any]] = []
    _append_event(
        events,
        output_dir=output_dir,
        field=source,
        z_m=0.0,
        window_m=source_window_m,
        event_type="source",
    )
    for interval in schedule.intervals:
        if interval.screen_indices_at_end:
            _append_event(
                events,
                output_dir=output_dir,
                field=_propagate_direct_window(
                    source,
                    wavelength_m=wavelength_m,
                    source_window_m=source_window_m,
                    destination_window_m=float(interval.window_m),
                    z_m=float(interval.end_z_m),
                ),
                z_m=interval.end_z_m,
                window_m=float(interval.window_m),
                event_type="screen_plane",
            )
        if interval.zoom_to_window_m is not None:
            _append_event(
                events,
                output_dir=output_dir,
                field=_propagate_direct_window(
                    source,
                    wavelength_m=wavelength_m,
                    source_window_m=source_window_m,
                    destination_window_m=float(interval.zoom_to_window_m),
                    z_m=float(interval.end_z_m),
                ),
                z_m=interval.end_z_m,
                window_m=float(interval.zoom_to_window_m),
                event_type="zoom_plane",
            )

    _append_event(
        events,
        output_dir=output_dir,
        field=_propagate_direct_window(
            source,
            wavelength_m=wavelength_m,
            source_window_m=source_window_m,
            destination_window_m=receiver_window_m,
            z_m=float(cfg["channel"]["path_length_m"]),
        ),
        z_m=float(cfg["channel"]["path_length_m"]),
        window_m=receiver_window_m,
        event_type="receiver",
    )
    return events


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    apply_runtime_environment(cfg["runtime"])
    set_global_seed(
        int(cfg["runtime"]["seed"]),
        strict_reproducibility=bool(cfg["runtime"]["strict_reproducibility"]),
    )
    device = choose_device(cfg["runtime"])
    events = _trace_vacuum_schedule(cfg, device=device)
    dump_json(
        resolve_run_subdir(cfg, "vacuum_split_step") / "manifest.json",
        {"events": events},
    )


if __name__ == "__main__":
    main()
