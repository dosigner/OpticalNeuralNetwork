"""Frozen-flow helpers."""

from __future__ import annotations

import math

import torch


def extract_frozen_flow_window(
    canvas: torch.Tensor,
    *,
    output_n: int,
    frame_index: int,
    dt_s: float,
    dx_m: float,
    wind_speed_mps: float,
    wind_dir_rad: float,
) -> torch.Tensor:
    """Extract a wrapped window from a phase canvas under frozen-flow motion."""
    canvas_n = canvas.shape[-1]
    center = (canvas_n - output_n) // 2

    shift_x = int(round(float(wind_speed_mps) * math.cos(float(wind_dir_rad)) * float(dt_s) * int(frame_index) / float(dx_m)))
    shift_y = int(round(float(wind_speed_mps) * math.sin(float(wind_dir_rad)) * float(dt_s) * int(frame_index) / float(dx_m)))

    rolled = torch.roll(canvas, shifts=(shift_y, shift_x), dims=(-2, -1))
    start_y = center - shift_y
    start_x = center - shift_x
    y_idx = (torch.arange(output_n, device=canvas.device) + start_y) % canvas_n
    x_idx = (torch.arange(output_n, device=canvas.device) + start_x) % canvas_n
    return rolled.index_select(0, y_idx).index_select(1, x_idx)
