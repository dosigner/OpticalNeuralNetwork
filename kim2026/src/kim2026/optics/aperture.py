"""Aperture masks."""

from __future__ import annotations

import torch

from kim2026.optics.gaussian_beam import coordinate_axis


def circular_aperture(
    *,
    n: int,
    window_m: float,
    diameter_m: float,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create a centered circular aperture mask."""
    axis = coordinate_axis(n, window_m, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    radius_sq = xx.square() + yy.square()
    return (radius_sq <= (0.5 * float(diameter_m)) ** 2).to(torch.float32)
