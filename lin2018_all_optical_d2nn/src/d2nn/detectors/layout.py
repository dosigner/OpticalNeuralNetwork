"""Detector layout definitions and mask builders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from d2nn.physics.grid import make_spatial_grid
from d2nn.utils.io import load_json


@dataclass(frozen=True)
class DetectorRegion:
    """Single detector region definition.

    Units:
        center_xy: [m]
        size_xy: [m]
    """

    name: str
    center_xy: tuple[float, float]
    size_xy: tuple[float, float]


@dataclass(frozen=True)
class DetectorLayout:
    """Detector layout on output plane."""

    regions: list[DetectorRegion]
    plane_size_xy: tuple[float, float]


def load_layout(path: str | Path) -> DetectorLayout:
    """Load detector layout from JSON file."""

    payload = load_json(path)
    regions = [
        DetectorRegion(
            name=r["name"],
            center_xy=(float(r["center_xy"][0]), float(r["center_xy"][1])),
            size_xy=(float(r["size_xy"][0]), float(r["size_xy"][1])),
        )
        for r in payload["regions"]
    ]
    plane = payload.get("plane_size_xy")
    if plane is None:
        raise ValueError("layout json must contain plane_size_xy")
    return DetectorLayout(regions=regions, plane_size_xy=(float(plane[0]), float(plane[1])))


def build_region_masks(layout: DetectorLayout, N: int, dx: float) -> np.ndarray:
    """Convert physical detector layout to pixel masks.

    Args:
        layout: detector layout in meters
        N: grid size [pixels]
        dx: pixel pitch [m]

    Returns:
        masks: bool array, shape (K, N, N)
    """

    x, y = make_spatial_grid(N, dx, centered=True)
    X, Y = np.meshgrid(x, y, indexing="xy")

    masks: list[np.ndarray] = []
    for region in layout.regions:
        cx, cy = region.center_xy
        w, h = region.size_xy
        mask = (np.abs(X - cx) <= w / 2.0) & (np.abs(Y - cy) <= h / 2.0)
        masks.append(mask)

    return np.stack(masks, axis=0)
