"""Phase-to-height export utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from d2nn.physics.materials import phase_to_height as _phase_to_height


def phase_to_height(phase: np.ndarray, wavelength: float, delta_n: float) -> np.ndarray:
    """Convert phase [rad] to height [m]."""

    return _phase_to_height(phase=phase, wavelength=wavelength, delta_n=delta_n)


def export_height_map(path: str | Path, phase: np.ndarray, wavelength: float, delta_n: float, *, quantization_levels: int | None = None) -> np.ndarray:
    """Convert phase to height map and save it as .npy.

    Args:
        path: output .npy path
        phase: phase map [rad], shape (N, N)
        wavelength: [m]
        delta_n: refractive index difference
        quantization_levels: optional number of quantization levels
    """

    height = phase_to_height(phase, wavelength, delta_n)
    if quantization_levels is not None and quantization_levels > 1:
        h_min = height.min()
        h_max = height.max()
        if h_max > h_min:
            scaled = (height - h_min) / (h_max - h_min)
            q = np.round(scaled * (quantization_levels - 1)) / float(quantization_levels - 1)
            height = q * (h_max - h_min) + h_min

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, height)
    return height
