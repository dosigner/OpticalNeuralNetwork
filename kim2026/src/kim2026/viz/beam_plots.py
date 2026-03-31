"""Compact field-plot helpers for beam-cleanup evaluation outputs."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


plt.rcParams.update({
    "font.size": 9,
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
})


def _to_numpy(field: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(field, torch.Tensor):
        return field.detach().cpu().numpy()
    return np.asarray(field)


def _normalized_intensity(field: torch.Tensor | np.ndarray, *, reference_max: float | None = None) -> np.ndarray:
    array = _to_numpy(field)
    intensity = np.abs(array) ** 2
    denom = float(reference_max) if reference_max is not None else float(intensity.max())
    return intensity / max(denom, 1e-12)


def _phase(field: torch.Tensor | np.ndarray) -> np.ndarray:
    return np.angle(_to_numpy(field))


def save_triptych(
    path: str | Path,
    *,
    input_field: torch.Tensor | np.ndarray,
    baseline_field: torch.Tensor | np.ndarray,
    pred_field: torch.Tensor | np.ndarray,
    target_field: torch.Tensor | np.ndarray,
    vacuum_field: torch.Tensor | np.ndarray | None = None,
    title: str | None = None,
) -> Path:
    """Save a comparison figure for one representative sample field."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    panels = [
        ("Input", input_field),
        ("Baseline", baseline_field),
        ("Prediction", pred_field),
        ("Target", target_field),
    ]
    if vacuum_field is not None:
        panels.insert(1, ("Vacuum", vacuum_field))

    target_norm = _normalized_intensity(target_field)
    reference_max = float((np.abs(_to_numpy(target_field)) ** 2).max())
    fig, axes = plt.subplots(2, len(panels), figsize=(3.2 * len(panels), 5.2), squeeze=False)

    for idx, (label, field) in enumerate(panels):
        intensity = _normalized_intensity(field, reference_max=reference_max)
        phase = _phase(field)

        im0 = axes[0, idx].imshow(intensity, origin="lower", cmap="inferno", vmin=0.0, vmax=1.0)
        axes[0, idx].set_title(label)
        axes[0, idx].set_xticks([])
        axes[0, idx].set_yticks([])
        fig.colorbar(im0, ax=axes[0, idx], fraction=0.046, pad=0.02)

        if label == "Target":
            phase_view = phase
        else:
            phase_view = (phase - _phase(target_field) + math.pi) % (2.0 * math.pi) - math.pi
        im1 = axes[1, idx].imshow(phase_view, origin="lower", cmap="twilight_shifted", vmin=-math.pi, vmax=math.pi)
        axes[1, idx].set_xticks([])
        axes[1, idx].set_yticks([])
        fig.colorbar(im1, ax=axes[1, idx], fraction=0.046, pad=0.02)

        if idx == 0:
            axes[0, idx].set_ylabel("Normalized Irradiance")
            axes[1, idx].set_ylabel("Phase vs Target [rad]")

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
