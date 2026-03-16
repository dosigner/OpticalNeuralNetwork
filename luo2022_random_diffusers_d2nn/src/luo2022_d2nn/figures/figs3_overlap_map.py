"""Supplementary Fig. S3: overlap map of phase islands."""

from __future__ import annotations

from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import torch
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    binary_opening,
    gaussian_filter,
    label,
)

from luo2022_d2nn.config.schema import load_and_validate_config
from luo2022_d2nn.utils.viz import save_figure

_PAIRWISE_LAYER_INDICES = ((0, 1), (1, 2), (2, 3), (3, 0))
_ISLAND_SIGMA_PX = 2.5
_ISLAND_PERCENTILE = 90.0
_ISLAND_MIN_AREA_PX = 6
_ISLAND_MAX_AREA_PX = 300


def _load_wrapped_phases(checkpoint_path: str) -> np.ndarray:
    """Load wrapped phase maps from a checkpoint."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = state["model_state_dict"] if "model_state_dict" in state else state

    phase_keys = sorted(key for key in state_dict if key.endswith(".phase"))
    if len(phase_keys) != 4:
        raise ValueError(
            f"Supplementary Fig. S3 expects 4 phase layers, found {len(phase_keys)}"
        )

    wrapped = []
    for key in phase_keys:
        phase = state_dict[key].detach().cpu().numpy()
        wrapped.append(np.mod(phase, 2.0 * np.pi).astype(np.float32))
    return np.stack(wrapped, axis=0)


def _make_circular_roi(shape: tuple[int, int], radius_px: float) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    cy = (shape[0] - 1) / 2.0
    cx = (shape[1] - 1) / 2.0
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return rr <= radius_px


def _compute_island_mask(
    wrapped_phase: np.ndarray,
    roi_mask: np.ndarray,
    sigma_px: float = _ISLAND_SIGMA_PX,
    percentile: float = _ISLAND_PERCENTILE,
    min_area_px: int = _ISLAND_MIN_AREA_PX,
    max_area_px: int = _ISLAND_MAX_AREA_PX,
) -> np.ndarray:
    """Extract smooth phase islands using local phase coherence."""
    phasor = np.exp(1j * wrapped_phase)
    coherence = np.abs(
        gaussian_filter(phasor.real, sigma=sigma_px)
        + 1j * gaussian_filter(phasor.imag, sigma=sigma_px)
    )

    threshold = float(np.percentile(coherence[roi_mask], percentile))
    mask = (coherence >= threshold) & roi_mask

    structure = np.ones((2, 2), dtype=bool)
    mask = binary_opening(mask, structure=structure)
    mask = binary_closing(mask, structure=structure)
    mask = binary_fill_holes(mask)

    labeled, num_components = label(mask)
    filtered = np.zeros_like(mask, dtype=bool)
    for component_idx in range(1, num_components + 1):
        component = labeled == component_idx
        area = int(component.sum())
        if min_area_px <= area <= max_area_px:
            filtered |= component
    return filtered


def _compute_overlap_counts(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return pairwise overlap counts and the 4-layer count map."""
    if masks.shape[0] != 4:
        raise ValueError(f"Expected 4 masks, received {masks.shape[0]}")

    pairwise = []
    for left_idx, right_idx in _PAIRWISE_LAYER_INDICES:
        pairwise.append(masks[left_idx].astype(np.uint8) + masks[right_idx].astype(np.uint8))
    pairwise_counts = np.stack(pairwise, axis=0).astype(np.uint8)
    all_layer_count_map = np.sum(masks.astype(np.uint8), axis=0, dtype=np.uint8)
    return pairwise_counts, all_layer_count_map


def _add_scale_bar_lambda(
    ax,
    image_shape: tuple[int, int],
    bar_length_px: float,
    label_text: str = r"10 $\lambda$",
    color: str = "white",
) -> None:
    """Add a scale bar in pixel units with lambda-based label."""
    height, width = image_shape
    x_start = 0.08 * width
    x_end = x_start + bar_length_px
    y_pos = 0.88 * height

    ax.plot([x_start, x_end], [y_pos, y_pos], color=color, linewidth=3, solid_capstyle="butt")
    ax.text(
        x_start,
        y_pos - 8,
        label_text,
        color=color,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def make_figs3(
    checkpoint_path: str,
    config_path: str = "configs/baseline.yaml",
    save_path: str | None = None,
) -> dict[str, Any]:
    """Generate Supplementary Fig. S3 overlap map."""
    cfg = load_and_validate_config(config_path)
    num_layers = int(cfg["geometry"]["num_layers"])
    if num_layers != 4:
        raise ValueError(
            f"Supplementary Fig. S3 only supports 4-layer models, got {num_layers}"
        )

    wrapped_phases = _load_wrapped_phases(checkpoint_path)

    wavelength_mm = float(cfg["optics"]["wavelength_mm"])
    dx_mm = float(cfg["grid"]["pitch_mm"])
    roi_radius_px = 0.5 * (80.0 * wavelength_mm / dx_mm)
    scale_bar_px = 10.0 * wavelength_mm / dx_mm

    roi_mask = _make_circular_roi(wrapped_phases.shape[1:], radius_px=roi_radius_px)
    masks = np.stack(
        [_compute_island_mask(phase, roi_mask) for phase in wrapped_phases],
        axis=0,
    )
    pairwise_counts, all_layer_count_map = _compute_overlap_counts(masks)

    pairwise_cmap = ListedColormap(["black", "white", "red"])
    pairwise_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], pairwise_cmap.N)
    all_layers_cmap = ListedColormap(["black", "#555555", "#aaaaaa", "white", "red"])
    all_layers_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], all_layers_cmap.N)

    fig = plt.figure(figsize=(12, 12), facecolor="white")
    grid = fig.add_gridspec(
        4,
        5,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 0.06],
        height_ratios=[1.0, 1.0, 1.0, 1.3],
        wspace=0.16,
        hspace=0.22,
    )

    phase_axes = [fig.add_subplot(grid[0, col]) for col in range(4)]
    mask_axes = [fig.add_subplot(grid[1, col]) for col in range(4)]
    pair_axes = [fig.add_subplot(grid[2, col]) for col in range(4)]
    all_layers_ax = fig.add_subplot(grid[3, 1:3])

    phase_cax = fig.add_subplot(grid[0, 4])
    mask_cax = fig.add_subplot(grid[1, 4])
    pair_cax = fig.add_subplot(grid[2, 4])
    all_layers_cax = fig.add_subplot(grid[3, 4])

    pair_titles = ["Layer 1-2", "Layer 2-3", "Layer 3-4", "Layer 4-1"]

    for idx, ax in enumerate(phase_axes):
        im = ax.imshow(
            wrapped_phases[idx],
            cmap="viridis",
            vmin=0.0,
            vmax=2.0 * np.pi,
            interpolation="nearest",
        )
        ax.set_title(f"Layer {idx + 1}", fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        _add_scale_bar_lambda(ax, wrapped_phases[idx].shape, scale_bar_px)
    phase_axes[0].set_ylabel("Diffractive layers", fontsize=11, fontweight="bold")

    phase_cbar = fig.colorbar(im, cax=phase_cax, ticks=[0, np.pi, 2.0 * np.pi])
    phase_cbar.ax.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])

    for idx, ax in enumerate(mask_axes):
        im = ax.imshow(masks[idx], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"Layer {idx + 1}", fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
    mask_axes[0].set_ylabel("Phase islands", fontsize=11, fontweight="bold")
    mask_cbar = fig.colorbar(im, cax=mask_cax, ticks=[0, 1])
    mask_cbar.ax.set_yticklabels(["0", "1"])

    for idx, ax in enumerate(pair_axes):
        im = ax.imshow(
            pairwise_counts[idx],
            cmap=pairwise_cmap,
            norm=pairwise_norm,
            interpolation="nearest",
        )
        ax.set_title(pair_titles[idx], fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        _add_scale_bar_lambda(ax, pairwise_counts[idx].shape, scale_bar_px)
    pair_axes[0].set_ylabel("Overlap map", fontsize=11, fontweight="bold")
    pair_cbar = fig.colorbar(im, cax=pair_cax, ticks=[0, 1, 2])
    pair_cbar.ax.set_yticklabels(["0", "1", "2"])

    im = all_layers_ax.imshow(
        all_layer_count_map,
        cmap=all_layers_cmap,
        norm=all_layers_norm,
        interpolation="nearest",
    )
    all_layers_ax.set_title("All four layers", fontsize=12, fontweight="bold")
    all_layers_ax.set_xticks([])
    all_layers_ax.set_yticks([])
    _add_scale_bar_lambda(all_layers_ax, all_layer_count_map.shape, scale_bar_px)
    all_layers_cbar = fig.colorbar(im, cax=all_layers_cax, ticks=[0, 1, 2, 3, 4])
    all_layers_cbar.ax.set_yticklabels(["0", "1", "2", "3", "4"])

    fig.suptitle(
        "Supp. Fig. S3: Overlap Map of Phase Islands on Successive Diffractive Layers",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    if save_path is not None:
        save_figure(fig, save_path)

    plt.close(fig)
    return {
        "wrapped_phases": wrapped_phases,
        "masks": masks,
        "pairwise_counts": pairwise_counts,
        "all_layer_count_map": all_layer_count_map,
    }
