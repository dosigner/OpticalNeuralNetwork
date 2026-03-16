"""Supplementary Fig. S1: trained D2NN layer phase patterns."""

from __future__ import annotations

from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from luo2022_d2nn.config.schema import load_and_validate_config
from luo2022_d2nn.utils.viz import save_figure


def _load_wrapped_phases(checkpoint_path: str) -> np.ndarray:
    """Load wrapped phase maps from a checkpoint, shape (L, N, N)."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = state["model_state_dict"] if "model_state_dict" in state else state

    phase_keys = sorted(key for key in state_dict if key.endswith(".phase"))
    wrapped = []
    for key in phase_keys:
        phase = state_dict[key].detach().cpu().numpy()
        wrapped.append(np.mod(phase, 2.0 * np.pi).astype(np.float32))
    return np.stack(wrapped, axis=0)


def make_figs1(
    checkpoint_path: str,
    config_path: str = "configs/baseline.yaml",
    save_path: str | None = None,
    title_suffix: str = "",
) -> dict[str, Any]:
    """Generate Supplementary Fig. S1: layer phase patterns."""
    cfg = load_and_validate_config(config_path)
    wavelength_mm = float(cfg["optics"]["wavelength_mm"])
    dx_mm = float(cfg["grid"]["pitch_mm"])
    scale_bar_px = 10.0 * wavelength_mm / dx_mm

    wrapped_phases = _load_wrapped_phases(checkpoint_path)
    num_layers = wrapped_phases.shape[0]

    fig, axes = plt.subplots(
        1, num_layers, figsize=(4 * num_layers + 1, 4), facecolor="white",
    )
    if num_layers == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        im = ax.imshow(
            wrapped_phases[idx],
            cmap="hsv",
            vmin=0.0,
            vmax=2.0 * np.pi,
            interpolation="nearest",
        )
        ax.set_title(f"Layer {idx + 1}", fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        # Scale bar
        h, w = wrapped_phases[idx].shape
        x_start = 0.08 * w
        x_end = x_start + scale_bar_px
        y_pos = 0.88 * h
        ax.plot([x_start, x_end], [y_pos, y_pos], color="white",
                linewidth=3, solid_capstyle="butt")
        ax.text(x_start, y_pos - 8, r"10$\lambda$", color="white",
                fontsize=9, fontweight="bold", ha="left", va="bottom")

    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_ticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    cbar.set_ticklabels(["0", r"$\pi$/2", r"$\pi$", r"3$\pi$/2", r"$2\pi$"])
    cbar.set_label("Phase (rad)", fontsize=11)

    title = "Supp. Fig. S1: Trained D2NN Layer Phase Patterns"
    if title_suffix:
        title += f" ({title_suffix})"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.subplots_adjust(right=0.88)

    if save_path is not None:
        save_figure(fig, save_path)

    plt.close(fig)
    return {"wrapped_phases": wrapped_phases}


def make_figs1_comparison(
    checkpoint_paths: dict[str, str],
    config_path: str = "configs/baseline.yaml",
    save_path: str | None = None,
) -> dict[str, Any]:
    """Compare layer phases across different checkpoints (e.g. B=64 vs B=4).

    Parameters
    ----------
    checkpoint_paths : dict
        e.g. {"B=64 (100ep)": "runs/n20_L4/model.pt",
               "B=4 (100ep)": "runs_b4/n20_L4/model.pt"}
    """
    cfg = load_and_validate_config(config_path)
    wavelength_mm = float(cfg["optics"]["wavelength_mm"])
    dx_mm = float(cfg["grid"]["pitch_mm"])
    scale_bar_px = 10.0 * wavelength_mm / dx_mm

    labels = list(checkpoint_paths.keys())
    all_phases = {}
    num_layers = None
    for lbl, ckpt_path in checkpoint_paths.items():
        phases = _load_wrapped_phases(ckpt_path)
        all_phases[lbl] = phases
        if num_layers is None:
            num_layers = phases.shape[0]

    nrows = len(labels)
    ncols = num_layers

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols + 1, 3.5 * nrows),
        facecolor="white", squeeze=False,
    )

    for row, lbl in enumerate(labels):
        phases = all_phases[lbl]
        for col in range(ncols):
            ax = axes[row, col]
            im = ax.imshow(
                phases[col],
                cmap="hsv",
                vmin=0.0,
                vmax=2.0 * np.pi,
                interpolation="nearest",
            )
            if row == 0:
                ax.set_title(f"Layer {col + 1}", fontsize=12, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

            # Scale bar
            h, w = phases[col].shape
            x_start = 0.08 * w
            x_end = x_start + scale_bar_px
            y_pos = 0.88 * h
            ax.plot([x_start, x_end], [y_pos, y_pos], color="white",
                    linewidth=3, solid_capstyle="butt")
            ax.text(x_start, y_pos - 8, r"10$\lambda$", color="white",
                    fontsize=9, fontweight="bold", ha="left", va="bottom")

        axes[row, 0].set_ylabel(lbl, fontsize=11, fontweight="bold")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
    cbar.set_ticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    cbar.set_ticklabels(["0", r"$\pi$/2", r"$\pi$", r"3$\pi$/2", r"$2\pi$"])
    cbar.set_label("Phase (rad)", fontsize=11)

    fig.suptitle(
        "Phase Pattern Comparison: Layer Phases Across Training Conditions",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.subplots_adjust(right=0.88)

    if save_path is not None:
        save_figure(fig, save_path)

    plt.close(fig)
    return {"all_phases": all_phases}
