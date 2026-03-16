"""Explanation figure for Fig. 3 period semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import find_peaks

from luo2022_d2nn.config.schema import load_and_validate_config
from luo2022_d2nn.data.resolution_targets import generate_grating_target
from luo2022_d2nn.eval.grating_period import estimate_grating_period
from luo2022_d2nn.figures.fig3_period_sweep import (
    _BLIND_SEED_BASE,
    _forward_grating,
    _load_model,
    _make_diffuser,
)
from luo2022_d2nn.optics.bl_asm import bl_asm_propagate, bl_asm_transfer_function
from luo2022_d2nn.utils.viz import contrast_enhance, save_figure


EXPLANATION_PERIODS_MM = [7.2, 10.8, 12.0]


def _forward_grating_free_space(
    grating: torch.Tensor,
    diffuser_t: torch.Tensor,
    H_obj_to_diff: torch.Tensor,
    H_diff_to_out: torch.Tensor,
    pad_factor: int,
) -> torch.Tensor:
    """Forward a grating through diffuser-only free-space propagation."""
    field = grating.unsqueeze(0).to(torch.complex64)
    field = bl_asm_propagate(field, H_obj_to_diff, pad_factor=pad_factor)
    field = field * diffuser_t.unsqueeze(0)
    field = bl_asm_propagate(field, H_diff_to_out, pad_factor=pad_factor)
    return field.abs() ** 2


def _extract_profile_and_peak_positions(
    intensity: torch.Tensor,
    dx_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return averaged profile, coordinate axis, and three representative peaks."""
    img = intensity.detach().cpu().numpy()
    if img.ndim == 3:
        img = img.squeeze(0)

    profile = img.mean(axis=1)
    coords_mm = np.arange(profile.size, dtype=float) * dx_mm

    peak_indices, properties = find_peaks(
        profile,
        height=float(profile.max()) * 0.25,
        prominence=float(profile.max()) * 0.10,
    )
    if peak_indices.size >= 3:
        top3 = np.argsort(properties["peak_heights"])[-3:]
        chosen = np.sort(peak_indices[top3])
    else:
        order = np.argsort(profile)[::-1]
        chosen_list: list[int] = []
        min_sep = max(2, int(round(4.0 / dx_mm)))
        for idx in order:
            if all(abs(idx - prev) >= min_sep for prev in chosen_list):
                chosen_list.append(int(idx))
            if len(chosen_list) == 3:
                break
        chosen = np.sort(np.asarray(chosen_list[:3], dtype=int))

    return profile, coords_mm, coords_mm[chosen]


def make_fig3_explanation(
    checkpoint_path: str,
    config_path: str = "configs/baseline.yaml",
    save_path: str | None = None,
) -> dict[str, Any]:
    """Generate a companion figure explaining Fig. 3 axis semantics."""
    device = torch.device("cpu")
    cfg = load_and_validate_config(config_path)

    grid = cfg["grid"]
    N = int(grid["nx"])
    dx_mm = float(grid["pitch_mm"])
    wavelength_mm = float(cfg["optics"]["wavelength_mm"])
    pad_factor = int(grid.get("pad_factor", 2))
    geom = cfg["geometry"]
    obj_to_diff_mm = float(geom["object_to_diffuser_mm"])
    resize_to = int(cfg["dataset"].get("resize_to_px", 160))

    model = _load_model(checkpoint_path, cfg, device)
    diffuser_t = _make_diffuser(cfg, _BLIND_SEED_BASE, device)
    H_obj_to_diff = bl_asm_transfer_function(
        N, dx_mm, wavelength_mm, obj_to_diff_mm, pad_factor=pad_factor,
    )
    free_space_total_mm = (
        float(geom["diffuser_to_layer1_mm"])
        + (int(geom["num_layers"]) - 1) * float(geom["layer_to_layer_mm"])
        + float(geom["last_layer_to_output_mm"])
    )
    H_diff_to_out = bl_asm_transfer_function(
        N, dx_mm, wavelength_mm, free_space_total_mm, pad_factor=pad_factor,
    )

    fig, axes = plt.subplots(
        len(EXPLANATION_PERIODS_MM),
        4,
        figsize=(14.4, 9.2),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 1.35]},
    )
    col_titles = [
        "Input Resolution Target",
        "Propagation Through Diffuser",
        "D2NN Reconstruction",
        "Averaged Profile",
    ]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=14, fontweight="bold", pad=12)

    measured_periods: list[float] = []
    peak_positions_by_row: list[list[float]] = []
    free_space_images: list[np.ndarray] = []
    reconstruction_images: list[np.ndarray] = []

    for row_idx, period_mm in enumerate(EXPLANATION_PERIODS_MM):
        target = generate_grating_target(
            period_mm=period_mm,
            dx_mm=dx_mm,
            active_size=resize_to,
            final_size=N,
        ).squeeze(0)
        free_space = _forward_grating_free_space(
            target,
            diffuser_t,
            H_obj_to_diff,
            H_diff_to_out,
            pad_factor,
        )
        reconstruction = _forward_grating(target, diffuser_t, model, H_obj_to_diff, pad_factor)
        measured_period = estimate_grating_period(reconstruction, dx_mm=dx_mm)
        profile, coords_mm, peak_positions_mm = _extract_profile_and_peak_positions(
            reconstruction,
            dx_mm=dx_mm,
        )

        measured_periods.append(measured_period)
        peak_positions_by_row.append(peak_positions_mm.tolist())
        free_space_images.append(free_space.squeeze(0).detach().cpu().numpy())
        reconstruction_images.append(reconstruction.squeeze(0).detach().cpu().numpy())

        ax_target = axes[row_idx, 0]
        ax_target.imshow(target.detach().cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        ax_target.set_ylabel(f"True period\n{period_mm:.1f} mm", fontsize=12)

        ax_free = axes[row_idx, 1]
        ax_free.imshow(
            contrast_enhance(free_space.squeeze(0)),
            cmap="inferno",
            vmin=0.0,
            vmax=1.0,
        )

        ax_recon = axes[row_idx, 2]
        ax_recon.imshow(
            contrast_enhance(reconstruction.squeeze(0)),
            cmap="inferno",
            vmin=0.0,
            vmax=1.0,
        )

        ax_profile = axes[row_idx, 3]
        ax_profile.plot(coords_mm, profile, color="#1f1f1f", linewidth=2.0)
        ax_profile.scatter(
            peak_positions_mm,
            np.interp(peak_positions_mm, coords_mm, profile),
            color="#d62728",
            s=30,
            zorder=3,
        )
        for peak_mm in peak_positions_mm:
            ax_profile.axvline(peak_mm, color="#d62728", linestyle=":", linewidth=1.0)
        ax_profile.set_xlim(coords_mm[0], coords_mm[-1])
        ax_profile.set_xlabel("Output-plane position, mm", fontsize=10)
        ax_profile.set_ylabel("Mean intensity", fontsize=10)
        ax_profile.text(
            0.03,
            0.95,
            f"True period = {period_mm:.1f} mm\nMeasured period = {measured_period:.2f} mm",
            transform=ax_profile.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#999999"},
        )

        for ax in (ax_target, ax_free, ax_recon):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color("#666666")

    fig.text(
        0.5,
        0.04,
        "In Fig. 3, the Resolution Test Target Period is the known spacing of the input bars, "
        "the diffuser-only panel shows the degraded baseline before learning, and the Measured "
        "Grating Period is the spacing read from the D2NN reconstructed output profile.",
        ha="center",
        va="center",
        fontsize=11,
    )
    fig.tight_layout(rect=[0.03, 0.08, 0.98, 0.98])

    if save_path is not None:
        save_figure(fig, save_path)

    result = {
        "periods_mm": EXPLANATION_PERIODS_MM,
        "free_space_images": free_space_images,
        "reconstruction_images": reconstruction_images,
        "measured_periods_mm": measured_periods,
        "peak_positions_mm": peak_positions_by_row,
    }
    plt.close(fig)
    return result
