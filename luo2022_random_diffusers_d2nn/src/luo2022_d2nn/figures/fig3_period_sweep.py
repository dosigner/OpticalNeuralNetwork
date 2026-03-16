"""Fig 3 -- Grating period recovery vs training diffuser count n.

Evaluates how accurately the D2NN resolves 3-bar grating targets of
varying period when trained with different numbers of diffusers per
epoch.  Two sub-panels:
  (a) Test with the *last n* diffusers from training (known).
  (b) Test with 20 *new* (blind) diffusers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from luo2022_d2nn.config.schema import load_and_validate_config
from luo2022_d2nn.data.resolution_targets import (
    SUPPORTED_PERIODS_MM,
    generate_grating_target,
)
from luo2022_d2nn.diffuser.random_phase import generate_diffuser
from luo2022_d2nn.eval.grating_period import estimate_grating_period
from luo2022_d2nn.models.d2nn import D2NN
from luo2022_d2nn.optics.bl_asm import bl_asm_propagate, bl_asm_transfer_function
from luo2022_d2nn.utils.viz import save_figure


# Standard test periods (mm) from the paper
TEST_PERIODS_MM: list[float] = SUPPORTED_PERIODS_MM  # [7.2, 8.4, 9.6, 10.8, 12.0]

# Number of blind diffusers to test in panel (b)
N_BLIND_DIFFUSERS = 20

# Seed offset so blind seeds never overlap with training
_BLIND_SEED_BASE = 77777777

_BAR_COLORS = [
    "#1f77b4",  # blue
    "#e6550d",  # orange-red
    "#f1b514",  # golden yellow
    "#8e24aa",  # purple
]
_TRUE_PERIOD_COLOR = "#00ff33"
_TITLE_COLORS = {
    "known": "#2b8cde",
    "blind": "#dca41b",
}
_Y_LIMS = (4.0, 15.0)
_Y_TICKS = np.arange(4.0, 16.0, 2.0)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _load_model(checkpoint_path: str, cfg: dict, device: torch.device) -> D2NN:
    """Instantiate D2NN from *cfg* and load checkpoint weights."""
    geom = cfg["geometry"]
    grid = cfg["grid"]
    model_cfg = cfg["model"]

    model = D2NN(
        num_layers=int(geom["num_layers"]),
        grid_size=int(grid["nx"]),
        dx_mm=float(grid["pitch_mm"]),
        wavelength_mm=float(cfg["optics"]["wavelength_mm"]),
        diffuser_to_layer1_mm=float(geom["diffuser_to_layer1_mm"]),
        layer_to_layer_mm=float(geom["layer_to_layer_mm"]),
        last_layer_to_output_mm=float(geom["last_layer_to_output_mm"]),
        pad_factor=int(grid.get("pad_factor", 2)),
        init_phase_dist=model_cfg.get("init_phase_distribution", "uniform_0_2pi"),
    )

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device).eval()
    return model


def _make_diffuser(cfg: dict, seed: int, device: torch.device) -> torch.Tensor:
    """Return a single diffuser transmittance tensor."""
    grid = cfg["grid"]
    diff_cfg = cfg["diffuser"]
    result = generate_diffuser(
        int(grid["nx"]),
        float(grid["pitch_mm"]),
        float(cfg["optics"]["wavelength_mm"]),
        delta_n=float(diff_cfg.get("delta_n", 0.74)),
        height_mean_lambda=float(diff_cfg.get("height_mean_lambda", 25.0)),
        height_std_lambda=float(diff_cfg.get("height_std_lambda", 8.0)),
        smoothing_sigma_lambda=float(diff_cfg.get("smoothing_sigma_lambda", 4.0)),
        seed=seed,
        device=device,
    )
    return result["transmittance"]


def _known_diffuser_seeds(cfg: dict, n: int) -> list[int]:
    """Seed list for the last-epoch diffusers (known)."""
    base_seed = int(cfg["experiment"]["seed"])
    last_epoch = int(cfg["training"]["epochs"]) - 1
    epoch_seed = base_seed + last_epoch
    return [epoch_seed * 1000 + i for i in range(n)]


def _blind_diffuser_seeds(count: int) -> list[int]:
    """Seed list for fresh blind diffusers."""
    return [_BLIND_SEED_BASE + i for i in range(count)]


def _forward_grating(
    grating: torch.Tensor,
    diffuser_t: torch.Tensor,
    model: D2NN,
    H_obj_to_diff: torch.Tensor,
    pad_factor: int,
) -> torch.Tensor:
    """Forward a grating target through diffuser + D2NN, return intensity."""
    field = grating.unsqueeze(0).to(torch.complex64)
    field = bl_asm_propagate(field, H_obj_to_diff, pad_factor=pad_factor)
    field = field * diffuser_t.unsqueeze(0)
    with torch.no_grad():
        out = model(field)
    return out.abs() ** 2  # (1, N, N)


def _compute_period_stats(
    est_dict: dict[int, dict[float, list[float]]],
    n_values: list[int],
) -> tuple[dict[int, list[float]], dict[int, list[float]]]:
    """Compute mean/std statistics for each n and target period."""
    means_by_n: dict[int, list[float]] = {}
    stds_by_n: dict[int, list[float]] = {}
    for n in n_values:
        means = []
        stds = []
        for p in TEST_PERIODS_MM:
            vals = np.asarray(est_dict[n][p], dtype=float)
            vals = vals[np.isfinite(vals)]
            means.append(float(np.mean(vals)) if len(vals) else float("nan"))
            stds.append(float(np.std(vals)) if len(vals) else float("nan"))
        means_by_n[n] = means
        stds_by_n[n] = stds
    return means_by_n, stds_by_n


def _plot_period_panel(
    ax: plt.Axes,
    est_dict: dict[int, dict[float, list[float]]],
    n_values: list[int],
    title: str,
    title_color: str,
) -> None:
    """Render a single paper-style Fig. 3 panel."""
    means_by_n, stds_by_n = _compute_period_stats(est_dict, n_values)

    x = np.arange(len(TEST_PERIODS_MM), dtype=float)
    bar_width = 0.16
    offsets = np.linspace(-1.5 * bar_width, 1.5 * bar_width, len(n_values))

    for idx, period in enumerate(TEST_PERIODS_MM):
        x0 = x[idx] + offsets[0] - 0.06
        x1 = x[idx] + offsets[-1] + bar_width + 0.06
        label = "True Period" if idx == 0 else None
        ax.hlines(
            y=period,
            xmin=x0,
            xmax=x1,
            colors=_TRUE_PERIOD_COLOR,
            linestyles="--",
            linewidth=2.5,
            label=label,
            zorder=4,
        )

    for idx, n in enumerate(n_values):
        ax.bar(
            x + offsets[idx],
            means_by_n[n],
            width=bar_width,
            yerr=stds_by_n[n],
            color=_BAR_COLORS[idx],
            edgecolor="#333333",
            linewidth=1.0,
            capsize=5,
            label=f"n = {n}",
            zorder=3,
        )

    ax.set_title(title, fontsize=15, fontweight="bold", color=title_color, pad=10)
    ax.set_xlabel("Resolution Test Target Period, mm", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:g}" for p in TEST_PERIODS_MM], fontsize=10)
    ax.set_ylim(*_Y_LIMS)
    ax.set_yticks(_Y_TICKS)
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=10)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#666666")
    ax.set_axisbelow(True)


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def make_fig3(
    checkpoint_paths: dict[int, str],
    config_path: str = "configs/baseline.yaml",
    save_path: str | None = None,
) -> dict[str, Any]:
    """Generate Fig 3: grating period recovery for different *n* values.

    Parameters
    ----------
    checkpoint_paths : dict
        Mapping  ``n -> checkpoint_path``, e.g.
        ``{1: "runs/n1/model.pt", 10: "...", 15: "...", 20: "..."}``.
    config_path : str
        Path to the YAML config.
    save_path : str or None
        If given, saves figure PNG and raw-data ``.npy``.

    Returns
    -------
    dict
        Raw results with keys ``n_values``, ``periods``,
        ``known_estimated``, ``blind_estimated``.
    """
    device = torch.device("cpu")
    cfg = load_and_validate_config(config_path)

    grid = cfg["grid"]
    N = int(grid["nx"])
    dx_mm = float(grid["pitch_mm"])
    wavelength_mm = float(cfg["optics"]["wavelength_mm"])
    pad_factor = int(grid.get("pad_factor", 2))
    obj_to_diff_mm = float(cfg["geometry"]["object_to_diffuser_mm"])
    ds_cfg = cfg["dataset"]

    H_obj_to_diff = bl_asm_transfer_function(
        N, dx_mm, wavelength_mm, obj_to_diff_mm, pad_factor=pad_factor,
    )

    # Build grating targets --------------------------------------------------
    gratings: dict[float, torch.Tensor] = {}
    for p in TEST_PERIODS_MM:
        g = generate_grating_target(
            period_mm=p,
            dx_mm=dx_mm,
            active_size=int(ds_cfg.get("resize_to_px", 160)),
            final_size=N,
        ).squeeze(0)  # (N, N)
        gratings[p] = g

    n_values = sorted(checkpoint_paths.keys())

    # Results containers: known_estimated[n][period] = list-of-estimates
    known_estimated: dict[int, dict[float, list[float]]] = {}
    blind_estimated: dict[int, dict[float, list[float]]] = {}

    for n in n_values:
        model = _load_model(checkpoint_paths[n], cfg, device)

        # --- Panel (a): known diffusers ------------------------------------
        k_seeds = _known_diffuser_seeds(cfg, n)
        known_estimated[n] = {p: [] for p in TEST_PERIODS_MM}

        for seed in k_seeds:
            t_d = _make_diffuser(cfg, seed, device)
            for p in TEST_PERIODS_MM:
                I_out = _forward_grating(gratings[p], t_d, model, H_obj_to_diff, pad_factor)
                try:
                    p_hat = estimate_grating_period(I_out, dx_mm=dx_mm)
                except RuntimeError:
                    p_hat = float("nan")
                known_estimated[n][p].append(p_hat)

        # --- Panel (b): blind diffusers ------------------------------------
        b_seeds = _blind_diffuser_seeds(N_BLIND_DIFFUSERS)
        blind_estimated[n] = {p: [] for p in TEST_PERIODS_MM}

        for seed in b_seeds:
            t_d = _make_diffuser(cfg, seed, device)
            for p in TEST_PERIODS_MM:
                I_out = _forward_grating(gratings[p], t_d, model, H_obj_to_diff, pad_factor)
                try:
                    p_hat = estimate_grating_period(I_out, dx_mm=dx_mm)
                except RuntimeError:
                    p_hat = float("nan")
                blind_estimated[n][p].append(p_hat)

    # ---- Plotting -----------------------------------------------------------
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.0, 4.8), sharey=True)

    _plot_period_panel(
        ax_a,
        known_estimated,
        n_values,
        title="All-Optical Imaging Through\nLast n Diffusers in Training",
        title_color=_TITLE_COLORS["known"],
    )
    _plot_period_panel(
        ax_b,
        blind_estimated,
        n_values,
        title="All-Optical Imaging Through\n20 New Diffusers",
        title_color=_TITLE_COLORS["blind"],
    )

    ax_a.set_ylabel("Measured Grating Period, mm", fontsize=11)
    ax_b.set_ylabel("Measured Grating Period, mm", fontsize=11)
    ax_b.tick_params(labelleft=True)

    legend_handles = [
        Line2D([0], [0], color=_BAR_COLORS[idx], lw=12, label=f"n = {n_values[idx]}")
        for idx in range(len(n_values))
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=_TRUE_PERIOD_COLOR,
            linestyle="--",
            linewidth=2.5,
            label="True Period",
        )
    )
    ax_b.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=10)

    ax_a.text(-0.20, 1.05, "(a)", transform=ax_a.transAxes, fontsize=14, fontfamily="serif")
    ax_b.text(-0.20, 1.05, "(b)", transform=ax_b.transAxes, fontsize=14, fontfamily="serif")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.18, wspace=0.50)

    if save_path is not None:
        save_figure(fig, save_path)
        npy_path = Path(save_path).with_suffix(".npy")
        raw = {
            "n_values": n_values,
            "periods": TEST_PERIODS_MM,
            "known_estimated": {
                n: {p: known_estimated[n][p] for p in TEST_PERIODS_MM} for n in n_values
            },
            "blind_estimated": {
                n: {p: blind_estimated[n][p] for p in TEST_PERIODS_MM} for n in n_values
            },
        }
        np.save(str(npy_path), raw)

    plt.close(fig)
    return {
        "n_values": n_values,
        "periods": TEST_PERIODS_MM,
        "known_estimated": known_estimated,
        "blind_estimated": blind_estimated,
    }
