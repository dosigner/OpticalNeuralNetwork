"""Supp. Fig S5 — Imaging through diffusers with different correlation lengths.

Networks trained with L~10λ diffusers are tested with:
  - Known diffuser (from training, L~10λ)
  - New diffuser (L~10λ, same statistics)
  - New diffuser (L~5λ, finer features, σ=2λ)

Shows reconstructions for n=1, n=10, n=15, n=20.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch

from luo2022_d2nn.config.schema import load_and_validate_config
from luo2022_d2nn.data.mnist import MNISTAmplitude
from luo2022_d2nn.diffuser.random_phase import generate_diffuser
from luo2022_d2nn.eval.pcc import compute_pcc
from luo2022_d2nn.models.d2nn import D2NN
from luo2022_d2nn.optics.bl_asm import bl_asm_propagate, bl_asm_transfer_function
from luo2022_d2nn.utils.viz import contrast_enhance, save_figure


def _load_model(checkpoint_path: str, cfg: dict, device: torch.device) -> D2NN:
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


def _forward_single(amplitude, diffuser_t, model, H_obj_to_diff, pad_factor):
    field = amplitude.unsqueeze(0).to(torch.complex64)
    field_at_diff = bl_asm_propagate(field, H_obj_to_diff, pad_factor=pad_factor)
    field_at_diff = field_at_diff * diffuser_t.unsqueeze(0)
    with torch.no_grad():
        field_out = model(field_at_diff)
    return field_out.abs() ** 2


def _get_digit3(cfg: dict) -> dict:
    ds_cfg = cfg["dataset"]
    dataset = MNISTAmplitude(
        root="data", split="test",
        resize_to=int(ds_cfg.get("resize_to_px", 160)),
        final_size=int(cfg["grid"]["nx"]),
    )
    for i in range(len(dataset)):
        sample = dataset[i]
        if int(sample["label"]) == 3:
            return sample
    raise RuntimeError("Digit 3 not found")


def make_figs5(
    checkpoint_paths: dict[str, str],
    config_path: str = "configs/baseline.yaml",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Generate Supp. Fig S5.

    Parameters
    ----------
    checkpoint_paths : dict
        Mapping n-label to checkpoint path, e.g.
        {"n=1": "runs/n1_L4/model.pt", "n=10": "runs/n10_L4/model.pt", ...}
    config_path : str
    save_path : str or None
    """
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

    viz_cfg = cfg.get("visualization", {}).get("contrast_enhancement", {})
    lo_pct = float(viz_cfg.get("lower_percentile", 1.0))
    hi_pct = float(viz_cfg.get("upper_percentile", 99.0))

    H_obj_to_diff = bl_asm_transfer_function(
        N, dx_mm, wavelength_mm, obj_to_diff_mm, pad_factor=pad_factor,
    )

    # Get digit 3
    sample = _get_digit3(cfg)
    obj_amp = sample["amplitude"].squeeze(0)  # (N, N)

    # Active region crop
    pad_each = (N - resize_to) // 2
    s = slice(pad_each, pad_each + resize_to)
    gt_crop = obj_amp[s, s].detach().cpu().numpy()
    target_crop = obj_amp[s, s].unsqueeze(0)  # (1, H, W)

    # Generate 3 diffusers
    diff_cfg = cfg["diffuser"]
    diff_common = dict(
        N=N, dx_mm=dx_mm, wavelength_mm=wavelength_mm,
        delta_n=float(diff_cfg.get("delta_n", 0.74)),
        height_mean_lambda=float(diff_cfg.get("height_mean_lambda", 25.0)),
        height_std_lambda=float(diff_cfg.get("height_std_lambda", 8.0)),
        device=device,
    )

    # Known diffuser (from last training epoch, L~10λ)
    base_seed = int(cfg["experiment"]["seed"])
    last_epoch = int(cfg["training"]["epochs"]) - 1
    epoch_seed = base_seed + last_epoch
    diff_known = generate_diffuser(
        **diff_common,
        smoothing_sigma_lambda=float(diff_cfg.get("smoothing_sigma_lambda", 4.0)),
        seed=epoch_seed * 1000,
    )

    # New diffuser L~10λ (same stats, unseen seed)
    diff_new_10 = generate_diffuser(
        **diff_common,
        smoothing_sigma_lambda=float(diff_cfg.get("smoothing_sigma_lambda", 4.0)),
        seed=88888888,
    )

    # New diffuser L~5λ (σ=2λ, paper says this gives L~5λ)
    diff_new_5 = generate_diffuser(
        **diff_common,
        smoothing_sigma_lambda=2.0,
        seed=99999999,
    )

    diffusers = [diff_known, diff_new_10, diff_new_5]
    diff_col_labels = [
        "Known Diffuser\n(Training)",
        "New Diffuser\n$L = 10\\lambda$",
        "New Diffuser\n$L = 5\\lambda$",
    ]

    # Sort model labels
    n_labels = sorted(checkpoint_paths.keys(), key=lambda x: int(x.split("=")[1]))

    # --- Forward passes ---
    # results[model_idx][diff_idx] = cropped image (H, W)
    results = []
    pccs = []
    for n_label in n_labels:
        ckpt = checkpoint_paths[n_label]
        model = _load_model(ckpt, cfg, device)
        row_results = []
        row_pccs = []
        for diff_info in diffusers:
            I_out = _forward_single(
                obj_amp, diff_info["transmittance"], model, H_obj_to_diff, pad_factor,
            )
            I_crop = I_out[:, s, s]
            pcc_val = compute_pcc(I_crop, target_crop).item()
            row_results.append(I_crop.squeeze(0).detach().cpu().numpy())
            row_pccs.append(pcc_val)
        results.append(row_results)
        pccs.append(row_pccs)

    # --- Plot ---
    n_models = len(n_labels)
    n_diff = 3
    # Rows: 1 (diffuser phase) + n_models (reconstructions)
    # Cols: 1 (GT) + 3 (diffusers)
    n_rows = 1 + n_models
    n_cols = 1 + n_diff

    fig = plt.figure(figsize=(3.0 * n_cols, 3.0 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.06, hspace=0.12,
                  left=0.08, right=0.95, top=0.90, bottom=0.02)

    # Row 0: diffuser phase maps
    # GT cell: scale bars
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor("white")
    ax0.set_xticks([])
    ax0.set_yticks([])
    for sp in ax0.spines.values():
        sp.set_visible(False)
    ax0.text(0.5, 0.6, "10$\\lambda$", transform=ax0.transAxes,
             fontsize=16, ha="center", va="center", fontweight="bold")
    ax0.text(0.5, 0.25, "0–2$\\pi$", transform=ax0.transAxes,
             fontsize=12, ha="center", va="center", color="gray")

    for d_idx, diff_info in enumerate(diffusers):
        ax = fig.add_subplot(gs[0, 1 + d_idx])
        phase = diff_info["phase_map"].detach().cpu().numpy() % (2 * np.pi)
        ax.imshow(phase, cmap="hsv", vmin=0, vmax=2 * np.pi, interpolation="nearest")
        ax.set_title(diff_col_labels[d_idx], fontsize=10, fontweight="bold", pad=5,
                     color="#1976D2" if d_idx == 0 else "#E65100")
        ax.set_xticks([])
        ax.set_yticks([])

    # Rows 1..n_models: GT + reconstructions
    for m_idx, n_label in enumerate(n_labels):
        row = 1 + m_idx

        # GT column
        ax_gt = fig.add_subplot(gs[row, 0])
        ax_gt.imshow(gt_crop, cmap="gray", vmin=0,
                     vmax=gt_crop.max() if gt_crop.max() > 0 else 1)
        ax_gt.set_xticks([])
        ax_gt.set_yticks([])
        ax_gt.set_ylabel(n_label, fontsize=12, fontweight="bold", rotation=0,
                         ha="right", va="center", labelpad=30)

        # Reconstructions
        for d_idx in range(n_diff):
            ax = fig.add_subplot(gs[row, 1 + d_idx])
            img = results[m_idx][d_idx]
            display_img = contrast_enhance(img, lo_pct, hi_pct)
            ax.imshow(display_img, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])

            pcc_val = pccs[m_idx][d_idx]
            ax.text(0.5, -0.02, f"PCC={pcc_val:.3f}", transform=ax.transAxes,
                    fontsize=11, ha="center", va="top")

    fig.suptitle("Supp. Fig. S5: Imaging through diffusers with different "
                 "correlation lengths (digit 3)",
                 fontsize=13, fontweight="bold", y=0.97)

    if save_path is not None:
        save_figure(fig, save_path)
        npy_path = Path(save_path).with_suffix(".npy")
        raw_data = {
            "results": np.array(results),
            "pccs": np.array(pccs),
            "n_labels": n_labels,
            "diff_labels": [l.replace("\n", " ") for l in diff_col_labels],
        }
        np.save(str(npy_path), raw_data)

    plt.close(fig)
    return fig
