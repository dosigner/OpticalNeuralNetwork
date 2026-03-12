"""Fig 1b — Object distortion and reconstruction comparison.

Generates a 4-row grid comparing target objects, free-space propagation
through diffuser, lens-based imaging through diffuser, and D2NN
reconstruction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from luo2022_d2nn.config.schema import load_and_validate_config
from luo2022_d2nn.data.mnist import MNISTAmplitude
from luo2022_d2nn.diffuser.random_phase import generate_diffuser
from luo2022_d2nn.eval.pcc import compute_pcc
from luo2022_d2nn.models.d2nn import D2NN
from luo2022_d2nn.optics.bl_asm import bl_asm_propagate, bl_asm_transfer_function
from luo2022_d2nn.optics.lens import fresnel_lens_transmission
from luo2022_d2nn.utils.viz import contrast_enhance, save_figure


def _load_model(checkpoint_path: str, cfg: dict, device: torch.device) -> D2NN:
    """Instantiate D2NN from config and load checkpoint weights."""
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


def _get_digit_samples(dataset: MNISTAmplitude, digits: list[int]) -> list[dict]:
    """Return one sample per requested digit label."""
    found: dict[int, dict] = {}
    for i in range(len(dataset)):
        sample = dataset[i]
        label = int(sample["label"])
        if label in digits and label not in found:
            found[label] = sample
            if len(found) == len(digits):
                break
    return [found[d] for d in digits]


def _free_space_through_diffuser(
    amplitude: torch.Tensor,
    diffuser_t: torch.Tensor,
    H_obj_to_diff: torch.Tensor,
    H_diff_to_out: torch.Tensor,
    pad_factor: int,
) -> torch.Tensor:
    """Propagate: object -> diffuser -> free-space to output plane.

    Returns output intensity (B, N, N).
    """
    field = amplitude.to(torch.complex64)
    field_at_diff = bl_asm_propagate(field, H_obj_to_diff, pad_factor=pad_factor)
    field_at_diff = field_at_diff * diffuser_t.unsqueeze(0)
    # Total free-space distance from diffuser to output:
    # 2 + 3*2 + 7 = 15 mm (through all layer gaps without layers)
    field_out = bl_asm_propagate(field_at_diff, H_diff_to_out, pad_factor=pad_factor)
    return field_out.abs() ** 2


def _lens_imaging_through_diffuser(
    amplitude: torch.Tensor,
    diffuser_t: torch.Tensor,
    H_obj_to_diff: torch.Tensor,
    N: int,
    dx_mm: float,
    wavelength_mm: float,
    pad_factor: int,
) -> torch.Tensor:
    """Propagate: object -> diffuser -> lens (2f geometry) -> image plane.

    Lens params from paper: f = 145.6 * lambda, pupil_radius = 52 * lambda.
    Returns output intensity (B, N, N).
    """
    f_mm = 145.6 * wavelength_mm   # ~ 109.2 mm
    pupil_r_mm = 52.0 * wavelength_mm  # ~ 39 mm

    # Object -> diffuser
    field = amplitude.to(torch.complex64)
    field_at_diff = bl_asm_propagate(field, H_obj_to_diff, pad_factor=pad_factor)
    field_at_diff = field_at_diff * diffuser_t.unsqueeze(0)

    # Propagate diffuser -> lens (at distance f)
    H_to_lens = bl_asm_transfer_function(N, dx_mm, wavelength_mm, f_mm, pad_factor=pad_factor)
    field_at_lens = bl_asm_propagate(field_at_diff, H_to_lens, pad_factor=pad_factor)

    # Apply lens
    t_lens = fresnel_lens_transmission(N, dx_mm, wavelength_mm, f_mm, pupil_r_mm)
    field_at_lens = field_at_lens * t_lens.unsqueeze(0)

    # Propagate lens -> image plane (distance f for 2f imaging)
    H_to_img = bl_asm_transfer_function(N, dx_mm, wavelength_mm, f_mm, pad_factor=pad_factor)
    field_out = bl_asm_propagate(field_at_lens, H_to_img, pad_factor=pad_factor)

    return field_out.abs() ** 2


def _d2nn_forward(
    amplitude: torch.Tensor,
    diffuser_t: torch.Tensor,
    model: D2NN,
    H_obj_to_diff: torch.Tensor,
    pad_factor: int,
) -> torch.Tensor:
    """Full D2NN forward: object -> diffuser -> D2NN -> output intensity."""
    field = amplitude.to(torch.complex64)
    field_at_diff = bl_asm_propagate(field, H_obj_to_diff, pad_factor=pad_factor)
    field_at_diff = field_at_diff * diffuser_t.unsqueeze(0)

    with torch.no_grad():
        field_out = model(field_at_diff)
    return field_out.abs() ** 2


def make_fig1b(
    checkpoint_path: str,
    config_path: str = "configs/baseline.yaml",
    save_path: Optional[str] = None,
    digits: list[int] | None = None,
) -> plt.Figure:
    """Generate Fig 1b: distortion comparison.

    Creates a grid with 4 rows x len(digits) columns:

    - Row 1: Target objects (MNIST digits)
    - Row 2: Free-space propagation through diffuser (no D2NN layers)
    - Row 3: Lens-based imaging through diffuser
    - Row 4: D2NN reconstruction output

    PCC values are shown under each reconstructed image.
    Contrast enhancement is applied for DISPLAY only.

    Parameters
    ----------
    checkpoint_path : str
        Path to trained D2NN model checkpoint.
    config_path : str
        Path to YAML config file.
    save_path : str or None
        If given, saves the figure (PNG) and raw data (.npy).
    digits : list of int
        Which MNIST digits to show (default [2, 5, 6]).
    """
    if digits is None:
        digits = [2, 5, 6]

    device = torch.device("cpu")
    cfg = load_and_validate_config(config_path)

    grid = cfg["grid"]
    N = int(grid["nx"])
    dx_mm = float(grid["pitch_mm"])
    wavelength_mm = float(cfg["optics"]["wavelength_mm"])
    pad_factor = int(grid.get("pad_factor", 2))
    geom = cfg["geometry"]
    obj_to_diff_mm = float(geom["object_to_diffuser_mm"])

    # Viz config
    viz_cfg = cfg.get("visualization", {}).get("contrast_enhancement", {})
    lo_pct = float(viz_cfg.get("lower_percentile", 1.0))
    hi_pct = float(viz_cfg.get("upper_percentile", 99.0))

    # Load model
    model = _load_model(checkpoint_path, cfg, device)

    # Dataset — use test split
    ds_cfg = cfg["dataset"]
    dataset = MNISTAmplitude(
        root="data",
        split="test",
        resize_to=int(ds_cfg.get("resize_to_px", 160)),
        final_size=int(ds_cfg.get("final_resolution_px", 240)),
    )
    samples = _get_digit_samples(dataset, digits)
    n_cols = len(digits)

    # Stack amplitudes: (B, N, N)
    amplitudes = torch.stack([s["amplitude"].squeeze(0) for s in samples], dim=0)
    targets = amplitudes.clone()

    # Generate one diffuser
    diff_cfg = cfg["diffuser"]
    diff_result = generate_diffuser(
        N, dx_mm, wavelength_mm,
        delta_n=float(diff_cfg.get("delta_n", 0.74)),
        height_mean_lambda=float(diff_cfg.get("height_mean_lambda", 25.0)),
        height_std_lambda=float(diff_cfg.get("height_std_lambda", 8.0)),
        smoothing_sigma_lambda=float(diff_cfg.get("smoothing_sigma_lambda", 4.0)),
        seed=int(cfg["experiment"]["seed"]),
        device=device,
    )
    diffuser_t = diff_result["transmittance"]

    # Transfer functions
    H_obj_to_diff = bl_asm_transfer_function(
        N, dx_mm, wavelength_mm, obj_to_diff_mm, pad_factor=pad_factor,
    )
    # Free-space total distance: diffuser-to-layer1 + 3*layer_to_layer + last_to_output
    free_space_total_mm = (
        float(geom["diffuser_to_layer1_mm"])
        + (int(geom["num_layers"]) - 1) * float(geom["layer_to_layer_mm"])
        + float(geom["last_layer_to_output_mm"])
    )
    H_diff_to_out = bl_asm_transfer_function(
        N, dx_mm, wavelength_mm, free_space_total_mm, pad_factor=pad_factor,
    )

    # --- Compute rows ---
    I_free = _free_space_through_diffuser(
        amplitudes, diffuser_t, H_obj_to_diff, H_diff_to_out, pad_factor,
    )
    I_lens = _lens_imaging_through_diffuser(
        amplitudes, diffuser_t, H_obj_to_diff, N, dx_mm, wavelength_mm, pad_factor,
    )
    I_d2nn = _d2nn_forward(amplitudes, diffuser_t, model, H_obj_to_diff, pad_factor)

    # PCC values
    pcc_free = compute_pcc(I_free, targets).detach().cpu().numpy()
    pcc_lens = compute_pcc(I_lens, targets).detach().cpu().numpy()
    pcc_d2nn = compute_pcc(I_d2nn, targets).detach().cpu().numpy()

    # --- Plot ---
    row_labels = ["Target", "Free-space\n+ diffuser", "Lens\n+ diffuser", "D2NN"]
    all_rows = [
        targets.detach().cpu().numpy(),
        I_free.detach().cpu().numpy(),
        I_lens.detach().cpu().numpy(),
        I_d2nn.detach().cpu().numpy(),
    ]
    all_pccs = [None, pcc_free, pcc_lens, pcc_d2nn]

    fig, axes = plt.subplots(4, n_cols, figsize=(2.5 * n_cols, 9))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row_idx in range(4):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            img = all_rows[row_idx][col_idx]
            display_img = contrast_enhance(img, lo_pct, hi_pct)
            ax.imshow(display_img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")

            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx], fontsize=9, fontweight="bold",
                              rotation=0, ha="right", va="center", labelpad=50)

            if all_pccs[row_idx] is not None:
                pcc_val = all_pccs[row_idx][col_idx]
                ax.set_title(f"PCC={pcc_val:.3f}", fontsize=8, pad=2)

    fig.suptitle("Fig 1b: Distortion & Reconstruction", fontsize=12,
                 fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0.08, 0.0, 1.0, 0.95])

    if save_path is not None:
        save_figure(fig, save_path)
        # Save raw data
        npy_path = Path(save_path).with_suffix(".npy")
        raw_data = {
            "targets": all_rows[0],
            "free_space": all_rows[1],
            "lens": all_rows[2],
            "d2nn": all_rows[3],
            "pcc_free": pcc_free,
            "pcc_lens": pcc_lens,
            "pcc_d2nn": pcc_d2nn,
            "digits": digits,
        }
        np.save(str(npy_path), raw_data)

    plt.close(fig)
    return fig
