"""Fig 2 — Known vs New diffuser evaluation.

Generates a comparison figure showing D2NN reconstruction quality when
using diffusers seen during training (known) versus fresh unseen
diffusers (new).
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
from luo2022_d2nn.data.resolution_targets import generate_grating_target
from luo2022_d2nn.diffuser.random_phase import generate_diffuser
from luo2022_d2nn.eval.pcc import compute_pcc
from luo2022_d2nn.models.d2nn import D2NN
from luo2022_d2nn.optics.bl_asm import bl_asm_propagate, bl_asm_transfer_function
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


def _generate_known_diffusers(cfg: dict, n: int, device: torch.device) -> list[dict]:
    """Generate 'known' diffusers using the last training epoch's seeds."""
    grid = cfg["grid"]
    N = int(grid["nx"])
    dx_mm = float(grid["pitch_mm"])
    wavelength_mm = float(cfg["optics"]["wavelength_mm"])
    diff_cfg = cfg["diffuser"]
    base_seed = int(cfg["experiment"]["seed"])
    last_epoch = int(cfg["training"]["epochs"]) - 1
    epoch_seed = base_seed + last_epoch

    diffusers = []
    for i in range(n):
        seed = epoch_seed * 1000 + i
        result = generate_diffuser(
            N, dx_mm, wavelength_mm,
            delta_n=float(diff_cfg.get("delta_n", 0.74)),
            height_mean_lambda=float(diff_cfg.get("height_mean_lambda", 25.0)),
            height_std_lambda=float(diff_cfg.get("height_std_lambda", 8.0)),
            smoothing_sigma_lambda=float(diff_cfg.get("smoothing_sigma_lambda", 4.0)),
            seed=seed,
            device=device,
        )
        diffusers.append(result)
    return diffusers


def _generate_new_diffusers(cfg: dict, n: int, device: torch.device) -> list[dict]:
    """Generate 'new' diffusers with fresh seeds (offset well beyond training)."""
    grid = cfg["grid"]
    N = int(grid["nx"])
    dx_mm = float(grid["pitch_mm"])
    wavelength_mm = float(cfg["optics"]["wavelength_mm"])
    diff_cfg = cfg["diffuser"]
    # Use a seed range that never overlaps training
    fresh_base_seed = 77777777

    diffusers = []
    for i in range(n):
        seed = fresh_base_seed + i
        result = generate_diffuser(
            N, dx_mm, wavelength_mm,
            delta_n=float(diff_cfg.get("delta_n", 0.74)),
            height_mean_lambda=float(diff_cfg.get("height_mean_lambda", 25.0)),
            height_std_lambda=float(diff_cfg.get("height_std_lambda", 8.0)),
            smoothing_sigma_lambda=float(diff_cfg.get("smoothing_sigma_lambda", 4.0)),
            seed=seed,
            device=device,
        )
        diffusers.append(result)
    return diffusers


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


def _forward_single(
    amplitude: torch.Tensor,
    diffuser_t: torch.Tensor,
    model: D2NN,
    H_obj_to_diff: torch.Tensor,
    pad_factor: int,
) -> torch.Tensor:
    """Forward pass for a single amplitude through diffuser + D2NN.

    Returns intensity (1, N, N).
    """
    field = amplitude.unsqueeze(0).to(torch.complex64)
    field_at_diff = bl_asm_propagate(field, H_obj_to_diff, pad_factor=pad_factor)
    field_at_diff = field_at_diff * diffuser_t.unsqueeze(0)
    with torch.no_grad():
        field_out = model(field_at_diff)
    return field_out.abs() ** 2


def make_fig2(
    checkpoint_path: str,
    config_path: str = "configs/baseline.yaml",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Generate Fig 2: known vs new diffuser reconstruction.

    Panel layout:
    - Top row: diffuser phase maps (2 known + 2 new)
    - Middle rows: reconstructions of test images (digits 0, 2, 7)
    - Bottom rows: resolution targets (10.8 mm, 12.0 mm period)
    - Each reconstruction shows PCC value

    Parameters
    ----------
    checkpoint_path : str
        Path to trained D2NN model checkpoint.
    config_path : str
        Path to YAML config file.
    save_path : str or None
        If given, saves the figure (PNG) and raw data (.npy).
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

    viz_cfg = cfg.get("visualization", {}).get("contrast_enhancement", {})
    lo_pct = float(viz_cfg.get("lower_percentile", 1.0))
    hi_pct = float(viz_cfg.get("upper_percentile", 99.0))

    # Load model
    model = _load_model(checkpoint_path, cfg, device)

    # Transfer function: object -> diffuser
    H_obj_to_diff = bl_asm_transfer_function(
        N, dx_mm, wavelength_mm, obj_to_diff_mm, pad_factor=pad_factor,
    )

    # Generate diffusers (2 known, 2 new for display)
    n_show = 2
    known_diffs = _generate_known_diffusers(cfg, n_show, device)
    new_diffs = _generate_new_diffusers(cfg, n_show, device)
    all_diffs = known_diffs + new_diffs  # 4 diffusers total
    diff_labels = [f"Known {i+1}" for i in range(n_show)] + [f"New {i+1}" for i in range(n_show)]

    # Test images: MNIST digits
    test_digits = [0, 2, 7]
    ds_cfg = cfg["dataset"]
    dataset = MNISTAmplitude(
        root="data",
        split="test",
        resize_to=int(ds_cfg.get("resize_to_px", 160)),
        final_size=int(ds_cfg.get("final_resolution_px", 240)),
    )
    digit_samples = _get_digit_samples(dataset, test_digits)

    # Resolution targets
    res_periods_mm = [10.8, 12.0]
    res_targets = []
    for p in res_periods_mm:
        gt = generate_grating_target(
            period_mm=p, dx_mm=dx_mm,
            active_size=int(ds_cfg.get("resize_to_px", 160)),
            final_size=N,
        )
        res_targets.append(gt.squeeze(0))  # (N, N)

    # Collect all test objects: digits + resolution targets
    all_objects = [s["amplitude"].squeeze(0) for s in digit_samples] + res_targets
    obj_labels = [f"Digit {d}" for d in test_digits] + [f"{p} mm grating" for p in res_periods_mm]
    n_objects = len(all_objects)
    n_diffs_total = len(all_diffs)

    # --- Forward passes ---
    # results[obj_idx][diff_idx] = intensity (N, N) numpy
    results = []
    pccs = []
    for obj_idx, obj_amp in enumerate(all_objects):
        row_results = []
        row_pccs = []
        target = obj_amp.clone()
        for diff_idx, diff_info in enumerate(all_diffs):
            I_out = _forward_single(
                obj_amp, diff_info["transmittance"], model, H_obj_to_diff, pad_factor,
            )
            pcc_val = compute_pcc(I_out, target.unsqueeze(0)).item()
            row_results.append(I_out.squeeze(0).detach().cpu().numpy())
            row_pccs.append(pcc_val)
        results.append(row_results)
        pccs.append(row_pccs)

    # --- Plot ---
    # Layout: (1 + n_objects) rows x n_diffs_total columns
    n_rows = 1 + n_objects
    n_cols = n_diffs_total

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.8 * n_rows))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    # Row 0: diffuser phase maps
    for col_idx, diff_info in enumerate(all_diffs):
        ax = axes[0, col_idx]
        phase = diff_info["phase_map"].detach().cpu().numpy()
        ax.imshow(phase, cmap="twilight", interpolation="nearest")
        ax.set_title(diff_labels[col_idx], fontsize=9, fontweight="bold")
        ax.axis("off")
    # Row label
    axes[0, 0].set_ylabel("Diffuser\nphase", fontsize=9, fontweight="bold",
                          rotation=0, ha="right", va="center", labelpad=55)

    # Rows 1..n_objects: reconstructions
    for obj_idx in range(n_objects):
        for col_idx in range(n_cols):
            ax = axes[1 + obj_idx, col_idx]
            img = results[obj_idx][col_idx]
            display_img = contrast_enhance(img, lo_pct, hi_pct)
            ax.imshow(display_img, cmap="gray", vmin=0, vmax=1)
            pcc_val = pccs[obj_idx][col_idx]
            ax.set_title(f"PCC={pcc_val:.3f}", fontsize=8, pad=2)
            ax.axis("off")
        # Row label
        axes[1 + obj_idx, 0].set_ylabel(
            obj_labels[obj_idx], fontsize=8, fontweight="bold",
            rotation=0, ha="right", va="center", labelpad=55,
        )

    fig.suptitle("Fig 2: Known vs New Diffuser Reconstruction",
                 fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0.07, 0.0, 1.0, 0.96])

    if save_path is not None:
        save_figure(fig, save_path)
        # Save raw data
        npy_path = Path(save_path).with_suffix(".npy")
        raw_data = {
            "results": np.array(results),
            "pccs": np.array(pccs),
            "obj_labels": obj_labels,
            "diff_labels": diff_labels,
            "test_digits": test_digits,
            "res_periods_mm": res_periods_mm,
        }
        np.save(str(npy_path), raw_data)

    plt.close(fig)
    return fig
