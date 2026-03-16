"""Fig 2 — Known vs New diffuser evaluation.

Reproduces the paper's Figure 2 layout:
- Left column: ground truth objects
- (a) Known diffusers: K1, K2, K3 with blue border
- (b) New diffusers: B1, B2, B3 with orange border
- Row 0: diffuser phase maps
- Rows 1-3: MNIST digit reconstructions (0, 2, 7)
- Rows 4-5: resolution targets (10.8 mm, 12.0 mm) with horizontal bars
- PCC values below each reconstruction image
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
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

    Matches the paper layout:
    - Left column: ground truth
    - (a) 3 known diffusers with blue border
    - (b) 3 new diffusers with orange border
    - PCC values reported below each reconstruction
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

    # Generate diffusers (3 known, 3 new — matching paper K1-K3, B1-B3)
    n_show = 3
    known_diffs = _generate_known_diffusers(cfg, n_show, device)
    new_diffs = _generate_new_diffusers(cfg, n_show, device)
    all_diffs = known_diffs + new_diffs
    diff_labels = [f"K{i+1}" for i in range(n_show)] + [f"B{i+1}" for i in range(n_show)]

    # Test images: MNIST digits
    test_digits = [0, 2, 7]
    ds_cfg = cfg["dataset"]
    resize_to = int(ds_cfg.get("resize_to_px", 160))
    dataset = MNISTAmplitude(
        root="data", split="test",
        resize_to=resize_to, final_size=N,
    )
    digit_samples = _get_digit_samples(dataset, test_digits)

    # Resolution targets (horizontal bars, matching paper)
    res_periods_mm = [10.8, 12.0]
    res_targets = []
    for p in res_periods_mm:
        gt = generate_grating_target(
            period_mm=p, dx_mm=dx_mm,
            active_size=resize_to, final_size=N,
        )
        res_targets.append(gt.squeeze(0))

    # Collect all test objects
    all_objects = [s["amplitude"].squeeze(0) for s in digit_samples] + res_targets
    obj_labels = [f"Digit {d}" for d in test_digits] + [f"{p} mm" for p in res_periods_mm]
    n_objects = len(all_objects)
    n_diffs_total = len(all_diffs)

    # --- Forward passes (crop to active region for PCC and display) ---
    pad_each = (N - resize_to) // 2
    s = slice(pad_each, pad_each + resize_to)  # active region slice

    results = []  # cropped reconstruction images
    pccs = []
    for obj_idx, obj_amp in enumerate(all_objects):
        row_results = []
        row_pccs = []
        target_crop = obj_amp[s, s].unsqueeze(0)  # (1, resize_to, resize_to)
        for diff_idx, diff_info in enumerate(all_diffs):
            I_out = _forward_single(
                obj_amp, diff_info["transmittance"], model, H_obj_to_diff, pad_factor,
            )
            I_crop = I_out[:, s, s]  # crop to active region
            pcc_val = compute_pcc(I_crop, target_crop).item()
            row_results.append(I_crop.squeeze(0).detach().cpu().numpy())
            row_pccs.append(pcc_val)
        results.append(row_results)
        pccs.append(row_pccs)

    # --- Prepare ground truth images (cropped to active region) ---
    gt_images = []
    for obj_amp in all_objects:
        gt_np = obj_amp[s, s].detach().cpu().numpy()
        gt_images.append(gt_np)

    # --- Plot matching paper layout ---
    # Columns: GT | K1 K2 K3 | B1 B2 B3
    # Rows: diffuser phase | digit0 | digit2 | digit7 | 10.8mm | 12.0mm
    n_rows = 1 + n_objects  # 1 diffuser row + 5 object rows
    n_cols = 1 + n_diffs_total  # 1 GT column + 6 diffuser columns

    fig = plt.figure(figsize=(2.2 * n_cols, 2.4 * n_rows))

    # Use GridSpec — leave a gap between col n_show and n_show+1 for border separation
    # width_ratios: GT column slightly narrower, gap column between (a) and (b)
    n_cols_actual = 1 + n_show + 1 + n_show  # GT | K1 K2 K3 | gap | B1 B2 B3
    width_ratios = [1.0] + [1.0]*n_show + [0.15] + [1.0]*n_show
    gs = GridSpec(n_rows, n_cols_actual, figure=fig, wspace=0.06, hspace=0.08,
                  left=0.03, right=0.97, top=0.91, bottom=0.02,
                  width_ratios=width_ratios)

    # Column mapping: 0=GT, 1..n_show=known, n_show+1=gap, n_show+2..end=new
    gap_col = 1 + n_show  # index of the gap column
    known_cols = list(range(1, 1 + n_show))             # [1, 2, 3]
    new_cols = list(range(gap_col + 1, n_cols_actual))   # [5, 6, 7]
    data_cols = known_cols + new_cols                     # [1,2,3,5,6,7]

    axes = np.empty((n_rows, n_cols_actual), dtype=object)
    for r in range(n_rows):
        for c in range(n_cols_actual):
            if c == gap_col:
                # Gap column: invisible spacer
                ax = fig.add_subplot(gs[r, c])
                ax.set_visible(False)
            else:
                axes[r, c] = fig.add_subplot(gs[r, c])

    # --- Row 0: diffuser phase maps ---
    ax_gt0 = axes[0, 0]
    ax_gt0.set_facecolor("white")
    ax_gt0.set_xticks([])
    ax_gt0.set_yticks([])
    for spine in ax_gt0.spines.values():
        spine.set_visible(False)
    ax_gt0.text(0.5, 0.5, "10$\\lambda$", transform=ax_gt0.transAxes,
                fontsize=18, ha="center", va="center", fontweight="bold")
    ax_gt0.text(0.5, 0.15, "0–2$\\pi$", transform=ax_gt0.transAxes,
                fontsize=14, ha="center", va="center", color="gray")

    col_headers = [f"Diffuser K{i+1}" for i in range(n_show)] + [f"Diffuser B{i+1}" for i in range(n_show)]
    for i, (col, diff_info) in enumerate(zip(data_cols, all_diffs)):
        ax = axes[0, col]
        phase_raw = diff_info["phase_map"].detach().cpu().numpy()
        phase = phase_raw % (2 * np.pi)
        ax.imshow(phase, cmap="hsv", vmin=0, vmax=2 * np.pi, interpolation="nearest")
        ax.set_title(col_headers[i], fontsize=9, fontweight="bold", pad=3)
        ax.set_xticks([])
        ax.set_yticks([])

    # --- Rows 1..n_objects: GT + reconstructions ---
    for obj_idx in range(n_objects):
        row = 1 + obj_idx

        # Ground truth in column 0
        ax_gt = axes[row, 0]
        gt_img = gt_images[obj_idx]
        ax_gt.imshow(gt_img, cmap="gray", vmin=0, vmax=gt_img.max() if gt_img.max() > 0 else 1)
        ax_gt.set_xticks([])
        ax_gt.set_yticks([])
        if obj_idx >= len(test_digits):
            period_mm = res_periods_mm[obj_idx - len(test_digits)]
            ax_gt.text(0.05, 0.08, f"{period_mm} mm", transform=ax_gt.transAxes,
                       fontsize=10, fontweight="bold", color="red", va="bottom")

        # Reconstructions
        for diff_idx, col in enumerate(data_cols):
            ax = axes[row, col]
            img = results[obj_idx][diff_idx]
            display_img = contrast_enhance(img, lo_pct, hi_pct)
            ax.imshow(display_img, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])

            pcc_val = pccs[obj_idx][diff_idx]
            ax.text(0.5, -0.02, f"{pcc_val:.4f}", transform=ax.transAxes,
                    fontsize=13, ha="center", va="top")

    # --- Panel labels ---
    pos_k_left = gs[0, known_cols[0]].get_position(fig)
    pos_k_right = gs[0, known_cols[-1]].get_position(fig)
    fig.text(0.5 * (pos_k_left.x0 + pos_k_right.x1),
             0.95, "(a)    All-Optical Reconstruction\n      with Known Diffusers",
             fontsize=11, fontweight="bold", ha="center", va="bottom",
             color="#1976D2")

    pos_n_left = gs[0, new_cols[0]].get_position(fig)
    pos_n_right = gs[0, new_cols[-1]].get_position(fig)
    fig.text(0.5 * (pos_n_left.x0 + pos_n_right.x1),
             0.95, "(b)    All-Optical Reconstruction\n      with New Diffusers",
             fontsize=11, fontweight="bold", ha="center", va="bottom",
             color="#E65100")

    # --- Colored borders around (a) and (b) panel groups ---
    margin = 0.008
    # Blue border for known diffusers
    pos_tl = gs[0, known_cols[0]].get_position(fig)
    pos_br = gs[n_rows - 1, known_cols[-1]].get_position(fig)
    rect_known = mpatches.FancyBboxPatch(
        (pos_tl.x0 - margin, pos_br.y0 - margin),
        pos_br.x1 - pos_tl.x0 + 2 * margin,
        pos_tl.y1 - pos_br.y0 + 2 * margin,
        boxstyle="round,pad=0.005",
        linewidth=3.5, edgecolor="#1976D2", facecolor="none",
        transform=fig.transFigure, clip_on=False,
    )
    fig.patches.append(rect_known)

    # Orange border for new diffusers
    pos_tl2 = gs[0, new_cols[0]].get_position(fig)
    pos_br2 = gs[n_rows - 1, new_cols[-1]].get_position(fig)
    rect_new = mpatches.FancyBboxPatch(
        (pos_tl2.x0 - margin, pos_br2.y0 - margin),
        pos_br2.x1 - pos_tl2.x0 + 2 * margin,
        pos_tl2.y1 - pos_br2.y0 + 2 * margin,
        boxstyle="round,pad=0.005",
        linewidth=3.5, edgecolor="#E65100", facecolor="none",
        transform=fig.transFigure, clip_on=False,
    )
    fig.patches.append(rect_new)

    # --- Intensity colorbar on the right edge ---
    cbar_ax = fig.add_axes([0.985, 0.05, 0.008, 0.08])
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    cbar_ax.imshow(gradient, aspect="auto", cmap="gray", origin="lower")
    cbar_ax.set_xticks([])
    cbar_ax.set_yticks([0, 255])
    cbar_ax.set_yticklabels(["0", "1"], fontsize=8)
    cbar_ax.yaxis.tick_right()

    if save_path is not None:
        save_figure(fig, save_path)
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
