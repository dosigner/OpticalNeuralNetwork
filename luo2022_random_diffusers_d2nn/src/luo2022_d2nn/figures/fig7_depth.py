"""Fig 7 -- Depth (number of layers) advantage.

Shows how increasing the number of D2NN phase layers improves
reconstruction PCC across different training diffuser counts *n*.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

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
from luo2022_d2nn.utils.viz import contrast_enhance, save_figure


# Seed for blind diffusers
_BLIND_SEED_BASE = 77777777
# Number of blind diffusers per evaluation
_N_BLIND = 20
# Number of test samples
_N_TEST = 200


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _load_model_depth(
    checkpoint_path: str,
    cfg: dict,
    num_layers: int,
    device: torch.device,
) -> D2NN:
    """Load a D2NN with a specific *num_layers* (overriding config)."""
    geom = cfg["geometry"]
    grid = cfg["grid"]
    model_cfg = cfg["model"]

    model = D2NN(
        num_layers=num_layers,
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


def _get_test_batch(
    cfg: dict, count: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load first *count* test samples, return (amps, targets)."""
    ds_cfg = cfg["dataset"]
    dataset = MNISTAmplitude(
        root="data",
        split="test",
        resize_to=int(ds_cfg.get("resize_to_px", 160)),
        final_size=int(ds_cfg.get("final_resolution_px", 240)),
    )
    amps, tgts = [], []
    for i in range(min(count, len(dataset))):
        s = dataset[i]
        amp = s["amplitude"]
        if amp.ndim == 3 and amp.shape[0] == 1:
            amp = amp.squeeze(0)
        amps.append(amp)
        tgts.append(amp.clone())
    return torch.stack(amps).to(device), torch.stack(tgts).to(device)


def _eval_blind(
    model: D2NN,
    amps: torch.Tensor,
    targets: torch.Tensor,
    cfg: dict,
    device: torch.device,
    n_blind: int = _N_BLIND,
    batch_size: int = 16,
) -> dict[str, float]:
    """Evaluate *model* with blind diffusers, return mean/std PCC."""
    grid = cfg["grid"]
    N = int(grid["nx"])
    dx_mm = float(grid["pitch_mm"])
    wavelength_mm = float(cfg["optics"]["wavelength_mm"])
    pad_factor = int(grid.get("pad_factor", 2))
    obj_to_diff_mm = float(cfg["geometry"]["object_to_diffuser_mm"])

    H_obj_to_diff = bl_asm_transfer_function(
        N, dx_mm, wavelength_mm, obj_to_diff_mm, pad_factor=pad_factor,
    )

    all_pccs: list[float] = []
    for d_idx in range(n_blind):
        seed = _BLIND_SEED_BASE + d_idx
        t_d = _make_diffuser(cfg, seed, device)

        for start in range(0, amps.shape[0], batch_size):
            end = min(start + batch_size, amps.shape[0])
            field = amps[start:end].to(torch.complex64)
            field = bl_asm_propagate(field, H_obj_to_diff, pad_factor=pad_factor)
            field = field * t_d.unsqueeze(0)
            with torch.no_grad():
                out = model(field)
            I_out = out.abs() ** 2
            pcc_vals = compute_pcc(I_out, targets[start:end])
            all_pccs.extend(pcc_vals.cpu().tolist())

    arr = np.array(all_pccs)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "values": all_pccs}


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def make_fig7(
    checkpoint_paths: dict[tuple[int, int], str],
    config_path: str = "configs/baseline.yaml",
    save_path: str | None = None,
) -> dict[str, Any]:
    """Generate Fig 7: depth (number of layers) advantage.

    Parameters
    ----------
    checkpoint_paths : dict
        Mapping ``(num_layers, n) -> checkpoint_path``.  For example::

            {(2, 1): "runs/d2_n1/model.pt",
             (2, 10): "runs/d2_n10/model.pt",
             (4, 1): "runs/d4_n1/model.pt", ...}

    config_path : str
        Path to the YAML config.
    save_path : str or None
        If given, saves the figure and raw data.

    Returns
    -------
    dict
        Raw results with keys ``depths``, ``n_values``,
        ``results[(depth, n)]``.
    """
    device = torch.device("cpu")
    cfg = load_and_validate_config(config_path)

    # Test data
    amps, tgts = _get_test_batch(cfg, _N_TEST, device)

    # Extract unique depths and n values
    depths = sorted(set(d for d, _ in checkpoint_paths.keys()))
    n_values = sorted(set(n for _, n in checkpoint_paths.keys()))

    results: dict[tuple[int, int], dict] = {}

    for (depth, n), ckpt in checkpoint_paths.items():
        model = _load_model_depth(ckpt, cfg, depth, device)
        res = _eval_blind(model, amps, tgts, cfg, device)
        results[(depth, n)] = res

    # ---- Plotting -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(depths)))
    markers = ["o", "s", "^", "D", "v", "p", "*"]

    for di, depth in enumerate(depths):
        means = []
        stds = []
        plot_n = []
        for n in n_values:
            key = (depth, n)
            if key in results:
                means.append(results[key]["mean"])
                stds.append(results[key]["std"])
                plot_n.append(n)
        ax.errorbar(
            plot_n, means, yerr=stds,
            label=f"{depth} layers",
            color=colors[di],
            marker=markers[di % len(markers)],
            capsize=4, linewidth=1.5, markersize=6,
        )

    ax.set_xlabel("n (diffusers per epoch)", fontsize=10)
    ax.set_ylabel("Mean PCC (blind diffusers)", fontsize=10)
    ax.set_title("Fig 7: Depth Advantage", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    fig.tight_layout()

    if save_path is not None:
        save_figure(fig, save_path)
        npy_path = Path(save_path).with_suffix(".npy")
        raw = {
            "depths": depths,
            "n_values": n_values,
            "results": {
                f"d{d}_n{n}": results[(d, n)]
                for (d, n) in results
            },
        }
        np.save(str(npy_path), raw)

    plt.close(fig)
    return {
        "depths": depths,
        "n_values": n_values,
        "results": results,
    }
