"""Fig 6 -- Known vs New vs No-diffuser comparison.

For each training diffuser count *n*, evaluates reconstruction quality
under three conditions:
  (a) Last known diffuser from final training epoch.
  (b) A new (blind) diffuser.
  (c) No diffuser (direct propagation, skipping diffuser multiplication).
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


# Seed offset for blind diffusers
_BLIND_SEED_BASE = 77777777
# Number of test samples
_N_TEST = 200


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


def _eval_condition(
    model: D2NN,
    amps: torch.Tensor,
    targets: torch.Tensor,
    H_obj_to_diff: torch.Tensor,
    pad_factor: int,
    diffuser_t: torch.Tensor | None,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Evaluate a single condition (with or without diffuser).

    If *diffuser_t* is ``None`` the diffuser step is skipped
    (no-diffuser condition).

    Returns dict with ``mean_pcc``, ``std_pcc``, ``values``.
    """
    N_total = amps.shape[0]
    pccs: list[float] = []

    for start in range(0, N_total, batch_size):
        end = min(start + batch_size, N_total)
        amp_b = amps[start:end]
        tgt_b = targets[start:end]

        field = amp_b.to(torch.complex64)
        field = bl_asm_propagate(field, H_obj_to_diff, pad_factor=pad_factor)

        if diffuser_t is not None:
            field = field * diffuser_t.unsqueeze(0)

        with torch.no_grad():
            out = model(field)
        I_out = out.abs() ** 2
        pcc_vals = compute_pcc(I_out, tgt_b)
        pccs.extend(pcc_vals.cpu().tolist())

    arr = np.array(pccs)
    return {
        "mean_pcc": float(arr.mean()),
        "std_pcc": float(arr.std()),
        "values": pccs,
    }


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def make_fig6(
    checkpoint_paths: dict[int, str],
    config_path: str = "configs/baseline.yaml",
    save_path: str | None = None,
) -> dict[str, Any]:
    """Generate Fig 6: known / new / no-diffuser comparison.

    Parameters
    ----------
    checkpoint_paths : dict
        Mapping ``n -> checkpoint_path``.
    config_path : str
        Path to the YAML config.
    save_path : str or None
        If given, saves the figure and raw data.

    Returns
    -------
    dict
        Raw results with keys ``n_values``, ``known``, ``new``,
        ``no_diffuser``.  Each sub-dict has ``mean_pcc``, ``std_pcc``.
    """
    device = torch.device("cpu")
    cfg = load_and_validate_config(config_path)

    grid = cfg["grid"]
    N = int(grid["nx"])
    dx_mm = float(grid["pitch_mm"])
    wavelength_mm = float(cfg["optics"]["wavelength_mm"])
    pad_factor = int(grid.get("pad_factor", 2))
    obj_to_diff_mm = float(cfg["geometry"]["object_to_diffuser_mm"])
    base_seed = int(cfg["experiment"]["seed"])
    total_epochs = int(cfg["training"]["epochs"])

    H_obj_to_diff = bl_asm_transfer_function(
        N, dx_mm, wavelength_mm, obj_to_diff_mm, pad_factor=pad_factor,
    )

    amps, tgts = _get_test_batch(cfg, _N_TEST, device)

    n_values = sorted(checkpoint_paths.keys())
    results_known: dict[int, dict] = {}
    results_new: dict[int, dict] = {}
    results_none: dict[int, dict] = {}

    for n in n_values:
        model = _load_model(checkpoint_paths[n], cfg, device)

        # Known: last diffuser from final training epoch
        last_epoch_seed = base_seed + (total_epochs - 1)
        known_seed = last_epoch_seed * 1000 + 0  # diffuser index 0
        t_known = _make_diffuser(cfg, known_seed, device)
        results_known[n] = _eval_condition(
            model, amps, tgts, H_obj_to_diff, pad_factor, t_known,
        )

        # New: blind diffuser
        t_new = _make_diffuser(cfg, _BLIND_SEED_BASE, device)
        results_new[n] = _eval_condition(
            model, amps, tgts, H_obj_to_diff, pad_factor, t_new,
        )

        # No diffuser
        results_none[n] = _eval_condition(
            model, amps, tgts, H_obj_to_diff, pad_factor, None,
        )

    # ---- Plotting -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    bar_width = 0.25
    x_pos = np.arange(len(n_values))

    known_means = [results_known[n]["mean_pcc"] for n in n_values]
    known_stds = [results_known[n]["std_pcc"] for n in n_values]
    new_means = [results_new[n]["mean_pcc"] for n in n_values]
    new_stds = [results_new[n]["std_pcc"] for n in n_values]
    none_means = [results_none[n]["mean_pcc"] for n in n_values]
    none_stds = [results_none[n]["std_pcc"] for n in n_values]

    ax.bar(x_pos - bar_width, known_means, bar_width,
           yerr=known_stds, capsize=4, label="Known diffuser", color="steelblue")
    ax.bar(x_pos, new_means, bar_width,
           yerr=new_stds, capsize=4, label="New diffuser", color="coral")
    ax.bar(x_pos + bar_width, none_means, bar_width,
           yerr=none_stds, capsize=4, label="No diffuser", color="seagreen")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"n={n}" for n in n_values])
    ax.set_ylabel("Mean PCC", fontsize=10)
    ax.set_title("Fig 6: Known vs New vs No Diffuser", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    fig.tight_layout()

    if save_path is not None:
        save_figure(fig, save_path)
        npy_path = Path(save_path).with_suffix(".npy")
        raw = {
            "n_values": n_values,
            "known": {n: results_known[n] for n in n_values},
            "new": {n: results_new[n] for n in n_values},
            "no_diffuser": {n: results_none[n] for n in n_values},
        }
        np.save(str(npy_path), raw)

    plt.close(fig)
    return {
        "n_values": n_values,
        "known": results_known,
        "new": results_new,
        "no_diffuser": results_none,
    }
