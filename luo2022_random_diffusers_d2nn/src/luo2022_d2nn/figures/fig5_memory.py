"""Fig 5 -- Network memory analysis.

Shows how D2NN PCC evolves as new diffusers are introduced across
training epochs, and compares recently-seen (known) diffusers against
fresh (blind) ones.
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


# Seed offset for blind diffusers (must not overlap training)
_BLIND_SEED_BASE = 77777777
# Number of blind diffusers for panel (b)
_N_BLIND = 20
# Number of test samples to evaluate per diffuser
_N_TEST_SAMPLES = 200
# Number of sampled epochs for the memory curve (to keep runtime practical)
_EPOCH_SAMPLE_COUNT = 40
# How many diffusers from the last N epochs count as "recently seen"
_RECENT_LAST_EPOCHS = 10


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
    """Load the first *count* test samples, return (amplitudes, targets)."""
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


def _eval_diffuser_pcc(
    model: D2NN,
    diffuser_t: torch.Tensor,
    amps: torch.Tensor,
    targets: torch.Tensor,
    H_obj_to_diff: torch.Tensor,
    pad_factor: int,
    batch_size: int = 16,
) -> float:
    """Return mean PCC for one diffuser evaluated on all samples."""
    N_total = amps.shape[0]
    pccs: list[float] = []
    for start in range(0, N_total, batch_size):
        end = min(start + batch_size, N_total)
        amp_b = amps[start:end]
        tgt_b = targets[start:end]

        field = amp_b.to(torch.complex64)
        field = bl_asm_propagate(field, H_obj_to_diff, pad_factor=pad_factor)
        field = field * diffuser_t.unsqueeze(0)
        with torch.no_grad():
            out = model(field)
        I_out = out.abs() ** 2
        pcc_vals = compute_pcc(I_out, tgt_b)
        pccs.extend(pcc_vals.cpu().tolist())
    return float(np.mean(pccs))


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def make_fig5(
    checkpoint_paths: dict[int, str],
    config_path: str = "configs/baseline.yaml",
    save_path: str | None = None,
) -> dict[str, Any]:
    """Generate Fig 5: network memory analysis.

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
        Raw results with keys ``n_values``, ``memory_curves``,
        ``recent_pcc``, ``blind_pcc``.

    Notes
    -----
    Full evaluation over every diffuser in every epoch is very expensive.
    This implementation samples a representative subset of epochs evenly
    spaced across training (``_EPOCH_SAMPLE_COUNT`` epochs).  For each
    sampled epoch it evaluates *one* diffuser (diffuser index 0) on a
    small test set (``_N_TEST_SAMPLES`` samples).
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

    # Test images
    amps, tgts = _get_test_batch(cfg, _N_TEST_SAMPLES, device)

    n_values = sorted(checkpoint_paths.keys())

    # -- Fig 5(a): memory curve over all training epochs ----------------------
    # Sample epochs evenly
    if total_epochs <= _EPOCH_SAMPLE_COUNT:
        sampled_epochs = list(range(total_epochs))
    else:
        sampled_epochs = np.linspace(0, total_epochs - 1, _EPOCH_SAMPLE_COUNT, dtype=int).tolist()

    memory_curves: dict[int, dict] = {}  # n -> {global_indices, pccs}

    for n in n_values:
        model = _load_model(checkpoint_paths[n], cfg, device)
        indices: list[int] = []
        pccs: list[float] = []

        for epoch in sampled_epochs:
            epoch_seed = base_seed + epoch
            # Evaluate diffuser 0 of this epoch as representative
            seed = epoch_seed * 1000 + 0
            t_d = _make_diffuser(cfg, seed, device)
            pcc = _eval_diffuser_pcc(model, t_d, amps, tgts, H_obj_to_diff, pad_factor)
            global_idx = epoch * n  # diffuser introduction order
            indices.append(global_idx)
            pccs.append(pcc)

        memory_curves[n] = {"global_indices": indices, "pccs": pccs}

    # -- Fig 5(b): recent known vs blind bar chart ----------------------------
    recent_pcc: dict[int, dict] = {}  # n -> {mean, std}
    blind_pcc: dict[int, dict] = {}

    for n in n_values:
        model = _load_model(checkpoint_paths[n], cfg, device)

        # Recent diffusers: last _RECENT_LAST_EPOCHS epochs
        recent_pccs: list[float] = []
        start_epoch = max(0, total_epochs - _RECENT_LAST_EPOCHS)
        for epoch in range(start_epoch, total_epochs):
            epoch_seed = base_seed + epoch
            for i in range(n):
                seed = epoch_seed * 1000 + i
                t_d = _make_diffuser(cfg, seed, device)
                pcc = _eval_diffuser_pcc(model, t_d, amps, tgts, H_obj_to_diff, pad_factor)
                recent_pccs.append(pcc)
        recent_pcc[n] = {
            "mean": float(np.mean(recent_pccs)),
            "std": float(np.std(recent_pccs)),
            "values": recent_pccs,
        }

        # Blind diffusers
        blind_pccs: list[float] = []
        for i in range(_N_BLIND):
            seed = _BLIND_SEED_BASE + i
            t_d = _make_diffuser(cfg, seed, device)
            pcc = _eval_diffuser_pcc(model, t_d, amps, tgts, H_obj_to_diff, pad_factor)
            blind_pccs.append(pcc)
        blind_pcc[n] = {
            "mean": float(np.mean(blind_pccs)),
            "std": float(np.std(blind_pccs)),
            "values": blind_pccs,
        }

    # ---- Plotting -----------------------------------------------------------
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(n_values)))

    # Panel (a): memory curve
    for ci, n in enumerate(n_values):
        mc = memory_curves[n]
        ax_a.plot(mc["global_indices"], mc["pccs"],
                  label=f"n={n}", color=colors[ci], linewidth=1.2, marker=".", markersize=3)
    ax_a.set_xlabel("Diffuser global index (epoch x n)", fontsize=10)
    ax_a.set_ylabel("Mean PCC", fontsize=10)
    ax_a.set_title("(a) Memory: PCC vs diffuser introduction order",
                    fontsize=11, fontweight="bold")
    ax_a.legend(fontsize=8)

    # Panel (b): bar chart
    bar_width = 0.35
    x_pos = np.arange(len(n_values))
    recent_means = [recent_pcc[n]["mean"] for n in n_values]
    recent_stds = [recent_pcc[n]["std"] for n in n_values]
    blind_means = [blind_pcc[n]["mean"] for n in n_values]
    blind_stds = [blind_pcc[n]["std"] for n in n_values]

    ax_b.bar(x_pos - bar_width / 2, recent_means, bar_width,
             yerr=recent_stds, capsize=4, label="Recent known", color="steelblue")
    ax_b.bar(x_pos + bar_width / 2, blind_means, bar_width,
             yerr=blind_stds, capsize=4, label="Blind (new)", color="coral")
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels([f"n={n}" for n in n_values])
    ax_b.set_ylabel("Mean PCC", fontsize=10)
    ax_b.set_title("(b) Recent known vs blind diffusers",
                    fontsize=11, fontweight="bold")
    ax_b.legend(fontsize=9)

    fig.suptitle("Fig 5: Network Memory", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()

    if save_path is not None:
        save_figure(fig, save_path)
        npy_path = Path(save_path).with_suffix(".npy")
        raw = {
            "n_values": n_values,
            "memory_curves": memory_curves,
            "recent_pcc": recent_pcc,
            "blind_pcc": blind_pcc,
        }
        np.save(str(npy_path), raw)

    plt.close(fig)
    return {
        "n_values": n_values,
        "memory_curves": memory_curves,
        "recent_pcc": recent_pcc,
        "blind_pcc": blind_pcc,
    }
