#!/usr/bin/env python
"""Physics visualization for FD2NN phase range sweep.

Fig 1: 6-model irradiance + phase comparison (aperture zoom)
Fig 2: Layer-by-layer propagation (best model)
Fig 3: Learned phase masks comparison (6 models × 5 layers)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from kim2026.data.dataset import CachedFieldDataset
from kim2026.models.fd2nn import BeamCleanupFD2NN
from kim2026.optics.angular_spectrum import propagate_same_window
from kim2026.optics.lens_2f import lens_2f_forward, lens_2f_inverse
from kim2026.training.targets import apply_receiver_aperture

plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "figure.facecolor": "white"})

SWEEP_DIR = Path(__file__).resolve().parent.parent / "runs" / "fd2nn_phase_range_sweep"
FIG_DIR = SWEEP_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kim2026" / "1km_cn2e-14_w2m_n1024_dx2mm" / "cache"
MANIFEST = DATA_DIR.parent / "split_manifest.json"

CONFIGS = {
    "sig_pi":   {"constraint": "sigmoid",        "phase_max": math.pi,     "label": "[0,π]"},
    "tanh_pi2": {"constraint": "symmetric_tanh", "phase_max": math.pi / 2, "label": "[-π/2,π/2]"},
    "sig_2pi":  {"constraint": "sigmoid",        "phase_max": 2 * math.pi, "label": "[0,2π]"},
    "tanh_pi":  {"constraint": "symmetric_tanh", "phase_max": math.pi,     "label": "[-π,π]"},
    "sig_4pi":  {"constraint": "sigmoid",        "phase_max": 4 * math.pi, "label": "[0,4π]"},
    "tanh_2pi": {"constraint": "symmetric_tanh", "phase_max": 2 * math.pi, "label": "[-2π,2π]"},
}

COMMON = dict(
    wavelength_m=1.55e-6, receiver_window_m=0.002048, aperture_diameter_m=0.002,
    n=1024, num_layers=5, layer_spacing_m=1e-3,
    dual_2f_f1_m=1e-3, dual_2f_f2_m=1e-3, dual_2f_na1=0.16, dual_2f_na2=0.16,
)

M = 100  # crop half-width for aperture zoom


def crop(f, m=M):
    c = f.shape[-1] // 2
    return f[..., c - m:c + m, c - m:c + m]


def load_model(name):
    cfg = CONFIGS[name]
    ckpt = torch.load(SWEEP_DIR / name / "checkpoint.pt", map_location="cpu", weights_only=False)
    model = BeamCleanupFD2NN(
        n=COMMON["n"], wavelength_m=COMMON["wavelength_m"], window_m=COMMON["receiver_window_m"],
        num_layers=COMMON["num_layers"], layer_spacing_m=COMMON["layer_spacing_m"],
        phase_max=cfg["phase_max"], phase_constraint=cfg["constraint"],
        dual_2f_f1_m=COMMON["dual_2f_f1_m"], dual_2f_f2_m=COMMON["dual_2f_f2_m"],
        dual_2f_na1=COMMON["dual_2f_na1"], dual_2f_na2=COMMON["dual_2f_na2"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def get_test_sample():
    ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="test")
    sample = ds[0]
    u_t = sample["u_turb"].unsqueeze(0)
    u_v = sample["u_vacuum"].unsqueeze(0)
    w, ap = COMMON["receiver_window_m"], COMMON["aperture_diameter_m"]
    inp = apply_receiver_aperture(u_t, receiver_window_m=w, aperture_diameter_m=ap)
    tgt = apply_receiver_aperture(u_v, receiver_window_m=w, aperture_diameter_m=ap)
    return inp, tgt


# ═══════════════════════════════════════════════════════════════════
# Fig 1: 6-model Irradiance + Phase Comparison
# ═══════════════════════════════════════════════════════════════════
def fig1(inp, tgt):
    names = list(CONFIGS.keys())
    fig, axes = plt.subplots(len(names) * 2, 4, figsize=(16, len(names) * 4))

    for i, name in enumerate(names):
        model = load_model(name)
        with torch.no_grad():
            pred = model(inp)

        test_m = json.load(open(SWEEP_DIR / name / "test_metrics.json"))
        co = test_m["complex_overlap"]
        pr = test_m["phase_rmse_rad"]
        lbl = CONFIGS[name]["label"]

        fields = [
            ("Turbulent", inp[0].numpy()),
            ("FD2NN Output", pred[0].numpy()),
            ("Vacuum Target", tgt[0].numpy()),
            ("Residual", (pred[0] - tgt[0]).numpy()),
        ]

        row_amp = i * 2
        row_ph = i * 2 + 1

        for col, (title, f) in enumerate(fields):
            amp = crop(np.abs(f))
            if col == 3:
                im = axes[row_amp, col].imshow(amp, cmap="hot", origin="lower")
            else:
                im = axes[row_amp, col].imshow(amp ** 2, cmap="inferno", origin="lower")
            plt.colorbar(im, ax=axes[row_amp, col], shrink=0.7, pad=0.02)
            if i == 0:
                axes[row_amp, col].set_title(f"{title}\nIrradiance |E|²" if col < 3 else f"{title}\n|ΔE|")

            if col < 3:
                ph = crop(np.angle(f))
            else:
                ph = crop(np.angle(pred[0].numpy() * np.conj(tgt[0].numpy())))
            im = axes[row_ph, col].imshow(ph, cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi, origin="lower")
            plt.colorbar(im, ax=axes[row_ph, col], shrink=0.7, pad=0.02)
            if i == 0:
                axes[row_ph, col].set_title("Phase arg(E)" if col < 3 else "Phase Error")

        axes[row_amp, 0].set_ylabel(f"{lbl}\nco={co:.4f}\n|E|²", fontsize=9, fontweight="bold")
        axes[row_ph, 0].set_ylabel(f"pr={pr:.3f}\nphase", fontsize=9)

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("FD2NN Phase Range Sweep: Irradiance & Phase (200×200 aperture crop, dx=2μm)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG_DIR / "fig1_irradiance_phase_comparison.png", bbox_inches="tight")
    plt.close()
    print("  fig1_irradiance_phase_comparison.png")


# ═══════════════════════════════════════════════════════════════════
# Fig 2: Layer-by-layer Propagation (best model)
# ═══════════════════════════════════════════════════════════════════
def fig2(inp, tgt):
    # Find best model
    best_name = max(CONFIGS.keys(),
                    key=lambda n: json.load(open(SWEEP_DIR / n / "test_metrics.json"))["complex_overlap"])
    model = load_model(best_name)

    # Manual step-by-step forward pass to capture intermediates
    snapshots = []
    labels = []

    with torch.no_grad():
        field = inp.to(torch.complex64)
        snapshots.append(field[0].numpy()); labels.append("0: Input\n(receiver plane)")

        # lens_2f_forward
        dx_m = model.window_m / model.n
        out, dx_f = lens_2f_forward(
            field, dx_in_m=dx_m, wavelength_m=model.wavelength_m,
            f_m=model.dual_2f_f1_m, na=model.dual_2f_na1, apply_scaling=model.dual_2f_apply_scaling,
        )
        snapshots.append(out[0].numpy()); labels.append("1: After 2f\n(Fourier plane)")

        fourier_window = dx_f * model.n
        for idx, layer in enumerate(model.layers):
            if idx > 0 and model.layer_spacing_m > 0:
                out = propagate_same_window(
                    out, wavelength_m=model.wavelength_m,
                    window_m=fourier_window, z_m=model.layer_spacing_m,
                )
                snapshots.append(out[0].numpy())
                labels.append(f"{len(snapshots)}: ASM {model.layer_spacing_m*1e3:.0f}mm")

            out = layer(out)
            snapshots.append(out[0].numpy())
            labels.append(f"{len(snapshots)}: Layer {idx}\nphase mask")

        # lens_2f_inverse
        out, _ = lens_2f_inverse(
            out, dx_fourier_m=dx_f, wavelength_m=model.wavelength_m,
            f_m=model.dual_2f_f2_m, na=model.dual_2f_na2, apply_scaling=model.dual_2f_apply_scaling,
        )
        snapshots.append(out[0].numpy()); labels.append(f"{len(snapshots)}: Output\n(after inv 2f)")

    n_steps = len(snapshots)
    fig, axes = plt.subplots(2, n_steps, figsize=(2.5 * n_steps, 5))

    for col in range(n_steps):
        f = snapshots[col]
        amp = crop(np.abs(f))
        ph = crop(np.angle(f))

        # Use log scale for Fourier plane (steps 1 to n-2)
        if 1 <= col <= n_steps - 2:
            amp_display = np.log10(amp ** 2 + 1e-20)
            cmap_amp = "viridis"
            amp_label = "log₁₀|E|²"
        else:
            amp_display = amp ** 2
            cmap_amp = "inferno"
            amp_label = "|E|²"

        im0 = axes[0, col].imshow(amp_display, cmap=cmap_amp, origin="lower")
        plt.colorbar(im0, ax=axes[0, col], shrink=0.6, pad=0.02)
        axes[0, col].set_title(labels[col], fontsize=8)
        if col == 0:
            axes[0, col].set_ylabel(amp_label, fontsize=10)

        im1 = axes[1, col].imshow(ph, cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi, origin="lower")
        plt.colorbar(im1, ax=axes[1, col], shrink=0.6, pad=0.02)
        if col == 0:
            axes[1, col].set_ylabel("Phase [rad]", fontsize=10)

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    co = json.load(open(SWEEP_DIR / best_name / "test_metrics.json"))["complex_overlap"]
    fig.suptitle(f"Layer-by-Layer Propagation: {best_name} ({CONFIGS[best_name]['label']}, co={co:.4f})\n"
                 f"Input → 2f Forward → [Phase Mask + ASM]×5 → 2f Inverse → Output",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIG_DIR / "fig2_layer_propagation.png", bbox_inches="tight")
    plt.close()
    print(f"  fig2_layer_propagation.png ({best_name})")


# ═══════════════════════════════════════════════════════════════════
# Fig 3: Learned Phase Masks (6 models × 5 layers)
# ═══════════════════════════════════════════════════════════════════
def fig3():
    names = list(CONFIGS.keys())
    n_layers = COMMON["num_layers"]

    fig, axes = plt.subplots(len(names), n_layers + 1, figsize=(3 * (n_layers + 1), 3 * len(names)))

    for row, name in enumerate(names):
        model = load_model(name)
        lbl = CONFIGS[name]["label"]
        pm = CONFIGS[name]["phase_max"]

        for col in range(n_layers):
            phase = model.layers[col].phase().detach().numpy()
            ph_crop = crop(phase)

            im = axes[row, col].imshow(ph_crop, cmap="twilight_shifted",
                                       vmin=-pm, vmax=pm, origin="lower")
            if row == 0:
                axes[row, col].set_title(f"Layer {col}", fontsize=10)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])

        axes[row, 0].set_ylabel(f"{lbl}", fontsize=11, fontweight="bold")

        # Last column: wrapped to [0, 2π] (fabrication view)
        phase_all = sum(layer.phase().detach() for layer in model.layers)
        wrapped = crop((phase_all.numpy() % (2 * np.pi)))
        im = axes[row, n_layers].imshow(wrapped, cmap="hsv", vmin=0, vmax=2 * np.pi, origin="lower")
        if row == 0:
            axes[row, n_layers].set_title("Sum→mod 2π\n(fabrication)", fontsize=10)
        axes[row, n_layers].set_xticks([]); axes[row, n_layers].set_yticks([])

    plt.colorbar(im, ax=axes[:, -1].tolist(), shrink=0.3, label="Phase [rad]", pad=0.02)

    fig.suptitle("Learned Phase Masks: 6 Configs × 5 Layers + Fabrication Wrap\n(200×200 aperture crop)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    fig.savefig(FIG_DIR / "fig3_learned_phase_masks.png", bbox_inches="tight")
    plt.close()
    print("  fig3_learned_phase_masks.png")


def main():
    print("Loading test sample...")
    inp, tgt = get_test_sample()
    print(f"  Input shape: {inp.shape}, Target shape: {tgt.shape}")

    print("\nGenerating Fig 1: 6-model irradiance + phase comparison...")
    fig1(inp, tgt)

    print("\nGenerating Fig 2: Layer-by-layer propagation (best model)...")
    fig2(inp, tgt)

    print("\nGenerating Fig 3: Learned phase masks comparison...")
    fig3()

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
