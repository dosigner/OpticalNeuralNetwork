#!/usr/bin/env python
"""Compare FD2NN vs D2NN architecture — how beams interact with masks.

Shows WHY D2NN should work better on telescope data:
- FD2NN: beam concentrated in 7px of Fourier plane → masks can't correct
- D2NN: beam fills entire mask → full correction capability

Generates comparison even if D2NN sweep isn't done yet (uses zero-phase D2NN).
Updates with trained D2NN results when available.

Usage:
    cd /root/dj/D2NN/kim2026 && python scripts/visualize_fdnn_vs_d2nn.py
"""
from __future__ import annotations
import json, math
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

from kim2026.data.dataset import CachedFieldDataset
from kim2026.models.fd2nn import BeamCleanupFD2NN
from kim2026.models.d2nn import BeamCleanupD2NN
from kim2026.optics.lens_2f import lens_2f_forward
from kim2026.training.targets import apply_receiver_aperture
from kim2026.training.metrics import complex_overlap

λ = 1.55e-6
N = 1024
WIN = 0.002048
APT = 0.002
dx = WIN / N

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kim2026" / "1km_cn2e-14_tel15cm_n1024_br75"
FDNN_SWEEP = Path(__file__).resolve().parent.parent / "autoresearch" / "runs" / "loss_sweep_telescope"
D2NN_SWEEP = Path(__file__).resolve().parent.parent / "autoresearch" / "runs" / "d2nn_sweep_telescope"
OUT = Path(__file__).resolve().parent.parent / "autoresearch" / "runs" / "single_case_viz"


def load_fdnn(name="baseline_co"):
    ckpt = FDNN_SWEEP / name / "checkpoint.pt"
    if not ckpt.exists():
        return None
    model = BeamCleanupFD2NN(
        n=N, wavelength_m=λ, window_m=WIN, num_layers=5, layer_spacing_m=5e-3,
        phase_constraint="unconstrained", phase_max=math.pi,
        phase_init="uniform", phase_init_scale=0.1,
        dual_2f_f1_m=25e-3, dual_2f_f2_m=25e-3,
        dual_2f_na1=0.508, dual_2f_na2=0.508, dual_2f_apply_scaling=False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["model_state_dict"])
    model.eval()
    return model


def load_d2nn(name="baseline_co"):
    ckpt = D2NN_SWEEP / name / "checkpoint.pt"
    if not ckpt.exists():
        return None
    model = BeamCleanupD2NN(
        n=N, wavelength_m=λ, window_m=WIN, num_layers=5,
        layer_spacing_m=10e-3, detector_distance_m=10e-3)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["model_state_dict"])
    model.eval()
    return model


def zoom(f, z=50):
    c = f.shape[-1] // 2
    return f[..., c-z:c+z, c-z:c+z]


def ext(n, dx, unit=1e-6):
    h = n * dx / 2 / unit
    return [-h, h, -h, h]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    ds = CachedFieldDataset(cache_dir=str(DATA_DIR / "cache"),
                             manifest_path=str(DATA_DIR / "split_manifest.json"), split="test")
    sample = ds[0]
    u_turb = apply_receiver_aperture(
        sample["u_turb"].unsqueeze(0).to(device),
        receiver_window_m=WIN, aperture_diameter_m=APT)
    u_vac = apply_receiver_aperture(
        sample["u_vacuum"].unsqueeze(0).to(device),
        receiver_window_m=WIN, aperture_diameter_m=APT)

    baseline_co = complex_overlap(u_turb, u_vac).item()
    print(f"Baseline CO (no correction): {baseline_co:.4f}")

    # Load models
    fdnn = load_fdnn()
    d2nn = load_d2nn()

    fdnn_trained = fdnn is not None
    d2nn_trained = d2nn is not None

    # Zero-phase models as fallback
    fdnn_zero = BeamCleanupFD2NN(
        n=N, wavelength_m=λ, window_m=WIN, num_layers=5, layer_spacing_m=5e-3,
        phase_constraint="unconstrained", phase_max=math.pi,
        phase_init="uniform", phase_init_scale=0.1,
        dual_2f_f1_m=25e-3, dual_2f_f2_m=25e-3,
        dual_2f_na1=0.508, dual_2f_na2=0.508, dual_2f_apply_scaling=False).to(device)
    for l in fdnn_zero.layers:
        l.raw.data.zero_()
    fdnn_zero.eval()

    d2nn_zero = BeamCleanupD2NN(
        n=N, wavelength_m=λ, window_m=WIN, num_layers=5,
        layer_spacing_m=10e-3, detector_distance_m=10e-3).to(device)
    d2nn_zero.eval()

    if fdnn is not None:
        fdnn = fdnn.to(device)
    if d2nn is not None:
        d2nn = d2nn.to(device)

    # Run inference
    with torch.no_grad():
        # FD2NN
        fdnn_out = fdnn(u_turb) if fdnn_trained else fdnn_zero(u_turb)
        fdnn_co = complex_overlap(fdnn_out, u_vac).item()

        # D2NN
        d2nn_out = d2nn(u_turb) if d2nn_trained else d2nn_zero(u_turb)
        d2nn_co = complex_overlap(d2nn_out, u_vac).item()

        # Fourier plane for FD2NN (to show under-resolution)
        u_turb_fourier, dx_f = lens_2f_forward(
            u_turb.to(torch.complex64), dx_in_m=dx, wavelength_m=λ,
            f_m=25e-3, na=0.508, apply_scaling=False)
        u_vac_fourier, _ = lens_2f_forward(
            u_vac.to(torch.complex64), dx_in_m=dx, wavelength_m=λ,
            f_m=25e-3, na=0.508, apply_scaling=False)
        fdnn_out_fourier, _ = lens_2f_forward(
            fdnn_out.to(torch.complex64), dx_in_m=dx, wavelength_m=λ,
            f_m=25e-3, na=0.508, apply_scaling=False)

    fdnn_label = f"FD2NN {'(trained)' if fdnn_trained else '(zero-phase)'}"
    d2nn_label = f"D2NN {'(trained)' if d2nn_trained else '(zero-phase)'}"

    print(f"FD2NN CO: {fdnn_co:.4f} {'(trained)' if fdnn_trained else '(zero-phase)'}")
    print(f"D2NN  CO: {d2nn_co:.4f} {'(trained)' if d2nn_trained else '(zero-phase)'}")

    # ═══════════════════════════════════════════════════════════
    # FIGURE: 5 rows × 4 cols
    # ═══════════════════════════════════════════════════════════
    fig, axes = plt.subplots(5, 4, figsize=(22, 28))

    title = (f"FD2NN vs D2NN Architecture Comparison — Telescope Data\n"
             f"Baseline CO={baseline_co:.4f} | "
             f"FD2NN CO={fdnn_co:.4f} | D2NN CO={d2nn_co:.4f}")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    recv_ext = ext(N, dx, 1e-3)  # mm

    # ── Row 0: Input & Output Irradiance ──────────────────────
    fields_row0 = [
        (u_turb[0].cpu().numpy(), "Turbulent input"),
        (u_vac[0].cpu().numpy(), "Vacuum target"),
        (fdnn_out[0].cpu().numpy(), f"{fdnn_label}\nCO={fdnn_co:.4f}"),
        (d2nn_out[0].cpu().numpy(), f"{d2nn_label}\nCO={d2nn_co:.4f}"),
    ]
    imax = max((np.abs(f)**2).max() for f, _ in fields_row0)
    for col, (f, t) in enumerate(fields_row0):
        im = axes[0, col].imshow(np.abs(f)**2, extent=recv_ext, origin="lower",
                                  cmap="inferno", vmin=0, vmax=imax)
        axes[0, col].set_title(t, fontsize=9)
        axes[0, col].set_xlabel("mm")
    axes[0, 0].set_ylabel("Row 0: Irradiance", fontsize=10, fontweight="bold")

    # ── Row 1: Phase ──────────────────────────────────────────
    for col, (f, t) in enumerate(fields_row0):
        axes[1, col].imshow(np.angle(f), extent=recv_ext, origin="lower",
                             cmap="twilight_shifted", vmin=-math.pi, vmax=math.pi)
        axes[1, col].set_title(f"Phase: {t.split(chr(10))[0]}", fontsize=9)
        axes[1, col].set_xlabel("mm")
    axes[1, 0].set_ylabel("Row 1: Phase", fontsize=10, fontweight="bold")

    # ── Row 2: WHERE the beam hits the mask ───────────────────
    # FD2NN: show Fourier plane (beam hits only 7px center)
    # D2NN: show receiver plane (beam hits entire mask)
    zp = 30  # zoom pixels
    f_ext = ext(2 * zp, dx_f, 1e-6)  # μm
    r_ext = ext(N, dx, 1e-3)  # mm

    # FD2NN: Fourier plane irradiance (zoomed)
    vf = zoom(u_vac_fourier[0].cpu().numpy(), zp)
    tf = zoom(u_turb_fourier[0].cpu().numpy(), zp)
    f_imax = max(np.abs(vf).max()**2, np.abs(tf).max()**2)
    axes[2, 0].imshow(np.abs(vf)**2, extent=f_ext, origin="lower", cmap="inferno", vmin=0, vmax=f_imax)
    axes[2, 0].set_title(f"FD2NN: Vacuum at mask\n(Fourier, zoom {2*zp}px, dx={dx_f*1e6:.0f}μm)", fontsize=8)
    axes[2, 0].set_xlabel("μm")

    axes[2, 1].imshow(np.abs(tf)**2, extent=f_ext, origin="lower", cmap="inferno", vmin=0, vmax=f_imax)
    axes[2, 1].set_title(f"FD2NN: Turbulent at mask\n(7px spot = 0.015% of mask)", fontsize=8)
    axes[2, 1].set_xlabel("μm")

    # D2NN: receiver plane irradiance (full — beam hits entire mask)
    vi = np.abs(u_vac[0].cpu().numpy())**2
    ti = np.abs(u_turb[0].cpu().numpy())**2
    r_imax = max(vi.max(), ti.max())
    axes[2, 2].imshow(vi, extent=r_ext, origin="lower", cmap="inferno", vmin=0, vmax=r_imax)
    axes[2, 2].set_title("D2NN: Vacuum at mask\n(실공간, 빔이 mask 전체 커버)", fontsize=8)
    axes[2, 2].set_xlabel("mm")

    axes[2, 3].imshow(ti, extent=r_ext, origin="lower", cmap="inferno", vmin=0, vmax=r_imax)
    axes[2, 3].set_title("D2NN: Turbulent at mask\n(100% of mask 활용)", fontsize=8)
    axes[2, 3].set_xlabel("mm")

    axes[2, 0].set_ylabel("Row 2: Beam at Mask\n(핵심 차이!)", fontsize=10, fontweight="bold", color="red")

    # ── Row 3: Phase masks (trained or zero) ──────────────────
    # FD2NN masks (Fourier plane, zoomed to center 100px)
    mz = 100
    if fdnn_trained:
        fdnn_phases = [l.wrapped_phase().detach().cpu().numpy() for l in fdnn.layers]
    else:
        fdnn_phases = [np.zeros((N, N)) for _ in range(5)]
    if d2nn_trained:
        d2nn_phases = [torch.remainder(l.phase, 2*math.pi).detach().cpu().numpy() for l in d2nn.layers]
    else:
        d2nn_phases = [np.zeros((N, N)) for _ in range(5)]

    # FD2NN: layer 0 and 2 (zoomed — most activity near center)
    for col, (idx, label) in enumerate([(0, "FD2NN Layer 0"), (2, "FD2NN Layer 2")]):
        ph = zoom(fdnn_phases[idx], mz)
        axes[3, col].imshow(ph, cmap="twilight_shifted", vmin=0, vmax=2*math.pi)
        axes[3, col].set_title(f"{label}\n(zoom {2*mz}px of {N}px)", fontsize=8)

    # D2NN: layer 0 and 2 (full — the entire mask is used)
    for col, (idx, label) in enumerate([(0, "D2NN Layer 0"), (2, "D2NN Layer 2")]):
        axes[3, col+2].imshow(d2nn_phases[idx], cmap="twilight_shifted", vmin=0, vmax=2*math.pi)
        axes[3, col+2].set_title(f"{label}\n(full {N}px)", fontsize=8)

    axes[3, 0].set_ylabel("Row 3: Phase Masks", fontsize=10, fontweight="bold")

    # ── Row 4: Summary metrics bar chart ──────────────────────
    # Load all available results
    all_results = []
    for sweep_dir, model_type in [(FDNN_SWEEP, "FD2NN"), (D2NN_SWEEP, "D2NN")]:
        if not sweep_dir.exists():
            continue
        for d in sorted(sweep_dir.iterdir()):
            rpath = d / "results.json"
            if rpath.exists():
                r = json.loads(rpath.read_text())
                r["model_type"] = model_type
                r["label"] = f"{model_type}:{r['name']}"
                all_results.append(r)

    if all_results:
        all_results.sort(key=lambda r: r["complex_overlap"], reverse=True)
        labels = [r["label"] for r in all_results]
        cos = [r["complex_overlap"] for r in all_results]
        colors = ['#2ecc71' if r["model_type"] == "D2NN" else '#3498db' for r in all_results]

        # CO bar chart
        axes[4, 0].barh(range(len(labels)), cos, color=colors)
        axes[4, 0].axvline(baseline_co, color='k', ls='--', lw=1.5, label=f'Baseline={baseline_co:.4f}')
        axes[4, 0].set_yticks(range(len(labels)))
        axes[4, 0].set_yticklabels(labels, fontsize=7)
        axes[4, 0].set_xlabel("Complex Overlap")
        axes[4, 0].set_title("CO Ranking (green=D2NN, blue=FD2NN)", fontsize=9)
        axes[4, 0].legend(fontsize=7)
        axes[4, 0].invert_yaxis()

        # Phase RMSE
        prmses = [r["phase_rmse_rad"] for r in all_results]
        axes[4, 1].barh(range(len(labels)), prmses, color=colors)
        axes[4, 1].set_yticks(range(len(labels)))
        axes[4, 1].set_yticklabels(labels, fontsize=7)
        axes[4, 1].set_xlabel("Phase RMSE [rad]")
        axes[4, 1].set_title("Phase RMSE (lower=better)", fontsize=9)
        axes[4, 1].invert_yaxis()

        # Throughput
        tps = [r["throughput"] for r in all_results]
        axes[4, 2].barh(range(len(labels)), tps, color=colors)
        axes[4, 2].set_yticks(range(len(labels)))
        axes[4, 2].set_yticklabels(labels, fontsize=7)
        axes[4, 2].set_xlabel("Throughput")
        axes[4, 2].set_title("Throughput", fontsize=9)
        axes[4, 2].invert_yaxis()

        # Summary text
        axes[4, 3].axis("off")
        summary_lines = [
            f"Baseline CO: {baseline_co:.4f}",
            f"",
            f"FD2NN best: {max((r['complex_overlap'] for r in all_results if r['model_type']=='FD2NN'), default=0):.4f}",
        ]
        d2nn_results = [r for r in all_results if r["model_type"] == "D2NN"]
        if d2nn_results:
            summary_lines.append(f"D2NN best:  {max(r['complex_overlap'] for r in d2nn_results):.4f}")
        else:
            summary_lines.append("D2NN: sweep 진행 중...")
        summary_lines += [
            "",
            "FD2NN 한계:",
            "  Fourier spot = 7px",
            "  mask 활용률 = 0.015%",
            "  → 학습 거의 불가",
            "",
            "D2NN 장점:",
            "  실공간 mask",
            "  mask 활용률 = 100%",
            "  layer간 62px 확산",
        ]
        axes[4, 3].text(0.05, 0.95, "\n".join(summary_lines), transform=axes[4, 3].transAxes,
                         fontsize=9, verticalalignment="top", fontfamily="monospace",
                         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    axes[4, 0].set_ylabel("Row 4: Ranking", fontsize=10, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUT / "fdnn_vs_d2nn_comparison.png"
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
