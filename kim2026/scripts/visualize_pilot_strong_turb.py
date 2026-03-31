#!/usr/bin/env python
"""Pilot Strong Turbulence Visualization — 6 Figures.

Figure 1: Input field (strong turbulence vs vacuum)
Figure 2: D2NN output (3 configs vs baseline)
Figure 3: Detector plane (focused, 1D cross-section, encircled energy)
Figure 4: Phase masks (2 layers × 3 configs)
Figure 5: Training curves (loss, val CO)
Figure 6: Multi-realization statistics (box plots)

Usage:
    cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python scripts/visualize_pilot_strong_turb.py
"""
from __future__ import annotations
import json, math
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

from kim2026.data.dataset import CachedFieldDataset
from kim2026.models.d2nn import BeamCleanupD2NN
from kim2026.optics.lens_2f import lens_2f_forward
from kim2026.training.targets import apply_receiver_aperture, center_crop_field
from kim2026.training.metrics import complex_overlap

# ─── Constants ────────────────────────────────────────────────
W = 1.55e-6
N_FULL = 1024
PILOT_N = 512
WIN_FULL = 0.002048
WIN = WIN_FULL * (PILOT_N / N_FULL)  # 1.024mm for N=512
APT = 0.002
FOCUS_F = 4.5e-3
DX = WIN / PILOT_N

ARCH = dict(num_layers=2, layer_spacing_m=10e-3, detector_distance_m=10e-3)
CONFIGS = ["baseline_co", "co_phasor", "co_amp"]

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kim2026" / "1km_cn2_5e-14_tel15cm_n1024_br75"
SWEEP_DIR = Path(__file__).resolve().parent.parent / "autoresearch" / "runs" / "d2nn_pilot_strong_turb"
OUT = SWEEP_DIR


def load_model(name):
    ckpt = SWEEP_DIR / name / "checkpoint.pt"
    if not ckpt.exists():
        return None
    m = BeamCleanupD2NN(n=PILOT_N, wavelength_m=W, window_m=WIN, **ARCH)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["model_state_dict"])
    m.eval()
    return m


def prepare(field):
    a = apply_receiver_aperture(field, receiver_window_m=WIN_FULL, aperture_diameter_m=APT)
    return center_crop_field(a, crop_n=PILOT_N)


def ext(n, dx, unit):
    h = n * dx / 2 / unit
    return [-h, h, -h, h]


def focus(field):
    with torch.no_grad():
        f, dx_f = lens_2f_forward(field.to(torch.complex64), dx_in_m=DX,
                                   wavelength_m=W, f_m=FOCUS_F, na=None, apply_scaling=False)
    return f, dx_f


def wrap_phase(p):
    return (p + np.pi) % (2 * np.pi) - np.pi


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ─── Load data ────────────────────────────────────────────
    ds = CachedFieldDataset(cache_dir=str(DATA_DIR / "cache"),
                             manifest_path=str(DATA_DIR / "split_manifest.json"), split="test")
    s = ds[0]
    u_turb_raw = s["u_turb"].unsqueeze(0).to(device)
    u_vac_raw = s["u_vacuum"].unsqueeze(0).to(device)

    u_turb = prepare(u_turb_raw)
    u_vac = prepare(u_vac_raw)

    # ─── Load models + inference ──────────────────────────────
    outputs = {}
    for name in CONFIGS:
        model = load_model(name)
        if model is not None:
            model = model.to(device)
            with torch.no_grad():
                outputs[name] = model(u_turb)
            co = complex_overlap(outputs[name], u_vac).item()
            print(f"  {name:>15}: CO={co:.4f}")

    # Zero-phase D2NN (no correction baseline)
    d2nn_zero = BeamCleanupD2NN(n=PILOT_N, wavelength_m=W, window_m=WIN, **ARCH).to(device)
    d2nn_zero.eval()
    with torch.no_grad():
        u_vac_d2nn = d2nn_zero(u_vac)
        u_turb_d2nn = d2nn_zero(u_turb)
    co_nocorr = complex_overlap(u_turb_d2nn, u_vac_d2nn).item()
    print(f"  {'no_correction':>15}: CO={co_nocorr:.4f}")

    # Focus
    u_vac_det, dx_det = focus(u_vac_d2nn)
    u_turb_det, _ = focus(u_turb_d2nn)
    det_outputs = {}
    for name, out in outputs.items():
        det_outputs[name], _ = focus(out)

    # To numpy
    vac_np = u_vac[0].cpu().numpy()
    turb_np = u_turb[0].cpu().numpy()
    vac_d2nn_np = u_vac_d2nn[0].cpu().numpy()
    turb_d2nn_np = u_turb_d2nn[0].cpu().numpy()
    out_np = {n: o[0].cpu().numpy() for n, o in outputs.items()}
    vac_det_np = u_vac_det[0].cpu().numpy()
    turb_det_np = u_turb_det[0].cpu().numpy()
    det_np = {n: o[0].cpu().numpy() for n, o in det_outputs.items()}

    mid = PILOT_N // 2
    e_mm = ext(PILOT_N, DX, 1e-3)

    # ═══════════════════════════════════════════════════════════
    # FIGURE 1: Input Field
    # ═══════════════════════════════════════════════════════════
    print("\nGenerating Figure 1: Input field...")
    fig1, axes1 = plt.subplots(2, 4, figsize=(24, 12))
    fig1.suptitle("Figure 1: Input Field (Strong Turbulence, Cn²=5e-14, D/r₀=5.0)",
                  fontsize=16, fontweight="bold")

    imax = (np.abs(vac_np)**2).max()

    axes1[0, 0].imshow(np.abs(vac_np)**2, extent=e_mm, origin="lower", cmap="inferno", vmin=0, vmax=imax)
    axes1[0, 0].set_title("Vacuum irradiance", fontsize=11)
    axes1[0, 0].set_ylabel("Before D2NN\n(after aperture+crop)", fontsize=11, fontweight="bold")

    axes1[0, 1].imshow(np.angle(vac_np), extent=e_mm, origin="lower", cmap="twilight_shifted",
                       vmin=-math.pi, vmax=math.pi)
    axes1[0, 1].set_title("Vacuum phase", fontsize=11)

    axes1[0, 2].imshow(np.abs(turb_np)**2, extent=e_mm, origin="lower", cmap="inferno", vmin=0, vmax=imax)
    axes1[0, 2].set_title("Turbulent irradiance", fontsize=11)

    im_ph = axes1[0, 3].imshow(np.angle(turb_np), extent=e_mm, origin="lower", cmap="twilight_shifted",
                               vmin=-math.pi, vmax=math.pi)
    axes1[0, 3].set_title("Turbulent phase", fontsize=11)

    # Row 2: Phase difference + 1D profiles
    phase_diff = wrap_phase(np.angle(turb_np) - np.angle(vac_np))
    vmax_pd = min(math.pi, np.percentile(np.abs(phase_diff[np.abs(vac_np) > np.abs(vac_np).max()*0.05]), 99))
    axes1[1, 0].imshow(phase_diff, extent=e_mm, origin="lower", cmap="RdBu_r", vmin=-vmax_pd, vmax=vmax_pd)
    axes1[1, 0].set_title("Phase difference (turb - vac)", fontsize=11)
    axes1[1, 0].set_ylabel("Diagnostics", fontsize=11, fontweight="bold")

    # 1D irradiance profile
    x_mm = np.linspace(e_mm[0], e_mm[1], PILOT_N)
    axes1[1, 1].plot(x_mm, np.abs(vac_np[mid, :])**2, 'b-', lw=1.5, label='Vacuum')
    axes1[1, 1].plot(x_mm, np.abs(turb_np[mid, :])**2, 'r-', lw=1.5, alpha=0.7, label='Turbulent')
    axes1[1, 1].set_title("1D irradiance (center row)", fontsize=11)
    axes1[1, 1].legend(fontsize=9)
    axes1[1, 1].set_xlabel("mm")
    axes1[1, 1].grid(True, alpha=0.3)

    # 1D phase profile
    axes1[1, 2].plot(x_mm, np.angle(vac_np[mid, :]), 'b-', lw=1.5, label='Vacuum')
    axes1[1, 2].plot(x_mm, np.angle(turb_np[mid, :]), 'r-', lw=1.5, alpha=0.7, label='Turbulent')
    axes1[1, 2].set_title("1D phase (center row)", fontsize=11)
    axes1[1, 2].legend(fontsize=9)
    axes1[1, 2].set_xlabel("mm")
    axes1[1, 2].grid(True, alpha=0.3)

    # Text summary
    axes1[1, 3].axis("off")
    summary = (
        f"Cn² = 5e-14\n"
        f"D/r₀ = 5.0\n"
        f"Grid: {PILOT_N}×{PILOT_N}\n"
        f"Window: {WIN*1e3:.3f} mm\n"
        f"dx = {DX*1e6:.1f} μm\n"
        f"Aperture: {APT*1e3:.0f} mm\n\n"
        f"CO (no correction) = {co_nocorr:.4f}"
    )
    axes1[1, 3].text(0.1, 0.5, summary, fontsize=14, va="center", family="monospace",
                     bbox=dict(facecolor="lightyellow", edgecolor="gray"))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p1 = OUT / "pilot_fig1_input.png"
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    print(f"  Saved: {p1}")
    plt.close(fig1)

    # ═══════════════════════════════════════════════════════════
    # FIGURE 2: D2NN Output
    # ═══════════════════════════════════════════════════════════
    print("Generating Figure 2: D2NN output...")
    ncols = 2 + len(outputs)  # Vacuum, Turb(no D2NN), configs...
    fig2, axes2 = plt.subplots(4, ncols, figsize=(6*ncols, 24))
    fig2.suptitle("Figure 2: D2NN Output (Pilot: 2 layers, N=512, 30 epochs)",
                  fontsize=16, fontweight="bold")

    all_fields = [("Vacuum", vac_d2nn_np), ("Turb\n(no D2NN)", turb_d2nn_np)] + \
                 [(n, out_np[n]) for n in CONFIGS if n in out_np]
    imax_out = max((np.abs(f)**2).max() for _, f in all_fields)

    for col, (label, field) in enumerate(all_fields):
        # Row 0: Irradiance
        axes2[0, col].imshow(np.abs(field)**2, extent=e_mm, origin="lower", cmap="inferno",
                             vmin=0, vmax=imax_out)
        axes2[0, col].set_title(label, fontsize=12, fontweight="bold")
        # Row 1: Phase
        axes2[1, col].imshow(np.angle(field), extent=e_mm, origin="lower", cmap="twilight_shifted",
                             vmin=-math.pi, vmax=math.pi)
        # Row 2: Residual phase vs vacuum
        res = wrap_phase(np.angle(field) - np.angle(vac_d2nn_np))
        mask = np.abs(vac_d2nn_np) > np.abs(vac_d2nn_np).max() * 0.05
        vmax_r = min(math.pi, np.percentile(np.abs(res[mask]), 99) if mask.any() else math.pi)
        axes2[2, col].imshow(res, extent=e_mm, origin="lower", cmap="RdBu_r", vmin=-vmax_r, vmax=vmax_r)
        # Row 3: 1D irradiance
        axes2[3, col].plot(x_mm, np.abs(vac_d2nn_np[mid, :])**2, 'b--', lw=1, alpha=0.5, label='Vacuum')
        axes2[3, col].plot(x_mm, np.abs(field[mid, :])**2, 'r-', lw=1.5, label=label)
        axes2[3, col].legend(fontsize=8)
        axes2[3, col].grid(True, alpha=0.3)

    row_labels = ["Irradiance", "Phase", "Residual phase\n(vs vacuum)", "1D profile"]
    for row, label in enumerate(row_labels):
        axes2[row, 0].set_ylabel(label, fontsize=11, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p2 = OUT / "pilot_fig2_d2nn_output.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    print(f"  Saved: {p2}")
    plt.close(fig2)

    # ═══════════════════════════════════════════════════════════
    # FIGURE 3: Detector Plane
    # ═══════════════════════════════════════════════════════════
    print("Generating Figure 3: Detector plane...")
    fig3, axes3 = plt.subplots(3, ncols, figsize=(6*ncols, 18))
    fig3.suptitle("Figure 3: Detector Plane (after f=4.5mm focusing lens)",
                  fontsize=16, fontweight="bold")

    all_det = [("Vacuum", vac_det_np), ("Turb\n(no D2NN)", turb_det_np)] + \
              [(n, det_np[n]) for n in CONFIGS if n in det_np]

    # Zoom to central 64 pixels
    Z = 64
    c_det = PILOT_N // 2
    e_det_um = ext(2*Z, float(dx_det), 1e-6)
    imax_det = max((np.abs(f[c_det-Z:c_det+Z, c_det-Z:c_det+Z])**2).max() for _, f in all_det)

    for col, (label, field) in enumerate(all_det):
        crop = field[c_det-Z:c_det+Z, c_det-Z:c_det+Z]
        # Row 0: Irradiance
        axes3[0, col].imshow(np.abs(crop)**2, extent=e_det_um, origin="lower", cmap="inferno",
                             vmin=0, vmax=imax_det)
        axes3[0, col].set_title(label, fontsize=12, fontweight="bold")

        # Row 1: 1D cross-section
        x_um = np.linspace(e_det_um[0], e_det_um[1], 2*Z)
        axes3[1, col].plot(x_um, np.abs(vac_det_np[c_det, c_det-Z:c_det+Z])**2, 'b--', lw=1, alpha=0.5, label='Vacuum')
        axes3[1, col].plot(x_um, np.abs(field[c_det, c_det-Z:c_det+Z])**2, 'r-', lw=1.5, label=label)
        axes3[1, col].legend(fontsize=8)
        axes3[1, col].grid(True, alpha=0.3)
        axes3[1, col].set_xlabel("μm")

        # Row 2: Encircled energy
        irr_full = np.abs(field)**2
        yy, xx = np.mgrid[-c_det:PILOT_N-c_det, -c_det:PILOT_N-c_det]
        rsq = (xx * float(dx_det))**2 + (yy * float(dx_det))**2
        radii_um = np.linspace(1, 100, 100) * 1e-6
        ee_vac = []
        ee_field = []
        irr_vac_full = np.abs(vac_det_np)**2
        total_vac = irr_vac_full.sum()
        total_field = irr_full.sum()
        for r in radii_um:
            mask_r = rsq <= r**2
            ee_vac.append(irr_vac_full[mask_r].sum() / max(total_vac, 1e-30))
            ee_field.append(irr_full[mask_r].sum() / max(total_field, 1e-30))
        axes3[2, col].plot(radii_um*1e6, ee_vac, 'b--', lw=1, alpha=0.5, label='Vacuum')
        axes3[2, col].plot(radii_um*1e6, ee_field, 'r-', lw=1.5, label=label)
        axes3[2, col].axvline(50, color='gray', ls=':', lw=1, label='50μm')
        axes3[2, col].legend(fontsize=8)
        axes3[2, col].grid(True, alpha=0.3)
        axes3[2, col].set_xlabel("Radius [μm]")
        axes3[2, col].set_ylabel("Encircled energy")

    row_labels = ["Irradiance\n(zoomed)", "1D cross-section", "Encircled energy"]
    for row, label in enumerate(row_labels):
        axes3[row, 0].set_ylabel(label, fontsize=11, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p3 = OUT / "pilot_fig3_detector.png"
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    print(f"  Saved: {p3}")
    plt.close(fig3)

    # ═══════════════════════════════════════════════════════════
    # FIGURE 4: Phase Masks
    # ═══════════════════════════════════════════════════════════
    print("Generating Figure 4: Phase masks...")
    fig4, axes4 = plt.subplots(3, len(CONFIGS), figsize=(8*len(CONFIGS), 24))
    fig4.suptitle("Figure 4: Learned Phase Masks (2 layers × 3 configs)",
                  fontsize=16, fontweight="bold")

    for col, name in enumerate(CONFIGS):
        model = load_model(name)
        if model is None:
            continue
        for layer_idx in range(2):
            phase = torch.remainder(model.layers[layer_idx].phase, 2*math.pi).detach().cpu().numpy()
            im = axes4[layer_idx, col].imshow(phase, cmap="twilight_shifted", vmin=0, vmax=2*math.pi)
            axes4[layer_idx, col].set_title(f"{name}\nLayer {layer_idx}" if layer_idx == 0 else f"Layer {layer_idx}",
                                            fontsize=12)
            axes4[layer_idx, col].axis("off")
            if col == 0:
                axes4[layer_idx, 0].set_ylabel(f"Layer {layer_idx}", fontsize=12, fontweight="bold")

        # Row 2: FFT of Layer 0
        phase0 = torch.remainder(model.layers[0].phase, 2*math.pi).detach().cpu().numpy()
        spectrum = np.abs(np.fft.fftshift(np.fft.fft2(phase0)))**2
        spectrum_log = np.log10(spectrum + 1e-10)
        axes4[2, col].imshow(spectrum_log, cmap="viridis")
        axes4[2, col].set_title("FFT of Layer 0 (log)", fontsize=12)
        axes4[2, col].axis("off")

    axes4[2, 0].set_ylabel("Frequency\nspectrum", fontsize=12, fontweight="bold")
    fig4.colorbar(im, ax=axes4[1, :].tolist(), shrink=0.5, label="Phase [0, 2π)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p4 = OUT / "pilot_fig4_masks.png"
    fig4.savefig(p4, dpi=150, bbox_inches="tight")
    print(f"  Saved: {p4}")
    plt.close(fig4)

    # ═══════════════════════════════════════════════════════════
    # FIGURE 5: Training Curves
    # ═══════════════════════════════════════════════════════════
    print("Generating Figure 5: Training curves...")
    fig5, axes5 = plt.subplots(2, 1, figsize=(14, 12))
    fig5.suptitle("Figure 5: Training Curves (Pilot: 2-layer, N=512, 30 epochs)",
                  fontsize=16, fontweight="bold")

    colors = {'baseline_co': '#2ecc71', 'co_phasor': '#9b59b6', 'co_amp': '#e74c3c'}

    for name in CONFIGS:
        rpath = SWEEP_DIR / name / "results.json"
        if not rpath.exists():
            continue
        with open(rpath) as f:
            r = json.load(f)
        h = r.get("history", {})
        if not h.get("epoch"):
            continue
        # Loss
        axes5[0].plot(h["epoch"], h["loss"], color=colors.get(name, 'gray'), lw=2, marker='o',
                      markersize=4, label=name)
        # Val CO
        axes5[1].plot(h["epoch"], h["val_co"], color=colors.get(name, 'gray'), lw=2, marker='o',
                      markersize=4, label=name)

    axes5[0].set_xlabel("Epoch", fontsize=12)
    axes5[0].set_ylabel("Loss", fontsize=12)
    axes5[0].set_title("Training Loss", fontsize=13, fontweight="bold")
    axes5[0].legend(fontsize=11)
    axes5[0].grid(True, alpha=0.3)

    # Add baseline CO line
    bl_co = 0.4985  # from pilot results
    axes5[1].axhline(bl_co, color='k', ls='--', lw=1.5, label=f'no_correction ({bl_co:.4f})')
    axes5[1].set_xlabel("Epoch", fontsize=12)
    axes5[1].set_ylabel("Validation CO", fontsize=12)
    axes5[1].set_title("Validation CO (dashed = no correction baseline)", fontsize=13, fontweight="bold")
    axes5[1].legend(fontsize=11)
    axes5[1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p5 = OUT / "pilot_fig5_training_curves.png"
    fig5.savefig(p5, dpi=150, bbox_inches="tight")
    print(f"  Saved: {p5}")
    plt.close(fig5)

    # ═══════════════════════════════════════════════════════════
    # FIGURE 6: Multi-Realization Statistics
    # ═══════════════════════════════════════════════════════════
    print("Generating Figure 6: Multi-realization statistics...")

    # Collect metrics over all test samples
    all_data = {n: {"co": [], "wf_rms": []} for n in ["no_d2nn"] + CONFIGS}

    models = {}
    for name in CONFIGS:
        m = load_model(name)
        if m:
            models[name] = m.to(device)

    d0 = BeamCleanupD2NN(n=PILOT_N, wavelength_m=W, window_m=WIN, **ARCH).to(device)
    d0.eval()

    def wf_rms(pred, target):
        p_ph = torch.angle(pred[0])
        t_ph = torch.angle(target[0])
        diff = torch.remainder(p_ph - t_ph + math.pi, 2*math.pi) - math.pi
        w = target[0].abs().square()
        w = w / w.sum()
        return torch.sqrt((w * diff.square()).sum()).item()

    for si in range(len(ds)):
        s_i = ds[si]
        ut = prepare(s_i["u_turb"].unsqueeze(0).to(device))
        uv = prepare(s_i["u_vacuum"].unsqueeze(0).to(device))
        with torch.no_grad():
            ut_out = d0(ut)
            uv_out = d0(uv)

        co_val = complex_overlap(ut_out, uv_out).item()
        wr = wf_rms(ut_out, uv_out)
        all_data["no_d2nn"]["co"].append(co_val)
        all_data["no_d2nn"]["wf_rms"].append(wr)

        for name, model in models.items():
            with torch.no_grad():
                pred = model(ut)
            co_val = complex_overlap(pred, uv_out).item()
            wr = wf_rms(pred, uv_out)
            all_data[name]["co"].append(co_val)
            all_data[name]["wf_rms"].append(wr)

        if (si + 1) % 10 == 0:
            print(f"  [{si+1}/{len(ds)}]")

    del d0
    torch.cuda.empty_cache()

    # Plot
    fig6, axes6 = plt.subplots(1, 2, figsize=(18, 8))
    fig6.suptitle("Figure 6: Multi-Realization Statistics (test set, 50 samples)",
                  fontsize=16, fontweight="bold")

    all_names = ["no_d2nn"] + CONFIGS
    bp_colors = ['gray', '#2ecc71', '#9b59b6', '#e74c3c']

    # Panel A: CO box plot
    co_data = [all_data[n]["co"] for n in all_names]
    bp = axes6[0].boxplot(co_data, labels=["no_d2nn"] + CONFIGS, patch_artist=True)
    for patch, color in zip(bp['boxes'], bp_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    axes6[0].set_ylabel("Complex Overlap", fontsize=12)
    axes6[0].set_title("CO Distribution", fontsize=13, fontweight="bold")
    axes6[0].tick_params(axis='x', rotation=20)
    axes6[0].grid(True, alpha=0.3, axis='y')

    # Panel B: WF RMS box plot (in nm)
    wf_data = [np.array(all_data[n]["wf_rms"]) * W / (2*math.pi) * 1e9 for n in all_names]
    bp = axes6[1].boxplot(wf_data, labels=["no_d2nn"] + CONFIGS, patch_artist=True)
    for patch, color in zip(bp['boxes'], bp_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    axes6[1].set_ylabel("WF RMS [nm]", fontsize=12)
    axes6[1].set_title("Wavefront RMS Distribution", fontsize=13, fontweight="bold")
    axes6[1].tick_params(axis='x', rotation=20)
    axes6[1].grid(True, alpha=0.3, axis='y')

    # Print stats
    print("\n--- Statistics ---")
    for name in all_names:
        co_arr = np.array(all_data[name]["co"])
        wf_arr = np.array(all_data[name]["wf_rms"]) * W / (2*math.pi) * 1e9
        print(f"  {name:>15}: CO={co_arr.mean():.4f}±{co_arr.std():.4f}, WF={wf_arr.mean():.1f}±{wf_arr.std():.1f}nm")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p6 = OUT / "pilot_fig6_statistics.png"
    fig6.savefig(p6, dpi=150, bbox_inches="tight")
    print(f"  Saved: {p6}")
    plt.close(fig6)

    print(f"\nDone! 6 figures saved to {OUT}")


if __name__ == "__main__":
    main()
