#!/usr/bin/env python
"""Phase 2 Comprehensive Visualization — D2NN Telescope Sweep.

Figure 1: Input (telescope receiver → beam reducer)
Figure 2: D2NN output (5 configs vs vacuum reference)
Figure 3: Detector plane (after focusing lens f=4.5mm)

Usage:
    cd /root/dj/D2NN/kim2026 && python scripts/visualize_phase2_report.py
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
from kim2026.training.targets import apply_receiver_aperture
from kim2026.training.metrics import complex_overlap

# ─── Constants ────────────────────────────────────────────────
W = 1.55e-6
N = 1024
WIN = 0.002048          # after 75:1 beam reducer
PHYS_WIN = 0.1536       # telescope aperture (153.6mm)
APT = 0.002
FOCUS_F = 4.5e-3
DX = WIN / N
PHYS_DX = PHYS_WIN / N

D2NN_ARCH = dict(num_layers=5, layer_spacing_m=10e-3, detector_distance_m=10e-3)
CONFIGS = ["baseline_co", "co_amp", "co_ffp", "co_phasor", "roi80"]

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kim2026" / "1km_cn2e-14_tel15cm_n1024_br75"
SWEEP_DIR = Path(__file__).resolve().parent.parent / "autoresearch" / "runs" / "d2nn_sweep_telescope"
OUT = SWEEP_DIR

# ─── Helpers ──────────────────────────────────────────────────
def load_d2nn(name):
    ckpt = SWEEP_DIR / name / "checkpoint.pt"
    if not ckpt.exists():
        return None
    m = BeamCleanupD2NN(n=N, wavelength_m=W, window_m=WIN, **D2NN_ARCH)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["model_state_dict"])
    m.eval()
    return m

def ext(n, dx, unit):
    h = n * dx / 2 / unit
    return [-h, h, -h, h]

def zoom(f, z):
    c = f.shape[-1] // 2
    return f[..., c-z:c+z, c-z:c+z]

def wrap_phase(p):
    return (p + np.pi) % (2 * np.pi) - np.pi

def plot_irr(ax, f, extent, title, vmax=None):
    d = np.abs(f)**2
    im = ax.imshow(d, extent=extent, origin="lower", cmap="inferno", vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=10)
    return im

def plot_phase(ax, f, extent, title):
    im = ax.imshow(np.angle(f), extent=extent, origin="lower",
                    cmap="twilight_shifted", vmin=-math.pi, vmax=math.pi)
    ax.set_title(title, fontsize=10)
    return im

def plot_residual(ax, f, f_vac, extent, title):
    res = wrap_phase(np.angle(f) - np.angle(f_vac))
    vmax = min(np.pi, np.percentile(np.abs(res[np.abs(f_vac) > np.abs(f_vac).max()*0.05]), 99))
    im = ax.imshow(res, extent=extent, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=10)
    return im


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ─── Load data ────────────────────────────────────────────
    ds = CachedFieldDataset(cache_dir=str(DATA_DIR / "cache"),
                             manifest_path=str(DATA_DIR / "split_manifest.json"), split="test")
    s = ds[0]
    u_turb_raw = s["u_turb"].unsqueeze(0).to(device)
    u_vac_raw = s["u_vacuum"].unsqueeze(0).to(device)

    u_turb = apply_receiver_aperture(u_turb_raw, receiver_window_m=WIN, aperture_diameter_m=APT)
    u_vac = apply_receiver_aperture(u_vac_raw, receiver_window_m=WIN, aperture_diameter_m=APT)

    # ─── Load models + inference ──────────────────────────────
    outputs = {}  # name → output field
    for name in CONFIGS:
        model = load_d2nn(name)
        if model is not None:
            model = model.to(device)
            with torch.no_grad():
                outputs[name] = model(u_turb)
            co = complex_overlap(outputs[name], u_vac).item()
            print(f"  {name:>15}: CO={co:.4f}")
        else:
            print(f"  {name:>15}: checkpoint not found, skipping")

    # Vacuum & turbulent through zero-phase D2NN (= no correction baseline)
    d2nn_zero = BeamCleanupD2NN(n=N, wavelength_m=W, window_m=WIN, **D2NN_ARCH).to(device)
    d2nn_zero.eval()
    with torch.no_grad():
        u_vac_d2nn = d2nn_zero(u_vac)    # vacuum through zero-phase D2NN
        u_turb_d2nn = d2nn_zero(u_turb)  # turbulent through zero-phase D2NN (= no correction)
    co_nocorr = complex_overlap(u_turb_d2nn, u_vac_d2nn).item()
    print(f"  {'no_correction':>15}: CO={co_nocorr:.4f} (turbulent, no D2NN)")

    # ─── Focus to detector ────────────────────────────────────
    def focus(field):
        with torch.no_grad():
            f, dx_f = lens_2f_forward(field.to(torch.complex64), dx_in_m=DX,
                                       wavelength_m=W, f_m=FOCUS_F, na=None, apply_scaling=False)
        return f, dx_f

    u_vac_det, dx_det = focus(u_vac_d2nn)
    u_turb_det, _ = focus(u_turb_d2nn)  # turbulent through zero-phase D2NN
    det_outputs = {}
    for name, out in outputs.items():
        det_outputs[name], _ = focus(out)

    # ─── To numpy ─────────────────────────────────────────────
    vac_np = u_vac[0].cpu().numpy()
    turb_np = u_turb[0].cpu().numpy()
    vac_d2nn_np = u_vac_d2nn[0].cpu().numpy()
    turb_d2nn_np = u_turb_d2nn[0].cpu().numpy()
    out_np = {n: o[0].cpu().numpy() for n, o in outputs.items()}

    vac_det_np = u_vac_det[0].cpu().numpy()
    turb_det_np = u_turb_det[0].cpu().numpy()
    det_np = {n: o[0].cpu().numpy() for n, o in det_outputs.items()}

    mid = N // 2

    # ═══════════════════════════════════════════════════════════
    # FIGURE 1: Input (Stage 1-2)
    # ═══════════════════════════════════════════════════════════
    print("\nGenerating Figure 1: Input...")
    fig1, axes1 = plt.subplots(3, 4, figsize=(24, 18))
    fig1.suptitle("Stage 1-2: Input Field (Telescope → Beam Reducer)", fontsize=16, fontweight="bold")

    # Row 0: Physical telescope coordinates
    pe = ext(N, PHYS_DX, 1e-3)  # mm
    imax_phys = (np.abs(vac_np)**2).max().item()
    plot_irr(axes1[0,0], vac_np, pe, "Vacuum irradiance\n(telescope, 153.6mm)", imax_phys)
    axes1[0,0].set_xlabel("mm"); axes1[0,0].set_ylabel("Stage 1: Telescope\n(before beam reducer)", fontsize=11, fontweight="bold")
    plot_phase(axes1[0,1], vac_np, pe, "Vacuum phase\n(telescope)")
    axes1[0,1].set_xlabel("mm")
    plot_irr(axes1[0,2], turb_np, pe, "Turbulent irradiance\n(telescope)", imax_phys)
    axes1[0,2].set_xlabel("mm")
    plot_phase(axes1[0,3], turb_np, pe, "Turbulent phase\n(telescope)")
    axes1[0,3].set_xlabel("mm")

    # Row 1: After beam reducer (2mm coordinates)
    re = ext(N, DX, 1e-3)  # mm
    imax_recv = (np.abs(vac_np)**2).max().item()
    plot_irr(axes1[1,0], vac_np, re, "Vacuum irradiance\n(after 75:1, 2.048mm)", imax_recv)
    axes1[1,0].set_xlabel("mm"); axes1[1,0].set_ylabel("Stage 2: After beam reducer\n(D2NN input plane)", fontsize=11, fontweight="bold")
    plot_phase(axes1[1,1], vac_np, re, "Vacuum phase\n(after 75:1)")
    axes1[1,1].set_xlabel("mm")
    plot_irr(axes1[1,2], turb_np, re, "Turbulent irradiance\n(after 75:1)", imax_recv)
    axes1[1,2].set_xlabel("mm")
    plot_phase(axes1[1,3], turb_np, re, "Turbulent phase\n(after 75:1)")
    axes1[1,3].set_xlabel("mm")

    # Row 2: 1D cross-sections
    x_mm = (np.arange(N) - mid) * DX * 1e3
    axes1[2,0].plot(x_mm, np.abs(vac_np[mid,:])**2, 'b-', lw=1)
    axes1[2,0].set_title("Vacuum irradiance profile", fontsize=10)
    axes1[2,0].set_xlabel("mm"); axes1[2,0].set_ylabel("1D Cross-section", fontsize=11, fontweight="bold")

    axes1[2,1].plot(x_mm, np.angle(vac_np[mid,:]), 'b-', lw=1)
    axes1[2,1].set_title("Vacuum phase profile", fontsize=10)
    axes1[2,1].set_xlabel("mm"); axes1[2,1].set_ylim(-np.pi, np.pi)

    axes1[2,2].plot(x_mm, np.abs(turb_np[mid,:])**2, 'r-', lw=1)
    axes1[2,2].plot(x_mm, np.abs(vac_np[mid,:])**2, 'b--', lw=0.5, alpha=0.5, label='vacuum')
    axes1[2,2].set_title("Turbulent irradiance profile", fontsize=10)
    axes1[2,2].set_xlabel("mm"); axes1[2,2].legend(fontsize=8)

    axes1[2,3].plot(x_mm, np.angle(turb_np[mid,:]), 'r-', lw=1)
    axes1[2,3].plot(x_mm, np.angle(vac_np[mid,:]), 'b--', lw=0.5, alpha=0.5, label='vacuum')
    axes1[2,3].set_title("Turbulent phase profile", fontsize=10)
    axes1[2,3].set_xlabel("mm"); axes1[2,3].set_ylim(-np.pi, np.pi); axes1[2,3].legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p1 = OUT / "phase2_fig1_input.png"
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    print(f"  Saved: {p1}")
    plt.close(fig1)

    # ═══════════════════════════════════════════════════════════
    # FIGURE 2: D2NN Output (Stage 3)
    # ═══════════════════════════════════════════════════════════
    print("Generating Figure 2: D2NN output...")
    cols = (["Vacuum\n(target)"]
            + [f"Turbulent\n(no D2NN)\nCO={co_nocorr:.4f}"]
            + [f"{n}\nCO={complex_overlap(outputs[n], u_vac).item():.4f}" for n in CONFIGS if n in outputs])
    fields_3 = [vac_d2nn_np] + [turb_d2nn_np] + [out_np[n] for n in CONFIGS if n in out_np]
    ncols = len(cols)

    fig2, axes2 = plt.subplots(4, ncols, figsize=(5*ncols, 20))
    fig2.suptitle("Stage 3: D2NN Output (before focusing lens)", fontsize=16, fontweight="bold")

    # Vacuum irradiance를 absolute reference로 사용 (relative 금지)
    imax3 = (np.abs(vac_d2nn_np)**2).max()

    for col, (f, label) in enumerate(zip(fields_3, cols)):
        plot_irr(axes2[0,col], f, re, f"Irradiance: {label}", imax3)
        axes2[0,col].set_xlabel("mm")

        plot_phase(axes2[1,col], f, re, f"Phase: {label.split(chr(10))[0]}")
        axes2[1,col].set_xlabel("mm")

        if col == 0:
            # Vacuum residual = 0 (self-reference)
            axes2[2,col].imshow(np.zeros((N,N)), extent=re, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
            axes2[2,col].set_title("Residual: (reference=0)", fontsize=10)
        else:
            plot_residual(axes2[2,col], f, vac_np, re, f"Residual: {label.split(chr(10))[0]}")
        axes2[2,col].set_xlabel("mm")

        # 1D cross-section
        axes2[3,col].plot(x_mm, np.abs(f[mid,:])**2 / max(imax3, 1e-30), 'k-', lw=1, label='irr')
        axes2[3,col].plot(x_mm, np.abs(vac_np[mid,:])**2 / max(imax3, 1e-30), 'b--', lw=0.5, alpha=0.5, label='vac')
        ax2r = axes2[3,col].twinx()
        phase_diff = wrap_phase(np.angle(f[mid,:]) - np.angle(vac_np[mid,:]))
        ax2r.plot(x_mm, phase_diff, 'r-', lw=0.5, alpha=0.7, label='res.phase')
        ax2r.set_ylim(-np.pi, np.pi)
        ax2r.set_ylabel("Residual phase [rad]", fontsize=8, color='r')
        axes2[3,col].set_xlabel("mm")
        axes2[3,col].set_title(f"1D: {label.split(chr(10))[0]}", fontsize=9)
        if col == 0:
            axes2[3,col].legend(fontsize=7, loc="upper left")

    axes2[0,0].set_ylabel("Irradiance", fontsize=12, fontweight="bold")
    axes2[1,0].set_ylabel("Phase", fontsize=12, fontweight="bold")
    axes2[2,0].set_ylabel("Residual Phase", fontsize=12, fontweight="bold")
    axes2[3,0].set_ylabel("1D Cross-section", fontsize=12, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p2 = OUT / "phase2_fig2_d2nn_output.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    print(f"  Saved: {p2}")
    plt.close(fig2)

    # ═══════════════════════════════════════════════════════════
    # FIGURE 3: Detector Plane (Stage 4)
    # ═══════════════════════════════════════════════════════════
    print("Generating Figure 3: Detector plane...")
    det_fields = [vac_det_np, turb_det_np] + [det_np[n] for n in CONFIGS if n in det_np]

    fig3 = plt.figure(figsize=(5*ncols, 28))
    gs3 = gridspec.GridSpec(5, ncols, figure=fig3, hspace=0.35, wspace=0.3,
                            height_ratios=[1, 1, 1, 0.6, 0.6])
    fig3.suptitle("Stage 4: Detector Plane (after focusing lens f=4.5mm)", fontsize=16, fontweight="bold")

    zp = 80  # zoom pixels
    de = ext(2*zp, dx_det, 1e-6)  # um
    # Vacuum detector를 absolute reference로 사용
    det_imax = (np.abs(zoom(vac_det_np, zp))**2).max()

    for col, (f, label) in enumerate(zip(det_fields, cols)):
        fz = zoom(f, zp)

        ax0 = fig3.add_subplot(gs3[0, col])
        plot_irr(ax0, fz, de, f"Irradiance: {label}", det_imax)
        ax0.set_xlabel("um")
        if col == 0:
            ax0.set_ylabel("Irradiance", fontsize=12, fontweight="bold")

        ax1 = fig3.add_subplot(gs3[1, col])
        plot_phase(ax1, fz, de, f"Phase: {label.split(chr(10))[0]}")
        ax1.set_xlabel("um")
        if col == 0:
            ax1.set_ylabel("Phase", fontsize=12, fontweight="bold")

        ax2 = fig3.add_subplot(gs3[2, col])
        vac_det_z = zoom(vac_det_np, zp)
        if col == 0:
            ax2.imshow(np.zeros((2*zp, 2*zp)), extent=de, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
            ax2.set_title("Residual: (reference=0)", fontsize=10)
        else:
            plot_residual(ax2, fz, vac_det_z, de, f"Residual: {label.split(chr(10))[0]}")
        ax2.set_xlabel("um")
        if col == 0:
            ax2.set_ylabel("Residual Phase", fontsize=12, fontweight="bold")

    # Row 3: 1D cross-sections (all overlaid)
    ax_1d = fig3.add_subplot(gs3[3, :])
    x_um = (np.arange(2*zp) - zp) * dx_det * 1e6
    colors = ['blue', 'gray', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']
    labels_1d = ["Vacuum", "Turb(no D2NN)"] + [n for n in CONFIGS if n in det_np]
    for i, (f, lbl) in enumerate(zip(det_fields, labels_1d)):
        fz = zoom(f, zp)
        profile = np.abs(fz[zp, :])**2
        ax_1d.plot(x_um, profile / max(det_imax, 1e-30), color=colors[i], lw=1.5 if i==0 else 1, label=lbl,
                    ls='--' if i==0 else '-')
    ax_1d.set_xlabel("Position [um]", fontsize=12)
    ax_1d.set_ylabel("Normalized irradiance", fontsize=12)
    ax_1d.set_title("1D Cross-section (all configs)", fontsize=12, fontweight="bold")
    ax_1d.legend(fontsize=9, ncol=3)
    ax_1d.set_xlim(-150, 150)
    ax_1d.grid(True, alpha=0.3)

    # Row 4: Encircled energy
    ax_ee = fig3.add_subplot(gs3[4, :])
    radii_px = np.arange(1, zp)
    radii_um = radii_px * dx_det * 1e6
    c = N // 2
    yy, xx = np.mgrid[-c:N-c, -c:N-c]
    r_sq = (xx * dx_det)**2 + (yy * dx_det)**2

    for i, (f_full, lbl) in enumerate(zip([vac_det_np] + [det_np[n] for n in CONFIGS if n in det_np], labels_1d)):
        irr = np.abs(f_full)**2
        total = irr.sum()
        ee = []
        for r_px in radii_px:
            r_m = r_px * dx_det
            ee.append((irr[r_sq <= r_m**2].sum() / max(total, 1e-30)))
        ax_ee.plot(radii_um, ee, color=colors[i], lw=1.5 if i==0 else 1, label=lbl,
                    ls='--' if i==0 else '-')
    ax_ee.set_xlabel("Bucket radius [um]", fontsize=12)
    ax_ee.set_ylabel("Encircled energy fraction", fontsize=12)
    ax_ee.set_title("Encircled Energy (Power in Bucket)", fontsize=12, fontweight="bold")
    ax_ee.legend(fontsize=9, ncol=3)
    ax_ee.set_xlim(0, 200)
    ax_ee.set_ylim(0, 1.05)
    ax_ee.grid(True, alpha=0.3)
    ax_ee.axhline(0.84, color='gray', ls=':', lw=0.5, label='84% (Strehl)')

    p3 = OUT / "phase2_fig3_detector.png"
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    print(f"  Saved: {p3}")
    plt.close(fig3)

    print("\nDone! All 3 figures generated.")


if __name__ == "__main__":
    main()
