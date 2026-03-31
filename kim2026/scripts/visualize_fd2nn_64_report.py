#!/usr/bin/env python
"""Phase 2 Visualization for FD2NN-64 Cropped Fourier Sweep.

Figure 1: Input field (strong turbulence)
Figure 2: FD2NN output (n_mask=32/48/64, baseline_co)
Figure 3: Detector plane (focused)
Figure 4: Phase masks (5 layers × 3 n_mask sizes)
Figure 5: Training curves
Figure 6: n_mask comparison bar chart + multi-realization stats

Usage:
    cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python scripts/visualize_fd2nn_64_report.py
"""
from __future__ import annotations
import json, math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

from kim2026.data.dataset import CachedFieldDataset
from kim2026.models.fd2nn import CroppedFourierD2NN
from kim2026.optics.lens_2f import lens_2f_forward
from kim2026.training.targets import apply_receiver_aperture, center_crop_field
from kim2026.training.metrics import complex_overlap

W = 1.55e-6
N = 1024
WIN = 0.002048
APT = 0.002
DX = WIN / N
FOCUS_F = 4.5e-3

OPTICS = dict(dual_2f_f1_m=25e-3, dual_2f_f2_m=25e-3,
              dual_2f_na1=0.508, dual_2f_na2=0.508, dual_2f_apply_scaling=False)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kim2026" / "1km_cn2_5e-14_tel15cm_n1024_br75"
SWEEP_DIR = Path(__file__).resolve().parent.parent / "autoresearch" / "runs" / "fd2nn_64_strong_turb"
OUT = SWEEP_DIR

# Best config per n_mask (baseline_co since co_phasor is worse)
SHOW_RUNS = ["nm32_baseline_co", "nm48_baseline_co", "nm64_baseline_co"]
N_MASKS = [32, 48, 64]


def load_model(name, n_mask):
    ckpt = SWEEP_DIR / name / "checkpoint.pt"
    if not ckpt.exists():
        return None
    m = CroppedFourierD2NN(n_input=N, n_mask=n_mask, wavelength_m=W, window_m=WIN,
                            num_layers=5, layer_spacing_m=5e-3, **OPTICS)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["model_state_dict"])
    m.eval()
    return m


def prepare(field):
    a = apply_receiver_aperture(field, receiver_window_m=WIN, aperture_diameter_m=APT)
    return center_crop_field(a, crop_n=N)


def focus(field):
    with torch.no_grad():
        f, dx_f = lens_2f_forward(field.to(torch.complex64), dx_in_m=DX,
                                   wavelength_m=W, f_m=FOCUS_F, na=None, apply_scaling=False)
    return f, dx_f


def ext(n, dx, unit):
    h = n * dx / 2 / unit
    return [-h, h, -h, h]


def wrap_phase(p):
    return (p + np.pi) % (2 * np.pi) - np.pi


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = CachedFieldDataset(cache_dir=str(DATA_DIR / "cache"),
                             manifest_path=str(DATA_DIR / "split_manifest.json"), split="test")
    s = ds[0]
    u_turb_raw = s["u_turb"].unsqueeze(0).to(device)
    u_vac_raw = s["u_vacuum"].unsqueeze(0).to(device)
    u_turb = prepare(u_turb_raw)
    u_vac = prepare(u_vac_raw)

    # Load models
    outputs = {}
    for run_name, nm in zip(SHOW_RUNS, N_MASKS):
        model = load_model(run_name, nm)
        if model:
            model = model.to(device)
            with torch.no_grad():
                outputs[run_name] = model(u_turb)
            co = complex_overlap(outputs[run_name], u_vac).item()
            print(f"  {run_name}: CO={co:.4f}")

    # No-correction baseline: raw turbulent vs vacuum, NO 2f system
    # Fair comparison = what happens without FD2NN at all
    co_bl = complex_overlap(u_turb, u_vac).item()
    print(f"  no_correction (no 2f): CO={co_bl:.4f}")

    # Focus
    u_vac_det, dx_det = focus(u_vac)
    u_turb_det, _ = focus(u_turb)  # turbulent WITHOUT 2f, directly focused
    det_outputs = {n: focus(o)[0] for n, o in outputs.items()}

    # numpy
    vac_np = u_vac[0].cpu().numpy()
    turb_np = u_turb[0].cpu().numpy()
    out_np = {n: o[0].cpu().numpy() for n, o in outputs.items()}
    vac_det_np = u_vac_det[0].cpu().numpy()
    turb_det_np = u_turb_det[0].cpu().numpy()
    det_np = {n: o[0].cpu().numpy() for n, o in det_outputs.items()}

    e_mm = ext(N, DX, 1e-3)
    mid = N // 2

    # ═══ FIGURE 1: Input ═══
    print("\nFigure 1: Input...")
    fig1, ax1 = plt.subplots(2, 4, figsize=(24, 12))
    fig1.suptitle("Figure 1: Input Field (Cn²=5e-14, D/r₀=5.0)", fontsize=16, fontweight="bold")
    imax = (np.abs(vac_np)**2).max()
    ax1[0,0].imshow(np.abs(vac_np)**2, extent=e_mm, origin="lower", cmap="inferno", vmin=0, vmax=imax)
    ax1[0,0].set_title("Vacuum irradiance"); ax1[0,0].set_ylabel("Before FD2NN", fontsize=11, fontweight="bold")
    ax1[0,1].imshow(np.angle(vac_np), extent=e_mm, origin="lower", cmap="twilight_shifted", vmin=-math.pi, vmax=math.pi)
    ax1[0,1].set_title("Vacuum phase")
    ax1[0,2].imshow(np.abs(turb_np)**2, extent=e_mm, origin="lower", cmap="inferno", vmin=0, vmax=imax)
    ax1[0,2].set_title("Turbulent irradiance")
    ax1[0,3].imshow(np.angle(turb_np), extent=e_mm, origin="lower", cmap="twilight_shifted", vmin=-math.pi, vmax=math.pi)
    ax1[0,3].set_title("Turbulent phase")

    x_mm = np.linspace(e_mm[0], e_mm[1], N)
    ax1[1,0].imshow(wrap_phase(np.angle(turb_np) - np.angle(vac_np)), extent=e_mm, origin="lower", cmap="RdBu_r")
    ax1[1,0].set_title("Phase diff"); ax1[1,0].set_ylabel("Diagnostics", fontsize=11, fontweight="bold")
    ax1[1,1].plot(x_mm, np.abs(vac_np[mid,:])**2, 'b-', lw=1.5, label='Vac')
    ax1[1,1].plot(x_mm, np.abs(turb_np[mid,:])**2, 'r-', lw=1.5, alpha=0.7, label='Turb')
    ax1[1,1].legend(); ax1[1,1].set_title("1D irradiance"); ax1[1,1].grid(True, alpha=0.3)
    ax1[1,2].plot(x_mm, np.angle(vac_np[mid,:]), 'b-', lw=1.5, label='Vac')
    ax1[1,2].plot(x_mm, np.angle(turb_np[mid,:]), 'r-', lw=1.5, alpha=0.7, label='Turb')
    ax1[1,2].legend(); ax1[1,2].set_title("1D phase"); ax1[1,2].grid(True, alpha=0.3)
    ax1[1,3].axis("off")
    ax1[1,3].text(0.1, 0.5, f"Cn²=5e-14\nD/r₀=5.0\nBaseline CO={co_bl:.4f}\nFD2NN crop/pad", fontsize=14, va="center",
                  family="monospace", bbox=dict(facecolor="lightyellow", edgecolor="gray"))
    plt.tight_layout(rect=[0,0,1,0.95])
    fig1.savefig(OUT / "phase2_fig1_input.png", dpi=150, bbox_inches="tight"); plt.close(fig1)
    print(f"  Saved: phase2_fig1_input.png")

    # ═══ FIGURE 2: FD2NN Output ═══
    print("Figure 2: FD2NN output...")
    cols = [("Vacuum", vac_np), ("Turb\n(no FD2NN)", turb_np)]
    for rn in SHOW_RUNS:
        if rn in out_np:
            nm = rn.split("_")[0].replace("nm","")
            cols.append((f"n_mask={nm}", out_np[rn]))
    ncols = len(cols)
    fig2, ax2 = plt.subplots(4, ncols, figsize=(6*ncols, 24))
    fig2.suptitle("Figure 2: FD2NN Output (Cropped Fourier, baseline_co)", fontsize=16, fontweight="bold")
    imax_out = max((np.abs(f)**2).max() for _, f in cols)
    for col, (label, field) in enumerate(cols):
        ax2[0,col].imshow(np.abs(field)**2, extent=e_mm, origin="lower", cmap="inferno", vmin=0, vmax=imax_out)
        ax2[0,col].set_title(label, fontsize=12, fontweight="bold")
        ax2[1,col].imshow(np.angle(field), extent=e_mm, origin="lower", cmap="twilight_shifted", vmin=-math.pi, vmax=math.pi)
        res = wrap_phase(np.angle(field) - np.angle(vac_np))
        ax2[2,col].imshow(res, extent=e_mm, origin="lower", cmap="RdBu_r", vmin=-math.pi, vmax=math.pi)
        ax2[3,col].plot(x_mm, np.abs(vac_np[mid,:])**2, 'b--', lw=1, alpha=0.5, label='Vacuum')
        ax2[3,col].plot(x_mm, np.abs(field[mid,:])**2, 'r-', lw=1.5, label=label)
        ax2[3,col].legend(fontsize=8); ax2[3,col].grid(True, alpha=0.3)
    for r, lbl in enumerate(["Irradiance", "Phase", "Residual phase", "1D profile"]):
        ax2[r,0].set_ylabel(lbl, fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.95])
    fig2.savefig(OUT / "phase2_fig2_fd2nn_output.png", dpi=150, bbox_inches="tight"); plt.close(fig2)
    print(f"  Saved: phase2_fig2_fd2nn_output.png")

    # ═══ FIGURE 3: Detector ═══
    print("Figure 3: Detector...")
    det_cols = [("Vacuum", vac_det_np), ("Turb\n(no FD2NN)", turb_det_np)]
    for rn in SHOW_RUNS:
        if rn in det_np:
            nm = rn.split("_")[0].replace("nm","")
            det_cols.append((f"n_mask={nm}", det_np[rn]))
    ncols = len(det_cols)
    fig3, ax3 = plt.subplots(3, ncols, figsize=(6*ncols, 18))
    fig3.suptitle("Figure 3: Detector Plane (f=4.5mm focus)", fontsize=16, fontweight="bold")
    Z = 64; c_det = N // 2
    e_um = ext(2*Z, float(dx_det), 1e-6)
    imax_d = max((np.abs(f[c_det-Z:c_det+Z, c_det-Z:c_det+Z])**2).max() for _, f in det_cols)
    for col, (label, field) in enumerate(det_cols):
        crop = field[c_det-Z:c_det+Z, c_det-Z:c_det+Z]
        ax3[0,col].imshow(np.abs(crop)**2, extent=e_um, origin="lower", cmap="inferno", vmin=0, vmax=imax_d)
        ax3[0,col].set_title(label, fontsize=12, fontweight="bold")
        x_um = np.linspace(e_um[0], e_um[1], 2*Z)
        ax3[1,col].plot(x_um, np.abs(vac_det_np[c_det,c_det-Z:c_det+Z])**2, 'b--', lw=1, alpha=0.5, label='Vac')
        ax3[1,col].plot(x_um, np.abs(field[c_det,c_det-Z:c_det+Z])**2, 'r-', lw=1.5, label=label)
        ax3[1,col].legend(fontsize=8); ax3[1,col].grid(True, alpha=0.3); ax3[1,col].set_xlabel("μm")
        irr = np.abs(field)**2; irr_v = np.abs(vac_det_np)**2
        yy, xx = np.mgrid[-c_det:N-c_det, -c_det:N-c_det]
        rsq = (xx*float(dx_det))**2 + (yy*float(dx_det))**2
        radii = np.linspace(1,100,100)*1e-6
        ee_v = [irr_v[rsq<=r**2].sum()/max(irr_v.sum(),1e-30) for r in radii]
        ee_f = [irr[rsq<=r**2].sum()/max(irr.sum(),1e-30) for r in radii]
        ax3[2,col].plot(radii*1e6, ee_v, 'b--', lw=1, alpha=0.5, label='Vac')
        ax3[2,col].plot(radii*1e6, ee_f, 'r-', lw=1.5, label=label)
        ax3[2,col].axvline(50, color='gray', ls=':', lw=1); ax3[2,col].legend(fontsize=8)
        ax3[2,col].grid(True, alpha=0.3); ax3[2,col].set_xlabel("Radius [μm]")
    for r, lbl in enumerate(["Irradiance (zoom)", "1D cross-section", "Encircled energy"]):
        ax3[r,0].set_ylabel(lbl, fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.95])
    fig3.savefig(OUT / "phase2_fig3_detector.png", dpi=150, bbox_inches="tight"); plt.close(fig3)
    print(f"  Saved: phase2_fig3_detector.png")

    # ═══ FIGURE 4: Phase masks ═══
    print("Figure 4: Phase masks...")
    fig4, ax4 = plt.subplots(6, 3, figsize=(18, 36))
    fig4.suptitle("Figure 4: Learned Fourier Phase Masks", fontsize=16, fontweight="bold")
    for col, (rn, nm) in enumerate(zip(SHOW_RUNS, N_MASKS)):
        model = load_model(rn, nm)
        if not model: continue
        for li in range(5):
            phase = model.layers[li].phase().detach().cpu().numpy()
            ax4[li,col].imshow(phase, cmap="twilight_shifted", vmin=-math.pi, vmax=math.pi)
            ax4[li,col].set_title(f"n_mask={nm}, Layer {li}" if li == 0 else f"Layer {li}", fontsize=11)
            ax4[li,col].axis("off")
        spec = np.abs(np.fft.fftshift(np.fft.fft2(model.layers[0].phase().detach().cpu().numpy())))**2
        ax4[5,col].imshow(np.log10(spec+1e-10), cmap="viridis")
        ax4[5,col].set_title("FFT Layer 0 (log)"); ax4[5,col].axis("off")
    for li in range(5):
        ax4[li,0].set_ylabel(f"Layer {li}", fontsize=11, fontweight="bold")
    ax4[5,0].set_ylabel("Spectrum", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.97])
    fig4.savefig(OUT / "phase2_fig4_masks.png", dpi=150, bbox_inches="tight"); plt.close(fig4)
    print(f"  Saved: phase2_fig4_masks.png")

    # ═══ FIGURE 5: Training curves ═══
    print("Figure 5: Training curves...")
    fig5, ax5 = plt.subplots(2, 1, figsize=(16, 12))
    fig5.suptitle("Figure 5: Training Curves (FD2NN Cropped Fourier)", fontsize=16, fontweight="bold")
    colors = {'nm32': '#e74c3c', 'nm48': '#f39c12', 'nm64': '#2ecc71'}
    for rn in SHOW_RUNS:
        rp = SWEEP_DIR / rn / "results.json"
        if not rp.exists(): continue
        r = json.load(open(rp))
        h = r.get("history", {})
        if not h.get("epoch"): continue
        nm_str = rn.split("_")[0]
        c = colors.get(nm_str, 'gray')
        ax5[0].plot(h["epoch"], h["loss"], color=c, lw=2, marker='o', ms=4, label=rn)
        ax5[1].plot(h["epoch"], h["val_co"], color=c, lw=2, marker='o', ms=4, label=rn)
    ax5[1].axhline(co_bl, color='k', ls='--', lw=1.5, label=f'no_correction ({co_bl:.3f})')
    for a, t in zip(ax5, ["Training Loss", "Validation CO"]):
        a.set_xlabel("Epoch", fontsize=12); a.set_title(t, fontsize=13, fontweight="bold")
        a.legend(fontsize=10); a.grid(True, alpha=0.3)
    ax5[0].set_ylabel("Loss"); ax5[1].set_ylabel("CO")
    plt.tight_layout(rect=[0,0,1,0.95])
    fig5.savefig(OUT / "phase2_fig5_training.png", dpi=150, bbox_inches="tight"); plt.close(fig5)
    print(f"  Saved: phase2_fig5_training.png")

    # ═══ FIGURE 6: n_mask comparison ═══
    print("Figure 6: n_mask comparison...")
    fig6, ax6 = plt.subplots(1, 2, figsize=(18, 8))
    fig6.suptitle("Figure 6: n_mask Size Comparison (all configs)", fontsize=16, fontweight="bold")
    all_results = []
    for d in sorted((SWEEP_DIR).iterdir()):
        rp = d / "results.json"
        if rp.exists():
            all_results.append(json.load(open(rp)))
    # Group by n_mask — add turbulence baseline as first group
    loss_names = ["baseline_co", "co_amp", "co_phasor"]
    n_losses = len(loss_names)
    bar_w = 0.18
    # Turbulence (no correction) bars
    bl_co = all_results[0]["baseline_co"] if all_results else 0.3044
    ax6[0].bar(np.arange(n_losses) - bar_w*0.5, [bl_co]*n_losses, bar_w,
               label='Turbulence\n(no correction)', color='gray', alpha=0.6)
    for nm in N_MASKS:
        subset = [r for r in all_results if r["n_mask"] == nm]
        cos = [r["complex_overlap"] for r in subset]
        x = np.arange(len(cos))
        offset = (N_MASKS.index(nm) + 1) * bar_w - bar_w*0.5
        ax6[0].bar(x + offset, cos, bar_w, label=f"n_mask={nm}",
                   color=list(colors.values())[N_MASKS.index(nm)], alpha=0.7)
    ax6[0].set_xticks(np.arange(n_losses) + bar_w); ax6[0].set_xticklabels(loss_names)
    ax6[0].set_ylabel("CO"); ax6[0].set_title("CO by config × n_mask (gray=turbulence baseline)")
    ax6[0].legend(fontsize=9); ax6[0].grid(True, alpha=0.3)

    # Delta bar chart — turbulence delta = 0 by definition
    ax6[1].bar(np.arange(n_losses) - bar_w*0.5, [0]*n_losses, bar_w,
               label='Turbulence\n(no correction)', color='gray', alpha=0.6)
    for nm in N_MASKS:
        subset = [r for r in all_results if r["n_mask"] == nm]
        deltas = [r["complex_overlap"] - r["baseline_co"] for r in subset]
        x = np.arange(len(deltas))
        offset = (N_MASKS.index(nm) + 1) * bar_w - bar_w*0.5
        ax6[1].bar(x + offset, deltas, bar_w, label=f"n_mask={nm}",
                   color=list(colors.values())[N_MASKS.index(nm)], alpha=0.7)
    ax6[1].axhline(0, color='k', lw=1)
    ax6[1].set_xticks(np.arange(n_losses) + bar_w); ax6[1].set_xticklabels(loss_names)
    ax6[1].set_ylabel("CO Delta"); ax6[1].set_title("CO Delta (vs turbulence baseline)")
    ax6[1].legend(fontsize=9); ax6[1].grid(True, alpha=0.3)
    plt.tight_layout(rect=[0,0,1,0.95])
    fig6.savefig(OUT / "phase2_fig6_comparison.png", dpi=150, bbox_inches="tight"); plt.close(fig6)
    print(f"  Saved: phase2_fig6_comparison.png")

    print(f"\nDone! 6 figures saved to {OUT}")


if __name__ == "__main__":
    main()
