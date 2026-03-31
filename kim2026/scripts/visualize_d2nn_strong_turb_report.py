#!/usr/bin/env python
"""Phase 2 Visualization for D2NN Strong Turbulence Sweep.

Same style as 중요_d2nn_sweep_telescope figures.

Figure 1: Input field
Figure 2: D2NN output (vacuum, turb(no D2NN), configs...)
Figure 3: Detector plane
Figure 4: Phase masks (5 layers × completed configs)
Figure 5: Training curves
Figure 6: Multi-realization statistics

Usage:
    cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python scripts/visualize_d2nn_strong_turb_report.py
"""
from __future__ import annotations
import json, math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

from kim2026.data.dataset import CachedFieldDataset
from kim2026.models.d2nn import BeamCleanupD2NN
from kim2026.optics.lens_2f import lens_2f_forward
from kim2026.training.targets import apply_receiver_aperture, center_crop_field
from kim2026.training.metrics import complex_overlap

W = 1.55e-6; N = 1024; WIN = 0.002048; APT = 0.002; DX = WIN / N; FOCUS_F = 4.5e-3
ARCH = dict(num_layers=5, layer_spacing_m=10e-3, detector_distance_m=10e-3)
ALL_CONFIGS = ["baseline_co", "co_amp", "co_phasor", "co_ffp", "roi80"]

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kim2026" / "1km_cn2_5e-14_tel15cm_n1024_br75"
SWEEP_DIR = Path(__file__).resolve().parent.parent / "autoresearch" / "runs" / "d2nn_strong_turb_sweep"
OUT = SWEEP_DIR


def load_model(name):
    ckpt = SWEEP_DIR / name / "checkpoint.pt"
    if not ckpt.exists(): return None
    m = BeamCleanupD2NN(n=N, wavelength_m=W, window_m=WIN, **ARCH)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["model_state_dict"])
    m.eval(); return m

def prepare(field):
    return center_crop_field(apply_receiver_aperture(field, receiver_window_m=WIN, aperture_diameter_m=APT), crop_n=N)

def focus(field):
    with torch.no_grad():
        f, dx = lens_2f_forward(field.to(torch.complex64), dx_in_m=DX, wavelength_m=W, f_m=FOCUS_F, na=None, apply_scaling=False)
    return f, dx

def ext(n, dx, unit):
    h = n*dx/2/unit; return [-h,h,-h,h]

def wrap_phase(p):
    return (p+np.pi)%(2*np.pi)-np.pi

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = CachedFieldDataset(cache_dir=str(DATA_DIR/"cache"), manifest_path=str(DATA_DIR/"split_manifest.json"), split="test")
    s = ds[0]
    u_turb = prepare(s["u_turb"].unsqueeze(0).to(device))
    u_vac = prepare(s["u_vacuum"].unsqueeze(0).to(device))

    # Load completed configs
    configs_done = []
    outputs = {}
    for name in ALL_CONFIGS:
        model = load_model(name)
        if model:
            model = model.to(device)
            with torch.no_grad():
                outputs[name] = model(u_turb)
            co = complex_overlap(outputs[name], u_vac).item()
            print(f"  {name:>15}: CO={co:.4f}")
            configs_done.append(name)
        else:
            print(f"  {name:>15}: not completed yet")

    if not configs_done:
        print("No configs completed. Exiting."); return

    # Zero-phase D2NN
    d0 = BeamCleanupD2NN(n=N, wavelength_m=W, window_m=WIN, **ARCH).to(device); d0.eval()
    with torch.no_grad():
        u_vac_d = d0(u_vac); u_turb_d = d0(u_turb)
    co_bl = complex_overlap(u_turb_d, u_vac_d).item()
    print(f"  {'no_correction':>15}: CO={co_bl:.4f}")

    # Focus
    u_vac_det, dx_det = focus(u_vac_d); u_turb_det, _ = focus(u_turb_d)
    det_out = {n: focus(o)[0] for n, o in outputs.items()}

    # numpy
    vac_np = u_vac[0].cpu().numpy(); turb_np = u_turb[0].cpu().numpy()
    vac_d_np = u_vac_d[0].cpu().numpy(); turb_d_np = u_turb_d[0].cpu().numpy()
    out_np = {n: o[0].cpu().numpy() for n, o in outputs.items()}
    vac_det_np = u_vac_det[0].cpu().numpy(); turb_det_np = u_turb_det[0].cpu().numpy()
    det_np = {n: o[0].cpu().numpy() for n, o in det_out.items()}
    e_mm = ext(N, DX, 1e-3); mid = N//2; x_mm = np.linspace(e_mm[0], e_mm[1], N)

    # ═══ FIGURE 1: Input ═══
    print("\nFig 1: Input...")
    fig1, ax1 = plt.subplots(2, 4, figsize=(24, 12))
    fig1.suptitle("Stage 1-2: Input Field (Telescope → Beam Reducer)", fontsize=16, fontweight="bold")
    imax = (np.abs(vac_np)**2).max()
    ax1[0,0].imshow(np.abs(vac_np)**2, extent=e_mm, origin="lower", cmap="inferno", vmin=0, vmax=imax)
    ax1[0,0].set_title("Vacuum irradiance"); ax1[0,0].set_ylabel("Before D2NN", fontsize=11, fontweight="bold")
    ax1[0,1].imshow(np.angle(vac_np), extent=e_mm, origin="lower", cmap="twilight_shifted", vmin=-math.pi, vmax=math.pi)
    ax1[0,1].set_title("Vacuum phase")
    ax1[0,2].imshow(np.abs(turb_np)**2, extent=e_mm, origin="lower", cmap="inferno", vmin=0, vmax=imax)
    ax1[0,2].set_title("Turbulent irradiance")
    ax1[0,3].imshow(np.angle(turb_np), extent=e_mm, origin="lower", cmap="twilight_shifted", vmin=-math.pi, vmax=math.pi)
    ax1[0,3].set_title("Turbulent phase")
    ax1[1,0].imshow(wrap_phase(np.angle(turb_np)-np.angle(vac_np)), extent=e_mm, origin="lower", cmap="RdBu_r")
    ax1[1,0].set_title("Phase diff"); ax1[1,0].set_ylabel("Diagnostics", fontsize=11, fontweight="bold")
    ax1[1,1].plot(x_mm, np.abs(vac_np[mid,:])**2, 'b-', lw=1.5, label='Vac')
    ax1[1,1].plot(x_mm, np.abs(turb_np[mid,:])**2, 'r-', lw=1.5, alpha=0.7, label='Turb')
    ax1[1,1].legend(); ax1[1,1].set_title("1D irradiance"); ax1[1,1].grid(True, alpha=0.3)
    ax1[1,2].plot(x_mm, np.angle(vac_np[mid,:]), 'b-', lw=1.5, label='Vac')
    ax1[1,2].plot(x_mm, np.angle(turb_np[mid,:]), 'r-', lw=1.5, alpha=0.7, label='Turb')
    ax1[1,2].legend(); ax1[1,2].set_title("1D phase"); ax1[1,2].grid(True, alpha=0.3)
    ax1[1,3].axis("off")
    ax1[1,3].text(0.1, 0.5, f"Cn²=5e-14\nD/r₀=5.02\nBaseline CO={co_bl:.4f}\nConfigs done: {len(configs_done)}/{len(ALL_CONFIGS)}",
                  fontsize=14, va="center", family="monospace", bbox=dict(facecolor="lightyellow", edgecolor="gray"))
    plt.tight_layout(rect=[0,0,1,0.95])
    fig1.savefig(OUT/"phase2_fig1_input.png", dpi=150, bbox_inches="tight"); plt.close(fig1)
    print("  Saved")

    # ═══ FIGURE 2: D2NN Output ═══
    print("Fig 2: D2NN output...")
    cols = [("Vacuum", vac_d_np), ("Turb\n(no D2NN)", turb_d_np)]
    for c in configs_done: cols.append((c, out_np[c]))
    ncols = len(cols)
    fig2, ax2 = plt.subplots(4, ncols, figsize=(6*ncols, 24))
    fig2.suptitle("Stage 3: D2NN Output (before focusing lens)", fontsize=16, fontweight="bold")
    imax_o = max((np.abs(f)**2).max() for _, f in cols)
    for col, (label, field) in enumerate(cols):
        ax2[0,col].imshow(np.abs(field)**2, extent=e_mm, origin="lower", cmap="inferno", vmin=0, vmax=imax_o)
        ax2[0,col].set_title(label, fontsize=12, fontweight="bold")
        ax2[1,col].imshow(np.angle(field), extent=e_mm, origin="lower", cmap="twilight_shifted", vmin=-math.pi, vmax=math.pi)
        res = wrap_phase(np.angle(field)-np.angle(vac_d_np))
        ax2[2,col].imshow(res, extent=e_mm, origin="lower", cmap="RdBu_r", vmin=-math.pi, vmax=math.pi)
        ax2[3,col].plot(x_mm, np.abs(vac_d_np[mid,:])**2, 'b--', lw=1, alpha=0.5, label='Vac')
        ax2[3,col].plot(x_mm, np.abs(field[mid,:])**2, 'r-', lw=1.5, label=label)
        ax2[3,col].legend(fontsize=8); ax2[3,col].grid(True, alpha=0.3)
    for r, lbl in enumerate(["Irradiance", "Phase", "Residual phase", "1D profile"]):
        ax2[r,0].set_ylabel(lbl, fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.95])
    fig2.savefig(OUT/"phase2_fig2_d2nn_output.png", dpi=150, bbox_inches="tight"); plt.close(fig2)
    print("  Saved")

    # ═══ FIGURE 3: Detector ═══
    print("Fig 3: Detector...")
    det_cols = [("Vacuum", vac_det_np), ("Turb\n(no D2NN)", turb_det_np)]
    for c in configs_done: det_cols.append((c, det_np[c]))
    ncols = len(det_cols)
    fig3, ax3 = plt.subplots(3, ncols, figsize=(6*ncols, 18))
    fig3.suptitle("Stage 4: Detector Plane (after focusing lens f=4.5mm)", fontsize=16, fontweight="bold")
    Z = 64; cd = N//2; e_um = ext(2*Z, float(dx_det), 1e-6)
    imax_d = max((np.abs(f[cd-Z:cd+Z,cd-Z:cd+Z])**2).max() for _, f in det_cols)
    yy, xx = np.mgrid[-cd:N-cd,-cd:N-cd]; rsq = (xx*float(dx_det))**2+(yy*float(dx_det))**2
    radii = np.linspace(1,100,100)*1e-6
    for col, (label, field) in enumerate(det_cols):
        crop = field[cd-Z:cd+Z,cd-Z:cd+Z]
        ax3[0,col].imshow(np.abs(crop)**2, extent=e_um, origin="lower", cmap="inferno", vmin=0, vmax=imax_d)
        ax3[0,col].set_title(label, fontsize=12, fontweight="bold")
        x_um = np.linspace(e_um[0], e_um[1], 2*Z)
        ax3[1,col].plot(x_um, np.abs(vac_det_np[cd,cd-Z:cd+Z])**2, 'b--', lw=1, alpha=0.5, label='Vac')
        ax3[1,col].plot(x_um, np.abs(field[cd,cd-Z:cd+Z])**2, 'r-', lw=1.5, label=label)
        ax3[1,col].legend(fontsize=8); ax3[1,col].grid(True, alpha=0.3); ax3[1,col].set_xlabel("μm")
        irr = np.abs(field)**2; irr_v = np.abs(vac_det_np)**2
        ee_v = [irr_v[rsq<=r**2].sum()/max(irr_v.sum(),1e-30) for r in radii]
        ee_f = [irr[rsq<=r**2].sum()/max(irr.sum(),1e-30) for r in radii]
        ax3[2,col].plot(radii*1e6, ee_v, 'b--', lw=1, alpha=0.5, label='Vac')
        ax3[2,col].plot(radii*1e6, ee_f, 'r-', lw=1.5, label=label)
        ax3[2,col].axvline(50, color='gray', ls=':', lw=1); ax3[2,col].legend(fontsize=8)
        ax3[2,col].grid(True, alpha=0.3); ax3[2,col].set_xlabel("Radius [μm]")
    for r, lbl in enumerate(["Irradiance (zoom)", "1D cross-section", "Encircled energy"]):
        ax3[r,0].set_ylabel(lbl, fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.95])
    fig3.savefig(OUT/"phase2_fig3_detector.png", dpi=150, bbox_inches="tight"); plt.close(fig3)
    print("  Saved")

    # ═══ FIGURE 4: Phase masks ═══
    print("Fig 4: Phase masks...")
    nc = len(configs_done)
    fig4, ax4 = plt.subplots(6, nc, figsize=(7*nc, 36))
    fig4.suptitle("Figure 4: Learned Phase Masks", fontsize=16, fontweight="bold")
    for col, name in enumerate(configs_done):
        model = load_model(name)
        if not model: continue
        for li in range(5):
            phase = torch.remainder(model.layers[li].phase, 2*math.pi).detach().cpu().numpy()
            ax4[li,col].imshow(phase, cmap="twilight_shifted", vmin=0, vmax=2*math.pi)
            ax4[li,col].set_title(f"{name}\nLayer {li}" if li==0 else f"Layer {li}", fontsize=10)
            ax4[li,col].axis("off")
        spec = np.abs(np.fft.fftshift(np.fft.fft2(torch.remainder(model.layers[0].phase, 2*math.pi).detach().cpu().numpy())))**2
        ax4[5,col].imshow(np.log10(spec+1e-10), cmap="viridis")
        ax4[5,col].set_title("FFT Layer 0"); ax4[5,col].axis("off")
    for li in range(5): ax4[li,0].set_ylabel(f"Layer {li}", fontsize=11, fontweight="bold")
    ax4[5,0].set_ylabel("Spectrum", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.97])
    fig4.savefig(OUT/"phase2_fig4_masks.png", dpi=150, bbox_inches="tight"); plt.close(fig4)
    print("  Saved")

    # ═══ FIGURE 5: Training curves ═══
    print("Fig 5: Training curves...")
    fig5, ax5 = plt.subplots(2, 1, figsize=(16, 12))
    fig5.suptitle("Figure 5: Training Curves (D2NN Strong Turbulence)", fontsize=16, fontweight="bold")
    colors = {'baseline_co':'#2ecc71','co_amp':'#e74c3c','co_phasor':'#9b59b6','co_ffp':'#3498db','roi80':'#f39c12'}
    for name in configs_done:
        rp = SWEEP_DIR/name/"results.json"
        if not rp.exists(): continue
        r = json.load(open(rp)); h = r.get("history",{})
        if not h.get("epoch"): continue
        c = colors.get(name, 'gray')
        ax5[0].plot(h["epoch"], h["loss"], color=c, lw=2, marker='o', ms=4, label=name)
        ax5[1].plot(h["epoch"], h["val_co"], color=c, lw=2, marker='o', ms=4, label=name)
    ax5[1].axhline(co_bl, color='k', ls='--', lw=1.5, label=f'no_correction ({co_bl:.4f})')
    for a, t, yl in zip(ax5, ["Training Loss", "Validation CO"], ["Loss", "CO"]):
        a.set_xlabel("Epoch", fontsize=12); a.set_title(t, fontsize=13, fontweight="bold")
        a.legend(fontsize=10); a.grid(True, alpha=0.3); a.set_ylabel(yl)
    plt.tight_layout(rect=[0,0,1,0.95])
    fig5.savefig(OUT/"phase2_fig5_training.png", dpi=150, bbox_inches="tight"); plt.close(fig5)
    print("  Saved")

    # ═══ FIGURE 6: Multi-realization stats ═══
    print("Fig 6: Multi-realization...")
    all_data = {n: {"co": []} for n in ["no_d2nn"] + configs_done}
    models = {n: load_model(n).to(device) for n in configs_done if load_model(n)}
    for si in range(len(ds)):
        s_i = ds[si]; ut = prepare(s_i["u_turb"].unsqueeze(0).to(device))
        uv = prepare(s_i["u_vacuum"].unsqueeze(0).to(device))
        with torch.no_grad(): ut_d = d0(ut); uv_d = d0(uv)
        all_data["no_d2nn"]["co"].append(complex_overlap(ut_d, uv_d).item())
        for name, model in models.items():
            with torch.no_grad(): pred = model(ut)
            all_data[name]["co"].append(complex_overlap(pred, uv_d).item())
        if (si+1)%10==0: print(f"  [{si+1}/{len(ds)}]")

    fig6, ax6 = plt.subplots(1, 2, figsize=(18, 8))
    fig6.suptitle("Figure 6: Multi-Realization Statistics (test set)", fontsize=16, fontweight="bold")
    all_names = ["no_d2nn"] + configs_done
    bp_colors = ['gray'] + [colors.get(c, 'gray') for c in configs_done]
    co_data = [all_data[n]["co"] for n in all_names]
    bp = ax6[0].boxplot(co_data, tick_labels=all_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], bp_colors):
        patch.set_facecolor(color); patch.set_alpha(0.5)
    ax6[0].set_ylabel("Complex Overlap"); ax6[0].set_title("CO Distribution")
    ax6[0].tick_params(axis='x', rotation=20); ax6[0].grid(True, alpha=0.3, axis='y')

    # Mean comparison bar
    means = [np.mean(all_data[n]["co"]) for n in all_names]
    stds = [np.std(all_data[n]["co"]) for n in all_names]
    ax6[1].bar(range(len(all_names)), means, yerr=stds, color=bp_colors, alpha=0.6, capsize=5)
    ax6[1].set_xticks(range(len(all_names))); ax6[1].set_xticklabels(all_names, rotation=20)
    ax6[1].set_ylabel("Mean CO"); ax6[1].set_title("Mean CO ± std")
    ax6[1].grid(True, alpha=0.3, axis='y')
    for i, (m, s) in enumerate(zip(means, stds)):
        ax6[1].text(i, m+s+0.01, f"{m:.4f}", ha="center", fontsize=10, fontweight="bold")

    print("\n--- Statistics ---")
    for n in all_names:
        arr = np.array(all_data[n]["co"])
        print(f"  {n:>15}: CO={arr.mean():.4f}±{arr.std():.4f}")

    plt.tight_layout(rect=[0,0,1,0.95])
    fig6.savefig(OUT/"phase2_fig6_statistics.png", dpi=150, bbox_inches="tight"); plt.close(fig6)
    print("  Saved")
    del d0; torch.cuda.empty_cache()
    print(f"\nDone! Figures saved to {OUT}")

if __name__ == "__main__":
    main()
