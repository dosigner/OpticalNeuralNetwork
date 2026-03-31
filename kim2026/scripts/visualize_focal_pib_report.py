#!/usr/bin/env python
"""Phase 2 Visualization for D2NN Focal-Plane PIB Sweep.

NOTE: Shared constants and utilities available in kim2026.eval.focal_utils.

Key difference from previous visualization: metric plane separation.
  - D2NN output plane: CO, WF RMS (unitary theorem verification)
  - Focal plane (f=4.5mm): PIB@10μm, PIB@50μm, Strehl (detector)

Figure 1: Input field (vacuum + turbulent)
Figure 2: D2NN output plane — CO/WF verification plane (BEFORE lens)
Figure 3: Focal plane — PIB/Strehl visualization (AFTER lens, detector)
Figure 4: Learned phase masks
Figure 5: Training curves (loss, focal PIB@10μm, CO)
Figure 6: Dual-plane metric comparison bar chart

Usage:
    cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python scripts/visualize_focal_pib_report.py
"""
from __future__ import annotations
import json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from kim2026.data.dataset import CachedFieldDataset
from kim2026.models.d2nn import BeamCleanupD2NN
from kim2026.optics.lens_2f import lens_2f_forward
from kim2026.training.targets import apply_receiver_aperture, center_crop_field
from kim2026.training.metrics import complex_overlap

W = 1.55e-6; N = 1024; WIN = 0.002048; APT = 0.002; DX = WIN / N
FOCUS_F = 4.5e-3
DX_FOCAL = W * FOCUS_F / (N * DX)  # 3.406 μm
ARCH = dict(num_layers=5, layer_spacing_m=10e-3, detector_distance_m=10e-3)

ALL_CONFIGS = ["focal_pib_only", "focal_strehl_only", "focal_intensity_overlap", "focal_co_pib_hybrid"]
COLORS = {
    "focal_pib_only": "#e74c3c",
    "focal_strehl_only": "#3498db",
    "focal_intensity_overlap": "#2ecc71",
    "focal_co_pib_hybrid": "#9b59b6",
}
LABELS = {
    "focal_pib_only": "Focal PIB",
    "focal_strehl_only": "Focal Strehl",
    "focal_intensity_overlap": "Focal IO",
    "focal_co_pib_hybrid": "CO+fPIB",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kim2026" / "1km_cn2_5e-14_tel15cm_n1024_br75"
SWEEP_DIR = Path(__file__).resolve().parent.parent / "autoresearch" / "runs" / "d2nn_focal_pib_sweep"
OUT = SWEEP_DIR

def load_model(name):
    ckpt = SWEEP_DIR / name / "checkpoint.pt"
    if not ckpt.exists(): return None
    m = BeamCleanupD2NN(n=N, wavelength_m=W, window_m=WIN, **ARCH)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["model_state_dict"])
    m.eval(); return m

def prepare(f):
    return center_crop_field(apply_receiver_aperture(f, receiver_window_m=WIN, aperture_diameter_m=APT), crop_n=N)

def focus(field):
    with torch.no_grad():
        f, dx = lens_2f_forward(field.to(torch.complex64), dx_in_m=DX, wavelength_m=W,
                                 f_m=FOCUS_F, na=None, apply_scaling=False)
    return f, dx

def ext(n, dx, unit): h = n * dx / 2 / unit; return [-h, h, -h, h]

def compute_pib(field, dx, radius_um):
    irr = np.abs(field)**2
    n = field.shape[-1]; c = n // 2
    yy, xx = np.mgrid[-c:n-c, -c:n-c]
    r = np.sqrt((xx * dx)**2 + (yy * dx)**2)
    mask = r <= (radius_um * 1e-6)
    return irr[mask].sum() / max(irr.sum(), 1e-30)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"dx_focal = {DX_FOCAL*1e6:.3f} μm, focal window = {N*DX_FOCAL*1e6:.0f} μm")

    ds = CachedFieldDataset(cache_dir=str(DATA_DIR / "cache"),
                             manifest_path=str(DATA_DIR / "split_manifest.json"), split="test")
    s = ds[0]
    u_turb = prepare(s["u_turb"].unsqueeze(0).to(device))
    u_vac = prepare(s["u_vacuum"].unsqueeze(0).to(device))

    # Load completed models
    configs_done = []; outputs = {}
    for name in ALL_CONFIGS:
        model = load_model(name)
        if model:
            model = model.to(device)
            with torch.no_grad(): outputs[name] = model(u_turb)
            co = complex_overlap(outputs[name], u_vac).item()
            print(f"  {name:>28}: CO={co:.4f}")
            configs_done.append(name)
        else:
            print(f"  {name:>28}: not completed")

    if not configs_done:
        print("No completed configs found! Training still in progress?")
        return

    # No-correction baseline
    d0 = BeamCleanupD2NN(n=N, wavelength_m=W, window_m=WIN, **ARCH).to(device); d0.eval()
    with torch.no_grad(): u_vac_d = d0(u_vac); u_turb_d = d0(u_turb)

    # Focus all
    focal_vac, dx_det = focus(u_vac_d)
    focal_turb, _ = focus(u_turb_d)
    focal_out = {n: focus(o)[0] for n, o in outputs.items()}

    # numpy
    vac_np = u_vac[0].cpu().numpy()
    turb_np = u_turb[0].cpu().numpy()
    vac_d_np = u_vac_d[0].cpu().numpy()
    turb_d_np = u_turb_d[0].cpu().numpy()
    out_np = {n: o[0].cpu().numpy() for n, o in outputs.items()}
    fvac_np = focal_vac[0].cpu().numpy()
    fturb_np = focal_turb[0].cpu().numpy()
    fout_np = {n: o[0].cpu().numpy() for n, o in focal_out.items()}

    e_mm = ext(N, DX, 1e-3)
    e_um = ext(N, DX, 1e-6)
    mid = N // 2

    # ═══ FIGURE 1: Input ═══
    print("\nFig 1: Input field...")
    fig1, ax1 = plt.subplots(1, 4, figsize=(24, 6))
    fig1.suptitle("Figure 1: Input Field (Cn²=5e-14, D/r₀=5.0)\n"
                  "D2NN 입력면: receiver aperture 적용 후",
                  fontsize=14, fontweight="bold")
    imax = (np.abs(vac_np)**2).max()
    for ax, field, title in zip(ax1, [vac_np, vac_np, turb_np, turb_np],
                                     ["Vacuum irradiance", "Vacuum phase",
                                      "Turbulent irradiance", "Turbulent phase"]):
        if "phase" in title:
            ax.imshow(np.angle(field), extent=e_mm, origin="lower",
                      cmap="twilight_shifted", vmin=-math.pi, vmax=math.pi)
        else:
            ax.imshow(np.abs(field)**2, extent=e_mm, origin="lower",
                      cmap="inferno", vmin=0, vmax=imax)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("mm"); ax.set_ylabel("mm")
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig1.savefig(OUT / "phase2_fig1_input.png", dpi=150, bbox_inches="tight"); plt.close(fig1)
    print("  Saved")

    # ═══ FIGURE 2: D2NN Output Plane (BEFORE lens) ═══
    print("Fig 2: D2NN output plane (CO/WF verification)...")
    cols = [("Vacuum", vac_d_np), ("Turb\n(no D2NN)", turb_d_np)]
    for c in configs_done: cols.append((LABELS.get(c, c), out_np[c]))
    ncols = len(cols)
    fig2, ax2 = plt.subplots(3, ncols, figsize=(6*ncols, 18))
    fig2.suptitle("Figure 2: D2NN Output Plane — CO/WF Verification (BEFORE focus lens)\n"
                  f"Window: {WIN*1e3:.3f}mm, dx={DX*1e6:.1f}μm | "
                  "CO, WF RMS는 이 평면에서 계산 (유니터리 정리)",
                  fontsize=13, fontweight="bold")

    for col, (label, field) in enumerate(cols):
        irr = np.abs(field)**2
        log_irr = np.log10(irr + 1e-30)
        ax2[0, col].imshow(log_irr, extent=e_um, origin="lower", cmap="inferno",
                           vmin=log_irr.max()-6, vmax=log_irr.max())
        ax2[0, col].set_title(label, fontsize=11, fontweight="bold")
        ax2[1, col].imshow(np.angle(field), extent=e_um, origin="lower",
                           cmap="twilight_shifted", vmin=-math.pi, vmax=math.pi)
        x_um = np.linspace(e_um[0], e_um[1], N)
        ax2[2, col].plot(x_um, np.abs(vac_d_np[mid, :])**2, 'b--', lw=1, alpha=0.5, label='Vacuum')
        ax2[2, col].plot(x_um, np.abs(field[mid, :])**2, 'r-', lw=1.5, label=label)
        ax2[2, col].legend(fontsize=8); ax2[2, col].grid(True, alpha=0.3)
        ax2[2, col].set_xlabel("μm")
    for r, lbl in enumerate(["Irradiance (log₁₀)\nμm", "Phase [rad]\nμm", "1D profile"]):
        ax2[r, 0].set_ylabel(lbl, fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig2.savefig(OUT / "phase2_fig2_d2nn_output.png", dpi=150, bbox_inches="tight"); plt.close(fig2)
    print("  Saved")

    # ═══ FIGURE 3: Focal Plane (AFTER lens) — PIB/Strehl ═══
    print("Fig 3: Focal plane (PIB/Strehl at detector)...")
    fcols = [("Vacuum", fvac_np), ("Turb\n(no D2NN)", fturb_np)]
    for c in configs_done: fcols.append((LABELS.get(c, c), fout_np[c]))
    ncols = len(fcols)
    fig3, ax3 = plt.subplots(3, ncols, figsize=(6*ncols, 18))
    fig3.suptitle("Figure 3: Focal Plane — PIB Analysis (AFTER f=4.5mm lens)\n"
                  f"dx_focal={DX_FOCAL*1e6:.3f}μm | "
                  "PIB, Strehl은 이 평면에서 계산 (실제 검출기 위치)",
                  fontsize=13, fontweight="bold")

    Z = 64; cd = N // 2
    e_zoom = ext(2*Z, float(dx_det), 1e-6)
    imax_f = max((np.abs(f[cd-Z:cd+Z, cd-Z:cd+Z])**2).max() for _, f in fcols)
    theta = np.linspace(0, 2*np.pi, 200)

    # Precompute radial grid
    yy, xx = np.mgrid[-cd:N-cd, -cd:N-cd]
    rsq = (xx * float(dx_det))**2 + (yy * float(dx_det))**2
    radii = np.linspace(1, 100, 200) * 1e-6

    for col, (label, field) in enumerate(fcols):
        crop = field[cd-Z:cd+Z, cd-Z:cd+Z]
        irr_crop = np.abs(crop)**2

        # Row 0: Irradiance with 10μm and 50μm circles
        ax3[0, col].imshow(irr_crop, extent=e_zoom, origin="lower", cmap="inferno", vmin=0, vmax=imax_f)
        pib10 = compute_pib(field, float(dx_det), 10)
        pib50 = compute_pib(field, float(dx_det), 50)
        ax3[0, col].set_title(f"{label}\nPIB@10μm={pib10:.4f} | @50μm={pib50:.4f}",
                               fontsize=10, fontweight="bold")
        # 10μm circle (red)
        ax3[0, col].plot(10*np.cos(theta), 10*np.sin(theta), 'r-', lw=2.0, alpha=0.9, label='10μm')
        # 50μm circle (white dashed)
        ax3[0, col].plot(50*np.cos(theta), 50*np.sin(theta), 'w--', lw=1.5, alpha=0.7, label='50μm')
        if col == 0: ax3[0, col].legend(fontsize=8, loc='upper right')

        # Row 1: 1D cross-section
        x_zoom = np.linspace(e_zoom[0], e_zoom[1], 2*Z)
        ax3[1, col].plot(x_zoom, np.abs(fvac_np[cd, cd-Z:cd+Z])**2, 'b--', lw=1, alpha=0.5, label='Vacuum')
        ax3[1, col].plot(x_zoom, np.abs(field[cd, cd-Z:cd+Z])**2, 'r-', lw=1.5, label=label)
        ax3[1, col].axvline(-10, color='red', ls=':', lw=1, alpha=0.5)
        ax3[1, col].axvline(10, color='red', ls=':', lw=1, alpha=0.5)
        ax3[1, col].axvline(-50, color='cyan', ls='--', lw=0.8, alpha=0.5)
        ax3[1, col].axvline(50, color='cyan', ls='--', lw=0.8, alpha=0.5)
        ax3[1, col].legend(fontsize=8); ax3[1, col].grid(True, alpha=0.3)
        ax3[1, col].set_xlabel("μm")

        # Row 2: Encircled energy curve
        irr_full = np.abs(field)**2
        irr_vac = np.abs(fvac_np)**2
        ee_v = [irr_vac[rsq <= r**2].sum() / max(irr_vac.sum(), 1e-30) for r in radii]
        ee_f = [irr_full[rsq <= r**2].sum() / max(irr_full.sum(), 1e-30) for r in radii]
        ax3[2, col].plot(radii*1e6, ee_v, 'b--', lw=1.5, alpha=0.6, label='Vacuum')
        ax3[2, col].plot(radii*1e6, ee_f, 'r-', lw=2, label=label)
        ax3[2, col].axvline(10, color='red', ls=':', lw=1.5, alpha=0.7, label='10μm')
        ax3[2, col].axvline(50, color='gray', ls=':', lw=1, alpha=0.5, label='50μm')
        ax3[2, col].legend(fontsize=8); ax3[2, col].grid(True, alpha=0.3)
        ax3[2, col].set_xlabel("Radius [μm]"); ax3[2, col].set_ylabel("EE fraction")

    for r, lbl in enumerate(["Focal irradiance\n(10μm=red, 50μm=white)", "1D cross-section", "Encircled energy"]):
        ax3[r, 0].set_ylabel(lbl, fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig3.savefig(OUT / "phase2_fig3_focal_plane.png", dpi=150, bbox_inches="tight"); plt.close(fig3)
    print("  Saved")

    # ═══ FIGURE 4: Phase masks ═══
    print("Fig 4: Learned phase masks...")
    nc = len(configs_done)
    if nc > 0:
        fig4, ax4 = plt.subplots(6, nc, figsize=(7*nc, 36))
        if nc == 1: ax4 = ax4.reshape(-1, 1)
        fig4.suptitle("Figure 4: Learned Phase Masks — Focal PIB Strategies",
                      fontsize=16, fontweight="bold")
        for col, name in enumerate(configs_done):
            model = load_model(name)
            if not model: continue
            for li in range(5):
                phase = torch.remainder(model.layers[li].phase, 2*math.pi).detach().cpu().numpy()
                ax4[li, col].imshow(phase, cmap="twilight_shifted", vmin=0, vmax=2*math.pi)
                title = f"{LABELS.get(name, name)}\nLayer {li}" if li == 0 else f"Layer {li}"
                ax4[li, col].set_title(title, fontsize=10)
                ax4[li, col].axis("off")
            # FFT of layer 0
            spec = np.abs(np.fft.fftshift(np.fft.fft2(
                torch.remainder(model.layers[0].phase, 2*math.pi).detach().cpu().numpy())))**2
            ax4[5, col].imshow(np.log10(spec + 1e-10), cmap="viridis")
            ax4[5, col].set_title("FFT L0"); ax4[5, col].axis("off")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fig4.savefig(OUT / "phase2_fig4_masks.png", dpi=150, bbox_inches="tight"); plt.close(fig4)
        print("  Saved")

    # ═══ FIGURE 5: Training curves ═══
    print("Fig 5: Training curves...")
    fig5, ax5 = plt.subplots(3, 1, figsize=(16, 18))
    fig5.suptitle("Figure 5: Training Curves — Focal PIB Strategies\n"
                  "Loss targets focal plane; CO measured at D2NN output",
                  fontsize=14, fontweight="bold")
    for name in configs_done:
        rp = SWEEP_DIR / name / "results.json"
        if not rp.exists(): continue
        r = json.load(open(rp)); h = r.get("history", {})
        if not h.get("epoch"): continue
        c = COLORS.get(name, 'gray'); lb = LABELS.get(name, name)
        ax5[0].plot(h["epoch"], h["loss"], color=c, lw=2, marker='o', ms=3, label=lb)
        ax5[1].plot(h["epoch"], h["val_co"], color=c, lw=2, marker='o', ms=3, label=lb)
        if "val_focal_pib_10" in h:
            ax5[2].plot(h["epoch"], h["val_focal_pib_10"], color=c, lw=2, marker='s', ms=3, label=lb)

    # Baseline reference lines
    r0 = None
    for name in configs_done:
        rp = SWEEP_DIR / name / "results.json"
        if rp.exists():
            r0 = json.load(open(rp)); break
    if r0:
        ax5[1].axhline(r0.get("co_baseline", 0.3044), color='k', ls='--', lw=1.5, label='Baseline CO')
        ax5[2].axhline(r0.get("focal_pib_10um_baseline", 0), color='k', ls='--', lw=1.5, label='Turb baseline')
        ax5[2].axhline(r0.get("focal_pib_10um_vacuum", 0), color='blue', ls=':', lw=1.5, label='Vacuum')

    for a, t, yl in zip(ax5,
                         ["Training Loss", "Validation CO (D2NN output plane)",
                          "Validation Focal PIB@10μm (detector plane)"],
                         ["Loss", "CO", "Focal PIB@10μm"]):
        a.set_xlabel("Epoch"); a.set_title(t, fontsize=13, fontweight="bold")
        a.legend(fontsize=10); a.grid(True, alpha=0.3); a.set_ylabel(yl)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig5.savefig(OUT / "phase2_fig5_training.png", dpi=150, bbox_inches="tight"); plt.close(fig5)
    print("  Saved")

    # ═══ FIGURE 6: Dual-plane metric comparison ═══
    print("Fig 6: Dual-plane metrics...")
    all_r = []
    for name in configs_done:
        rp = SWEEP_DIR / name / "results.json"
        if rp.exists(): all_r.append(json.load(open(rp)))
    if all_r:
        fig6, ax6 = plt.subplots(2, 3, figsize=(24, 14))
        fig6.suptitle("Figure 6: Dual-Plane Metric Comparison\n"
                      "Top: D2NN output plane (CO, IO, WF RMS) | Bottom: Focal plane (PIB@10μm, PIB@50μm, Strehl)",
                      fontsize=14, fontweight="bold")

        names = ["No D2NN"] + [LABELS.get(r["name"], r["name"]) for r in all_r]
        bar_colors = ['gray'] + [COLORS.get(r["name"], 'gray') for r in all_r]
        x = np.arange(len(names))
        width = 0.6

        # Row 0: D2NN output plane metrics
        cos = [all_r[0]["co_baseline"]] + [r["co_output"] for r in all_r]
        ax6[0, 0].bar(x, cos, width, color=bar_colors, alpha=0.8, edgecolor='black', lw=0.5)
        ax6[0, 0].set_title("(a) CO (D2NN output)", fontsize=12, fontweight="bold")
        ax6[0, 0].set_ylabel("Complex Overlap")
        for i, v in enumerate(cos): ax6[0, 0].text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=9)

        ios = [0] + [r["io_output"] for r in all_r]
        ax6[0, 1].bar(x, ios, width, color=bar_colors, alpha=0.8, edgecolor='black', lw=0.5)
        ax6[0, 1].set_title("(b) IO (D2NN output)", fontsize=12, fontweight="bold")
        ax6[0, 1].set_ylabel("Intensity Overlap")
        for i, v in enumerate(ios):
            if v > 0: ax6[0, 1].text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=9)

        wfs = [all_r[0].get("wf_rms_baseline_nm", 460)] + [r.get("wf_rms_nm", 0) for r in all_r]
        ax6[0, 2].bar(x, wfs, width, color=bar_colors, alpha=0.8, edgecolor='black', lw=0.5)
        ax6[0, 2].set_title("(c) WF RMS (D2NN output)", fontsize=12, fontweight="bold")
        ax6[0, 2].set_ylabel("WF RMS [nm]")
        for i, v in enumerate(wfs): ax6[0, 2].text(i, v + 5, f"{v:.0f}", ha="center", fontsize=9)

        # Row 1: Focal plane metrics
        pibs10 = [all_r[0]["focal_pib_10um_baseline"]] + [r["focal_pib_10um"] for r in all_r]
        ax6[1, 0].bar(x, pibs10, width, color=bar_colors, alpha=0.8, edgecolor='black', lw=0.5)
        ax6[1, 0].axhline(all_r[0]["focal_pib_10um_vacuum"], color='blue', ls=':', lw=2, label='Vacuum')
        ax6[1, 0].set_title("(d) Focal PIB@10μm (detector)", fontsize=12, fontweight="bold")
        ax6[1, 0].set_ylabel("PIB@10μm")
        ax6[1, 0].legend(fontsize=10)
        for i, v in enumerate(pibs10): ax6[1, 0].text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=9)

        pibs50 = [all_r[0]["focal_pib_50um_baseline"]] + [r["focal_pib_50um"] for r in all_r]
        ax6[1, 1].bar(x, pibs50, width, color=bar_colors, alpha=0.8, edgecolor='black', lw=0.5)
        ax6[1, 1].axhline(all_r[0]["focal_pib_50um_vacuum"], color='blue', ls=':', lw=2, label='Vacuum')
        ax6[1, 1].set_title("(e) Focal PIB@50μm (detector)", fontsize=12, fontweight="bold")
        ax6[1, 1].set_ylabel("PIB@50μm")
        ax6[1, 1].legend(fontsize=10)
        for i, v in enumerate(pibs50): ax6[1, 1].text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=9)

        strehls = [0] + [r["focal_strehl"] for r in all_r]
        ax6[1, 2].bar(x, strehls, width, color=bar_colors, alpha=0.8, edgecolor='black', lw=0.5)
        ax6[1, 2].set_title("(f) Focal Strehl (detector)", fontsize=12, fontweight="bold")
        ax6[1, 2].set_ylabel("Strehl ratio")
        for i, v in enumerate(strehls):
            if v > 0: ax6[1, 2].text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=9)

        for row in range(2):
            for col in range(3):
                ax6[row, col].set_xticks(x)
                ax6[row, col].set_xticklabels(names, rotation=25, ha="right", fontsize=10)
                ax6[row, col].grid(True, alpha=0.3, axis='y')

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        fig6.savefig(OUT / "phase2_fig6_metrics.png", dpi=150, bbox_inches="tight"); plt.close(fig6)
        print("  Saved")

    del d0; torch.cuda.empty_cache()
    print(f"\nDone! Phase 2 figures saved to {OUT}")


if __name__ == "__main__":
    main()
