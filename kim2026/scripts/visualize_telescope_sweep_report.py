#!/usr/bin/env python
"""Generate telescope sweep report with 5-section visualization.

Reads completed results from loss_sweep_telescope/ and generates:
  Section A: Cross-config ranking (bar charts)
  Section B: Beam comparison (best config vs baseline vs vacuum)
  Section C: Fourier plane analysis (under-resolution effect)
  Section D: Detector focal plane
  Section E: Phase mask analysis (5 layers)

Usage:
    cd /root/dj/D2NN/kim2026 && python scripts/visualize_telescope_sweep_report.py
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
from kim2026.optics.lens_2f import lens_2f_forward
from kim2026.training.targets import apply_receiver_aperture
from kim2026.training.metrics import complex_overlap

WAVELENGTH_M = 1.55e-6
N = 1024
WINDOW_M = 0.002048
APERTURE_M = 0.002
FOCUS_F_M = 4.5e-3

ARCH = dict(
    num_layers=5, layer_spacing_m=5.0e-3,
    phase_constraint="unconstrained", phase_max=math.pi,
    phase_init="uniform", phase_init_scale=0.1,
    dual_2f_f1_m=25.0e-3, dual_2f_f2_m=25.0e-3,
    dual_2f_na1=0.508, dual_2f_na2=0.508,
    dual_2f_apply_scaling=False,
)

SWEEP_DIR = Path(__file__).resolve().parent.parent / "autoresearch" / "runs" / "loss_sweep_telescope"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kim2026" / "1km_cn2e-14_tel15cm_n1024_br75"
OUT_DIR = SWEEP_DIR


def load_results():
    results = []
    for d in sorted(SWEEP_DIR.iterdir()):
        rpath = d / "results.json"
        if rpath.exists():
            r = json.loads(rpath.read_text())
            r["dir"] = d
            results.append(r)
    return results


def load_model(name):
    ckpt_path = SWEEP_DIR / name / "checkpoint.pt"
    if not ckpt_path.exists():
        return None
    model = BeamCleanupFD2NN(n=N, wavelength_m=WAVELENGTH_M, window_m=WINDOW_M, **ARCH)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def get_zoom(f, z=50):
    c = f.shape[-1] // 2
    return f[..., c-z:c+z, c-z:c+z]


def make_ext(n, dx, unit=1e-6):
    h = n * dx / 2 / unit
    return [-h, h, -h, h]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = load_results()
    if not results:
        print("No results found!")
        return

    # Sort by CO descending
    results.sort(key=lambda r: r["complex_overlap"], reverse=True)
    names = [r["name"] for r in results]
    cos = [r["complex_overlap"] for r in results]
    tps = [r["throughput"] for r in results]
    phase_rmses = [r["phase_rmse_rad"] for r in results]
    baseline_co = results[0]["baseline_co"]
    best = results[0]

    print(f"Results: {len(results)} configs")
    print(f"Baseline CO (no correction): {baseline_co:.4f}")
    for r in results:
        delta = r["complex_overlap"] - baseline_co
        print(f"  {r['name']:>22}: CO={r['complex_overlap']:.4f} ({delta:+.4f}) "
              f"PhRMSE={r['phase_rmse_rad']:.3f} TP={r['throughput']:.4f}")

    # Load test sample
    ds = CachedFieldDataset(
        cache_dir=str(DATA_DIR / "cache"),
        manifest_path=str(DATA_DIR / "split_manifest.json"), split="test")
    sample = ds[0]
    u_turb = sample["u_turb"].unsqueeze(0).to(device)
    u_vac = sample["u_vacuum"].unsqueeze(0).to(device)
    u_turb = apply_receiver_aperture(u_turb, receiver_window_m=WINDOW_M, aperture_diameter_m=APERTURE_M)
    u_vac = apply_receiver_aperture(u_vac, receiver_window_m=WINDOW_M, aperture_diameter_m=APERTURE_M)

    # Load best model and run inference
    best_model = load_model(best["name"])
    if best_model is not None:
        best_model = best_model.to(device)
        with torch.no_grad():
            u_best = best_model(u_turb)
    else:
        u_best = u_turb  # fallback

    dx = WINDOW_M / N

    # ═══════════════════════════════════════════════════════════
    # FIGURE: 5-section report
    # ═══════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(28, 35))
    gs = gridspec.GridSpec(6, 4, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle(
        f"FD2NN Telescope Sweep Report — 15cm + 75:1, 100 epochs\n"
        f"Best: {best['name']} (CO={best['complex_overlap']:.4f}), "
        f"Baseline: {baseline_co:.4f}",
        fontsize=15, fontweight="bold", y=0.98)

    # ── Section A: Cross-config ranking ───────────────────────
    colors = ['#2ecc71' if c > baseline_co else '#e74c3c' for c in cos]

    ax_a1 = fig.add_subplot(gs[0, 0:2])
    bars = ax_a1.barh(range(len(names)), cos, color=colors)
    ax_a1.axvline(baseline_co, color='k', ls='--', lw=1.5, label=f'Baseline={baseline_co:.4f}')
    ax_a1.set_yticks(range(len(names)))
    ax_a1.set_yticklabels(names, fontsize=8)
    ax_a1.set_xlabel("Complex Overlap (higher=better)")
    ax_a1.set_title("Section A: Complex Overlap Ranking", fontweight="bold")
    ax_a1.legend(fontsize=8)
    ax_a1.invert_yaxis()

    ax_a2 = fig.add_subplot(gs[0, 2:4])
    ax_a2.barh(range(len(names)), phase_rmses, color='#3498db')
    ax_a2.set_yticks(range(len(names)))
    ax_a2.set_yticklabels(names, fontsize=8)
    ax_a2.set_xlabel("Phase RMSE [rad] (lower=better)")
    ax_a2.set_title("Section A: Phase RMSE Ranking", fontweight="bold")
    ax_a2.invert_yaxis()

    # ── Section B: Beam comparison ────────────────────────────
    recv_ext = make_ext(N, dx, unit=1e-3)
    fields_b = [
        (u_turb[0].cpu().numpy(), "Turbulent input"),
        (u_best[0].cpu().numpy(), f"Best: {best['name']}"),
        (u_vac[0].cpu().numpy(), "Vacuum target"),
    ]
    diff = np.abs(u_best[0].cpu().numpy())**2 - np.abs(u_vac[0].cpu().numpy())**2

    # Row 1: Irradiance
    for col, (f, title) in enumerate(fields_b):
        ax = fig.add_subplot(gs[1, col])
        im = ax.imshow(np.abs(f)**2, extent=recv_ext, origin="lower", cmap="inferno")
        ax.set_title(f"Irradiance: {title}", fontsize=9)
        ax.set_xlabel("mm")
        fig.colorbar(im, ax=ax, shrink=0.7)

    ax_diff = fig.add_subplot(gs[1, 3])
    vmax = np.abs(diff).max() * 0.3
    ax_diff.imshow(diff, extent=recv_ext, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax_diff.set_title("Difference (Best - Vacuum)", fontsize=9)
    ax_diff.set_xlabel("mm")

    # Row 2: Phase
    for col, (f, title) in enumerate(fields_b):
        ax = fig.add_subplot(gs[2, col])
        ax.imshow(np.angle(f), extent=recv_ext, origin="lower",
                  cmap="twilight_shifted", vmin=-math.pi, vmax=math.pi)
        ax.set_title(f"Phase: {title}", fontsize=9)
        ax.set_xlabel("mm")

    # Residual phase
    res_phase = np.angle(u_best[0].cpu().numpy()) - np.angle(u_vac[0].cpu().numpy())
    res_phase = (res_phase + np.pi) % (2 * np.pi) - np.pi
    ax_res = fig.add_subplot(gs[2, 3])
    ax_res.imshow(res_phase, extent=recv_ext, origin="lower",
                  cmap="twilight_shifted", vmin=-math.pi, vmax=math.pi)
    ax_res.set_title("Residual phase (Best - Vacuum)", fontsize=9)
    ax_res.set_xlabel("mm")

    fig.text(0.02, 0.72, "Section B: Beam Comparison", fontsize=12,
             fontweight="bold", rotation=90, va="center")

    # ── Section C: Fourier plane ──────────────────────────────
    with torch.no_grad():
        u_turb_f, dx_f = lens_2f_forward(
            u_turb.to(torch.complex64), dx_in_m=dx, wavelength_m=WAVELENGTH_M,
            f_m=25e-3, na=0.508, apply_scaling=False)
        u_best_f, _ = lens_2f_forward(
            u_best.to(torch.complex64), dx_in_m=dx, wavelength_m=WAVELENGTH_M,
            f_m=25e-3, na=0.508, apply_scaling=False)
        u_vac_f, _ = lens_2f_forward(
            u_vac.to(torch.complex64), dx_in_m=dx, wavelength_m=WAVELENGTH_M,
            f_m=25e-3, na=0.508, apply_scaling=False)

    zoom_px = 20  # zoom to central 40px to see the under-resolved spot
    f_ext = make_ext(2 * zoom_px, dx_f, unit=1e-6)

    fourier_fields = [
        (u_turb_f[0].cpu().numpy(), "Turbulent"),
        (u_best_f[0].cpu().numpy(), f"Best: {best['name']}"),
        (u_vac_f[0].cpu().numpy(), "Vacuum"),
    ]
    f_imax = max((np.abs(get_zoom(f, zoom_px))**2).max() for f, _ in fourier_fields)

    for col, (f, title) in enumerate(fourier_fields):
        ax = fig.add_subplot(gs[3, col])
        z = get_zoom(f, zoom_px)
        ax.imshow(np.abs(z)**2, extent=f_ext, origin="lower", cmap="inferno", vmin=0, vmax=f_imax)
        ax.set_title(f"Fourier: {title}\n(dx={dx_f*1e6:.1f}μm, zoom {2*zoom_px}px)", fontsize=8)
        ax.set_xlabel("μm")

    # Phase mask visualization (best model, layer 0 and 4)
    ax_mask = fig.add_subplot(gs[3, 3])
    if best_model is not None:
        phases = [l.wrapped_phase().detach().cpu().numpy() for l in best_model.layers]
        # Show layer 0 and layer 4 side by side
        combined = np.concatenate([get_zoom(phases[0], 100), get_zoom(phases[-1], 100)], axis=1)
        ax_mask.imshow(combined, cmap="twilight_shifted", vmin=0, vmax=2*math.pi)
        ax_mask.set_title(f"Phase masks: Layer 0 (left) | Layer 4 (right)\n(zoom 200px)", fontsize=8)
    else:
        ax_mask.text(0.5, 0.5, "No checkpoint", ha="center", va="center")
    ax_mask.set_xlabel("pixels")

    fig.text(0.02, 0.42, "Section C: Fourier Plane", fontsize=12,
             fontweight="bold", rotation=90, va="center")

    # ── Section D: Detector focal plane ───────────────────────
    with torch.no_grad():
        u_turb_det, dx_det = lens_2f_forward(
            u_turb.to(torch.complex64), dx_in_m=dx, wavelength_m=WAVELENGTH_M,
            f_m=FOCUS_F_M, na=None, apply_scaling=False)
        u_best_det, _ = lens_2f_forward(
            u_best.to(torch.complex64), dx_in_m=dx, wavelength_m=WAVELENGTH_M,
            f_m=FOCUS_F_M, na=None, apply_scaling=False)
        u_vac_det, _ = lens_2f_forward(
            u_vac.to(torch.complex64), dx_in_m=dx, wavelength_m=WAVELENGTH_M,
            f_m=FOCUS_F_M, na=None, apply_scaling=False)

    det_zoom = 80
    det_ext = make_ext(2 * det_zoom, dx_det, unit=1e-6)
    det_fields = [
        (u_turb_det[0].cpu().numpy(), "Turbulent"),
        (u_best_det[0].cpu().numpy(), f"Best: {best['name']}"),
        (u_vac_det[0].cpu().numpy(), "Vacuum"),
    ]
    det_imax = max((np.abs(get_zoom(f, det_zoom))**2).max() for f, _ in det_fields)

    for col, (f, title) in enumerate(det_fields):
        ax = fig.add_subplot(gs[4, col])
        z = get_zoom(f, det_zoom)
        ax.imshow(np.abs(z)**2, extent=det_ext, origin="lower", cmap="inferno", vmin=0, vmax=det_imax)
        ax.set_title(f"Detector: {title}\n(f={FOCUS_F_M*1e3:.1f}mm)", fontsize=8)
        ax.set_xlabel("μm")

    # PIB comparison
    ax_pib = fig.add_subplot(gs[4, 3])
    pib_radii = np.arange(1, 100) * dx_det
    pib_data = {}
    for f, label in det_fields:
        irr = np.abs(f)**2
        total = irr.sum()
        c = N // 2
        yy, xx = np.mgrid[-c:N-c, -c:N-c]
        r_sq = (xx * dx_det)**2 + (yy * dx_det)**2
        pibs = []
        for r in pib_radii:
            pibs.append((irr[r_sq <= r**2].sum() / total).item() if total > 0 else 0)
        pib_data[label] = pibs

    for label, pibs in pib_data.items():
        ax_pib.plot(pib_radii * 1e6, pibs, label=label[:15], lw=1.5)
    ax_pib.set_xlabel("Bucket radius [μm]")
    ax_pib.set_ylabel("Power in Bucket")
    ax_pib.set_title("PIB Comparison", fontsize=9)
    ax_pib.legend(fontsize=7)
    ax_pib.set_xlim(0, 200)
    ax_pib.grid(True, alpha=0.3)

    fig.text(0.02, 0.22, "Section D: Detector", fontsize=12,
             fontweight="bold", rotation=90, va="center")

    # ── Section E: Phase mask all 5 layers ────────────────────
    if best_model is not None:
        for layer_idx in range(5):
            ax = fig.add_subplot(gs[5, layer_idx]) if layer_idx < 4 else fig.add_subplot(gs[5, 3])
            if layer_idx < 4:
                ax = fig.add_subplot(gs[5, layer_idx])
            phase = best_model.layers[layer_idx].wrapped_phase().detach().cpu().numpy()
            ax.imshow(phase, cmap="twilight_shifted", vmin=0, vmax=2*math.pi)
            ax.set_title(f"Layer {layer_idx} phase [0,2π)", fontsize=8)
            ax.axis("off")

    fig.text(0.02, 0.08, "Section E: Masks", fontsize=12,
             fontweight="bold", rotation=90, va="center")

    # Save
    out_path = OUT_DIR / "telescope_sweep_report.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
