#!/usr/bin/env python
"""Visualize the telescope-modeled data: receiver → Fourier → detector.

Loads the new 15cm telescope data and runs through FD2NN (zero-phase).
Shows the under-resolution effect in Fourier plane.

Usage:
    cd /root/dj/D2NN/kim2026 && python scripts/visualize_telescope_data.py
"""
from __future__ import annotations
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

from kim2026.data.dataset import CachedFieldDataset
from kim2026.models.fd2nn import BeamCleanupFD2NN
from kim2026.optics.gaussian_beam import coordinate_axis
from kim2026.optics.lens_2f import lens_2f_forward
from kim2026.training.targets import apply_receiver_aperture
from kim2026.training.metrics import complex_overlap

WAVELENGTH_M = 1.55e-6
N = 1024
FDNN_WINDOW_M = 0.002048      # 2.048mm (75:1 from 153.6mm)
APERTURE_M = 0.002             # 2mm
FOCUS_F_M = 4.5e-3

ARCH = dict(
    num_layers=5, layer_spacing_m=5.0e-3,
    phase_constraint="unconstrained", phase_max=math.pi,
    phase_init="uniform", phase_init_scale=0.1,
    dual_2f_f1_m=25.0e-3, dual_2f_f2_m=25.0e-3,
    dual_2f_na1=0.508, dual_2f_na2=0.508,
    dual_2f_apply_scaling=False,
)

TEL_DATA = Path(__file__).resolve().parent.parent / "data" / "kim2026" / "1km_cn2e-14_tel15cm_n1024_br75"
OLD_DATA = Path(__file__).resolve().parent.parent / "data" / "kim2026" / "1km_cn2e-14_w2m_n1024_dx2mm"
OUT_DIR = Path(__file__).resolve().parent.parent / "autoresearch" / "runs" / "single_case_viz"


def measure_spot(f2d, dx):
    irr = np.abs(f2d)**2
    irr_n = irr / max(irr.max(), 1e-30)
    r_px = math.sqrt((irr_n > 1/math.e**2).sum() / math.pi)
    return r_px * dx, r_px


def get_zoom(f, zoom_px=50):
    c = f.shape[0] // 2
    return f[c-zoom_px:c+zoom_px, c-zoom_px:c+zoom_px]


def make_extent(n, dx, unit=1e-6):
    h = n * dx / 2 / unit
    return [-h, h, -h, h]


def plot_field(ax, f, ext, title, mode="irr", vmin=None, vmax=None):
    if mode == "irr":
        d = np.abs(f)**2
        im = ax.imshow(d, extent=ext, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    else:
        d = np.angle(f)
        im = ax.imshow(d, extent=ext, origin="lower", cmap="twilight_shifted",
                       vmin=-math.pi, vmax=math.pi)
    ax.set_title(title, fontsize=8)
    return im


def run_case(label, data_dir, window_m, device):
    """Run analysis on one dataset. Returns dict of results."""
    ds = CachedFieldDataset(
        cache_dir=str(data_dir / "cache"),
        manifest_path=str(data_dir / "split_manifest.json"),
        split="test")
    s = ds[0]
    u_t = s["u_turb"].unsqueeze(0).to(device)
    u_v = s["u_vacuum"].unsqueeze(0).to(device)

    # Aperture
    apt_d = window_m  # beam fills aperture in telescope case
    u_t = apply_receiver_aperture(u_t, receiver_window_m=window_m, aperture_diameter_m=apt_d)
    u_v = apply_receiver_aperture(u_v, receiver_window_m=window_m, aperture_diameter_m=apt_d)

    dx = window_m / N

    # Beam size
    spot_recv, spot_recv_px = measure_spot(u_v[0].cpu().numpy(), dx)
    print(f"  [{label}] Beam 1/e²: {spot_recv*1e6:.0f}μm = {spot_recv_px:.0f}px, fill={2*spot_recv/window_m:.0%}")

    # FD2NN (zero-phase)
    model = BeamCleanupFD2NN(n=N, wavelength_m=WAVELENGTH_M, window_m=window_m, **ARCH).to(device)
    for layer in model.layers:
        layer.raw.data.zero_()
    model.eval()
    with torch.no_grad():
        u_out = model(u_t)
    tp = u_out.abs().square().sum().item() / max(u_t.abs().square().sum().item(), 1e-30)
    co = complex_overlap(u_t, u_v).item()
    print(f"  [{label}] TP={tp:.4f}, Baseline CO={co:.4f}")

    # Fourier plane
    with torch.no_grad():
        u_vf, dx_f = lens_2f_forward(
            u_v.to(torch.complex64), dx_in_m=dx, wavelength_m=WAVELENGTH_M,
            f_m=25e-3, na=0.508, apply_scaling=False)
        u_tf, _ = lens_2f_forward(
            u_t.to(torch.complex64), dx_in_m=dx, wavelength_m=WAVELENGTH_M,
            f_m=25e-3, na=0.508, apply_scaling=False)
    spot_f, spot_f_px = measure_spot(u_vf[0].cpu().numpy(), dx_f)
    print(f"  [{label}] Fourier: dx={dx_f*1e6:.1f}μm, spot={spot_f*1e6:.0f}μm={spot_f_px:.0f}px, spot/px={spot_f/dx_f:.2f}")

    # Detector
    with torch.no_grad():
        u_t_foc, dx_foc = lens_2f_forward(
            u_t.to(torch.complex64), dx_in_m=dx, wavelength_m=WAVELENGTH_M,
            f_m=FOCUS_F_M, na=None, apply_scaling=False)
        u_v_foc, _ = lens_2f_forward(
            u_v.to(torch.complex64), dx_in_m=dx, wavelength_m=WAVELENGTH_M,
            f_m=FOCUS_F_M, na=None, apply_scaling=False)

    return {
        "label": label, "u_t": u_t, "u_v": u_v, "u_out": u_out,
        "u_vf": u_vf, "u_tf": u_tf, "dx": dx, "dx_f": dx_f,
        "u_t_foc": u_t_foc, "u_v_foc": u_v_foc, "dx_foc": dx_foc,
        "tp": tp, "co": co, "spot_f_px": spot_f_px, "spot_f": spot_f,
        "window": window_m,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Run both cases
    r_old = run_case("Old (1000:1)", OLD_DATA, 0.002048, device)
    print()
    r_tel = run_case("Telescope (75:1)", TEL_DATA, 0.002048, device)

    # ─── FIGURE: 4 rows × 6 cols ─────────────────────────────
    fig, axes = plt.subplots(4, 6, figsize=(30, 20))
    fig.suptitle(
        f"Old Simulation (1000:1) vs Telescope Data (15cm + 75:1)\n"
        f"λ=1.55μm, f=25mm, NA=0.508",
        fontsize=14, fontweight="bold")

    for case_idx, r in enumerate([r_old, r_tel]):
        col_off = case_idx * 3
        ext_r = make_extent(N, r["dx"], unit=1e-3)
        ext_f = make_extent(100, r["dx_f"], unit=1e-6)
        foc_zoom = 100
        ext_foc = make_extent(2*foc_zoom, r["dx_foc"], unit=1e-6)

        # Row 0: Receiver irradiance
        for ci, (f, t) in enumerate([
            (r["u_t"][0].cpu().numpy(), f"{r['label']}\nTurbulent"),
            (r["u_v"][0].cpu().numpy(), f"{r['label']}\nVacuum"),
            (r["u_v"][0].cpu().numpy(), f"{r['label']}\nVacuum phase"),
        ]):
            mode = "phase" if ci == 2 else "irr"
            plot_field(axes[0, col_off+ci], f, ext_r, t, mode)
            axes[0, col_off+ci].set_xlabel("mm")

        # Row 1: Receiver phase detail
        plot_field(axes[1, col_off], r["u_t"][0].cpu().numpy(), ext_r,
                   "Turbulent phase", "phase")
        axes[1, col_off].set_xlabel("mm")
        # Phase difference
        tp = np.angle(r["u_t"][0].cpu().numpy())
        vp = np.angle(r["u_v"][0].cpu().numpy())
        diff = (tp - vp + np.pi) % (2*np.pi) - np.pi
        axes[1, col_off+1].imshow(diff, extent=ext_r, origin="lower",
                                   cmap="twilight_shifted", vmin=-math.pi, vmax=math.pi)
        axes[1, col_off+1].set_title("Phase diff (turb-vac)", fontsize=8)
        axes[1, col_off+1].set_xlabel("mm")
        # Irradiance cross-section
        irr_v = np.abs(r["u_v"][0].cpu().numpy())**2
        mid = N // 2
        axes[1, col_off+2].plot(np.arange(N) * r["dx"]*1e3 - N//2*r["dx"]*1e3,
                                 irr_v[mid, :] / max(irr_v.max(), 1e-30), 'b-', lw=1)
        axes[1, col_off+2].set_title(f"Vacuum irradiance cross-section\nfill={2*r.get('spot_f_px',0)*r['dx_f']:.0f}", fontsize=8)
        axes[1, col_off+2].set_xlabel("mm")
        axes[1, col_off+2].set_ylabel("Normalized")
        axes[1, col_off+2].axhline(1/math.e**2, color='r', ls='--', lw=0.5, label='1/e²')
        axes[1, col_off+2].legend(fontsize=6)

        # Row 2: Fourier plane (zoomed)
        vf_z = get_zoom(r["u_vf"][0].cpu().numpy())
        tf_z = get_zoom(r["u_tf"][0].cpu().numpy())
        plot_field(axes[2, col_off], vf_z, ext_f,
                   f"Vacuum Fourier\nspot={r['spot_f_px']:.0f}px, dx={r['dx_f']*1e6:.1f}μm", "irr")
        axes[2, col_off].set_xlabel("μm")
        plot_field(axes[2, col_off+1], tf_z, ext_f, "Turbulent Fourier", "irr")
        axes[2, col_off+1].set_xlabel("μm")
        plot_field(axes[2, col_off+2], vf_z, ext_f, "Vacuum Fourier phase", "phase")
        axes[2, col_off+2].set_xlabel("μm")

        # Row 3: Detector
        for ci, (f, t) in enumerate([
            (r["u_t_foc"][0].cpu().numpy(), "Turbulent focused"),
            (r["u_v_foc"][0].cpu().numpy(), "Vacuum focused"),
            (r["u_v_foc"][0].cpu().numpy(), "Vacuum focused phase"),
        ]):
            mode = "phase" if ci == 2 else "irr"
            fz = get_zoom(f, foc_zoom)
            plot_field(axes[3, col_off+ci], fz, ext_foc, t, mode)
            axes[3, col_off+ci].set_xlabel("μm")

    for r, lbl in enumerate(["Receiver Irr/Phase", "Receiver Phase/Cross-section",
                              "Fourier Plane (zoom 100px)", "Detector (zoom)"]):
        axes[r, 0].set_ylabel(lbl, fontsize=10, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / "telescope_data_comparison.png"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
