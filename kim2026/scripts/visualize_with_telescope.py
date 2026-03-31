#!/usr/bin/env python
"""Visualize with realistic 15cm telescope + 75:1 beam reducer.

Compares:
  A) Current simulation: 2m grid → 1000:1 implicit reducer (beam = 30% of aperture)
  B) Physical system: 15cm telescope truncation + wavefront curvature + 75:1 reducer
     (beam fills aperture)

Usage:
    cd /root/dj/D2NN/kim2026 && python scripts/visualize_with_telescope.py
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from kim2026.data.dataset import CachedFieldDataset
from kim2026.models.fd2nn import BeamCleanupFD2NN
from kim2026.optics.gaussian_beam import coordinate_axis
from kim2026.optics.lens_2f import lens_2f_forward
from kim2026.optics.aperture import circular_aperture
from kim2026.training.targets import apply_receiver_aperture
from kim2026.training.metrics import complex_overlap

# ─── Physical constants ──────────────────────────────────────
WAVELENGTH_M = 1.55e-6
N = 1024

# Physical grid (data was generated on this)
PHYS_WINDOW_M = 2.048           # 2.048m receiver grid
PHYS_DX_M = PHYS_WINDOW_M / N  # 2mm per pixel

# Telescope
TELESCOPE_DIAMETER_M = 0.15    # 15cm
BEAM_REDUCER_RATIO = 75        # 75:1

# FD2NN input (after beam reducer)
FDNN_WINDOW_M = TELESCOPE_DIAMETER_M / BEAM_REDUCER_RATIO  # 2mm
FDNN_APERTURE_M = FDNN_WINDOW_M  # beam fills aperture
FOCUS_F_M = 4.5e-3

# Wavefront curvature at 1km (far-field spherical wave)
R_CURVATURE_M = 1000.0

ARCH = dict(
    num_layers=5, layer_spacing_m=5.0e-3,
    phase_constraint="unconstrained", phase_max=math.pi,
    phase_init="uniform", phase_init_scale=0.1,
    dual_2f_f1_m=25.0e-3, dual_2f_f2_m=25.0e-3,
    dual_2f_na1=0.508, dual_2f_na2=0.508,
    dual_2f_apply_scaling=False,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kim2026" / "1km_cn2e-14_w2m_n1024_dx2mm" / "cache"
MANIFEST = DATA_DIR.parent / "split_manifest.json"
OUT_DIR = Path(__file__).resolve().parent.parent / "autoresearch" / "runs" / "single_case_viz"


def apply_telescope(field: torch.Tensor, device: torch.device, reducer_ratio: int = 75) -> torch.Tensor:
    """Apply 15cm telescope aperture + wavefront curvature + beam reducer.

    Input: field on physical 2m grid (1024x1024)
    Output: field on FD2NN grid (1024x1024), beam fills/partially fills aperture

    reducer_ratio: 75 → D_out=2mm, 50 → D_out=3mm
    """
    n = field.shape[-1]
    has_batch = field.ndim == 3
    if has_batch:
        f2d = field[0]
    else:
        f2d = field

    # Step 1: Apply 15cm circular aperture on the 2m physical grid
    telescope_mask = circular_aperture(
        n=n, window_m=PHYS_WINDOW_M, diameter_m=TELESCOPE_DIAMETER_M, device=device,
    )
    f_apertured = f2d * telescope_mask

    # Step 2: Telescope collimates the beam — REMOVE wavefront curvature
    # Real Keplerian telescope converts spherical wave → plane wave.
    # The propagation code already includes the curvature in the field phase.
    # We remove it here (telescope collimation effect).
    axis = coordinate_axis(n, PHYS_WINDOW_M, device=device).to(torch.float64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    r_sq = xx.square() + yy.square()
    # Undo the spherical wavefront: multiply by conjugate phase
    curv_phase = math.pi * r_sq / (WAVELENGTH_M * R_CURVATURE_M)  # +sign = undo
    curv_phasor = torch.exp(1j * curv_phase).to(torch.complex64)
    f_collimated = f_apertured * curv_phasor

    # Step 3: Crop to telescope aperture region
    telescope_pixels = int(round(TELESCOPE_DIAMETER_M / PHYS_DX_M))  # ~75 pixels
    c = n // 2
    half = telescope_pixels // 2
    f_cropped = f_collimated[c - half:c + half, c - half:c + half]  # ~75x75

    # Step 4: Beam reducer = resample to 1024x1024
    # Interpolate real and imaginary parts separately
    real_part = f_cropped.real.unsqueeze(0).unsqueeze(0).float()
    imag_part = f_cropped.imag.unsqueeze(0).unsqueeze(0).float()
    real_up = F.interpolate(real_part, size=(n, n), mode="bilinear", align_corners=False).squeeze()
    imag_up = F.interpolate(imag_part, size=(n, n), mode="bilinear", align_corners=False).squeeze()
    f_reduced = (real_up + 1j * imag_up).to(torch.complex64)

    if has_batch:
        return f_reduced.unsqueeze(0)
    return f_reduced


def focus_to_detector(field, window_m, f_m, wavelength_m):
    n = field.shape[-1]
    focused, dx_f = lens_2f_forward(
        field.to(torch.complex64), dx_in_m=window_m / n,
        wavelength_m=wavelength_m, f_m=f_m, na=None, apply_scaling=False)
    return focused, dx_f


def make_extent(n, dx_m, unit=1e-6):
    half = n * dx_m / 2 / unit
    return [-half, half, -half, half]


def plot_field(ax, f2d, extent, title, mode="irradiance", vmin=None, vmax=None, cmap=None):
    if mode == "irradiance":
        data = np.abs(f2d) ** 2
        cmap = cmap or "inferno"
        im = ax.imshow(data, extent=extent, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        data = np.angle(f2d)
        cmap = cmap or "twilight_shifted"
        im = ax.imshow(data, extent=extent, origin="lower", cmap=cmap, vmin=-math.pi, vmax=math.pi)
    ax.set_title(title, fontsize=8)
    return im


def measure_spot(field_2d, dx_m):
    """Measure 1/e² spot radius."""
    irr = np.abs(field_2d) ** 2
    irr_norm = irr / max(irr.max(), 1e-30)
    above = irr_norm > (1 / math.e ** 2)
    r_px = math.sqrt(above.sum() / math.pi)
    return r_px * dx_m, r_px


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="test")
    sample = ds[0]
    u_turb_raw = sample["u_turb"].to(device)
    u_vac_raw = sample["u_vacuum"].to(device)

    # ─── Case A: Current simulation (1000:1 implicit) ────────
    print("=" * 60)
    print("Case A: Current simulation (2m → 2mm, 1000:1)")
    print("=" * 60)
    u_turb_a = apply_receiver_aperture(
        u_turb_raw.unsqueeze(0), receiver_window_m=0.002048, aperture_diameter_m=0.002)
    u_vac_a = apply_receiver_aperture(
        u_vac_raw.unsqueeze(0), receiver_window_m=0.002048, aperture_diameter_m=0.002)

    dx_a = 0.002048 / N
    spot_a, spot_a_px = measure_spot(u_vac_a[0].cpu().numpy(), dx_a)
    print(f"  Beam 1/e² radius: {spot_a*1e6:.0f}μm = {spot_a_px:.0f}px")
    print(f"  Beam / aperture: {2*spot_a/0.002:.1%}")

    with torch.no_grad():
        u_vac_fourier_a, dx_f_a = lens_2f_forward(
            u_vac_a.to(torch.complex64), dx_in_m=dx_a, wavelength_m=WAVELENGTH_M,
            f_m=25e-3, na=0.508, apply_scaling=False)
    spot_f_a, spot_f_a_px = measure_spot(u_vac_fourier_a[0].cpu().numpy(), dx_f_a)
    print(f"  Fourier spot: {spot_f_a*1e6:.0f}μm = {spot_f_a_px:.0f}px")
    print(f"  Fourier dx: {dx_f_a*1e6:.1f}μm")

    # ─── Case B: Physical telescope (15cm + 75:1) ────────────
    print()
    print("=" * 60)
    print("Case B: 15cm telescope + wavefront curvature + 75:1 reducer")
    print("=" * 60)

    with torch.no_grad():
        u_turb_b = apply_telescope(u_turb_raw, device).unsqueeze(0)
        u_vac_b = apply_telescope(u_vac_raw, device).unsqueeze(0)

    dx_b = FDNN_WINDOW_M / N
    spot_b, spot_b_px = measure_spot(u_vac_b[0].cpu().numpy(), dx_b)
    print(f"  FD2NN window: {FDNN_WINDOW_M*1e3:.2f}mm")
    print(f"  FD2NN dx: {dx_b*1e6:.2f}μm")
    print(f"  Beam 1/e² radius: {spot_b*1e6:.0f}μm = {spot_b_px:.0f}px")
    print(f"  Beam / aperture: {2*spot_b/FDNN_WINDOW_M:.1%}")

    # Energy captured by telescope
    energy_total = (u_vac_raw.abs() ** 2).sum().item()
    tel_mask = circular_aperture(n=N, window_m=PHYS_WINDOW_M,
                                  diameter_m=TELESCOPE_DIAMETER_M, device=device)
    energy_captured = ((u_vac_raw.abs() ** 2) * tel_mask).sum().item()
    print(f"  Telescope capture: {energy_captured/energy_total:.1%}")

    # Wavefront curvature effect
    axis = coordinate_axis(N, PHYS_WINDOW_M, device=device).to(torch.float64)
    r_edge = TELESCOPE_DIAMETER_M / 2
    phase_at_edge = math.pi * r_edge ** 2 / (WAVELENGTH_M * R_CURVATURE_M)
    print(f"  Wavefront curvature at telescope edge: {phase_at_edge:.1f} rad = {phase_at_edge/(2*math.pi):.1f} × 2π")

    with torch.no_grad():
        u_vac_fourier_b, dx_f_b = lens_2f_forward(
            u_vac_b.to(torch.complex64), dx_in_m=dx_b, wavelength_m=WAVELENGTH_M,
            f_m=25e-3, na=0.508, apply_scaling=False)
    spot_f_b, spot_f_b_px = measure_spot(u_vac_fourier_b[0].cpu().numpy(), dx_f_b)
    print(f"  Fourier dx: {dx_f_b*1e6:.1f}μm")
    print(f"  Fourier spot: {spot_f_b*1e6:.0f}μm = {spot_f_b_px:.0f}px")

    # ─── FD2NN throughput test ────────────────────────────────
    print()
    print("=" * 60)
    print("FD2NN Zero-phase Throughput")
    print("=" * 60)
    for label, u_t, u_v, win in [
        ("Case A (1000:1)", u_turb_a, u_vac_a, 0.002048),
        ("Case B (75:1)", u_turb_b, u_vac_b, FDNN_WINDOW_M),
    ]:
        model = BeamCleanupFD2NN(n=N, wavelength_m=WAVELENGTH_M, window_m=win, **ARCH).to(device)
        for layer in model.layers:
            layer.raw.data.zero_()
        model.eval()
        with torch.no_grad():
            out = model(u_t)
        tp = out.abs().square().sum().item() / max(u_t.abs().square().sum().item(), 1e-30)
        co = complex_overlap(out, u_v).item()
        print(f"  {label}: TP={tp:.4f}, CO={co:.4f}")

    # ─── FIGURE: Side-by-side comparison ──────────────────────
    fig, axes = plt.subplots(4, 9, figsize=(36, 20))
    fig.suptitle(
        "A: Simulation (1000:1)  |  B: Telescope 75:1 (D_out=2mm)  |  C: Telescope 50:1 (D_out=3mm)\n"
        f"λ=1.55μm, f=25mm, NA=0.508, N={N}",
        fontsize=14, fontweight="bold")

    def get_zoom(f2d, zoom_px=50):
        c = N // 2
        return f2d[c - zoom_px:c + zoom_px, c - zoom_px:c + zoom_px]

    # ── Col 0-2: Case A (current simulation) ─────────────────
    ext_a = make_extent(N, dx_a, unit=1e-3)
    ext_fa = make_extent(100, dx_f_a, unit=1e-6)

    # ─── Case C: 15cm telescope + 50:1 reducer (D_out=3mm) ─────
    print()
    print("=" * 60)
    print("Case C: 15cm telescope + wavefront curvature + 50:1 reducer")
    print("=" * 60)
    FDNN_WINDOW_C = TELESCOPE_DIAMETER_M / 50  # 3mm
    with torch.no_grad():
        u_turb_c = apply_telescope(u_turb_raw, device, reducer_ratio=50).unsqueeze(0)
        u_vac_c = apply_telescope(u_vac_raw, device, reducer_ratio=50).unsqueeze(0)
    dx_c = FDNN_WINDOW_C / N
    spot_c, spot_c_px = measure_spot(u_vac_c[0].cpu().numpy(), dx_c)
    print(f"  FD2NN window: {FDNN_WINDOW_C*1e3:.2f}mm")
    print(f"  FD2NN dx: {dx_c*1e6:.2f}μm")
    print(f"  Beam 1/e² radius: {spot_c*1e6:.0f}μm = {spot_c_px:.0f}px")
    print(f"  Beam / aperture: {2*spot_c/FDNN_WINDOW_C:.1%}")

    with torch.no_grad():
        u_vac_fourier_c, dx_f_c = lens_2f_forward(
            u_vac_c.to(torch.complex64), dx_in_m=dx_c, wavelength_m=WAVELENGTH_M,
            f_m=25e-3, na=0.508, apply_scaling=False)
    spot_f_c, spot_f_c_px = measure_spot(u_vac_fourier_c[0].cpu().numpy(), dx_f_c)
    print(f"  Fourier dx: {dx_f_c*1e6:.1f}μm")
    print(f"  Fourier spot: {spot_f_c*1e6:.0f}μm = {spot_f_c_px:.0f}px")

    # Case C throughput
    model_c = BeamCleanupFD2NN(n=N, wavelength_m=WAVELENGTH_M, window_m=FDNN_WINDOW_C, **ARCH).to(device)
    for layer in model_c.layers:
        layer.raw.data.zero_()
    model_c.eval()
    with torch.no_grad():
        out_c = model_c(u_turb_c)
    tp_c = out_c.abs().square().sum().item() / max(u_turb_c.abs().square().sum().item(), 1e-30)
    co_c = complex_overlap(out_c, u_vac_c).item()
    print(f"  Zero-phase: TP={tp_c:.4f}, CO={co_c:.4f}")

    cases = [
        ("A: Sim (1000:1)", u_turb_a, u_vac_a, dx_a, 0.002048,
         u_vac_fourier_a, dx_f_a, 0, "Beam=30%"),
        ("B: Tel 75:1", u_turb_b, u_vac_b, dx_b, FDNN_WINDOW_M,
         u_vac_fourier_b, dx_f_b, 3, "Beam=100%"),
        ("C: Tel 50:1", u_turb_c, u_vac_c, dx_c, FDNN_WINDOW_C,
         u_vac_fourier_c, dx_f_c, 6, "Beam=67%"),
    ]

    for label, u_t, u_v, dx, win, u_vf, dx_f, col_off, note in cases:
        ext_recv = make_extent(N, dx, unit=1e-3)
        ext_four = make_extent(100, dx_f, unit=1e-6)

        # Row 0: Receiver irradiance
        imax = (np.abs(u_t[0].cpu().numpy())**2).max()
        for ci, (f, t) in enumerate([
            (u_t[0].cpu().numpy(), f"{label}\nTurbulent"),
            (u_v[0].cpu().numpy(), f"{label}\nVacuum"),
        ]):
            plot_field(axes[0, col_off + ci], f, ext_recv, t, "irradiance")
            axes[0, col_off + ci].set_xlabel("mm")
        # Vacuum phase
        plot_field(axes[0, col_off + 2], u_v[0].cpu().numpy(), ext_recv,
                   f"{label}\nVacuum phase", "phase")
        axes[0, col_off + 2].set_xlabel("mm")

        # Row 1: Receiver phase
        plot_field(axes[1, col_off], u_t[0].cpu().numpy(), ext_recv,
                   "Turbulent phase", "phase")
        plot_field(axes[1, col_off + 1], u_v[0].cpu().numpy(), ext_recv,
                   "Vacuum phase", "phase")
        # Difference
        turb_phase = np.angle(u_t[0].cpu().numpy())
        vac_phase = np.angle(u_v[0].cpu().numpy())
        diff_phase = (turb_phase - vac_phase + np.pi) % (2 * np.pi) - np.pi
        axes[1, col_off + 2].imshow(diff_phase, extent=ext_recv, origin="lower",
                                     cmap="twilight_shifted", vmin=-math.pi, vmax=math.pi)
        axes[1, col_off + 2].set_title("Phase diff (turb-vac)", fontsize=8)
        axes[1, col_off + 2].set_xlabel("mm")

        # Row 2: Fourier plane (zoomed)
        vf_z = get_zoom(u_vf[0].cpu().numpy())
        plot_field(axes[2, col_off], vf_z, ext_four,
                   f"Vacuum Fourier irr\ndx={dx_f*1e6:.1f}μm", "irradiance")
        axes[2, col_off].set_xlabel("μm")
        plot_field(axes[2, col_off + 1], vf_z, ext_four,
                   "Vacuum Fourier phase", "phase")
        axes[2, col_off + 1].set_xlabel("μm")
        # Turbulent Fourier
        with torch.no_grad():
            u_tf, _ = lens_2f_forward(
                u_t.to(torch.complex64), dx_in_m=dx, wavelength_m=WAVELENGTH_M,
                f_m=25e-3, na=0.508, apply_scaling=False)
        tf_z = get_zoom(u_tf[0].cpu().numpy())
        plot_field(axes[2, col_off + 2], tf_z, ext_four,
                   "Turbulent Fourier irr", "irradiance")
        axes[2, col_off + 2].set_xlabel("μm")

        # Row 3: Detector focal plane
        with torch.no_grad():
            u_t_foc, dx_foc = focus_to_detector(u_t, win, FOCUS_F_M, WAVELENGTH_M)
            u_v_foc, _ = focus_to_detector(u_v, win, FOCUS_F_M, WAVELENGTH_M)
        foc_zoom = 100
        ext_foc = make_extent(2 * foc_zoom, dx_foc, unit=1e-6)
        airy = 1.22 * WAVELENGTH_M * FOCUS_F_M / (2 * dx * N)  # approximate

        for ci, (f, t) in enumerate([
            (u_t_foc[0].cpu().numpy(), f"Turbulent focused"),
            (u_v_foc[0].cpu().numpy(), f"Vacuum focused"),
        ]):
            fz = get_zoom(f, foc_zoom)
            plot_field(axes[3, col_off + ci], fz, ext_foc, t, "irradiance")
            axes[3, col_off + ci].set_xlabel("μm")
        fz = get_zoom(u_v_foc[0].cpu().numpy(), foc_zoom)
        plot_field(axes[3, col_off + 2], fz, ext_foc, "Vacuum focused phase", "phase")
        axes[3, col_off + 2].set_xlabel("μm")

    row_labels = [
        "Receiver — Irradiance & Phase",
        "Receiver — Phase details",
        "Fourier Plane (zoomed 100px)",
        "Detector (focused, zoomed)",
    ]
    for r, lbl in enumerate(row_labels):
        axes[r, 0].set_ylabel(lbl, fontsize=10, fontweight="bold")

    # Add divider annotations
    for r in range(4):
        axes[r, 2].yaxis.set_label_position("right")
        axes[r, 3].yaxis.set_label_position("left")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUT_DIR / "telescope_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
