#!/usr/bin/env python
"""Visualize a single FD2NN case: receiver plane + Fourier plane + detector focal plane.

Shows turbulent input, vacuum reference, and (untrained) FD2NN output at each stage.
Purpose: understand beam structure and under-resolution before running sweeps.

Usage:
    cd /root/dj/D2NN/kim2026 && python scripts/visualize_single_case.py
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
from kim2026.optics.lens_2f import lens_2f_forward, fourier_plane_pitch
from kim2026.optics.aperture import circular_aperture
from kim2026.training.targets import apply_receiver_aperture
from kim2026.training.metrics import complex_overlap

# ─── Parameters ───────────────────────────────────────────────
WAVELENGTH_M = 1.55e-6
RECEIVER_WINDOW_M = 0.002048        # 2.048mm
APERTURE_DIAMETER_M = 0.002          # 2mm
N = 1024
FOCUS_F_M = 4.5e-3                   # detector focusing lens f=4.5mm

# FD2NN architecture (matches loss_sweep.py)
ARCH = dict(
    num_layers=5,
    layer_spacing_m=5.0e-3,
    phase_constraint="unconstrained",
    phase_max=math.pi,
    phase_init="uniform",
    phase_init_scale=0.1,
    dual_2f_f1_m=25.0e-3,
    dual_2f_f2_m=25.0e-3,
    dual_2f_na1=0.508,               # AC254-025-C
    dual_2f_na2=0.508,
    dual_2f_apply_scaling=False,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kim2026" / "1km_cn2e-14_w2m_n1024_dx2mm" / "cache"
MANIFEST = DATA_DIR.parent / "split_manifest.json"
OUT_DIR = Path(__file__).resolve().parent.parent / "autoresearch" / "runs" / "single_case_viz"


def thin_lens_phase(n: int, window_m: float, f_m: float, wavelength_m: float, device: torch.device) -> torch.Tensor:
    """Quadratic phase of an ideal thin lens: exp(-j*pi*(x^2+y^2)/(lambda*f))."""
    axis = coordinate_axis(n, window_m, device=device).to(torch.float64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    r_sq = xx.square() + yy.square()
    phase = -math.pi * r_sq / (wavelength_m * f_m)
    return torch.exp(1j * phase).to(torch.complex64)


def focus_to_detector(field: torch.Tensor, window_m: float, f_m: float, wavelength_m: float) -> tuple[torch.Tensor, float]:
    """Focus field through a thin lens to the focal plane using 2f Fourier relation.

    Returns (focused_field, dx_focal_m).
    """
    n = field.shape[-1]
    dx_m = window_m / n
    # Focusing lens + propagation to focal plane ≈ Fourier transform
    # Use lens_2f_forward which computes the exact Fourier relation
    focused, dx_focal = lens_2f_forward(
        field.to(torch.complex64),
        dx_in_m=dx_m,
        wavelength_m=wavelength_m,
        f_m=f_m,
        na=None,  # no NA clipping for detector lens
        apply_scaling=False,
    )
    return focused, dx_focal


def make_extent(n: int, dx_m: float, unit: float = 1e-6) -> list[float]:
    """Return [left, right, bottom, top] extent in given units (default μm)."""
    half = n * dx_m / 2 / unit
    return [-half, half, -half, half]


def plot_field(ax, field_2d: np.ndarray, extent, title: str, mode: str = "irradiance",
               vmin=None, vmax=None, cmap=None):
    """Plot irradiance or phase on a given axis."""
    if mode == "irradiance":
        data = np.abs(field_2d) ** 2
        if cmap is None:
            cmap = "inferno"
        im = ax.imshow(data, extent=extent, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    elif mode == "phase":
        data = np.angle(field_2d)
        if cmap is None:
            cmap = "twilight_shifted"
        im = ax.imshow(data, extent=extent, origin="lower", cmap=cmap,
                        vmin=-math.pi, vmax=math.pi)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    ax.set_title(title, fontsize=9)
    return im


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ─── Load one sample ──────────────────────────────────────
    ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="test")
    sample = ds[0]
    u_turb_raw = sample["u_turb"].unsqueeze(0).to(device)   # (1, 1024, 1024)
    u_vac_raw = sample["u_vacuum"].unsqueeze(0).to(device)

    # Apply receiver aperture
    u_turb = apply_receiver_aperture(u_turb_raw, receiver_window_m=RECEIVER_WINDOW_M,
                                     aperture_diameter_m=APERTURE_DIAMETER_M)
    u_vac = apply_receiver_aperture(u_vac_raw, receiver_window_m=RECEIVER_WINDOW_M,
                                    aperture_diameter_m=APERTURE_DIAMETER_M)

    # ─── FD2NN (zero-phase = identity) ────────────────────────
    model = BeamCleanupFD2NN(n=N, wavelength_m=WAVELENGTH_M, window_m=RECEIVER_WINDOW_M, **ARCH).to(device)
    for layer in model.layers:
        layer.raw.data.zero_()
    model.eval()

    with torch.no_grad():
        u_fdnn_out = model(u_turb)

    # Throughput check
    energy_in = u_turb.abs().square().sum().item()
    energy_out = u_fdnn_out.abs().square().sum().item()
    tp = energy_out / max(energy_in, 1e-12)
    co_bl = complex_overlap(u_turb, u_vac).item()
    co_fdnn = complex_overlap(u_fdnn_out, u_vac).item()
    print(f"Zero-phase throughput: {tp:.4f}")
    print(f"Baseline CO (turb vs vac): {co_bl:.4f}")
    print(f"FD2NN CO (out vs vac):     {co_fdnn:.4f}")

    # ─── Fourier plane fields ─────────────────────────────────
    dx_in = RECEIVER_WINDOW_M / N
    with torch.no_grad():
        u_turb_fourier, dx_fourier = lens_2f_forward(
            u_turb.to(torch.complex64), dx_in_m=dx_in, wavelength_m=WAVELENGTH_M,
            f_m=ARCH["dual_2f_f1_m"], na=ARCH["dual_2f_na1"], apply_scaling=False)
        u_vac_fourier, _ = lens_2f_forward(
            u_vac.to(torch.complex64), dx_in_m=dx_in, wavelength_m=WAVELENGTH_M,
            f_m=ARCH["dual_2f_f1_m"], na=ARCH["dual_2f_na1"], apply_scaling=False)

    fourier_window = dx_fourier * N
    print(f"\nFourier plane: dx={dx_fourier*1e6:.2f}μm, window={fourier_window*1e3:.2f}mm")

    # Measure vacuum beam Fourier spot size (1/e² radius)
    vac_f_intensity = u_vac_fourier[0].abs().square().cpu().numpy()
    vac_f_intensity_norm = vac_f_intensity / vac_f_intensity.max()
    # Find 1/e² radius: where intensity drops to 1/e² of peak
    above_threshold = vac_f_intensity_norm > (1 / math.e**2)
    n_pixels_above = above_threshold.sum()
    equiv_radius_px = math.sqrt(n_pixels_above / math.pi)
    spot_1e2_m = equiv_radius_px * dx_fourier
    print(f"Vacuum Fourier spot (1/e²): {spot_1e2_m*1e6:.1f}μm = {equiv_radius_px:.1f} pixels")
    print(f"Spot/pixel ratio: {spot_1e2_m/dx_fourier:.2f}")

    # ─── Detector focal plane ─────────────────────────────────
    with torch.no_grad():
        u_turb_focus, dx_focus = focus_to_detector(u_turb, RECEIVER_WINDOW_M, FOCUS_F_M, WAVELENGTH_M)
        u_vac_focus, _ = focus_to_detector(u_vac, RECEIVER_WINDOW_M, FOCUS_F_M, WAVELENGTH_M)
        u_fdnn_focus, _ = focus_to_detector(u_fdnn_out, RECEIVER_WINDOW_M, FOCUS_F_M, WAVELENGTH_M)

    focus_window = dx_focus * N
    airy_radius = 1.22 * WAVELENGTH_M * FOCUS_F_M / APERTURE_DIAMETER_M
    print(f"\nDetector plane: dx={dx_focus*1e6:.2f}μm, window={focus_window*1e3:.2f}mm")
    print(f"Airy disk first zero: {airy_radius*1e6:.1f}μm")

    # ─── FIGURE ───────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    fig.suptitle(
        f"FD2NN Single Case — f=25mm, NA=0.508, N={N}, λ=1.55μm\n"
        f"Zero-phase TP={tp:.4f}, Baseline CO={co_bl:.4f}",
        fontsize=13, fontweight="bold",
    )

    # Extents
    recv_ext = make_extent(N, RECEIVER_WINDOW_M / N, unit=1e-3)  # mm
    fourier_ext_full = make_extent(N, dx_fourier, unit=1e-3)      # mm
    focus_ext = make_extent(N, dx_focus, unit=1e-6)               # μm

    # Zoom for Fourier plane (central 100 pixels)
    zoom = 50
    fourier_ext_zoom = make_extent(2 * zoom, dx_fourier, unit=1e-6)  # μm

    def get_zoom(field, zoom_px=50):
        c = N // 2
        return field[c - zoom_px:c + zoom_px, c - zoom_px:c + zoom_px]

    # Focus zoom (central 200 pixels for detector)
    focus_zoom = 100
    focus_ext_zoom = make_extent(2 * focus_zoom, dx_focus, unit=1e-6)

    # ── Row 0: Receiver plane irradiance ──────────────────────
    fields_recv = [
        (u_turb[0].cpu().numpy(), "Turbulent input"),
        (u_vac[0].cpu().numpy(), "Vacuum reference"),
        (u_fdnn_out[0].cpu().numpy(), "FD2NN output (zero-phase)"),
    ]
    # Compute difference irradiance
    diff_irr = np.abs(u_fdnn_out[0].cpu().numpy())**2 - np.abs(u_vac[0].cpu().numpy())**2

    recv_imax = max((np.abs(f)**2).max() for _, (f, _) in enumerate(fields_recv))
    for col, (f, title) in enumerate(fields_recv):
        im = plot_field(axes[0, col], f, recv_ext, title, "irradiance", vmin=0, vmax=recv_imax)
        axes[0, col].set_xlabel("mm")
    # Difference
    im_diff = axes[0, 3].imshow(diff_irr, extent=recv_ext, origin="lower", cmap="RdBu_r",
                                 vmin=-recv_imax * 0.1, vmax=recv_imax * 0.1)
    axes[0, 3].set_title("Difference (FD2NN - Vacuum)", fontsize=9)
    axes[0, 3].set_xlabel("mm")
    fig.colorbar(im, ax=axes[0, 2], shrink=0.7, label="W/m²")
    fig.colorbar(im_diff, ax=axes[0, 3], shrink=0.7)

    # ── Row 1: Receiver plane phase ───────────────────────────
    for col, (f, title) in enumerate(fields_recv):
        im = plot_field(axes[1, col], f, recv_ext, f"Phase: {title}", "phase")
        axes[1, col].set_xlabel("mm")
    # Residual phase: FD2NN - Vacuum
    residual_phase = np.angle(u_fdnn_out[0].cpu().numpy()) - np.angle(u_vac[0].cpu().numpy())
    residual_phase = (residual_phase + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]
    im_res = axes[1, 3].imshow(residual_phase, extent=recv_ext, origin="lower",
                                cmap="twilight_shifted", vmin=-math.pi, vmax=math.pi)
    axes[1, 3].set_title("Residual phase (FD2NN - Vacuum)", fontsize=9)
    axes[1, 3].set_xlabel("mm")
    fig.colorbar(im_res, ax=axes[1, 3], shrink=0.7, label="rad")

    # ── Row 2: Fourier plane (zoomed) ─────────────────────────
    fourier_fields = [
        (u_turb_fourier[0].cpu().numpy(), "Turbulent (Fourier)"),
        (u_vac_fourier[0].cpu().numpy(), "Vacuum (Fourier)"),
    ]
    fourier_imax = max((np.abs(get_zoom(f))**2).max() for f, _ in fourier_fields)

    for col, (f, title) in enumerate(fourier_fields):
        zoomed = get_zoom(f)
        im = plot_field(axes[2, col], zoomed, fourier_ext_zoom,
                        f"{title}\n(central {2*zoom}px, dx={dx_fourier*1e6:.1f}μm)",
                        "irradiance", vmin=0, vmax=fourier_imax)
        axes[2, col].set_xlabel("μm")

    # Fourier plane phase (zoomed)
    for col_offset, (f, title) in enumerate(fourier_fields):
        zoomed = get_zoom(f)
        im = plot_field(axes[2, col_offset + 2], zoomed, fourier_ext_zoom,
                        f"Phase: {title}", "phase")
        axes[2, col_offset + 2].set_xlabel("μm")
    fig.colorbar(im, ax=axes[2, 3], shrink=0.7, label="rad")

    # ── Row 3: Detector focal plane (zoomed) ──────────────────
    focus_fields = [
        (u_turb_focus[0].cpu().numpy(), "Turbulent (focused)"),
        (u_vac_focus[0].cpu().numpy(), "Vacuum (focused)"),
        (u_fdnn_focus[0].cpu().numpy(), "FD2NN (focused)"),
    ]
    focus_imax = max((np.abs(get_zoom(f, focus_zoom))**2).max() for f, _ in focus_fields)

    for col, (f, title) in enumerate(focus_fields):
        zoomed = get_zoom(f, focus_zoom)
        im = plot_field(axes[3, col], zoomed, focus_ext_zoom,
                        f"{title}\n(f={FOCUS_F_M*1e3:.1f}mm, Airy={airy_radius*1e6:.1f}μm)",
                        "irradiance", vmin=0, vmax=focus_imax)
        axes[3, col].set_xlabel("μm")

    # Focused phase
    zoomed_vac = get_zoom(u_vac_focus[0].cpu().numpy(), focus_zoom)
    im = plot_field(axes[3, 3], zoomed_vac, focus_ext_zoom,
                    f"Phase: Vacuum (focused)", "phase")
    axes[3, 3].set_xlabel("μm")

    # ── Layout ────────────────────────────────────────────────
    row_labels = [
        "Receiver Plane — Irradiance",
        "Receiver Plane — Phase",
        "Fourier Plane (zoomed) — Irradiance & Phase",
        "Detector Focal Plane (zoomed) — Irradiance & Phase",
    ]
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=10, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUT_DIR / "single_case_4row.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()

    # ─── Print summary ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Receiver: N={N}, dx={RECEIVER_WINDOW_M/N*1e6:.1f}μm, window={RECEIVER_WINDOW_M*1e3:.2f}mm")
    print(f"Fourier:  dx={dx_fourier*1e6:.2f}μm, window={fourier_window*1e3:.2f}mm")
    print(f"Vacuum Fourier spot (1/e²): {spot_1e2_m*1e6:.1f}μm ({equiv_radius_px:.1f}px)")
    print(f"Spot/pixel ratio: {spot_1e2_m/dx_fourier:.2f}")
    print(f"Detector: f={FOCUS_F_M*1e3:.1f}mm, dx={dx_focus*1e6:.2f}μm, Airy={airy_radius*1e6:.1f}μm")
    print(f"Zero-phase throughput: {tp:.4f}")
    print(f"Baseline CO: {co_bl:.4f}")


if __name__ == "__main__":
    main()
