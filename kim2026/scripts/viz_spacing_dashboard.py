#!/usr/bin/env python
"""Spacing sweep dashboard: irradiance + phase + beam profile with physical scale.

Generates 3 figures for completed spacing runs (0, 1, 3, 6 mm):
  Fig A: 2D Irradiance maps (input, target, predictions) — physical µm scale
  Fig B: 2D Phase maps [0, 2π] — physical µm scale
  Fig C: Beam profiles (radial + centerline) — physical µm scale
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ─── Config ──────────────────────────────────────────────────────
PROJ = Path(__file__).resolve().parent.parent
SWEEP = PROJ / "runs" / "01_fd2nn_spacing_sweep_f10mm_claude"
FIG_DIR = PROJ / "figures" / "spacing_sweep_f10mm"

LAMBDA = 1.55e-6
F_M = 10e-3
N = 1024
WINDOW_M = 0.002048
DX = WINDOW_M / N  # 2 µm
DX_F = LAMBDA * F_M / (N * DX)
Z_R = math.pi * (10 * DX_F)**2 / LAMBDA

COMPLETED = ["spacing_0mm", "spacing_1mm", "spacing_3mm", "spacing_6mm",
             "spacing_12mm", "spacing_25mm", "spacing_50mm"]
SPACING_M = {"spacing_0mm": 0, "spacing_1mm": 1e-3, "spacing_3mm": 3e-3,
             "spacing_6mm": 6e-3, "spacing_12mm": 12e-3, "spacing_25mm": 25e-3,
             "spacing_50mm": 50e-3}
COLORS = {"spacing_0mm": "#888888", "spacing_1mm": "#e74c3c",
          "spacing_3mm": "#2ecc71", "spacing_6mm": "#3498db",
          "spacing_12mm": "#9b59b6", "spacing_25mm": "#f39c12",
          "spacing_50mm": "#e67e22"}

plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "figure.facecolor": "white"})


def _complex_from_npz(npz, prefix):
    return npz[f"{prefix}_real"] + 1j * npz[f"{prefix}_imag"]


def _physical_extent(n, dx_m):
    """Return extent in µm for imshow: [-half, +half, -half, +half]."""
    half = n * dx_m * 1e6 / 2
    return [-half, half, -half, half]


def _add_scalebar(ax, extent_um, bar_um=100):
    """Add a white scale bar in bottom-right."""
    xmin, xmax = extent_um[0], extent_um[1]
    ymin = extent_um[2]
    x0 = xmax - bar_um - 20
    y0 = ymin + 20
    ax.plot([x0, x0 + bar_um], [y0, y0], "w-", lw=2.5)
    ax.text(x0 + bar_um / 2, y0 + 15, f"{bar_um} µm", color="white",
            fontsize=6, ha="center", va="bottom", fontweight="bold")


def load_fields():
    """Load all completed runs' fields."""
    data = {}
    for name in COMPLETED:
        npz = np.load(SWEEP / name / "sample_fields.npz")
        data[name] = {
            "input": _complex_from_npz(npz, "input"),
            "pred": _complex_from_npz(npz, "pred"),
            "target": _complex_from_npz(npz, "target"),
        }
    return data


def load_metrics():
    metrics = {}
    for name in COMPLETED:
        with open(SWEEP / name / "test_metrics.json") as f:
            metrics[name] = json.load(f)
    return metrics


# ═══════════════════════════════════════════════════════════════════
# Figure A: Irradiance Dashboard
# ═══════════════════════════════════════════════════════════════════

def fig_irradiance(data, metrics):
    """6 cols: Input | Target | 4 predictions. 2 rows: full + zoom."""
    ncols = 2 + len(COMPLETED)  # input, target, 4 preds
    crop = 200  # pixels for zoom

    # Reference max for normalization
    ref_max = max(float((np.abs(data[n]["target"])**2).max()) for n in COMPLETED)

    fig, axes = plt.subplots(2, ncols, figsize=(3.0 * ncols, 6.5))

    # Column labels
    col_labels = ["Input\n(turbulent)", "Target\n(vacuum)"]
    for name in COMPLETED:
        sp = SPACING_M[name] * 1e3
        z_r = SPACING_M[name] / Z_R if SPACING_M[name] > 0 else 0
        m = metrics[name]
        col_labels.append(f"{sp:.0f}mm\nCO={m['complex_overlap']:.3f}")

    # Build field list
    first = data[COMPLETED[0]]
    fields = [first["input"], first["target"]]
    fields += [data[n]["pred"] for n in COMPLETED]

    extent_full = _physical_extent(N, DX)

    for col, (field, label) in enumerate(zip(fields, col_labels)):
        irr_full = np.abs(field)**2 / ref_max

        # Row 0: full view
        ax = axes[0, col]
        im = ax.imshow(irr_full, cmap="inferno", vmin=0, vmax=1,
                       origin="lower", extent=extent_full, interpolation="nearest")
        ax.set_title(label, fontsize=8, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Full field", fontsize=9)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("µm", fontsize=6)

        # Row 1: center zoom
        cy, cx = N // 2, N // 2
        crop_field = field[cy - crop:cy + crop, cx - crop:cx + crop]
        irr_crop = np.abs(crop_field)**2 / ref_max
        extent_crop = _physical_extent(2 * crop, DX)

        ax2 = axes[1, col]
        im2 = ax2.imshow(irr_crop, cmap="inferno", vmin=0, vmax=1,
                         origin="lower", extent=extent_crop, interpolation="nearest")
        if col == 0:
            ax2.set_ylabel("Center zoom", fontsize=9)
        ax2.tick_params(labelsize=6)
        ax2.set_xlabel("µm", fontsize=6)
        _add_scalebar(ax2, extent_crop, bar_um=100)

    # Colorbars
    cbar = fig.colorbar(im, ax=axes[0, :].tolist(), shrink=0.8, pad=0.02)
    cbar.set_label("I / I_max", fontsize=8)
    cbar.ax.tick_params(labelsize=6)
    cbar2 = fig.colorbar(im2, ax=axes[1, :].tolist(), shrink=0.8, pad=0.02)
    cbar2.set_label("I / I_max", fontsize=8)
    cbar2.ax.tick_params(labelsize=6)

    fig.suptitle(
        "FD2NN Spacing Sweep — Irradiance (f=10mm)\n"
        f"λ=1.55µm | dx_fourier={DX_F*1e6:.1f}µm | 5 layers | physical scale",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 0.95, 0.93])

    out = FIG_DIR / "dashboard_irradiance_spacing.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════
# Figure B: Phase Dashboard [0, 2π]
# ═══════════════════════════════════════════════════════════════════

def fig_phase(data, metrics):
    """Same layout as irradiance but showing phase [0, 2π]."""
    ncols = 2 + len(COMPLETED)
    crop = 200
    ref_max = max(float((np.abs(data[n]["target"])**2).max()) for n in COMPLETED)
    phase_thr = ref_max * 1e-3

    fig, axes = plt.subplots(2, ncols, figsize=(3.0 * ncols, 6.5))

    col_labels = ["Input\n(turbulent)", "Target\n(vacuum)"]
    for name in COMPLETED:
        sp = SPACING_M[name] * 1e3
        m = metrics[name]
        col_labels.append(f"{sp:.0f}mm\nφ_RMSE={m['phase_rmse_rad']:.2f}rad")

    first = data[COMPLETED[0]]
    fields = [first["input"], first["target"]]
    fields += [data[n]["pred"] for n in COMPLETED]

    extent_full = _physical_extent(N, DX)

    for col, (field, label) in enumerate(zip(fields, col_labels)):
        intensity = np.abs(field)**2
        phase = np.angle(field) % (2 * np.pi)
        masked = np.ma.masked_where(intensity < phase_thr, phase)

        # Row 0: full
        ax = axes[0, col]
        im = ax.imshow(masked, cmap="twilight", vmin=0, vmax=2 * np.pi,
                       origin="lower", extent=extent_full, interpolation="nearest")
        ax.set_title(label, fontsize=8, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Full field", fontsize=9)
        ax.tick_params(labelsize=6)

        # Row 1: zoom
        cy, cx = N // 2, N // 2
        crop_field = field[cy - crop:cy + crop, cx - crop:cx + crop]
        crop_int = np.abs(crop_field)**2
        crop_phase = np.angle(crop_field) % (2 * np.pi)
        crop_masked = np.ma.masked_where(crop_int < phase_thr, crop_phase)
        extent_crop = _physical_extent(2 * crop, DX)

        ax2 = axes[1, col]
        im2 = ax2.imshow(crop_masked, cmap="twilight", vmin=0, vmax=2 * np.pi,
                         origin="lower", extent=extent_crop, interpolation="nearest")
        if col == 0:
            ax2.set_ylabel("Center zoom", fontsize=9)
        ax2.tick_params(labelsize=6)
        _add_scalebar(ax2, extent_crop, bar_um=100)

    # Colorbars with π labels
    for row_axes, im_ref in [(axes[0, :], im), (axes[1, :], im2)]:
        cbar = fig.colorbar(im_ref, ax=row_axes.tolist(), shrink=0.8, pad=0.02)
        cbar.set_ticks([0, np.pi, 2 * np.pi])
        cbar.set_ticklabels(["0", "π", "2π"])
        cbar.set_label("Phase (rad)", fontsize=8)
        cbar.ax.tick_params(labelsize=6)

    fig.suptitle(
        "FD2NN Spacing Sweep — Phase [0, 2π] (f=10mm)\n"
        f"λ=1.55µm | dx_fourier={DX_F*1e6:.1f}µm | masked where I < 0.1% I_max",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 0.95, 0.93])

    out = FIG_DIR / "dashboard_phase_spacing.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════
# Figure C: Beam Profiles (physical scale)
# ═══════════════════════════════════════════════════════════════════

def fig_profiles(data, metrics):
    """4 panels: radial lin, radial log, centerline intensity, centerline phase error."""
    ref_max = max(float((np.abs(data[n]["target"])**2).max()) for n in COMPLETED)
    phase_thr = ref_max * 1e-3
    first = data[COMPLETED[0]]
    target = first["target"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Physical x-axis: µm
    x_px = np.arange(N) - N // 2
    x_um = x_px * DX * 1e6

    # ── (a) Radial irradiance (linear) ──
    ax = axes[0, 0]
    for label, field, color, ls in [
        ("Input (turb)", first["input"], "black", "--"),
        ("Target (vac)", target, "black", "-"),
    ]:
        irr = np.abs(field)**2 / ref_max
        cy, cx = N // 2, N // 2
        yy, xx = np.mgrid[:N, :N]
        rr = np.sqrt((xx - cx)**2 + (yy - cy)**2).astype(int)
        r_um = np.arange(min(rr.max() + 1, N // 2)) * DX * 1e6
        radial = np.zeros(len(r_um))
        for r in range(len(r_um)):
            mask = rr == r
            if mask.any():
                radial[r] = irr[mask].mean()
        ax.plot(r_um, radial, color=color, ls=ls, lw=1.5, alpha=0.6, label=label)

    for name in COMPLETED:
        field = data[name]["pred"]
        irr = np.abs(field)**2 / ref_max
        yy, xx = np.mgrid[:N, :N]
        rr = np.sqrt((xx - N//2)**2 + (yy - N//2)**2).astype(int)
        r_um = np.arange(min(rr.max() + 1, N // 2)) * DX * 1e6
        radial = np.zeros(len(r_um))
        for r in range(len(r_um)):
            mask = rr == r
            if mask.any():
                radial[r] = irr[mask].mean()
        sp = SPACING_M[name] * 1e3
        z_r = SPACING_M[name] / Z_R if SPACING_M[name] > 0 else 0
        ax.plot(r_um, radial, color=COLORS[name], lw=1.3,
                label=f"{sp:.0f}mm (z/z_R={z_r:.2f})")

    ax.set_xlabel("Radius (µm)")
    ax.set_ylabel("Normalized irradiance")
    ax.set_title("(a) Radial Intensity Profile", fontweight="bold")
    ax.set_xlim(0, 400)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ── (b) Radial irradiance (log) ──
    ax = axes[0, 1]
    for label, field, color, ls in [
        ("Input (turb)", first["input"], "black", "--"),
        ("Target (vac)", target, "black", "-"),
    ]:
        irr = np.abs(field)**2 / ref_max
        yy, xx = np.mgrid[:N, :N]
        rr = np.sqrt((xx - N//2)**2 + (yy - N//2)**2).astype(int)
        r_um = np.arange(min(rr.max() + 1, N // 2)) * DX * 1e6
        radial = np.zeros(len(r_um))
        for r in range(len(r_um)):
            mask = rr == r
            if mask.any():
                radial[r] = irr[mask].mean()
        ax.semilogy(r_um, np.clip(radial, 1e-7, None), color=color, ls=ls, lw=1.5, alpha=0.6, label=label)

    for name in COMPLETED:
        field = data[name]["pred"]
        irr = np.abs(field)**2 / ref_max
        yy, xx = np.mgrid[:N, :N]
        rr = np.sqrt((xx - N//2)**2 + (yy - N//2)**2).astype(int)
        r_um = np.arange(min(rr.max() + 1, N // 2)) * DX * 1e6
        radial = np.zeros(len(r_um))
        for r in range(len(r_um)):
            mask = rr == r
            if mask.any():
                radial[r] = irr[mask].mean()
        sp = SPACING_M[name] * 1e3
        ax.semilogy(r_um, np.clip(radial, 1e-7, None), color=COLORS[name], lw=1.3,
                     label=f"{sp:.0f}mm")

    ax.set_xlabel("Radius (µm)")
    ax.set_ylabel("Normalized irradiance (log)")
    ax.set_title("(b) Radial Profile (log scale)", fontweight="bold")
    ax.set_xlim(0, 600)
    ax.set_ylim(1e-5, 2)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ── (c) Centerline intensity ──
    ax = axes[1, 0]
    for label, field, color, ls in [
        ("Input (turb)", first["input"], "black", "--"),
        ("Target (vac)", target, "black", "-"),
    ]:
        irr = np.abs(field)**2 / ref_max
        center = irr[N // 2, :]
        ax.plot(x_um, center, color=color, ls=ls, lw=1.5, alpha=0.6, label=label)

    for name in COMPLETED:
        field = data[name]["pred"]
        irr = np.abs(field)**2 / ref_max
        center = irr[N // 2, :]
        sp = SPACING_M[name] * 1e3
        ax.plot(x_um, center, color=COLORS[name], lw=1.3, label=f"{sp:.0f}mm")

    ax.set_xlabel("Position (µm)")
    ax.set_ylabel("Normalized irradiance")
    ax.set_title("(c) Centerline Intensity", fontweight="bold")
    ax.set_xlim(-400, 400)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ── (d) Centerline phase error ──
    ax = axes[1, 1]
    for name in COMPLETED:
        field = data[name]["pred"]
        phase_err = np.angle(field * np.conj(target))
        f_i = np.abs(field)**2
        t_i = np.abs(target)**2
        masked = np.where((f_i < phase_thr) | (t_i < phase_thr), np.nan, phase_err)
        center = masked[N // 2, :]
        sp = SPACING_M[name] * 1e3
        z_r = SPACING_M[name] / Z_R if SPACING_M[name] > 0 else 0
        ax.plot(x_um, center, color=COLORS[name], lw=1.0, alpha=0.8,
                label=f"{sp:.0f}mm (z/z_R={z_r:.2f})")

    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.set_xlabel("Position (µm)")
    ax.set_ylabel("Phase error (rad)")
    ax.set_title("(d) Centerline Phase Error vs Vacuum", fontweight="bold")
    ax.set_xlim(-300, 300)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_yticklabels(["-π", "-π/2", "0", "π/2", "π"])
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "FD2NN Beam Profiles — Spacing Comparison (f=10mm)\n"
        f"λ=1.55µm | dx={DX*1e6:.0f}µm | physical scale",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = FIG_DIR / "dashboard_profiles_spacing.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading fields...")
    data = load_fields()
    metrics = load_metrics()

    print("\nGenerating irradiance dashboard...")
    fig_irradiance(data, metrics)

    print("Generating phase dashboard...")
    fig_phase(data, metrics)

    print("Generating beam profiles...")
    fig_profiles(data, metrics)

    print("\nDone.")


if __name__ == "__main__":
    main()
