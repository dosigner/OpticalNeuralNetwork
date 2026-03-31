#!/usr/bin/env python
"""Generate comprehensive dashboard figures for f=10mm spacing sweep + loss sweep.

Outputs:
  fig_spacing_field_dashboard.png  — 2D irradiance + phase + beam profiles (physical scale)
  fig_loss_field_dashboard.png     — Loss function comparison fields
  fig_combined_metrics.png         — Combined metrics table + CO vs spacing + CO vs loss
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
PROJ = Path(__file__).resolve().parent.parent
SPACING_DIR = PROJ / "runs" / "중요_01_fd2nn_spacing_sweep_f10mm_claude"
LOSS_DIR = PROJ / "runs" / "06_fd2nn_loss_sweep_f10mm_sp50mm_claude"
FIG_DIR = PROJ / "figures" / "f10mm_dashboard"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── System parameters ────────────────────────────────────────────────────────
LAM = 1.55e-6
N = 1024
DX_IN = 2.0e-6
F = 10.0e-3
DX_F = LAM * F / (N * DX_IN)
WINDOW = DX_IN * N  # 2.048 mm
Z_R_10PX = math.pi * (10 * DX_F) ** 2 / LAM

SPACING_NAMES = ["spacing_0mm", "spacing_1mm", "spacing_3mm",
                 "spacing_6mm", "spacing_12mm", "spacing_25mm", "spacing_50mm"]
SPACING_MM = [0, 1, 3, 6, 12, 25, 50]
SPACING_LABELS = ["0 mm", "1 mm", "3 mm", "6 mm", "12 mm", "25 mm", "50 mm"]

LOSS_NAMES = ["complex", "phasor", "irradiance", "hybrid"]
LOSS_LABELS = ["Complex\n(CO+IO+BR+EE)", "Phasor\n(phase-only)",
               "Irradiance\n(IO+BR+EE)", "Hybrid\n(CO+IO+BR)"]

plt.rcParams.update({
    "font.size": 9, "figure.dpi": 200, "figure.facecolor": "white",
    "axes.titlesize": 10, "axes.labelsize": 9,
})


def _complex_from_npz(npz, prefix):
    return npz[f"{prefix}_real"] + 1j * npz[f"{prefix}_imag"]


def _crop(arr, radius):
    cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
    return arr[cy - radius:cy + radius, cx - radius:cx + radius]


def _load_fields(run_dir):
    p = run_dir / "sample_fields.npz"
    if not p.exists():
        return None
    npz = np.load(p)
    return {
        "input": _complex_from_npz(npz, "input"),
        "pred": _complex_from_npz(npz, "pred"),
        "target": _complex_from_npz(npz, "target"),
    }


def _load_metrics(run_dir):
    p = run_dir / "test_metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _physical_extent(crop_radius):
    """Returns extent in µm for imshow."""
    half = crop_radius * DX_IN * 1e6  # µm
    return [-half, half, -half, half]


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Spacing sweep — field dashboard
# ═══════════════════════════════════════════════════════════════════════════

def fig_spacing_field_dashboard():
    crop = 200  # pixels
    ext = _physical_extent(crop)

    # Load all fields
    all_fields = {}
    for name in SPACING_NAMES:
        f = _load_fields(SPACING_DIR / name)
        if f:
            all_fields[name] = f

    if not all_fields:
        print("No spacing fields found, skipping fig1")
        return

    first = next(iter(all_fields.values()))
    target = first["target"]
    ref_max = float((np.abs(target) ** 2).max())
    phase_thr = ref_max * 1e-3

    n_runs = len(all_fields)
    fig = plt.figure(figsize=(20, 3.2 * n_runs + 2))
    gs = gridspec.GridSpec(n_runs, 6, hspace=0.25, wspace=0.12,
                           left=0.06, right=0.94, top=0.93, bottom=0.05)

    col_titles = ["Input irradiance", "Pred irradiance", "Target irradiance",
                  "|Pred−Target|", "Pred phase [0,2π]", "Phase error"]

    for row_idx, (name, fields) in enumerate(all_fields.items()):
        inp_c = _crop(fields["input"], crop)
        prd_c = _crop(fields["pred"], crop)
        tgt_c = _crop(fields["target"], crop)

        i_inp = np.abs(inp_c) ** 2 / ref_max
        i_prd = np.abs(prd_c) ** 2 / ref_max
        i_tgt = np.abs(tgt_c) ** 2 / ref_max
        i_err = np.abs(i_prd - i_tgt)

        p_prd = np.angle(prd_c) % (2 * np.pi)
        p_prd_masked = np.ma.masked_where(np.abs(prd_c) ** 2 < phase_thr, p_prd)

        p_err = np.angle(prd_c * np.conj(tgt_c))
        p_err_masked = np.ma.masked_where(
            (np.abs(prd_c) ** 2 < phase_thr) | (np.abs(tgt_c) ** 2 < phase_thr), p_err)

        images = [
            (i_inp, {"cmap": "inferno", "vmin": 0, "vmax": 1}),
            (i_prd, {"cmap": "inferno", "vmin": 0, "vmax": 1}),
            (i_tgt, {"cmap": "inferno", "vmin": 0, "vmax": 1}),
            (i_err, {"cmap": "magma", "vmin": 0, "vmax": 0.3}),
            (p_prd_masked, {"cmap": "twilight", "vmin": 0, "vmax": 2 * np.pi}),
            (p_err_masked, {"cmap": "RdBu_r", "vmin": -np.pi, "vmax": np.pi}),
        ]

        sp_mm = SPACING_MM[SPACING_NAMES.index(name)]
        z_zr = sp_mm * 1e-3 / Z_R_10PX if sp_mm > 0 else 0

        for col_idx, (img, kw) in enumerate(images):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(img, origin="lower", extent=ext, **kw)
            ax.set_xticks([]); ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=9, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{sp_mm}mm\nz/z_R={z_zr:.2f}",
                              fontsize=8, fontweight="bold")

    # Colorbars
    for col_idx, (cmap, vmin, vmax, label) in enumerate([
        ("inferno", 0, 1, "I/I_max"), ("inferno", 0, 1, "I/I_max"),
        ("inferno", 0, 1, "I/I_max"), ("magma", 0, 0.3, "|ΔI|/I_max"),
        ("twilight", 0, 2*np.pi, "Phase (rad)"), ("RdBu_r", -np.pi, np.pi, "Δφ (rad)"),
    ]):
        cbar_ax = fig.add_axes([0.06 + col_idx * 0.147, 0.015, 0.12, 0.01])
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=cmap)
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label(label, fontsize=7)
        if col_idx == 4:
            cbar.set_ticks([0, np.pi, 2 * np.pi])
            cbar.set_ticklabels(["0", "π", "2π"])
        elif col_idx == 5:
            cbar.set_ticks([-np.pi, 0, np.pi])
            cbar.set_ticklabels(["-π", "0", "π"])

    fig.suptitle(
        "FD2NN Spacing Sweep — Field Comparison (f=10mm, dx_f=7.57µm)\n"
        f"λ=1.55µm | 5 layers | tanh 2π | 30 epochs | physical scale (µm)",
        fontsize=12, fontweight="bold",
    )

    out = FIG_DIR / "fig_spacing_field_dashboard.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Loss sweep — field dashboard
# ═══════════════════════════════════════════════════════════════════════════

def fig_loss_field_dashboard():
    crop = 200
    ext = _physical_extent(crop)

    all_fields = {}
    all_metrics = {}
    for name in LOSS_NAMES:
        f = _load_fields(LOSS_DIR / name)
        m = _load_metrics(LOSS_DIR / name)
        if f and m:
            all_fields[name] = f
            all_metrics[name] = m

    if not all_fields:
        print("No loss sweep fields found, skipping fig2")
        return

    first = next(iter(all_fields.values()))
    target = first["target"]
    ref_max = float((np.abs(target) ** 2).max())
    phase_thr = ref_max * 1e-3

    n_runs = len(all_fields)
    fig = plt.figure(figsize=(20, 3.2 * n_runs + 2))
    gs = gridspec.GridSpec(n_runs, 6, hspace=0.25, wspace=0.12,
                           left=0.06, right=0.94, top=0.93, bottom=0.05)

    col_titles = ["Input irradiance", "Pred irradiance", "Target irradiance",
                  "|Pred−Target|", "Pred phase [0,2π]", "Phase error"]

    for row_idx, name in enumerate(LOSS_NAMES):
        if name not in all_fields:
            continue
        fields = all_fields[name]
        m = all_metrics[name]

        inp_c = _crop(fields["input"], crop)
        prd_c = _crop(fields["pred"], crop)
        tgt_c = _crop(fields["target"], crop)

        i_inp = np.abs(inp_c) ** 2 / ref_max
        i_prd = np.abs(prd_c) ** 2 / ref_max
        i_tgt = np.abs(tgt_c) ** 2 / ref_max
        i_err = np.abs(i_prd - i_tgt)

        p_prd = np.angle(prd_c) % (2 * np.pi)
        p_prd_masked = np.ma.masked_where(np.abs(prd_c) ** 2 < phase_thr, p_prd)
        p_err = np.angle(prd_c * np.conj(tgt_c))
        p_err_masked = np.ma.masked_where(
            (np.abs(prd_c) ** 2 < phase_thr) | (np.abs(tgt_c) ** 2 < phase_thr), p_err)

        images = [
            (i_inp, {"cmap": "inferno", "vmin": 0, "vmax": 1}),
            (i_prd, {"cmap": "inferno", "vmin": 0, "vmax": 1}),
            (i_tgt, {"cmap": "inferno", "vmin": 0, "vmax": 1}),
            (i_err, {"cmap": "magma", "vmin": 0, "vmax": 0.3}),
            (p_prd_masked, {"cmap": "twilight", "vmin": 0, "vmax": 2 * np.pi}),
            (p_err_masked, {"cmap": "RdBu_r", "vmin": -np.pi, "vmax": np.pi}),
        ]

        label = LOSS_LABELS[LOSS_NAMES.index(name)]
        co = m["complex_overlap"]
        io = m["intensity_overlap"]

        for col_idx, (img, kw) in enumerate(images):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(img, origin="lower", extent=ext, **kw)
            ax.set_xticks([]); ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=9, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{label}\nCO={co:.3f} IO={io:.3f}",
                              fontsize=7, fontweight="bold")

    # Colorbars (same as spacing fig)
    for col_idx, (cmap, vmin, vmax, lbl) in enumerate([
        ("inferno", 0, 1, "I/I_max"), ("inferno", 0, 1, "I/I_max"),
        ("inferno", 0, 1, "I/I_max"), ("magma", 0, 0.3, "|ΔI|/I_max"),
        ("twilight", 0, 2*np.pi, "Phase (rad)"), ("RdBu_r", -np.pi, np.pi, "Δφ (rad)"),
    ]):
        cbar_ax = fig.add_axes([0.06 + col_idx * 0.147, 0.015, 0.12, 0.01])
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=cmap)
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label(lbl, fontsize=7)
        if col_idx == 4:
            cbar.set_ticks([0, np.pi, 2 * np.pi])
            cbar.set_ticklabels(["0", "π", "2π"])
        elif col_idx == 5:
            cbar.set_ticks([-np.pi, 0, np.pi])
            cbar.set_ticklabels(["-π", "0", "π"])

    fig.suptitle(
        "FD2NN Loss Function Comparison — Field Dashboard (f=10mm, spacing=50mm)\n"
        f"λ=1.55µm | 5 layers | tanh 2π | 100 epochs | physical scale (µm)",
        fontsize=12, fontweight="bold",
    )

    out = FIG_DIR / "fig_loss_field_dashboard.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Combined metrics + beam profiles
# ═══════════════════════════════════════════════════════════════════════════

def fig_combined_metrics():
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30,
                           left=0.07, right=0.96, top=0.93, bottom=0.06)

    # ── (a) CO vs spacing ──
    ax_sp = fig.add_subplot(gs[0, 0])
    cos, ios, prs = [], [], []
    for name in SPACING_NAMES:
        m = _load_metrics(SPACING_DIR / name)
        if m:
            cos.append(m["complex_overlap"])
            ios.append(m["intensity_overlap"])
            prs.append(m["phase_rmse_rad"])
        else:
            cos.append(np.nan); ios.append(np.nan); prs.append(np.nan)

    z_zr = [s * 1e-3 / Z_R_10PX if s > 0 else 0 for s in SPACING_MM]

    ax_sp.plot(SPACING_MM, cos, "o-", color="#e74c3c", lw=2, ms=7, label="CO ↑")
    ax_sp.axhline(0.1913, color="gray", ls="--", lw=1, label="Baseline CO=0.191")
    ax_sp.set_xlabel("Layer spacing (mm)")
    ax_sp.set_ylabel("Complex Overlap", color="#e74c3c")
    ax_sp.set_title("(a) CO vs Layer Spacing (f=10mm)", fontweight="bold")
    ax_sp.legend(fontsize=8, loc="lower right")
    ax_sp.grid(alpha=0.3)

    ax_sp2 = ax_sp.twinx()
    ax_sp2.plot(SPACING_MM, ios, "s--", color="#3498db", lw=1.5, ms=5, label="IO ↑")
    ax_sp2.set_ylabel("Intensity Overlap", color="#3498db")
    ax_sp2.legend(fontsize=8, loc="center right")

    # z/z_R annotation
    for i, (sp, zz) in enumerate(zip(SPACING_MM, z_zr)):
        if sp > 0:
            ax_sp.annotate(f"z/z_R={zz:.1f}", (sp, cos[i]),
                          textcoords="offset points", xytext=(5, 8), fontsize=6, color="gray")

    # ── (b) CO vs loss function ──
    ax_loss = fig.add_subplot(gs[0, 1])
    loss_co, loss_io, loss_pr = [], [], []
    for name in LOSS_NAMES:
        m = _load_metrics(LOSS_DIR / name)
        if m:
            loss_co.append(m["complex_overlap"])
            loss_io.append(m["intensity_overlap"])
            loss_pr.append(m["phase_rmse_rad"])

    x = np.arange(len(LOSS_NAMES))
    w = 0.35
    bars1 = ax_loss.bar(x - w/2, loss_co, w, color="#e74c3c", alpha=0.8, label="CO")
    bars2 = ax_loss.bar(x + w/2, loss_io, w, color="#3498db", alpha=0.8, label="IO")
    ax_loss.axhline(0.1913, color="gray", ls="--", lw=1, label="Baseline CO")
    ax_loss.axhline(0.9725, color="gray", ls=":", lw=1, label="Baseline IO")
    ax_loss.set_xticks(x)
    ax_loss.set_xticklabels(["Complex", "Phasor", "Irradiance", "Hybrid"], fontsize=8)
    ax_loss.set_ylabel("Metric value")
    ax_loss.set_title("(b) CO vs IO by Loss Function (f=10mm, sp=50mm, 100ep)",
                      fontweight="bold")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(alpha=0.3, axis="y")

    # Value labels
    for bar, val in zip(bars1, loss_co):
        ax_loss.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=7, color="#e74c3c")
    for bar, val in zip(bars2, loss_io):
        ax_loss.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=7, color="#3498db")

    # ── (c) Radial beam profiles (spacing sweep) ──
    ax_rad = fig.add_subplot(gs[1, 0])
    colors_sp = ["#888888", "#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#f39c12", "#e67e22"]
    crop = 200

    first_fields = _load_fields(SPACING_DIR / SPACING_NAMES[0])
    if first_fields:
        tgt = first_fields["target"]
        ref_max = float((np.abs(tgt) ** 2).max())
        tgt_c = _crop(tgt, crop)

        # Target profile
        i_tgt = np.abs(tgt_c) ** 2 / ref_max
        cy = i_tgt.shape[0] // 2
        yy, xx = np.mgrid[:i_tgt.shape[0], :i_tgt.shape[1]]
        rr = np.sqrt((xx - cy) ** 2 + (yy - cy) ** 2)
        r_um = rr * DX_IN * 1e6
        r_max_um = crop * DX_IN * 1e6

        # Azimuthal average for target
        r_bins = np.arange(0, crop, 1)
        rad_tgt = np.zeros(len(r_bins))
        for ri, r in enumerate(r_bins):
            mask = (rr >= r) & (rr < r + 1)
            if mask.any():
                rad_tgt[ri] = i_tgt[mask].mean()
        r_um_ax = r_bins * DX_IN * 1e6

        ax_rad.plot(r_um_ax, rad_tgt, "k-", lw=2.5, label="Target (vacuum)")

        for si, name in enumerate(SPACING_NAMES):
            fields = _load_fields(SPACING_DIR / name)
            if not fields:
                continue
            prd_c = _crop(fields["pred"], crop)
            i_prd = np.abs(prd_c) ** 2 / ref_max
            rad = np.zeros(len(r_bins))
            for ri, r in enumerate(r_bins):
                mask = (rr >= r) & (rr < r + 1)
                if mask.any():
                    rad[ri] = i_prd[mask].mean()
            ax_rad.plot(r_um_ax, rad, color=colors_sp[si], lw=1.2, alpha=0.8,
                       label=f"{SPACING_LABELS[si]} (z/z_R={z_zr[si]:.1f})")

    ax_rad.set_xlabel("Radius (µm)")
    ax_rad.set_ylabel("Normalized irradiance")
    ax_rad.set_title("(c) Radial Beam Profile — Spacing Sweep (physical scale)",
                     fontweight="bold")
    ax_rad.set_xlim(0, 250)
    ax_rad.legend(fontsize=6.5, loc="upper right")
    ax_rad.grid(alpha=0.3)

    # ── (d) Radial beam profiles (loss sweep) ──
    ax_rad2 = fig.add_subplot(gs[1, 1])
    colors_loss = ["#e74c3c", "#9b59b6", "#f39c12", "#2ecc71"]

    first_loss = _load_fields(LOSS_DIR / LOSS_NAMES[0])
    if first_loss:
        tgt = first_loss["target"]
        ref_max = float((np.abs(tgt) ** 2).max())
        tgt_c = _crop(tgt, crop)
        i_tgt = np.abs(tgt_c) ** 2 / ref_max
        cy = i_tgt.shape[0] // 2
        yy, xx = np.mgrid[:i_tgt.shape[0], :i_tgt.shape[1]]
        rr = np.sqrt((xx - cy) ** 2 + (yy - cy) ** 2)
        r_bins = np.arange(0, crop, 1)
        r_um_ax = r_bins * DX_IN * 1e6

        rad_tgt = np.zeros(len(r_bins))
        for ri, r in enumerate(r_bins):
            mask = (rr >= r) & (rr < r + 1)
            if mask.any():
                rad_tgt[ri] = i_tgt[mask].mean()
        ax_rad2.plot(r_um_ax, rad_tgt, "k-", lw=2.5, label="Target (vacuum)")

        # Input
        inp_c = _crop(first_loss["input"], crop)
        i_inp = np.abs(inp_c) ** 2 / ref_max
        rad_inp = np.zeros(len(r_bins))
        for ri, r in enumerate(r_bins):
            mask = (rr >= r) & (rr < r + 1)
            if mask.any():
                rad_inp[ri] = i_inp[mask].mean()
        ax_rad2.plot(r_um_ax, rad_inp, "k--", lw=1.5, alpha=0.5, label="Input (turb)")

        for li, name in enumerate(LOSS_NAMES):
            fields = _load_fields(LOSS_DIR / name)
            if not fields:
                continue
            m = _load_metrics(LOSS_DIR / name)
            prd_c = _crop(fields["pred"], crop)
            i_prd = np.abs(prd_c) ** 2 / ref_max
            rad = np.zeros(len(r_bins))
            for ri, r in enumerate(r_bins):
                mask = (rr >= r) & (rr < r + 1)
                if mask.any():
                    rad[ri] = i_prd[mask].mean()
            co = m["complex_overlap"] if m else 0
            ax_rad2.plot(r_um_ax, rad, color=colors_loss[li], lw=1.5, alpha=0.9,
                        label=f"{LOSS_NAMES[li]} (CO={co:.3f})")

    ax_rad2.set_xlabel("Radius (µm)")
    ax_rad2.set_ylabel("Normalized irradiance")
    ax_rad2.set_title("(d) Radial Beam Profile — Loss Comparison (physical scale)",
                      fontweight="bold")
    ax_rad2.set_xlim(0, 250)
    ax_rad2.legend(fontsize=7, loc="upper right")
    ax_rad2.grid(alpha=0.3)

    fig.suptitle(
        "FD2NN f=10mm Combined Analysis Dashboard\n"
        r"$\lambda=1.55\,\mu$m | dx$_f$=7.57µm (4.9λ) | 5 layers | Cn²=1×10⁻¹⁴",
        fontsize=13, fontweight="bold",
    )

    out = FIG_DIR / "fig_combined_metrics.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=== F=10mm Dashboard Generation ===\n")

    print("1. Spacing sweep field dashboard...")
    fig_spacing_field_dashboard()

    print("\n2. Loss sweep field dashboard...")
    fig_loss_field_dashboard()

    print("\n3. Combined metrics dashboard...")
    fig_combined_metrics()

    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()