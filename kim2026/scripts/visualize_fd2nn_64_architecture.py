#!/usr/bin/env python
"""Phase 1: FD2NN-64 Architecture — Low-Resolution Fourier Mask.

FD2NN with n_mask=64 (cropped Fourier plane):
  Input(1024) → Lens1(f=25mm) → FFT → crop central 64×64 → 5 masks → pad to 1024 → Lens2 → Output

Key insight: beam spot ~7px in 1024 grid → 7/64 = 10.9% utilization in 64 grid (vs 0.015% in 1024).

Usage:
    cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python scripts/visualize_fd2nn_64_architecture.py
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def draw_fd2nn_64(ax):
    """FD2NN-64 optical layout with crop/pad in Fourier plane."""
    ax.set_xlim(-2, 120)
    ax.set_ylim(-18, 22)
    ax.set_aspect("equal")
    ax.set_title(
        "FD2NN-64: Low-Resolution Fourier Mask (Option E)\n"
        "Input(1024) → f=25mm → FFT(1024) → Crop(64) → 5 Masks(64×64) → Pad(1024) → IFFT → Output",
        fontsize=15, fontweight="bold", pad=10,
    )

    c_lens = "#3498db"
    c_mask = "#e74c3c"
    c_beam = "#f39c1240"
    c_beam_e = "#e67e22"
    c_focus = "#16a085"
    c_det = "#8e44ad"
    c_crop = "#9b59b6"

    # Positions
    input_x = 0
    lens1_x = 8
    fft_x = 20
    crop_x = 30
    mask_start = 38
    mask_sp = 5
    mask_xs = [mask_start + i * mask_sp for i in range(5)]
    pad_x = mask_xs[-1] + 6
    ifft_x = pad_x + 8
    lens2_x = ifft_x + 8
    output_x = lens2_x + 5
    focus_x = output_x + 5
    det_x = focus_x + 5

    bh_in = 5       # input beam half-height
    bh_fft = 5      # FFT output (full 1024)
    bh_crop = 1.5   # cropped 64 (small)
    bh_mask = 1.5   # mask region

    # ═══ Beam path ═══
    # Input → Lens1 → FFT (converging)
    ax.fill_between([input_x, lens1_x, fft_x],
                    [bh_in, bh_in, bh_fft],
                    [-bh_in, -bh_in, -bh_fft],
                    color=c_beam, edgecolor=c_beam_e, lw=0.8)

    # FFT → Crop (the spot is tiny in the full grid)
    # Show full FFT grid with tiny spot
    ax.fill_between([fft_x, crop_x - 1],
                    [bh_fft, bh_fft],
                    [-bh_fft, -bh_fft],
                    color="#f39c1208", edgecolor="#e67e2240", lw=0.5)
    # Actual beam (tiny spot)
    ax.fill_between([fft_x, crop_x - 1],
                    [0.3, 0.3],
                    [-0.3, -0.3],
                    color="#e74c3c30", edgecolor="#e74c3c", lw=1)

    # Crop region highlight
    ax.fill_between([crop_x - 1, crop_x + 1],
                    [bh_crop + 0.5, bh_crop + 0.5],
                    [-bh_crop - 0.5, -bh_crop - 0.5],
                    color=c_crop, alpha=0.1, edgecolor=c_crop, lw=2, ls="--")

    # Through masks (cropped beam, filling ~11%)
    ax.fill_between([crop_x + 1] + [mx for mx in mask_xs] + [pad_x - 1],
                    [bh_mask] + [bh_mask] * len(mask_xs) + [bh_mask],
                    [-bh_mask] + [-bh_mask] * len(mask_xs) + [-bh_mask],
                    color="#e74c3c20", edgecolor="#e74c3c80", lw=0.8)

    # Pad → IFFT → Lens2 → output (expanding back)
    ax.fill_between([pad_x - 1, pad_x + 1],
                    [bh_crop + 0.5, bh_fft],
                    [-bh_crop - 0.5, -bh_fft],
                    color=c_crop, alpha=0.05, edgecolor=c_crop, lw=1.5, ls="--")

    ax.fill_between([pad_x + 1, ifft_x, lens2_x, output_x, focus_x, det_x],
                    [bh_fft, bh_fft, bh_in, bh_in, bh_in, 0.3],
                    [-bh_fft, -bh_fft, -bh_in, -bh_in, -bh_in, -0.3],
                    color=c_beam, edgecolor=c_beam_e, lw=0.8)

    # ═══ Input ═══
    ax.annotate("Input\n1024×1024\n2.048mm", xy=(input_x, -9), fontsize=14, ha="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#ecf0f1", edgecolor="gray"))

    # ═══ Lens 1 ═══
    ax.plot([lens1_x, lens1_x], [-7, 7], color=c_lens, lw=3)
    ax.text(lens1_x, -10, "Lens1\nf=25mm", fontsize=14, ha="center", color=c_lens, fontweight="bold")

    # ═══ FFT ═══
    ax.add_patch(patches.FancyBboxPatch((fft_x - 1.5, -6), 3, 12,
                 boxstyle="round,pad=0.3", facecolor="#ecf0f1", alpha=0.3, edgecolor="gray", lw=1.5))
    ax.text(fft_x, 7.5, "FFT\n1024×1024", fontsize=13, ha="center", color="gray")

    # ═══ Crop ═══
    ax.text(crop_x, bh_crop + 3, "CROP\n→ 64×64", fontsize=15, ha="center", color=c_crop,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#d2b4de", alpha=0.3, edgecolor=c_crop))
    ax.annotate("", xy=(crop_x, bh_crop + 1), xytext=(crop_x, bh_crop + 2.5),
                arrowprops=dict(arrowstyle="->", color=c_crop, lw=2))

    # ═══ 5 Masks ═══
    for i, mx in enumerate(mask_xs):
        ax.add_patch(patches.Rectangle((mx - 0.3, -bh_mask - 1), 0.6, 2 * bh_mask + 2,
                     facecolor=c_mask, alpha=0.25, edgecolor=c_mask, lw=2))
        ax.text(mx, bh_mask + 2, f"M{i}", fontsize=14, ha="center", color=c_mask, fontweight="bold")

    # Mask label
    ax.text((mask_xs[0] + mask_xs[-1]) / 2, -5,
            "5 Fourier masks\n64×64, dx=302μm",
            fontsize=13, ha="center", color=c_mask,
            bbox=dict(boxstyle="round", facecolor=c_mask, alpha=0.08))

    # Beam utilization note
    ax.text((mask_xs[1] + mask_xs[3]) / 2, bh_mask - 0.3,
            "Spot fills ~11%\nof mask!",
            fontsize=16, ha="center", color="green", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.7, edgecolor="green"))

    # ═══ Pad ═══
    ax.text(pad_x, bh_crop + 3, "PAD\n→ 1024", fontsize=15, ha="center", color=c_crop,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#d2b4de", alpha=0.3, edgecolor=c_crop))

    # ═══ IFFT ═══
    ax.add_patch(patches.FancyBboxPatch((ifft_x - 1.5, -6), 3, 12,
                 boxstyle="round,pad=0.3", facecolor="#ecf0f1", alpha=0.3, edgecolor="gray", lw=1.5))
    ax.text(ifft_x, 7.5, "IFFT\n1024×1024", fontsize=13, ha="center", color="gray")

    # ═══ Lens 2 ═══
    ax.plot([lens2_x, lens2_x], [-7, 7], color=c_lens, lw=3)
    ax.text(lens2_x, -10, "Lens2\nf=25mm", fontsize=14, ha="center", color=c_lens, fontweight="bold")

    # ═══ Focus ═══
    ax.plot([focus_x, focus_x], [-7, 7], color=c_focus, lw=3)
    ax.text(focus_x, -10, "Focus\nf=4.5mm", fontsize=14, ha="center", color=c_focus)

    # ═══ Detector ═══
    ax.add_patch(patches.Rectangle((det_x - 0.5, -3.5), 1, 7,
                 facecolor=c_det, alpha=0.4, edgecolor=c_det, lw=2))
    ax.text(det_x, -7, "APD/MMF", fontsize=14, ha="center", color=c_det, fontweight="bold")

    # ═══ Comparison annotation ═══
    ax.text(fft_x + 3, -14,
            "Before: 7px spot in 1024² mask → 0.015% used\n"
            "Now:    7px spot in 64² mask → 10.9% used  (730× improvement)",
            fontsize=14, ha="left", color="#c0392b", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fdebd0", edgecolor="#e67e22"))

    # Dimensions
    ax.annotate("", xy=(lens1_x, 18), xytext=(lens2_x, 18),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1.5))
    ax.text((lens1_x + lens2_x) / 2, 19, "~70mm (2f system)", fontsize=15, ha="center", color="gray")

    ax.set_xlabel("Optical axis [mm]", fontsize=14)
    ax.axhline(0, color="gray", lw=0.3, ls=":")
    ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=13)


def draw_table(ax):
    """FD2NN-64 parameter table."""
    ax.axis("off")
    data = [
        ["Parameter", "FD2NN-1024\n(Previous, failed)", "FD2NN-64\n(Option E)"],
        ["Architecture", "Fourier (2f lens)", "Fourier (2f + crop/pad)"],
        ["Input grid", "1024 × 1024", "1024 × 1024"],
        ["Mask grid", "1024 × 1024", "64 × 64"],
        ["Mask pixel (dx)", "18.9 um", "302 um"],
        ["Fourier spot", "~7 px", "~7 px"],
        ["Beam utilization", "0.015%", "10.9%"],
        ["Parameters", "5.2M (800 used)", "20,480 (all used)"],
        ["Overfitting risk", "Very high", "Very low"],
        ["Lens", "AC254-025-C x2", "AC254-025-C x2"],
        ["NA", "0.508", "0.508"],
        ["Cn²", "—", "5.0e-14"],
        ["D/r₀", "1.91", "5.02"],
        ["Train / Val / Test", "160 / 20 / 20", "400 / 50 / 50"],
        ["Batch size", "2", "16"],
        ["Epochs", "100", "30"],
        ["Learning rate", "5e-4", "5e-4"],
    ]
    table = ax.table(cellText=data, loc="upper center", cellLoc="center",
                     colWidths=[0.25, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(17)
    table.scale(1, 2.1)

    for j in range(3):
        table[0, j].set_facecolor("#34495e")
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=15)
    for i in range(1, len(data)):
        table[i, 0].set_facecolor("#ecf0f1")
        table[i, 0].set_text_props(fontweight="bold")
        table[i, 1].set_facecolor("#fdebd0")
        table[i, 2].set_facecolor("#d5f5e3")

    # Highlight key improvements
    table[6, 1].set_facecolor("#f5b7b1")   # 0.015% bad
    table[6, 2].set_facecolor("#abebc6")   # 10.9% good
    table[7, 1].set_facecolor("#f5b7b1")   # 5.2M wasted
    table[7, 2].set_facecolor("#abebc6")   # 20K all used
    table[8, 1].set_facecolor("#f5b7b1")   # high overfitting
    table[8, 2].set_facecolor("#abebc6")   # low overfitting

    ax.set_title("FD2NN-64 vs FD2NN-1024: Parameter Comparison",
                 fontsize=20, fontweight="bold", pad=5)


def main():
    fig = plt.figure(figsize=(30, 34))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.08)

    draw_fd2nn_64(fig.add_subplot(gs[0]))
    draw_table(fig.add_subplot(gs[1]))

    fig.suptitle("FD2NN-64: Low-Resolution Fourier Mask Experiment",
                 fontsize=24, fontweight="bold", y=0.98)

    out_dir = "/root/dj/D2NN/kim2026/autoresearch/runs/fd2nn_64_strong_turb"
    import os
    os.makedirs(out_dir, exist_ok=True)
    out = f"{out_dir}/phase1_architecture.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
