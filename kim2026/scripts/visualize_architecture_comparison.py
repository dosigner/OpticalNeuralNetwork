#!/usr/bin/env python
"""FD2NN vs D2NN architecture layout comparison — optical schematic."""
from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def draw_fdnn(ax):
    """Draw FD2NN: Input → Lens1 → Masks(Fourier) → Lens2 → Focus → Detector."""
    ax.set_xlim(-2, 85)
    ax.set_ylim(-14, 16)
    ax.set_aspect("equal")
    ax.set_title("FD2NN (Fourier-space D2NN)\nf=25mm, NA=0.508, 5 masks, Total=70mm",
                 fontsize=16, fontweight="bold")

    c_lens = "#3498db"
    c_mask = "#e74c3c"
    c_beam_fill = "#f39c1240"
    c_beam_edge = "#e67e22"

    c_focus = "#16a085"
    c_det = "#8e44ad"

    # Positions along optical axis
    input_x = 0
    lens1_x = 5
    fourier_start = lens1_x + 25  # f=25mm
    mask_spacing = 2.5  # 5mm shown as 2.5 for visual
    fourier_end = fourier_start + 4 * mask_spacing  # last mask
    lens2_x = fourier_end + 25    # f=25mm
    output_x = lens2_x + 3
    focus_x = output_x + 5        # focusing lens
    det_x = focus_x + 5           # detector

    # Input beam half-height (2mm aperture)
    bh = 5

    # ── Continuous beam path ──
    beam_x = [input_x, lens1_x, fourier_start, fourier_end, lens2_x, output_x, focus_x, det_x]
    beam_top = [bh, bh, 0.4, 0.4, bh, bh, bh, 0.3]
    beam_bot = [-bh, -bh, -0.4, -0.4, -bh, -bh, -bh, -0.3]
    ax.fill_between(beam_x, beam_top, beam_bot, color=c_beam_fill, edgecolor=c_beam_edge, lw=1)

    # ── Input label ──
    ax.annotate("Input\n2mm", xy=(input_x, -8), fontsize=17, ha="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#ecf0f1", edgecolor="gray"))

    # ── Lens 1 ──
    ax.plot([lens1_x, lens1_x], [-8, 8], color=c_lens, lw=3)
    ax.annotate("Lens1\nf=25mm", xy=(lens1_x, -11), fontsize=16, ha="center", color=c_lens)

    # ── 5 Masks in Fourier plane ──
    for i in range(5):
        mx = fourier_start + i * mask_spacing
        ax.add_patch(patches.Rectangle((mx - 0.3, -7), 0.6, 14,
                     facecolor=c_mask, alpha=0.25, edgecolor=c_mask, lw=1.5))
        ax.text(mx, 8, f"M{i}", fontsize=16, ha="center", color=c_mask)

    # Fourier plane label
    ax.annotate("Fourier Plane\n19.4mm, dx=18.9um", xy=(fourier_start + 2 * mask_spacing, -11),
                fontsize=16, ha="center", color="#9b59b6",
                bbox=dict(boxstyle="round", facecolor="#9b59b6", alpha=0.1))

    # ── Lens 2 ──
    ax.plot([lens2_x, lens2_x], [-8, 8], color=c_lens, lw=3)
    ax.annotate("Lens2\nf=25mm", xy=(lens2_x, -11), fontsize=16, ha="center", color=c_lens)

    # ── Focusing lens ──
    ax.plot([focus_x, focus_x], [-8, 8], color=c_focus, lw=3)
    ax.annotate("Focus\nf=4.5mm", xy=(focus_x, -11), fontsize=16, ha="center", color=c_focus)

    # ── Detector ──
    ax.add_patch(patches.Rectangle((det_x - 0.5, -4), 1.0, 8,
                 facecolor=c_det, alpha=0.4, edgecolor=c_det, lw=2))
    ax.annotate("APD/MMF", xy=(det_x, -8), fontsize=16, ha="center", color=c_det,
                fontweight="bold")

    # ── Dimensions ──
    ax.annotate("", xy=(input_x, 14), xytext=(det_x, 14),
                arrowprops=dict(arrowstyle="<->", color="gray"))
    ax.text((input_x + det_x) / 2, 15, "~80mm total", fontsize=18, ha="center", color="gray")

    ax.annotate("", xy=(lens1_x, 12), xytext=(fourier_start, 12),
                arrowprops=dict(arrowstyle="<->", color=c_lens))
    ax.text((lens1_x + fourier_start) / 2, 12.8, "25mm", fontsize=16, ha="center", color=c_lens)

    ax.annotate("", xy=(fourier_end, 12), xytext=(lens2_x, 12),
                arrowprops=dict(arrowstyle="<->", color=c_lens))
    ax.text((fourier_end + lens2_x) / 2, 12.8, "25mm", fontsize=16, ha="center", color=c_lens)

    # ── Warning ──
    ax.text(fourier_start + 2 * mask_spacing, 5,
            "Beam hits only\n0.015% of mask!",
            fontsize=18, ha="center", color="red", fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="red"))

    ax.set_xlabel("Optical axis [mm]", fontsize=17)
    ax.axhline(0, color="gray", lw=0.3, ls=":")
    ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=16)


def draw_d2nn(ax):
    """Draw D2NN: Input → Mask0 → ... → Mask4 → Detector. Same x-scale as FD2NN."""
    ax.set_xlim(-2, 85)
    ax.set_ylim(-14, 16)
    ax.set_aspect("equal")
    ax.set_title("D2NN (Free-space D2NN)\n5 masks, 10mm spacing, Angular spectrum, Total=60mm",
                 fontsize=16, fontweight="bold")

    c_mask = "#2ecc71"
    c_beam_fill = "#f39c1240"
    c_beam_edge = "#e67e22"
    c_focus = "#16a085"
    c_det = "#8e44ad"

    input_x = 0
    mask_spacing = 8  # 10mm shown as 8 for visual
    last_mask_x = 5 + 4 * mask_spacing
    d2nn_out_x = last_mask_x + 8   # 10mm propagation to D2NN output
    focus_x = d2nn_out_x + 5       # focusing lens
    det_x = focus_x + 5            # detector
    bh = 5

    # ── Continuous beam path ──
    xs = [input_x]
    tops = [bh]
    bots = [-bh]
    for i in range(5):
        mx = 5 + i * mask_spacing
        xs.extend([mx - 0.3, mx + 0.3])
        spread = 0.3 * i
        tops.extend([bh + spread, bh + spread])
        bots.extend([-bh - spread, -bh - spread])
    # D2NN output → focus lens → detector (converging)
    xs.extend([d2nn_out_x, focus_x, det_x])
    tops.extend([bh + 1.5, bh + 1.5, 0.3])
    bots.extend([-bh - 1.5, -bh - 1.5, -0.3])

    ax.fill_between(xs, tops, bots, color=c_beam_fill, edgecolor=c_beam_edge, lw=1)

    # ── Input label ──
    ax.annotate("Input\n2mm", xy=(input_x, -8), fontsize=17, ha="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#ecf0f1", edgecolor="gray"))

    # ── 5 Masks in real space ──
    for i in range(5):
        mx = 5 + i * mask_spacing
        ax.add_patch(patches.Rectangle((mx - 0.3, -7), 0.6, 14,
                     facecolor=c_mask, alpha=0.25, edgecolor=c_mask, lw=1.5))
        ax.text(mx, 8, f"M{i}", fontsize=16, ha="center", color=c_mask)

    # ── Focusing lens ──
    ax.plot([focus_x, focus_x], [-8, 8], color=c_focus, lw=3)
    ax.annotate("Focus\nf=4.5mm", xy=(focus_x, -11), fontsize=16, ha="center", color=c_focus)

    # ── Detector ──
    ax.add_patch(patches.Rectangle((det_x - 0.5, -4), 1.0, 8,
                 facecolor=c_det, alpha=0.4, edgecolor=c_det, lw=2))
    ax.annotate("APD/MMF", xy=(det_x, -8), fontsize=16, ha="center", color=c_det,
                fontweight="bold")

    # ── Labels ──
    ax.annotate("Real-space\n2.048mm, dx=2.0um", xy=(5 + 2 * mask_spacing, -11),
                fontsize=16, ha="center", color=c_mask,
                bbox=dict(boxstyle="round", facecolor=c_mask, alpha=0.1))

    # ── Dimensions ──
    ax.annotate("", xy=(5, 12), xytext=(5 + mask_spacing, 12),
                arrowprops=dict(arrowstyle="<->", color="gray"))
    ax.text(5 + mask_spacing / 2, 12.8, "10mm", fontsize=16, ha="center", color="gray")

    ax.annotate("", xy=(input_x, 14), xytext=(det_x, 14),
                arrowprops=dict(arrowstyle="<->", color="gray"))
    ax.text((input_x + det_x) / 2, 15, "~60mm total", fontsize=18, ha="center", color="gray")

    # ── Advantage ──
    ax.text(5 + 2 * mask_spacing, 5,
            "Beam covers 100%\nof mask area",
            fontsize=18, ha="center", color="green", fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="lightgreen", edgecolor="green"))

    # Diffraction note
    ax.text(5 + 0.5 * mask_spacing, -7,
            "Spread: 124um = 62px", fontsize=16, ha="center", color="gray", style="italic")

    ax.set_xlabel("Optical axis [mm]", fontsize=17)
    ax.axhline(0, color="gray", lw=0.3, ls=":")
    ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=16)


def draw_table(ax):
    ax.axis("off")
    data = [
        ["", "FD2NN (Fourier)", "D2NN (Free-space)"],
        ["Propagation", "2f Lens Fourier transform", "Angular spectrum diffraction"],
        ["Lens", "AC254-025-C x2", "None"],
        ["Mask location", "Fourier plane (19.4mm)", "Real-space (2.0mm)"],
        ["Mask pixel", "18.92 um", "2.00 um"],
        ["Beam utilization", "0.015%", "100%"],
        ["Layer spacing", "5 mm", "10 mm"],
        ["Diffraction spread", "4.7 px", "62 px"],
        ["Total length", "70 mm", "50 mm"],
        ["Parameters", "5.2M (800 used)", "5.2M (all used)"],
    ]
    table = ax.table(cellText=data, loc="upper center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(20)
    table.scale(1, 2.5)
    for j in range(3):
        table[0, j].set_facecolor("#34495e")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(data)):
        table[i, 0].set_facecolor("#ecf0f1")
        table[i, 0].set_text_props(fontweight="bold")
        table[i, 1].set_facecolor("#ebf5fb")
        table[i, 2].set_facecolor("#eafaf1")
    table[5, 1].set_facecolor("#f5b7b1")
    table[5, 2].set_facecolor("#abebc6")
    ax.set_title("Parameter Comparison Summary", fontsize=20, fontweight="bold", pad=10)


def main():
    fig = plt.figure(figsize=(28, 36))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.7], hspace=0.15)
    draw_fdnn(fig.add_subplot(gs[0]))
    draw_d2nn(fig.add_subplot(gs[1]))
    draw_table(fig.add_subplot(gs[2]))
    fig.suptitle("FD2NN vs D2NN -- Optical Layout Comparison", fontsize=22, fontweight="bold", y=0.98)
    out = "/root/dj/D2NN/kim2026/autoresearch/runs/single_case_viz/architecture_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
