#!/usr/bin/env python
"""Full D2NN pipeline diagram: TX → Turbulence → Telescope → Lanczos BR → D2NN → Focal Lens → PIB.

Generates a comprehensive optical architecture diagram with physical dimensions
at each stage, showing the complete data generation + inference pipeline.

Usage:
    cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python scripts/draw_full_pipeline_diagram.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "autoresearch" / "runs" / "0401-datagen-dn100um-lanczos50"
OUT.mkdir(parents=True, exist_ok=True)

# ─── Colors ───────────────────────────────────────────────────
C_TX      = "#f39c12"   # orange
C_ATM     = "#e74c3c"   # red
C_TEL     = "#3498db"   # blue
C_BR      = "#16a085"   # teal
C_D2NN    = "#2ecc71"   # green
C_LENS    = "#9b59b6"   # purple
C_DET     = "#e67e22"   # dark orange
C_BG      = "#fafafa"
C_BEAM    = "#f1c40f"   # yellow beam
C_BEAM_T  = "#e74c3c88" # turbulent beam (semi-transparent red)
C_GRID    = "#ecf0f1"


def draw_full_pipeline(ax):
    """Draw the complete optical pipeline."""
    ax.set_xlim(-1, 32)
    ax.set_ylim(-6, 7)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Stage positions (x) ──
    x_tx = 0
    x_atm_start = 2
    x_atm_end = 7
    x_tel = 8.5
    x_crop = 11
    x_br = 13.5
    x_d2nn_start = 16.5
    x_d2nn_end = 21.5
    x_lens = 23.5
    x_det = 26

    # ══════════════════════════════════════════════════════════
    # 1. TX (Gaussian Source)
    # ══════════════════════════════════════════════════════════
    circle_tx = plt.Circle((x_tx, 0), 0.5, color=C_TX, ec="black", lw=1.5, zorder=5)
    ax.add_patch(circle_tx)
    ax.text(x_tx, 0, "TX", ha="center", va="center", fontsize=11, fontweight="bold", color="white", zorder=6)
    ax.text(x_tx, -1.3, "λ=1.55μm\nw₀=3.3mm\nθ=0.3mrad", ha="center", va="top", fontsize=8, color="#555")

    # ══════════════════════════════════════════════════════════
    # 2. Atmosphere (1km, turbulence)
    # ══════════════════════════════════════════════════════════
    atm_box = FancyBboxPatch((x_atm_start, -2.5), x_atm_end - x_atm_start, 5,
                              boxstyle="round,pad=0.2", facecolor="#e74c3c22",
                              edgecolor=C_ATM, linestyle="--", lw=2, zorder=2)
    ax.add_patch(atm_box)

    # Phase screens (wavy lines)
    n_scr_draw = 6
    for i in range(n_scr_draw):
        xs = x_atm_start + 0.4 + i * (x_atm_end - x_atm_start - 0.8) / (n_scr_draw - 1)
        ys = np.linspace(-1.8, 1.8, 40)
        xs_wave = xs + 0.12 * np.sin(ys * 3.5 + i * 1.3)
        ax.plot(xs_wave, ys, color=C_ATM, alpha=0.4, lw=1.2, zorder=3)

    ax.text((x_atm_start + x_atm_end) / 2, 3.2, "Atmosphere\n(1 km)", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=C_ATM)
    ax.text((x_atm_start + x_atm_end) / 2, -3.2, "Cn2=5e-14\n39 phase screens\nD/r0=5.0",
            ha="center", va="top", fontsize=8, color="#555")

    # Beam through atmosphere (diverging, getting distorted)
    beam_y_tx = 0.3
    beam_y_atm = 1.5
    # Upper edge
    ax.fill_between(
        [x_tx + 0.5, x_atm_start, x_atm_end],
        [beam_y_tx, beam_y_tx + 0.2, beam_y_atm],
        [0, 0, 0],
        color=C_BEAM, alpha=0.15, zorder=1)
    # Lower edge
    ax.fill_between(
        [x_tx + 0.5, x_atm_start, x_atm_end],
        [-beam_y_tx, -beam_y_tx - 0.2, -beam_y_atm],
        [0, 0, 0],
        color=C_BEAM, alpha=0.15, zorder=1)

    # ══════════════════════════════════════════════════════════
    # 3. Telescope Aperture
    # ══════════════════════════════════════════════════════════
    tel_hw = 2.0
    ax.plot([x_tel, x_tel], [-tel_hw, -0.3], color=C_TEL, lw=4, zorder=5, solid_capstyle="round")
    ax.plot([x_tel, x_tel], [0.3, tel_hw], color=C_TEL, lw=4, zorder=5, solid_capstyle="round")
    # Arrow heads pointing inward (aperture stop)
    ax.annotate("", xy=(x_tel, 0.4), xytext=(x_tel, tel_hw),
                arrowprops=dict(arrowstyle="->", color=C_TEL, lw=2))
    ax.annotate("", xy=(x_tel, -0.4), xytext=(x_tel, -tel_hw),
                arrowprops=dict(arrowstyle="->", color=C_TEL, lw=2))
    ax.text(x_tel, 3.2, "Telescope\nAperture", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=C_TEL)
    ax.text(x_tel, -3.2, "D=150mm\n(circular mask)", ha="center", va="top", fontsize=8, color="#555")

    # ══════════════════════════════════════════════════════════
    # 4. Crop (simulation step)
    # ══════════════════════════════════════════════════════════
    crop_box = FancyBboxPatch((x_crop - 0.6, -1.5), 1.2, 3,
                               boxstyle="round,pad=0.15", facecolor="#ecf0f133",
                               edgecolor="#999", linestyle=":", lw=1.5, zorder=4)
    ax.add_patch(crop_box)
    ax.text(x_crop, 0, "Crop\n1536²", ha="center", va="center", fontsize=9, color="#666", zorder=5)
    ax.text(x_crop, -2.3, "dx=100μm\n153.6mm 창", ha="center", va="top", fontsize=7, color="#888")

    # ══════════════════════════════════════════════════════════
    # 5. Beam Reducer (Lanczos)
    # ══════════════════════════════════════════════════════════
    br_box = FancyBboxPatch((x_br - 0.8, -1.8), 1.6, 3.6,
                             boxstyle="round,pad=0.2", facecolor="#16a08522",
                             edgecolor=C_BR, lw=2, zorder=4)
    ax.add_patch(br_box)
    ax.text(x_br, 0.3, "Lanczos\n50:1", ha="center", va="center",
            fontsize=11, fontweight="bold", color=C_BR, zorder=5)
    ax.text(x_br, -0.6, "sinc filter", ha="center", va="center", fontsize=8, color=C_BR, zorder=5)
    ax.text(x_br, 3.2, "Beam\nReducer", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=C_BR)
    ax.text(x_br, -2.8, "1536x1536 -> 1024x1024\n100um -> 2um\n+ defocus correction",
            ha="center", va="top", fontsize=8, color="#555")

    # Beam narrowing through BR
    ax.fill([x_br - 0.7, x_br + 0.7, x_br + 0.7, x_br - 0.7],
            [1.5, 0.7, -0.7, -1.5],
            color=C_BEAM, alpha=0.2, zorder=3)

    # ══════════════════════════════════════════════════════════
    # 6. D2NN (5 phase masks)
    # ══════════════════════════════════════════════════════════
    d2nn_box = FancyBboxPatch((x_d2nn_start - 0.3, -2.2), x_d2nn_end - x_d2nn_start + 0.6, 4.4,
                               boxstyle="round,pad=0.2", facecolor="#2ecc7111",
                               edgecolor=C_D2NN, lw=2, zorder=2)
    ax.add_patch(d2nn_box)

    n_masks = 5
    mask_spacing = (x_d2nn_end - x_d2nn_start) / (n_masks - 1)
    for i in range(n_masks):
        xm = x_d2nn_start + i * mask_spacing
        ax.plot([xm, xm], [-1.2, 1.2], color=C_D2NN, lw=3, zorder=5, solid_capstyle="round")
        ax.text(xm, -1.5, f"M{i+1}", ha="center", va="top", fontsize=7, color=C_D2NN)

    ax.text((x_d2nn_start + x_d2nn_end) / 2, 3.2, "D2NN\n(trainable)", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=C_D2NN)
    ax.text((x_d2nn_start + x_d2nn_end) / 2, -3.2,
            "5 phase masks\nspacing 10mm\n1024x1024, dx=2um\nwindow=2.048mm",
            ha="center", va="top", fontsize=8, color="#555")

    # Beam through D2NN
    beam_d2nn_y = 0.7
    ax.fill_between(
        [x_d2nn_start, x_d2nn_end],
        [beam_d2nn_y, beam_d2nn_y],
        [-beam_d2nn_y, -beam_d2nn_y],
        color=C_BEAM, alpha=0.12, zorder=1)

    # ══════════════════════════════════════════════════════════
    # 7. Focal Lens
    # ══════════════════════════════════════════════════════════
    lens_hw = 1.3
    # Draw lens shape (biconvex)
    theta = np.linspace(-np.pi/2, np.pi/2, 50)
    lens_r = 0.3
    ax.plot(x_lens + lens_r * np.cos(theta), lens_hw * np.sin(theta) / (np.pi/2) * np.ones_like(theta) * 0 + np.linspace(-lens_hw, lens_hw, 50),
            color=C_LENS, lw=3, zorder=5)
    # Simpler: just draw a vertical line with curves
    ax.plot([x_lens, x_lens], [-lens_hw, lens_hw], color=C_LENS, lw=4, zorder=5, solid_capstyle="round")
    # Small lens markers
    ax.plot([x_lens - 0.2, x_lens, x_lens + 0.2], [lens_hw, lens_hw + 0.2, lens_hw], color=C_LENS, lw=2, zorder=5)
    ax.plot([x_lens - 0.2, x_lens, x_lens + 0.2], [-lens_hw, -lens_hw - 0.2, -lens_hw], color=C_LENS, lw=2, zorder=5)

    ax.text(x_lens, 3.2, "Focusing\nLens", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=C_LENS)
    ax.text(x_lens, -2.3, "f = 4.5mm", ha="center", va="top", fontsize=9, fontweight="bold", color=C_LENS)

    # Converging beam after lens
    ax.fill([x_lens + 0.3, x_det, x_det, x_lens + 0.3],
            [0.7, 0.08, -0.08, -0.7],
            color=C_BEAM, alpha=0.25, zorder=1)

    # ══════════════════════════════════════════════════════════
    # 8. Detector / PIB measurement
    # ══════════════════════════════════════════════════════════
    det_box = FancyBboxPatch((x_det - 0.4, -0.8), 0.8, 1.6,
                              boxstyle="round,pad=0.1", facecolor=C_DET,
                              edgecolor="black", lw=1.5, zorder=5)
    ax.add_patch(det_box)
    ax.text(x_det, 0, "PIB", ha="center", va="center", fontsize=10, fontweight="bold", color="white", zorder=6)
    ax.text(x_det, 3.2, "Focal\nPlane", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=C_DET)
    ax.text(x_det, -1.5, "PIB@10um\ndx=3.4um", ha="center", va="top", fontsize=8, color="#555")

    # ══════════════════════════════════════════════════════════
    # Arrows connecting stages
    # ══════════════════════════════════════════════════════════
    arrow_kw = dict(arrowstyle="-|>", color="#555", lw=1.5, mutation_scale=15)
    for (x1, x2) in [(x_tx + 0.5, x_atm_start), (x_atm_end, x_tel - 0.15),
                      (x_tel + 0.15, x_crop - 0.6), (x_crop + 0.6, x_br - 0.8),
                      (x_br + 0.8, x_d2nn_start - 0.3),
                      (x_d2nn_end + 0.3, x_lens - 0.3), (x_lens + 0.3, x_det - 0.4)]:
        ax.annotate("", xy=(x2, 0), xytext=(x1, 0), arrowprops=arrow_kw)

    # ══════════════════════════════════════════════════════════
    # Scale annotations
    # ══════════════════════════════════════════════════════════
    y_scale = -5.0
    # Atmosphere distance
    ax.annotate("", xy=(x_atm_end, y_scale), xytext=(x_atm_start, y_scale),
                arrowprops=dict(arrowstyle="<->", color="#888", lw=1))
    ax.text((x_atm_start + x_atm_end)/2, y_scale - 0.3, "1 km", ha="center", fontsize=9, color="#888")

    # D2NN length
    ax.annotate("", xy=(x_d2nn_end, y_scale), xytext=(x_d2nn_start, y_scale),
                arrowprops=dict(arrowstyle="<->", color="#888", lw=1))
    ax.text((x_d2nn_start + x_d2nn_end)/2, y_scale - 0.3, "40mm", ha="center", fontsize=9, color="#888")

    # Lens to detector
    ax.annotate("", xy=(x_det, y_scale), xytext=(x_lens, y_scale),
                arrowprops=dict(arrowstyle="<->", color="#888", lw=1))
    ax.text((x_lens + x_det)/2, y_scale - 0.3, "4.5mm", ha="center", fontsize=9, color="#888")

    # Title
    ax.text(13, 5.8, "D2NN Full Pipeline: TX -> Atmosphere -> Telescope -> Beam Reducer -> D2NN -> Focal Plane",
            ha="center", va="bottom", fontsize=14, fontweight="bold", color="#2c3e50")


def draw_size_comparison(ax):
    """Draw beam size at each stage."""
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 4.5)
    ax.axis("off")

    stages = [
        ("Atmos\nOutput", "D~300mm\n(diverging)", 3.0, C_ATM),
        ("After\nTelescope", "D=150mm\n4096x4096\n100um", 1.8, C_TEL),
        ("After\nCrop", "153.6mm\n1536x1536\n100um", 1.5, C_BR),
        ("Lanczos\n50:1", "2.048mm\n1024x1024\n2um", 0.3, C_D2NN),
        ("D2NN\nOutput", "2.048mm\n1024x1024\n2um", 0.3, C_D2NN),
        ("Focal\nPlane", "~7um spot\ndx=3.4um", 0.08, C_DET),
    ]

    for i, (label, desc, size, color) in enumerate(stages):
        x = i * 1.1 + 0.3
        circle = plt.Circle((x, 2.0), size / 2 * 1.0, color=color, alpha=0.3, ec=color, lw=1.5)
        ax.add_patch(circle)
        ax.text(x, 3.8, label, ha="center", va="center", fontsize=9, fontweight="bold", color=color)
        ax.text(x, 0.2, desc, ha="center", va="top", fontsize=7, color="#555")
        if i < len(stages) - 1:
            ax.annotate("", xy=((i + 1) * 1.1 + 0.1, 2.0), xytext=(x + 0.3, 2.0),
                        arrowprops=dict(arrowstyle="->", color="#aaa", lw=1))

    ax.text(3.2, 4.3, "Beam Size at Each Stage", ha="center", fontsize=12, fontweight="bold", color="#2c3e50")


def draw_param_table(ax):
    """Parameter table."""
    ax.axis("off")

    data = [
        ["Parameter", "Value", "Note"],
        ["Wavelength", "1.55 um", "IR telecom"],
        ["Path length", "1 km", "FSO link"],
        ["Cn2", "5e-14 m^-2/3", "Strong turb"],
        ["r0 (Fried)", "3.0 cm", "D/r0 = 5.0"],
        ["Telescope", "D=150 mm", "Circular aperture"],
        ["Prop grid (N)", "4096", "delta_n=100um"],
        ["Phase screens", "39", "Kolmogorov+SH"],
        ["Crop size", "1536x1536", "153.6mm window"],
        ["Lanczos ratio", "50:1", "100um -> 2um"],
        ["D2NN grid", "1024x1024, dx=2um", "window=2.048mm"],
        ["D2NN layers", "5 masks, 10mm gap", "phase-only"],
        ["Focal lens", "f = 4.5 mm", "For PIB measurement"],
        ["Focal pitch", "3.4 um/pixel", "wvl*f/(N*dx)"],
        ["PIB bucket", "r = 10 um", "Target: >80%"],
        ["Dataset", "5000 (4000/500/500)", "train/val/test"],
        ["Gen time", "~3.2 s/realization", "A100 40GB"],
    ]

    table = ax.table(
        cellText=data[1:],
        colLabels=data[0],
        cellLoc="center",
        loc="center",
        colColours=["#34495e"] * 3,
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Style header
    for j in range(3):
        cell = table[0, j]
        cell.set_text_props(color="white", fontweight="bold", fontsize=10)
        cell.set_facecolor("#34495e")

    # Style data rows
    for i in range(1, len(data)):
        for j in range(3):
            cell = table[i, j]
            cell.set_facecolor("#f8f9fa" if i % 2 == 0 else "white")
            cell.set_edgecolor("#dee2e6")
            if j == 0:
                cell.set_text_props(fontweight="bold")

    ax.set_title("System Parameters", fontsize=13, fontweight="bold", color="#2c3e50", pad=10)


def main():
    fig = plt.figure(figsize=(28, 22))

    # Top: Full pipeline diagram (takes most space)
    ax1 = fig.add_axes([0.02, 0.45, 0.96, 0.50])
    draw_full_pipeline(ax1)

    # Bottom left: Size comparison
    ax2 = fig.add_axes([0.02, 0.02, 0.45, 0.38])
    draw_size_comparison(ax2)

    # Bottom right: Parameter table
    ax3 = fig.add_axes([0.50, 0.02, 0.48, 0.38])
    draw_param_table(ax3)

    fig.patch.set_facecolor("white")

    out_path = OUT / "phase1_architecture.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
