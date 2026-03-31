#!/usr/bin/env python
"""Phase 1: D2NN Architecture for Strong Turbulence Experiment.

Single connected diagram: TX → 1km atmosphere → Telescope → Beam Reducer → D2NN → Focus → Detector
+ Parameter comparison table (Weak vs Strong turbulence)

Same style as visualize_architecture_comparison.py.

Usage:
    cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python scripts/visualize_strong_turb_architecture.py
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def draw_full_system(ax):
    """TX → 1km → Telescope → BR → D2NN (detailed, 5 masks) → Focus → Detector."""
    ax.set_xlim(-5, 145)
    ax.set_ylim(-16, 20)
    ax.set_aspect("equal")
    ax.set_title(
        "D2NN Beam Cleanup System (Strong Turbulence)\n"
        "TX 1.55um → 1km (Cn²=5e-14, D/r₀=5.0) → 15cm Telescope → 75:1 BR → D2NN 5-layer → Focus → Detector",
        fontsize=16, fontweight="bold", pad=12,
    )

    # Colors
    c_tx = "#2c3e50"
    c_atm = "#e74c3c"
    c_tel = "#3498db"
    c_br = "#16a085"
    c_mask = "#2ecc71"
    c_beam = "#f39c1240"
    c_beam_e = "#e67e22"
    c_focus = "#9b59b6"
    c_det = "#8e44ad"

    # ═══ Positions ═══
    tx_x = 0
    atm_start = 6
    atm_end = 28
    tel_x = 33
    br_start = 40
    br_end = 48
    # D2NN: 5 masks at 10mm spacing (visual: 7 units apart)
    d2nn_input_x = 55
    mask_sp = 7
    mask_xs = [d2nn_input_x + i * mask_sp for i in range(5)]
    d2nn_out_x = mask_xs[-1] + mask_sp  # after last mask
    focus_x = d2nn_out_x + 7
    det_x = focus_x + 7

    bh_tx = 2       # TX beam half-height
    bh_atm = 8      # beam at 1km (diverged, >> telescope)
    bh_tel = 6      # telescope aperture half-height
    bh_br_out = 4   # after beam reducer
    bh_d2nn = 4     # D2NN input beam

    # ═══ Beam path (continuous) ═══
    # Segment 1: TX → atmosphere (diverging)
    seg1_x = [tx_x + 2, atm_start, (atm_start + atm_end) / 2, atm_end, tel_x - 1]
    seg1_top = [bh_tx, bh_tx + 1, bh_atm * 0.7, bh_atm, bh_atm]
    seg1_bot = [-bh_tx, -bh_tx - 1, -bh_atm * 0.7, -bh_atm, -bh_atm]
    ax.fill_between(seg1_x, seg1_top, seg1_bot, color=c_beam, edgecolor=c_beam_e, lw=0.8)

    # Segment 2: Telescope truncates → beam reducer
    seg2_x = [tel_x + 1, br_start, br_end]
    seg2_top = [bh_tel, bh_tel * 0.8, bh_br_out * 0.6]
    seg2_bot = [-bh_tel, -bh_tel * 0.8, -bh_br_out * 0.6]
    ax.fill_between(seg2_x, seg2_top, seg2_bot, color=c_beam, edgecolor=c_beam_e, lw=0.8)

    # Segment 3: BR → D2NN masks → out → focus → detector
    seg3_x = [br_end + 1, d2nn_input_x]
    seg3_top = [bh_br_out * 0.5, bh_d2nn]
    seg3_bot = [-bh_br_out * 0.5, -bh_d2nn]
    ax.fill_between(seg3_x, seg3_top, seg3_bot, color=c_beam, edgecolor=c_beam_e, lw=0.8)

    # Through D2NN masks (slight diffraction spread)
    d2nn_xs = [d2nn_input_x]
    d2nn_tops = [bh_d2nn]
    d2nn_bots = [-bh_d2nn]
    for i, mx in enumerate(mask_xs):
        spread = 0.15 * i
        d2nn_xs.extend([mx - 0.3, mx + 0.3])
        d2nn_tops.extend([bh_d2nn + spread, bh_d2nn + spread])
        d2nn_bots.extend([-bh_d2nn - spread, -bh_d2nn - spread])
    d2nn_xs.extend([d2nn_out_x, focus_x, det_x])
    d2nn_tops.extend([bh_d2nn + 0.8, bh_d2nn + 0.8, 0.3])
    d2nn_bots.extend([-bh_d2nn - 0.8, -bh_d2nn - 0.8, -0.3])
    ax.fill_between(d2nn_xs, d2nn_tops, d2nn_bots, color=c_beam, edgecolor=c_beam_e, lw=0.8)

    # ═══ TX ═══
    ax.add_patch(patches.FancyBboxPatch((tx_x - 2, -3.5), 4, 7,
                 boxstyle="round,pad=0.4", facecolor=c_tx, alpha=0.15, edgecolor=c_tx, lw=2))
    ax.text(tx_x, 0, "TX", fontsize=16, ha="center", va="center", color=c_tx, fontweight="bold")
    ax.text(tx_x, -6, "1.55um\n0.3mrad", fontsize=13, ha="center", color=c_tx)

    # ═══ Atmosphere ═══
    x_wave = np.linspace(atm_start, atm_end, 150)
    for offset in np.linspace(-bh_atm + 1, bh_atm - 1, 8):
        y_wave = offset + 0.6 * np.sin(x_wave * 0.7 + offset * 0.5)
        ax.plot(x_wave, y_wave, color=c_atm, alpha=0.12, lw=1.5)
    ax.add_patch(patches.FancyBboxPatch((atm_start - 0.5, -bh_atm - 1), atm_end - atm_start + 1, 2 * bh_atm + 2,
                 boxstyle="round,pad=0.3", facecolor="none", edgecolor=c_atm, lw=1.5, ls="--"))
    ax.text((atm_start + atm_end) / 2, -12,
            "1km Atmosphere\nCn²=5e-14",
            fontsize=14, ha="center", color=c_atm, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#f5b7b1", alpha=0.3, edgecolor=c_atm))

    # ═══ Telescope ═══
    ax.plot([tel_x, tel_x], [-bh_tel - 1, bh_tel + 1], color=c_tel, lw=4)
    # Lens curves
    ax.plot([tel_x - 1, tel_x], [-bh_tel - 1, -bh_tel + 0.5], color=c_tel, lw=2)
    ax.plot([tel_x - 1, tel_x], [bh_tel + 1, bh_tel - 0.5], color=c_tel, lw=2)
    ax.text(tel_x, -12, "Telescope\n15cm", fontsize=14, ha="center", color=c_tel, fontweight="bold")

    # Truncation indicator
    ax.annotate("truncates\nbeam", xy=(tel_x + 1, bh_tel + 2), fontsize=11, color=c_tel,
                ha="center", style="italic")

    # ═══ Beam Reducer ═══
    ax.add_patch(patches.FancyBboxPatch((br_start, -5), br_end - br_start, 10,
                 boxstyle="round,pad=0.3", facecolor=c_br, alpha=0.12, edgecolor=c_br, lw=2))
    ax.text((br_start + br_end) / 2, 0, "75:1\nBR", fontsize=15, ha="center", va="center",
            color=c_br, fontweight="bold")
    ax.text((br_start + br_end) / 2, -9, "153.6mm\n→ 2.048mm", fontsize=12, ha="center", color=c_br)

    # ═══ D2NN Masks ═══
    for i, mx in enumerate(mask_xs):
        ax.add_patch(patches.Rectangle((mx - 0.4, -bh_d2nn - 1.5), 0.8, 2 * bh_d2nn + 3,
                     facecolor=c_mask, alpha=0.2, edgecolor=c_mask, lw=2))
        ax.text(mx, bh_d2nn + 3, f"M{i}", fontsize=15, ha="center", color=c_mask, fontweight="bold")

    # D2NN label
    ax.text((mask_xs[0] + mask_xs[-1]) / 2, -10,
            "D2NN (5 layers)\nAngular spectrum\n2.048mm, dx=2.0um",
            fontsize=13, ha="center", color=c_mask, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=c_mask, alpha=0.08, edgecolor=c_mask))

    # Beam covers 100% note
    ax.text((mask_xs[1] + mask_xs[2]) / 2, bh_d2nn - 1,
            "Beam covers\n100% of mask",
            fontsize=14, ha="center", color="green", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.7, edgecolor="green"))

    # ═══ Focus Lens ═══
    ax.plot([focus_x, focus_x], [-bh_d2nn - 2, bh_d2nn + 2], color=c_focus, lw=3)
    ax.text(focus_x, -10, "Focus\nf=4.5mm", fontsize=14, ha="center", color=c_focus, fontweight="bold")

    # ═══ Detector ═══
    ax.add_patch(patches.Rectangle((det_x - 0.6, -3), 1.2, 6,
                 facecolor=c_det, alpha=0.4, edgecolor=c_det, lw=2))
    ax.text(det_x, -7, "APD/MMF", fontsize=14, ha="center", color=c_det, fontweight="bold")

    # ═══ Dimension arrows ═══
    # 1km
    ax.annotate("", xy=(tx_x, 17), xytext=(tel_x, 17),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1.5))
    ax.text((tx_x + tel_x) / 2, 18, "1 km", fontsize=17, ha="center", color="gray", fontweight="bold")

    # D2NN internal spacing
    ax.annotate("", xy=(mask_xs[0], bh_d2nn + 5.5), xytext=(mask_xs[1], bh_d2nn + 5.5),
                arrowprops=dict(arrowstyle="<->", color="gray"))
    ax.text((mask_xs[0] + mask_xs[1]) / 2, bh_d2nn + 6.3, "10mm", fontsize=14, ha="center", color="gray")

    # Total D2NN length
    ax.annotate("", xy=(d2nn_input_x, 15), xytext=(det_x, 15),
                arrowprops=dict(arrowstyle="<->", color=c_mask))
    ax.text((d2nn_input_x + det_x) / 2, 16, "~60mm total", fontsize=15, ha="center", color=c_mask)

    # D/r₀ annotation
    ax.text((atm_start + atm_end) / 2, bh_atm + 3,
            "D/r₀ = 5.0",
            fontsize=18, ha="center", color=c_atm, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor=c_atm, lw=1.5))

    ax.set_xlabel("Optical axis (not to scale)", fontsize=14)
    ax.axhline(0, color="gray", lw=0.3, ls=":")
    ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=13)


def draw_table(ax):
    """D2NN single-architecture parameter table."""
    ax.axis("off")
    data = [
        ["Parameter", "Value"],
        ["Architecture", "D2NN (Free-space, Angular Spectrum)"],
        ["Cn²", "5.0e-14"],
        ["D/r₀", "5.02"],
        ["r₀ (plane wave)", "2.99 cm"],
        ["Baseline CO", "0.499"],
        ["WF RMS (baseline)", "474 nm"],
        ["Wavelength", "1.55 um"],
        ["Telescope", "15cm aperture"],
        ["Beam Reducer", "75:1 (153.6mm → 2.048mm)"],
        ["Mask domain", "Real-space (2.048mm window)"],
        ["Grid", "1024 x 1024"],
        ["Mask pixel (dx)", "2.0 um"],
        ["Layers", "5 (10mm spacing)"],
        ["Detector distance", "10 mm"],
        ["Focus lens", "f = 4.5 mm"],
        ["Beam utilization", "100%"],
        ["Diffraction spread", "124um = 62px per layer"],
        ["Train / Val / Test", "400 / 50 / 50"],
        ["Epochs", "100"],
        ["Learning rate", "5e-4 (Adam)"],
        ["Batch size", "2"],
        ["Parameters", "5.2M (all used)"],
    ]
    table = ax.table(cellText=data, loc="upper center", cellLoc="center",
                     colWidths=[0.35, 0.55])
    table.auto_set_font_size(False)
    table.set_fontsize(18)
    table.scale(1, 2.0)

    # Header
    for j in range(2):
        table[0, j].set_facecolor("#34495e")
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=18)
    # Row styling
    for i in range(1, len(data)):
        table[i, 0].set_facecolor("#ecf0f1")
        table[i, 0].set_text_props(fontweight="bold")
        table[i, 1].set_facecolor("#eafaf1")

    # Highlight key values
    table[3, 1].set_facecolor("#abebc6")   # D/r₀ = 5.02 (good range)
    table[5, 1].set_facecolor("#fdebd0")   # baseline CO (room for improvement)
    table[17, 1].set_facecolor("#abebc6")  # 100% beam utilization

    ax.set_title("D2NN Experiment Parameters",
                 fontsize=20, fontweight="bold", pad=5)


def main():
    fig = plt.figure(figsize=(32, 30))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.9], hspace=0.08)

    draw_full_system(fig.add_subplot(gs[0]))
    draw_table(fig.add_subplot(gs[1]))

    fig.suptitle("D2NN Strong Turbulence Experiment -- Optical System & Parameters",
                 fontsize=24, fontweight="bold", y=0.98)

    out_dir = "/root/dj/D2NN/kim2026/autoresearch/runs/d2nn_strong_turb_sweep"
    import os
    os.makedirs(out_dir, exist_ok=True)
    out = f"{out_dir}/phase1_architecture.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
