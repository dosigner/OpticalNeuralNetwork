#!/usr/bin/env python3
"""Generate 5 high-quality matplotlib diagrams for the F-D2NN report."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUTDIR = os.path.dirname(os.path.abspath(__file__))
DPI = 300
FONT_FAMILY = 'DejaVu Sans'

# Color palette
C_BLUE = '#DBEAFE'
C_BLUE_EDGE = '#60A5FA'
C_PURPLE = '#E9D5FF'
C_PURPLE_EDGE = '#A78BFA'
C_ORANGE = '#FED7AA'
C_ORANGE_EDGE = '#FB923C'
C_GREEN = '#D1FAE5'
C_GREEN_EDGE = '#34D399'
C_GRAY = '#F3F4F6'
C_GRAY_EDGE = '#9CA3AF'


def _setup_ax(ax):
    """Remove all axes decorations."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_aspect('auto')


def _box(ax, cx, cy, w, h, text, fc, ec, fontsize=11, bold=False, text_color='#1F2937'):
    """Draw a rounded rectangle with centered text."""
    x0 = cx - w / 2
    y0 = cy - h / 2
    box = FancyBboxPatch((x0, y0), w, h,
                         boxstyle="round,pad=0.015",
                         facecolor=fc, edgecolor=ec, linewidth=1.5,
                         transform=ax.transAxes, zorder=2)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(cx, cy, text, transform=ax.transAxes, ha='center', va='center',
            fontsize=fontsize, fontfamily=FONT_FAMILY, fontweight=weight,
            color=text_color, zorder=3)
    return box


def _arrow_h(ax, x1, x2, y, color='#374151'):
    """Draw a horizontal arrow in axes coords."""
    arrow = FancyArrowPatch((x1, y), (x2, y),
                            arrowstyle='->', mutation_scale=15,
                            color=color, linewidth=1.5,
                            transform=ax.transAxes, zorder=1)
    ax.add_patch(arrow)


def _arrow_v(ax, x, y1, y2, color='#374151'):
    """Draw a vertical arrow in axes coords."""
    arrow = FancyArrowPatch((x, y1), (x, y2),
                            arrowstyle='->', mutation_scale=15,
                            color=color, linewidth=1.5,
                            transform=ax.transAxes, zorder=1)
    ax.add_patch(arrow)


def _label(ax, x, y, text, fontsize=9, color='#6B7280', ha='center', va='center',
           style='italic'):
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=fontsize, fontfamily=FONT_FAMILY, color=color,
            fontstyle=style, zorder=3)


# ──────────────────────────────────────────────────────────────────────
# Diagram 1: Dual 2f optical system
# ──────────────────────────────────────────────────────────────────────
def diagram_dual2f():
    fig, ax = plt.subplots(figsize=(12, 3))
    _setup_ax(ax)
    ax.set_xlim(0, 12)
    ax.set_ylim(-1.5, 1.8)

    # Positions along x
    positions = {
        'input': 0.8,
        'lens1': 3.2,
        'fourier': 5.6,
        'lens2': 8.0,
        'output': 10.4,
    }
    bw, bh = 1.6, 0.8  # box width / height for planes

    def draw_plane(x, label, fc, ec):
        rect = FancyBboxPatch((x - bw/2, -bh/2), bw, bh,
                              boxstyle="round,pad=0.12",
                              facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        ax.text(x, 0, label, ha='center', va='center', fontsize=11,
                fontfamily=FONT_FAMILY, fontweight='bold', color='#1F2937', zorder=3)

    def draw_lens(x, label):
        # Draw biconvex lens shape
        from matplotlib.patches import Arc
        lens_h = 1.1
        ax.plot([x, x], [-lens_h/2, lens_h/2], color='#D97706', linewidth=2.5, zorder=2)
        # Arrow tips
        ax.annotate('', xy=(x - 0.15, lens_h/2), xytext=(x + 0.15, lens_h/2),
                    arrowprops=dict(arrowstyle='<->', color='#D97706', lw=1.5))
        ax.annotate('', xy=(x - 0.15, -lens_h/2), xytext=(x + 0.15, -lens_h/2),
                    arrowprops=dict(arrowstyle='<->', color='#D97706', lw=1.5))
        # Ellipse for lens body
        from matplotlib.patches import Ellipse
        ell = Ellipse((x, 0), 0.35, lens_h, facecolor='#FDE68A', edgecolor='#D97706',
                      linewidth=1.5, alpha=0.8, zorder=2)
        ax.add_patch(ell)
        ax.text(x, -lens_h/2 - 0.35, label, ha='center', va='top', fontsize=10,
                fontfamily=FONT_FAMILY, fontweight='bold', color='#92400E', zorder=3)

    # Draw planes
    draw_plane(positions['input'], 'Input plane\n(real)', C_BLUE, C_BLUE_EDGE)
    draw_plane(positions['fourier'], 'Fourier plane\n(frequency)', C_GREEN, C_GREEN_EDGE)
    draw_plane(positions['output'], 'Output plane\n(real)', C_BLUE, C_BLUE_EDGE)

    # Draw lenses
    draw_lens(positions['lens1'], 'Lens\u2081')
    draw_lens(positions['lens2'], 'Lens\u2082')

    # Arrows (horizontal connecting lines)
    for (a, b) in [('input', 'lens1'), ('lens1', 'fourier'), ('fourier', 'lens2'), ('lens2', 'output')]:
        x1 = positions[a] + (bw/2 if a in ('input', 'fourier', 'output') else 0.22)
        x2 = positions[b] - (bw/2 if b in ('input', 'fourier', 'output') else 0.22)
        ax.annotate('', xy=(x2, 0), xytext=(x1, 0),
                    arrowprops=dict(arrowstyle='->', color='#374151', lw=1.5))

    # Distance labels
    def dist_label(x1, x2, label):
        xm = (x1 + x2) / 2
        ax.annotate('', xy=(x2, -1.0), xytext=(x1, -1.0),
                    arrowprops=dict(arrowstyle='<->', color='#6B7280', lw=1.0))
        ax.text(xm, -1.22, label, ha='center', va='top', fontsize=10,
                fontfamily=FONT_FAMILY, color='#6B7280', fontstyle='italic')

    dist_label(positions['input'], positions['lens1'], '$f_1$')
    dist_label(positions['lens1'], positions['fourier'], '$f_1$')
    dist_label(positions['fourier'], positions['lens2'], '$f_2$')
    dist_label(positions['lens2'], positions['output'], '$f_2$')

    # Total path label
    ax.annotate('', xy=(positions['output'], 1.4), xytext=(positions['input'], 1.4),
                arrowprops=dict(arrowstyle='<->', color='#374151', lw=1.2))
    ax.text((positions['input'] + positions['output'])/2, 1.55,
            'Total path:  $2f_1 + 2f_2$  (4f system)', ha='center', va='bottom',
            fontsize=10, fontfamily=FONT_FAMILY, color='#374151', fontweight='bold')

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'diagram_dual2f.png'), dpi=DPI,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  -> diagram_dual2f.png")


# ──────────────────────────────────────────────────────────────────────
# Diagram 2: F-D2NN full forward pass (vertical)
# ──────────────────────────────────────────────────────────────────────
def diagram_forward_pass():
    fig, ax = plt.subplots(figsize=(8, 14))
    _setup_ax(ax)

    bw = 0.60  # box width
    bh = 0.048  # box height
    cx = 0.50   # center x
    gap = 0.065  # vertical gap between box centers

    # Define stages: (label, fill_color, edge_color, domain_tag)
    stages = [
        ("Input  $U_0(x,y)$", C_BLUE, C_BLUE_EDGE, "Real domain"),
        ("Lens 2f Forward:  FFT + NA mask", C_ORANGE, C_ORANGE_EDGE, None),
        ("Phase Mask\u2081$(f_x, f_y)$  [learnable]", C_PURPLE, C_PURPLE_EDGE, "Fourier domain"),
        ("ASM propagation (100 \u00b5m)", C_GRAY, C_GRAY_EDGE, None),
        ("Phase Mask\u2082$(f_x, f_y)$  [learnable]", C_PURPLE, C_PURPLE_EDGE, None),
        ("ASM propagation (100 \u00b5m)", C_GRAY, C_GRAY_EDGE, None),
        ("Phase Mask\u2083$(f_x, f_y)$  [learnable]", C_PURPLE, C_PURPLE_EDGE, None),
        ("\u22ee  (repeat \u00d7 N layers)", C_GRAY, C_GRAY_EDGE, None),
        ("Phase Mask\u2099$(f_x, f_y)$  [learnable]", C_PURPLE, C_PURPLE_EDGE, None),
        ("[Optional]  SBN nonlinearity", C_ORANGE, C_ORANGE_EDGE, None),
        ("Lens 2f Inverse:  IFFT + NA mask", C_ORANGE, C_ORANGE_EDGE, None),
        ("Detector:  $I(x,y) = |U_{out}|^2$", C_GREEN, C_GREEN_EDGE, "Real domain"),
        ("Output:  classification / saliency", C_GREEN, C_GREEN_EDGE, None),
    ]

    n = len(stages)
    y_top = 0.95
    y_positions = [y_top - i * gap for i in range(n)]

    # Draw dashed group around repeating Fourier layers (stages 2-8, indices 2..8)
    group_top = y_positions[2] + bh/2 + 0.012
    group_bot = y_positions[8] - bh/2 - 0.012
    group_rect = FancyBboxPatch((cx - bw/2 - 0.06, group_bot),
                                bw + 0.12, group_top - group_bot,
                                boxstyle="round,pad=0.01",
                                facecolor='#F5F3FF', edgecolor='#A78BFA',
                                linewidth=1.2, linestyle='--',
                                transform=ax.transAxes, zorder=0)
    ax.add_patch(group_rect)
    ax.text(cx + bw/2 + 0.08, (group_top + group_bot)/2,
            'Fourier-domain\nlayer stack',
            transform=ax.transAxes, ha='left', va='center',
            fontsize=9, fontfamily=FONT_FAMILY, color='#7C3AED',
            fontstyle='italic', zorder=3)

    for i, (label, fc, ec, domain) in enumerate(stages):
        y = y_positions[i]
        _box(ax, cx, y, bw, bh, label, fc, ec, fontsize=10)

        # Arrow to next
        if i < n - 1:
            _arrow_v(ax, cx, y - bh/2 - 0.003, y_positions[i+1] + bh/2 + 0.003)

        # Domain tag
        if domain:
            ax.text(cx - bw/2 - 0.04, y, domain,
                    transform=ax.transAxes, ha='right', va='center',
                    fontsize=8, fontfamily=FONT_FAMILY, color='#6B7280',
                    fontstyle='italic', zorder=3,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='#D1D5DB',
                              lw=0.8, alpha=0.9))

    # Annotation: dx update
    ax.annotate('$dx_F = \\frac{\\lambda f}{N \\cdot dx}$',
                xy=(cx + bw/2 + 0.01, y_positions[1]),
                xytext=(cx + bw/2 + 0.12, y_positions[1] + 0.015),
                transform=ax.transAxes, fontsize=9,
                arrowprops=dict(arrowstyle='->', color='#9CA3AF', lw=1.0),
                color='#6B7280', fontfamily=FONT_FAMILY, va='center')

    fig.savefig(os.path.join(OUTDIR, 'diagram_forward_pass.png'), dpi=DPI,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  -> diagram_forward_pass.png")


# ──────────────────────────────────────────────────────────────────────
# Diagram 3: Real-space D2NN flow (horizontal)
# ──────────────────────────────────────────────────────────────────────
def diagram_real_d2nn_flow():
    fig, ax = plt.subplots(figsize=(14, 2.5))
    _setup_ax(ax)

    stages = [
        ("Input$(x,y)$", C_BLUE, C_BLUE_EDGE),
        ("Phase\u2081$(x,y)$", '#BFDBFE', C_BLUE_EDGE),
        ("ASM\n(3 mm)", C_GRAY, C_GRAY_EDGE),
        ("[SBN\u2081]", C_ORANGE, C_ORANGE_EDGE),
        ("Phase\u2082$(x,y)$", '#BFDBFE', C_BLUE_EDGE),
        ("ASM\n(3 mm)", C_GRAY, C_GRAY_EDGE),
        ("[SBN\u2082]", C_ORANGE, C_ORANGE_EDGE),
        ("\u2026", 'white', 'white'),
        ("Phase\u2099$(x,y)$", '#BFDBFE', C_BLUE_EDGE),
        ("Detector", C_GREEN, C_GREEN_EDGE),
    ]

    n = len(stages)
    bw = 0.078
    bh = 0.55
    x_start = 0.06
    x_end = 0.94
    xs = np.linspace(x_start, x_end, n)

    for i, (label, fc, ec) in enumerate(stages):
        fs = 10 if label != '\u2026' else 16
        _box(ax, xs[i], 0.50, bw, bh, label, fc, ec, fontsize=fs)
        if i < n - 1:
            _arrow_h(ax, xs[i] + bw/2 + 0.005, xs[i+1] - bw/2 - 0.005, 0.50)

    ax.text(0.5, 0.92, 'Real-space D2NN information flow', transform=ax.transAxes,
            ha='center', va='center', fontsize=13, fontfamily=FONT_FAMILY,
            fontweight='bold', color='#1F2937')

    fig.savefig(os.path.join(OUTDIR, 'diagram_real_d2nn_flow.png'), dpi=DPI,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  -> diagram_real_d2nn_flow.png")


# ──────────────────────────────────────────────────────────────────────
# Diagram 4: Fourier-space D2NN flow (horizontal)
# ──────────────────────────────────────────────────────────────────────
def diagram_fourier_d2nn_flow():
    fig, ax = plt.subplots(figsize=(14, 2.5))
    _setup_ax(ax)

    stages = [
        ("Input$(x,y)$", C_BLUE, C_BLUE_EDGE),
        ("2f (FT)", C_ORANGE, C_ORANGE_EDGE),
        ("Phase\u2081$(f_x,f_y)$", C_PURPLE, C_PURPLE_EDGE),
        ("ASM\n(100 \u00b5m)", C_GRAY, C_GRAY_EDGE),
        ("Phase\u2082$(f_x,f_y)$", C_PURPLE, C_PURPLE_EDGE),
        ("\u2026", 'white', 'white'),
        ("Phase\u2099$(f_x,f_y)$", C_PURPLE, C_PURPLE_EDGE),
        ("2f\u207b\u00b9 (IFT)", C_ORANGE, C_ORANGE_EDGE),
        ("Detector", C_GREEN, C_GREEN_EDGE),
    ]

    n = len(stages)
    bw = 0.085
    bh = 0.55
    xs = np.linspace(0.06, 0.94, n)

    for i, (label, fc, ec) in enumerate(stages):
        fs = 10 if label != '\u2026' else 16
        _box(ax, xs[i], 0.50, bw, bh, label, fc, ec, fontsize=fs)
        if i < n - 1:
            _arrow_h(ax, xs[i] + bw/2 + 0.005, xs[i+1] - bw/2 - 0.005, 0.50)

    ax.text(0.5, 0.92, 'Fourier-space D2NN information flow', transform=ax.transAxes,
            ha='center', va='center', fontsize=13, fontfamily=FONT_FAMILY,
            fontweight='bold', color='#1F2937')

    fig.savefig(os.path.join(OUTDIR, 'diagram_fourier_d2nn_flow.png'), dpi=DPI,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  -> diagram_fourier_d2nn_flow.png")


# ──────────────────────────────────────────────────────────────────────
# Diagram 5: Hybrid D2NN flow (horizontal)
# ──────────────────────────────────────────────────────────────────────
def diagram_hybrid_flow():
    fig, ax = plt.subplots(figsize=(14, 3.5))
    _setup_ax(ax)

    # (label, fc, ec, domain_label)  domain_label shown above
    stages = [
        ("2f (FT)", C_ORANGE, C_ORANGE_EDGE, None),
        ("Phase\u2081\n$(f_x,f_y)$", C_PURPLE, C_PURPLE_EDGE, "Fourier"),
        ("2f\u207b\u00b9 (IFT)", C_ORANGE, C_ORANGE_EDGE, None),
        ("Phase\u2082\n$(x,y)$", '#BFDBFE', C_BLUE_EDGE, "Real"),
        ("2f (FT)", C_ORANGE, C_ORANGE_EDGE, None),
        ("Phase\u2083\n$(f_x,f_y)$", C_PURPLE, C_PURPLE_EDGE, "Fourier"),
        ("\u2026", 'white', 'white', None),
        ("2f\u207b\u00b9 (IFT)", C_ORANGE, C_ORANGE_EDGE, None),
        ("Detector", C_GREEN, C_GREEN_EDGE, "Real"),
    ]

    n = len(stages)
    bw = 0.082
    bh = 0.35
    cy = 0.42
    xs = np.linspace(0.06, 0.94, n)

    for i, (label, fc, ec, dom) in enumerate(stages):
        fs = 10 if label != '\u2026' else 16
        _box(ax, xs[i], cy, bw, bh, label, fc, ec, fontsize=fs)
        if i < n - 1:
            _arrow_h(ax, xs[i] + bw/2 + 0.005, xs[i+1] - bw/2 - 0.005, cy)
        if dom:
            color = '#7C3AED' if dom == 'Fourier' else '#2563EB'
            ax.text(xs[i], cy + bh/2 + 0.08, f'{dom} domain',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=9, fontfamily=FONT_FAMILY, color=color,
                    fontstyle='italic',
                    bbox=dict(boxstyle='round,pad=0.15',
                              fc=C_PURPLE if dom == 'Fourier' else C_BLUE,
                              ec=color, lw=0.8, alpha=0.7))

    ax.text(0.5, 0.92, 'Hybrid D2NN structure  (alternating domains)',
            transform=ax.transAxes, ha='center', va='center', fontsize=13,
            fontfamily=FONT_FAMILY, fontweight='bold', color='#1F2937')

    fig.savefig(os.path.join(OUTDIR, 'diagram_hybrid_flow.png'), dpi=DPI,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  -> diagram_hybrid_flow.png")


# ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating diagrams...")
    diagram_dual2f()
    diagram_forward_pass()
    diagram_real_d2nn_flow()
    diagram_fourier_d2nn_flow()
    diagram_hybrid_flow()
    print("Done. All 5 diagrams saved to:", OUTDIR)
