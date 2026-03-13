#!/usr/bin/env python3
"""Generate optical-component-accurate schematics for the F-D2NN report.

Physical parameters (MNIST 5-layer, f=1mm config):
  λ = 532 nm,  f = 1 mm,  N = 200,  dx = 1 μm
  NA = 0.16,   physical aperture ≈ 200 μm

Real D2NN:    layer spacing 3 mm × 5 = 15 mm total
Fourier D2NN: 2f + 4×100μm + 2f = 4.4 mm total
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Arc,
                                 Ellipse, Rectangle, Polygon)
import numpy as np
import os

OUTDIR = os.path.dirname(os.path.abspath(__file__))
DPI = 300
FONT = 'DejaVu Sans'

# ── Color palette ──────────────────────────────────────────────────
C_BEAM = '#E63946'
C_BEAM_ALPHA = 0.18
C_LENS = '#B45309'
C_LENS_FILL = '#FDE68A'
C_PHASE_REAL = '#2563EB'
C_PHASE_REAL_FILL = '#DBEAFE'
C_PHASE_FOURIER = '#7C3AED'
C_PHASE_FOURIER_FILL = '#EDE9FE'
C_SBN = '#EA580C'
C_SBN_FILL = '#FED7AA'
C_DETECTOR = '#059669'
C_DETECTOR_FILL = '#D1FAE5'
C_INPUT_EDGE = '#2563EB'
C_AXIS = '#D1D5DB'
C_LABEL = '#1F2937'
C_DIM = '#6B7280'
C_DOMAIN_FOURIER = '#7C3AED'
C_DOMAIN_REAL = '#2563EB'


# ═══════════════════════════════════════════════════════════════════
#  Drawing primitives
# ═══════════════════════════════════════════════════════════════════

def draw_optical_axis(ax, x1, x2, y):
    ax.plot([x1, x2], [y, y], color=C_AXIS, linewidth=0.5, linestyle=':',
            zorder=0, alpha=0.6)


def draw_lens(ax, x, y, h, label=None, label_below=True):
    """Biconvex thin lens: two arcs + fill + arrow tips."""
    w = h * 0.18
    arc_l = Arc((x - w * 0.12, y), w, h, angle=0, theta1=-90, theta2=90,
                color=C_LENS, linewidth=2.0, zorder=5)
    ax.add_patch(arc_l)
    arc_r = Arc((x + w * 0.12, y), w, h, angle=0, theta1=90, theta2=270,
                color=C_LENS, linewidth=2.0, zorder=5)
    ax.add_patch(arc_r)
    ell = Ellipse((x, y), w * 0.85, h * 0.97, facecolor=C_LENS_FILL,
                  edgecolor='none', alpha=0.55, zorder=4)
    ax.add_patch(ell)
    tip = h * 0.055
    for sign in [1, -1]:
        yt = y + sign * h / 2
        ax.plot([x - tip, x, x + tip],
                [yt - sign * tip, yt, yt - sign * tip],
                color=C_LENS, linewidth=1.5, solid_capstyle='round', zorder=5)
    if label:
        ly = y - h / 2 - h * 0.15 if label_below else y + h / 2 + h * 0.15
        va = 'top' if label_below else 'bottom'
        ax.text(x, ly, label, ha='center', va=va, fontsize=12,
                fontfamily=FONT, color=C_LENS, fontweight='bold', zorder=6)


def draw_phase_mask(ax, x, y, h, label=None, fourier=False, label_above=True):
    """Phase mask / SLM: thin slab with wavy interior."""
    color = C_PHASE_FOURIER if fourier else C_PHASE_REAL
    fill = C_PHASE_FOURIER_FILL if fourier else C_PHASE_REAL_FILL
    w = h * 0.055
    rect = Rectangle((x - w / 2, y - h / 2), w, h,
                      facecolor=fill, edgecolor=color, linewidth=1.8, zorder=5)
    ax.add_patch(rect)
    for i in range(3):
        xi = x - w / 2 + w * (i + 1) / 4
        ys = np.linspace(y - h * 0.40, y + h * 0.40, 30)
        xs = xi + w * 0.10 * np.sin(ys * 28)
        ax.plot(xs, ys, color=color, linewidth=0.4, alpha=0.45, zorder=6)
    if label:
        ly = y + h / 2 + h * 0.10 if label_above else y - h / 2 - h * 0.10
        va = 'bottom' if label_above else 'top'
        ax.text(x, ly, label, ha='center', va=va, fontsize=11,
                fontfamily=FONT, color=color, fontweight='bold', zorder=6)


def draw_sbn(ax, x, y, h, label=None):
    """SBN nonlinear crystal: hatched rectangle."""
    w = h * 0.12
    rect = Rectangle((x - w / 2, y - h / 2), w, h,
                      facecolor=C_SBN_FILL, edgecolor=C_SBN, linewidth=1.5,
                      zorder=5)
    ax.add_patch(rect)
    for i in range(5):
        frac = (i + 1) / 6
        yh = y - h / 2 + h * frac
        ax.plot([x - w / 2, x + w / 2], [yh - h * 0.035, yh + h * 0.035],
                color=C_SBN, linewidth=0.5, alpha=0.45, zorder=6)
    if label:
        ax.text(x, y - h / 2 - h * 0.10, label, ha='center', va='top',
                fontsize=11, fontfamily=FONT, color=C_SBN, fontweight='bold',
                zorder=6)


def draw_detector(ax, x, y, h, label=None):
    """Detector: thick bar with pixel grid."""
    w = h * 0.07
    rect = Rectangle((x - w / 2, y - h / 2), w, h,
                      facecolor=C_DETECTOR_FILL, edgecolor=C_DETECTOR,
                      linewidth=2.0, zorder=5)
    ax.add_patch(rect)
    for i in range(1, 8):
        yp = y - h / 2 + h * i / 8
        ax.plot([x - w / 2, x + w / 2], [yp, yp],
                color=C_DETECTOR, linewidth=0.4, alpha=0.35, zorder=6)
    if label:
        ax.text(x, y - h / 2 - h * 0.10, label, ha='center', va='top',
                fontsize=12, fontfamily=FONT, color=C_DETECTOR,
                fontweight='bold', zorder=6)


def draw_input_plane(ax, x, y, h, label=None):
    """Input object plane: vertical line with upward arrow tip."""
    ax.plot([x, x], [y - h / 2, y + h / 2], color=C_INPUT_EDGE,
            linewidth=2.0, zorder=5)
    tip = h * 0.05
    ax.plot([x - tip * 0.7, x, x + tip * 0.7],
            [y + h / 2 - tip, y + h / 2, y + h / 2 - tip],
            color=C_INPUT_EDGE, linewidth=1.5, zorder=5)
    if label:
        ax.text(x, y - h / 2 - h * 0.10, label, ha='center', va='top',
                fontsize=12, fontfamily=FONT, color=C_DOMAIN_REAL,
                fontweight='bold', zorder=6)


def draw_beam(ax, x1, x2, y, h1, h2=None):
    """Beam propagation: shaded trapezoid with edge rays."""
    if h2 is None:
        h2 = h1
    verts = [(x1, y - h1 / 2), (x2, y - h2 / 2),
             (x2, y + h2 / 2), (x1, y + h1 / 2)]
    poly = Polygon(verts, closed=True, facecolor=C_BEAM,
                   alpha=C_BEAM_ALPHA, edgecolor='none', zorder=1)
    ax.add_patch(poly)
    ax.plot([x1, x2], [y + h1 / 2, y + h2 / 2],
            color=C_BEAM, linewidth=0.6, alpha=0.45, zorder=2)
    ax.plot([x1, x2], [y - h1 / 2, y - h2 / 2],
            color=C_BEAM, linewidth=0.6, alpha=0.45, zorder=2)


def draw_dim(ax, x1, x2, y, label, above=False):
    """Dimension arrow with label."""
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='<->', color=C_DIM, lw=0.8))
    offset = 0.06 if above else -0.06
    va = 'bottom' if above else 'top'
    ax.text((x1 + x2) / 2, y + offset, label, ha='center', va=va,
            fontsize=10, fontfamily=FONT, color=C_DIM, fontstyle='italic')


def draw_domain_bracket(ax, x1, x2, y, label, color):
    """Bracket above a region indicating domain."""
    tick = 0.06
    ax.plot([x1, x2], [y, y], color=color, linewidth=1.0, alpha=0.7, zorder=3)
    ax.plot([x1, x1], [y - tick, y], color=color, linewidth=1.0, alpha=0.7, zorder=3)
    ax.plot([x2, x2], [y - tick, y], color=color, linewidth=1.0, alpha=0.7, zorder=3)
    ax.text((x1 + x2) / 2, y + tick * 0.4, label, ha='center', va='bottom',
            fontsize=10, fontfamily=FONT, color=color, fontstyle='italic',
            zorder=6)


def draw_params_box(ax, x, y, lines, fontsize=9):
    """Small text box listing physical parameters."""
    text = '\n'.join(lines)
    ax.text(x, y, text, ha='left', va='top', fontsize=fontsize,
            fontfamily='DejaVu Sans Mono', color=C_DIM,
            bbox=dict(boxstyle='round,pad=0.4', fc='#F9FAFB', ec='#E5E7EB',
                      lw=0.8, alpha=0.9), zorder=7)


# ═══════════════════════════════════════════════════════════════════
#  Diagram 1: F-D2NN architecture  (replaces p.12)
# ═══════════════════════════════════════════════════════════════════
def diagram_fdnn_architecture():
    """Full F-D2NN horizontal optical bench schematic with accurate distances."""
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(-1, 21)
    ax.set_ylim(-3.8, 4.5)
    ax.axis('off')
    ax.set_aspect('equal')

    y0 = 0.0
    ch = 2.0  # component height

    # Positions — wider mask spacing for label clarity
    spacing_f = 2.2       # display units for focal length f = 1 mm
    spacing_mask = 0.70   # display units for 100 μm between masks (wider for labels)

    x = 0.5
    pos = {}
    pos['input'] = x
    x += spacing_f
    pos['lens1'] = x
    x += spacing_f
    pos['pm1'] = x
    x += spacing_mask
    pos['pm2'] = x
    x += spacing_mask
    pos['pm3'] = x
    x += spacing_mask
    pos['pm4'] = x
    x += spacing_mask
    pos['pm5'] = x
    x += 1.2   # more space before SBN
    pos['sbn'] = x
    x += 1.2   # more space SBN->Lens2
    pos['lens2'] = x
    x += spacing_f
    pos['det'] = x

    draw_optical_axis(ax, pos['input'] - 0.4, pos['det'] + 0.4, y0)

    # Beam regions
    bh = ch * 0.50
    pairs = [
        ('input', 'lens1', bh * 0.65, bh * 0.95),
        ('lens1', 'pm1', bh * 0.95, bh * 0.80),
        ('pm1', 'pm2', bh * 0.80, bh * 0.80),
        ('pm2', 'pm3', bh * 0.80, bh * 0.80),
        ('pm3', 'pm4', bh * 0.80, bh * 0.80),
        ('pm4', 'pm5', bh * 0.80, bh * 0.80),
        ('pm5', 'sbn', bh * 0.80, bh * 0.80),
        ('sbn', 'lens2', bh * 0.80, bh * 0.95),
        ('lens2', 'det', bh * 0.95, bh * 0.65),
    ]
    for a, b, h1, h2 in pairs:
        x1 = pos[a] + 0.08
        x2 = pos[b] - 0.08
        if x2 > x1:
            draw_beam(ax, x1, x2, y0, h1, h2)

    # Components — all PM labels above for clarity
    draw_input_plane(ax, pos['input'], y0, ch, '$U_0(x,y)$\nInput plane')
    draw_lens(ax, pos['lens1'], y0, ch * 1.1, 'Lens$_1$ (2f FT)\n$f$ = 1 mm')
    pm_labels = ['PM$_1$', 'PM$_2$', 'PM$_3$', 'PM$_4$', 'PM$_5$']
    for i, key in enumerate(['pm1', 'pm2', 'pm3', 'pm4', 'pm5']):
        draw_phase_mask(ax, pos[key], y0, ch, pm_labels[i],
                        fourier=True, label_above=True)
    draw_sbn(ax, pos['sbn'], y0, ch, 'SBN\n(optional)')
    draw_lens(ax, pos['lens2'], y0, ch * 1.1, 'Lens$_2$ (2f$^{-1}$ IFT)\n$f$ = 1 mm')
    draw_detector(ax, pos['det'], y0, ch, 'Detector\n$I = |U_{out}|^2$')

    # ── Dimension annotations ──
    dim_y = y0 - ch / 2 - 0.6
    draw_dim(ax, pos['input'], pos['lens1'], dim_y, '$f$ = 1 mm')
    draw_dim(ax, pos['lens1'], pos['pm1'], dim_y, '$f$ = 1 mm')

    # Inter-mask spacing — single example + total
    dim_y2 = y0 - ch / 2 - 1.3
    draw_dim(ax, pos['pm1'], pos['pm2'], dim_y2, '100 $\\mu$m')
    draw_dim(ax, pos['pm1'], pos['pm5'], dim_y2 - 0.7,
             '4 $\\times$ 100 $\\mu$m = 0.4 mm')

    draw_dim(ax, pos['lens2'], pos['det'], dim_y, '$f$ = 1 mm')
    draw_dim(ax, pos['sbn'], pos['lens2'], dim_y - 0.7, '$f$ = 1 mm')

    # Total system length
    total_y = y0 + ch / 2 + 1.2
    ax.annotate('', xy=(pos['det'] + 0.15, total_y),
                xytext=(pos['input'] - 0.15, total_y),
                arrowprops=dict(arrowstyle='<->', color=C_LABEL, lw=1.0))
    ax.text((pos['input'] + pos['det']) / 2, total_y + 0.15,
            'Total system length: 2$f$ + 0.4 mm + 2$f$ = 4.4 mm',
            ha='center', va='bottom', fontsize=13, fontfamily=FONT,
            fontweight='bold', color=C_LABEL)

    # Domain brackets
    bracket_y = y0 + ch / 2 + 0.4
    draw_domain_bracket(ax, pos['input'] - 0.15, pos['lens1'] - 0.3,
                        bracket_y, 'Real domain', C_DOMAIN_REAL)
    draw_domain_bracket(ax, pos['pm1'] - 0.15, pos['pm5'] + 0.15,
                        bracket_y + 0.55,
                        'Fourier domain  (learnable phase masks + ASM)',
                        C_DOMAIN_FOURIER)
    draw_domain_bracket(ax, pos['lens2'] + 0.3, pos['det'] + 0.15,
                        bracket_y, 'Real domain', C_DOMAIN_REAL)

    # Resolution annotation
    ax.annotate(r'$\Delta x_F = \frac{\lambda f}{N \cdot dx}$'
                '\n= 2.66 $\\mu$m',
                xy=(pos['lens1'] + 0.3, y0 + ch * 0.35),
                xytext=(pos['lens1'] + 1.8, y0 + ch * 0.8),
                fontsize=10, color=C_DIM, fontfamily=FONT,
                arrowprops=dict(arrowstyle='->', color=C_DIM, lw=0.8),
                bbox=dict(boxstyle='round,pad=0.3', fc='#F9FAFB',
                          ec='#E5E7EB', lw=0.6, alpha=0.9))

    # Parameters box
    draw_params_box(ax, pos['det'] + 0.5, y0 + ch * 0.5, [
        'lambda = 532 nm',
        'f      = 1 mm',
        'N      = 200x200',
        'dx     = 1 um',
        'NA     = 0.16',
        'Layers = 5',
    ])

    # Title
    ax.set_title('Fourier-space D$^2$NN Architecture  —  Optical Schematic',
                 fontsize=18, fontfamily=FONT, fontweight='bold',
                 color=C_LABEL, pad=15)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'optical_fdnn_architecture.png'),
                dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  -> optical_fdnn_architecture.png")


# ═══════════════════════════════════════════════════════════════════
#  Diagram 2: Real-space D2NN flow  (replaces p.15)
# ═══════════════════════════════════════════════════════════════════
def diagram_real_d2nn():
    """Real-space D2NN with 3 mm layer spacing."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_xlim(-0.5, 18)
    ax.set_ylim(-3.0, 3.2)
    ax.axis('off')
    ax.set_aspect('equal')

    y0 = 0.0
    ch = 1.8
    sp = 2.8  # display spacing for 3 mm (uniform)

    x = 0.5
    pos = {}
    pos['input'] = x;  x += sp
    pos['pm1'] = x;    x += sp * 0.55
    pos['sbn1'] = x;   x += sp * 0.45
    pos['pm2'] = x;    x += sp * 0.55
    pos['sbn2'] = x;   x += sp * 0.35
    pos['dots'] = x;   x += sp * 0.35
    pos['pm5'] = x;    x += sp
    pos['det'] = x

    draw_optical_axis(ax, pos['input'] - 0.3, pos['det'] + 0.3, y0)

    bh = ch * 0.50
    # Beams — in real D2NN, beam spreads significantly due to diffraction cone
    segs = [
        ('input', 'pm1', bh * 0.55, bh * 0.90),
        ('pm1', 'sbn1', bh * 0.90, bh * 1.05),
        ('sbn1', 'pm2', bh * 1.05, bh * 1.15),
        ('pm2', 'sbn2', bh * 1.15, bh * 1.25),
        ('sbn2', 'dots', bh * 1.25, bh * 1.30),
        ('dots', 'pm5', bh * 1.30, bh * 1.35),
        ('pm5', 'det', bh * 1.35, bh * 0.90),
    ]
    for a, b, h1, h2 in segs:
        x1 = pos[a] + (0.08 if a != 'dots' else 0.25)
        x2 = pos[b] - (0.08 if b != 'dots' else 0.25)
        if x2 > x1:
            draw_beam(ax, x1, x2, y0, h1, h2)

    draw_input_plane(ax, pos['input'], y0, ch, '$U_0(x,y)$')
    draw_phase_mask(ax, pos['pm1'], y0, ch, 'Phase$_1(x,y)$',
                    fourier=False, label_above=True)
    draw_sbn(ax, pos['sbn1'], y0, ch * 0.85, '[SBN$_1$]')
    draw_phase_mask(ax, pos['pm2'], y0, ch, 'Phase$_2(x,y)$',
                    fourier=False, label_above=True)
    draw_sbn(ax, pos['sbn2'], y0, ch * 0.85, '[SBN$_2$]')
    ax.text(pos['dots'], y0, r'$\cdots$', ha='center', va='center',
            fontsize=28, color=C_LABEL, zorder=6)
    draw_phase_mask(ax, pos['pm5'], y0, ch, 'Phase$_5(x,y)$',
                    fourier=False, label_above=True)
    draw_detector(ax, pos['det'], y0, ch, 'Detector\n$I=|U_{out}|^2$')

    # Dimension: inter-layer
    dim_y = y0 - ch / 2 - 0.55
    draw_dim(ax, pos['input'], pos['pm1'], dim_y, '3 mm (ASM)')
    draw_dim(ax, pos['pm2'] + 0.1, pos['sbn2'] - 0.1, dim_y - 0.55,
             '3 mm')

    # Total
    total_y = y0 + ch / 2 + 0.6
    ax.annotate('', xy=(pos['det'] + 0.15, total_y),
                xytext=(pos['input'] - 0.15, total_y),
                arrowprops=dict(arrowstyle='<->', color=C_LABEL, lw=1.0))
    ax.text((pos['input'] + pos['det']) / 2, total_y + 0.10,
            'Total: 5 × 3 mm = 15 mm',
            ha='center', va='bottom', fontsize=13, fontfamily=FONT,
            fontweight='bold', color=C_LABEL)

    # Note about convolutional nature
    note_y = y0 - ch / 2 - 1.7
    ax.text((pos['input'] + pos['det']) / 2, note_y,
            'Each ASM propagation acts as a fixed convolutional kernel '
            '(diffraction cone covers full previous layer)',
            ha='center', va='top', fontsize=10, fontfamily=FONT,
            color=C_DIM, fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.3', fc='#FFF7ED', ec='#FDBA74',
                      lw=0.6, alpha=0.9))

    # Parameters
    draw_params_box(ax, pos['det'] + 0.4, y0 + ch * 0.35, [
        'lambda = 532 nm',
        'dx     = 1 um',
        'N      = 200x200',
        'dz     = 3 mm',
        'N_F    << 1',
    ])

    ax.set_title('Real-space D$^2$NN Information Flow',
                 fontsize=18, fontfamily=FONT, fontweight='bold',
                 color=C_LABEL, pad=15)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'optical_real_d2nn_flow.png'),
                dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  -> optical_real_d2nn_flow.png")


# ═══════════════════════════════════════════════════════════════════
#  Diagram 3: Fourier-space D2NN flow  (replaces p.16 top)
# ═══════════════════════════════════════════════════════════════════
def diagram_fourier_d2nn():
    """Fourier-space D2NN horizontal flow with accurate distances."""
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.set_xlim(-0.5, 19)
    ax.set_ylim(-3.5, 3.5)
    ax.axis('off')
    ax.set_aspect('equal')

    y0 = 0.0
    ch = 1.8
    sp_f = 2.0    # display units for f = 1 mm
    sp_mask = 1.0 # display units for 100 μm (wider for labels)

    x = 0.5
    pos = {}
    pos['input'] = x;  x += sp_f
    pos['lens1'] = x;  x += sp_f
    pos['pm1'] = x;    x += sp_mask
    pos['pm2'] = x;    x += sp_mask
    pos['dots'] = x;   x += sp_mask
    pos['pm5'] = x;    x += sp_f
    pos['lens2'] = x;  x += sp_f
    pos['det'] = x

    draw_optical_axis(ax, pos['input'] - 0.3, pos['det'] + 0.3, y0)

    bh = ch * 0.50
    segs = [
        ('input', 'lens1', bh * 0.60, bh * 0.90),
        ('lens1', 'pm1', bh * 0.90, bh * 0.78),
        ('pm1', 'pm2', bh * 0.78, bh * 0.78),
        ('pm2', 'dots', bh * 0.78, bh * 0.78),
        ('dots', 'pm5', bh * 0.78, bh * 0.78),
        ('pm5', 'lens2', bh * 0.78, bh * 0.90),
        ('lens2', 'det', bh * 0.90, bh * 0.60),
    ]
    for a, b, h1, h2 in segs:
        x1 = pos[a] + (0.08 if a != 'dots' else 0.20)
        x2 = pos[b] - (0.08 if b != 'dots' else 0.20)
        if x2 > x1:
            draw_beam(ax, x1, x2, y0, h1, h2)

    draw_input_plane(ax, pos['input'], y0, ch, '$U_0(x,y)$')
    draw_lens(ax, pos['lens1'], y0, ch * 1.05, 'Lens$_1$\n(2f FT)')
    draw_phase_mask(ax, pos['pm1'], y0, ch,
                    'PM$_1(f_x,f_y)$', fourier=True, label_above=True)
    draw_phase_mask(ax, pos['pm2'], y0, ch,
                    'PM$_2(f_x,f_y)$', fourier=True, label_above=True)
    ax.text(pos['dots'], y0, r'$\cdots$', ha='center', va='center',
            fontsize=28, color=C_LABEL, zorder=6)
    draw_phase_mask(ax, pos['pm5'], y0, ch,
                    'PM$_5(f_x,f_y)$', fourier=True, label_above=True)
    draw_lens(ax, pos['lens2'], y0, ch * 1.05, 'Lens$_2$\n(2f$^{-1}$ IFT)')
    draw_detector(ax, pos['det'], y0, ch, 'Detector')

    # Dimensions
    dim_y = y0 - ch / 2 - 0.55
    draw_dim(ax, pos['input'], pos['lens1'], dim_y, '$f$ = 1 mm')
    draw_dim(ax, pos['lens1'], pos['pm1'], dim_y, '$f$ = 1 mm')
    draw_dim(ax, pos['pm1'], pos['pm2'], dim_y - 0.6, '100 μm')
    draw_dim(ax, pos['pm5'], pos['lens2'], dim_y, '$f$ = 1 mm')
    draw_dim(ax, pos['lens2'], pos['det'], dim_y, '$f$ = 1 mm')

    # Domain brackets
    bracket_y = y0 + ch / 2 + 0.35
    draw_domain_bracket(ax, pos['input'] - 0.15, pos['lens1'] - 0.25,
                        bracket_y, 'Real domain', C_DOMAIN_REAL)
    draw_domain_bracket(ax, pos['pm1'] - 0.15, pos['pm5'] + 0.15,
                        bracket_y + 0.5,
                        'Fourier domain  (global spectral access)',
                        C_DOMAIN_FOURIER)
    draw_domain_bracket(ax, pos['lens2'] + 0.25, pos['det'] + 0.15,
                        bracket_y, 'Real domain', C_DOMAIN_REAL)

    # Total
    total_y = y0 + ch / 2 + 1.3
    ax.annotate('', xy=(pos['det'] + 0.15, total_y),
                xytext=(pos['input'] - 0.15, total_y),
                arrowprops=dict(arrowstyle='<->', color=C_LABEL, lw=1.0))
    ax.text((pos['input'] + pos['det']) / 2, total_y + 0.10,
            'Total: 4$f$ + 4×100 μm = 4.4 mm  (3.4× more compact than Real D$^2$NN)',
            ha='center', va='bottom', fontsize=13, fontfamily=FONT,
            fontweight='bold', color=C_LABEL)

    # Key insight
    note_y = y0 - ch / 2 - 1.8
    ax.text((pos['input'] + pos['det']) / 2, note_y,
            '2f system gives instant Fourier transform → Phase Mask$_1$ already '
            'has direct access to all spatial frequencies\n'
            '100 μm propagation provides local mixing '
            '(equivalent to convolutional kernel in Fourier plane)',
            ha='center', va='top', fontsize=10, fontfamily=FONT,
            color=C_DIM, fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.3', fc='#F5F3FF', ec='#C4B5FD',
                      lw=0.6, alpha=0.9))

    ax.set_title('Fourier-space D$^2$NN Information Flow',
                 fontsize=18, fontfamily=FONT, fontweight='bold',
                 color=C_LABEL, pad=15)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'optical_fourier_d2nn_flow.png'),
                dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  -> optical_fourier_d2nn_flow.png")


# ═══════════════════════════════════════════════════════════════════
#  Diagram 4: Hybrid D2NN flow  (replaces p.16 bottom)
# ═══════════════════════════════════════════════════════════════════
def diagram_hybrid_d2nn():
    """Hybrid D2NN with alternating Fourier / Real domains."""
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(-0.5, 20)
    ax.set_ylim(-3.0, 4.2)
    ax.axis('off')
    ax.set_aspect('equal')

    y0 = 0.0
    ch = 1.7
    sp = 1.5  # uniform display spacing

    x = 0.5
    pos = {}
    pos['input'] = x;   x += sp
    pos['L1'] = x;      x += sp       # 2f FT
    pos['pm1'] = x;     x += sp       # Phase_1 (Fourier)
    pos['L2'] = x;      x += sp       # 2f^-1 IFT
    pos['pm2'] = x;     x += sp       # Phase_2 (Real)
    pos['L3'] = x;      x += sp       # 2f FT
    pos['pm3'] = x;     x += sp       # Phase_3 (Fourier)
    pos['dots'] = x;    x += sp
    pos['L4'] = x;      x += sp       # 2f^-1 IFT (final)
    pos['det'] = x

    draw_optical_axis(ax, pos['input'] - 0.3, pos['det'] + 0.3, y0)

    bh = ch * 0.45
    keys = list(pos.keys())
    for i in range(len(keys) - 1):
        a, b = keys[i], keys[i + 1]
        x1 = pos[a] + (0.08 if a != 'dots' else 0.25)
        x2 = pos[b] - (0.08 if b != 'dots' else 0.25)
        if x2 > x1:
            # Lens convergence/divergence effect
            a_is_lens = a.startswith('L')
            b_is_lens = b.startswith('L')
            h1 = bh * (0.85 if not a_is_lens else 0.95)
            h2 = bh * (0.85 if not b_is_lens else 0.95)
            if a == 'input':
                h1 = bh * 0.60
            if b == 'det':
                h2 = bh * 0.60
            draw_beam(ax, x1, x2, y0, h1, h2)

    draw_input_plane(ax, pos['input'], y0, ch, '$U_0(x,y)$')
    draw_lens(ax, pos['L1'], y0, ch, '2f\n(FT)')
    draw_phase_mask(ax, pos['pm1'], y0, ch,
                    'PM$_1(f_x,f_y)$', fourier=True, label_above=True)
    draw_lens(ax, pos['L2'], y0, ch, '2f$^{-1}$\n(IFT)')
    draw_phase_mask(ax, pos['pm2'], y0, ch,
                    'PM$_2(x,y)$', fourier=False, label_above=True)
    draw_lens(ax, pos['L3'], y0, ch, '2f\n(FT)')
    draw_phase_mask(ax, pos['pm3'], y0, ch,
                    'PM$_3(f_x,f_y)$', fourier=True, label_above=True)

    ax.text(pos['dots'], y0, r'$\cdots$', ha='center', va='center',
            fontsize=28, color=C_LABEL, zorder=6)

    draw_lens(ax, pos['L4'], y0, ch, '2f$^{-1}$\n(IFT)')
    draw_detector(ax, pos['det'], y0, ch, 'Detector')

    # Domain brackets (alternating) — above PM labels
    bracket_y = y0 + ch / 2 + 0.55
    draw_domain_bracket(ax, pos['L1'] + 0.2, pos['L2'] - 0.2,
                        bracket_y, 'Fourier domain', C_DOMAIN_FOURIER)
    draw_domain_bracket(ax, pos['L2'] + 0.2, pos['L3'] - 0.2,
                        bracket_y, 'Real domain', C_DOMAIN_REAL)
    draw_domain_bracket(ax, pos['L3'] + 0.2, pos['dots'] - 0.3,
                        bracket_y, 'Fourier domain', C_DOMAIN_FOURIER)
    draw_domain_bracket(ax, pos['L4'] + 0.2, pos['det'] + 0.15,
                        bracket_y, 'Real domain', C_DOMAIN_REAL)

    # Title
    ax.set_title('Hybrid D$^2$NN Structure  (alternating Fourier ↔ Real domains)',
                 fontsize=18, fontfamily=FONT, fontweight='bold',
                 color=C_LABEL, pad=15)

    # Key advantage annotation
    note_y = y0 - ch / 2 - 1.2
    ax.text((pos['input'] + pos['det']) / 2, note_y,
            'Alternating domains enable cross-domain feature learning:\n'
            'Fourier layers → global spectral filtering  |  '
            'Real layers → spatially localized modulation',
            ha='center', va='top', fontsize=11, fontfamily=FONT,
            color=C_DIM, fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.3', fc='#F0FDF4', ec='#86EFAC',
                      lw=0.6, alpha=0.9))

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'optical_hybrid_d2nn_flow.png'),
                dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  -> optical_hybrid_d2nn_flow.png")


# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating optical-component schematics...")
    diagram_fdnn_architecture()
    diagram_real_d2nn()
    diagram_fourier_d2nn()
    diagram_hybrid_d2nn()
    print(f"Done. All diagrams saved to: {OUTDIR}")
