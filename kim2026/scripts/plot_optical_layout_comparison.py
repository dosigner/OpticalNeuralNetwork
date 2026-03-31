"""
FD2NN vs D2NN Optical Layout Comparison — Corrected Parameters
Based on actual code: loss_sweep.py (FD2NN) and br15cm sweep config (D2NN)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig, axes = plt.subplots(2, 1, figsize=(14, 16))
fig.suptitle("FD2NN vs D2NN — Optical Layout Comparison (Actual Parameters)",
             fontsize=14, fontweight='bold', y=0.98)

# ════════════════════════════════════════════════════════════════
# Panel 1: FD2NN (Fourier-space D2NN)
# ════════════════════════════════════════════════════════════════
ax = axes[0]
ax.set_xlim(-5, 85)
ax.set_ylim(-22, 18)
ax.set_aspect('equal')
ax.set_title("FD2NN (Fourier-space D2NN)\nf=25mm, NA=0.508, 5 masks, Total=70mm",
             fontsize=11, fontweight='bold')
ax.set_xlabel("Optical axis [mm]", fontsize=10)

# Input beam (2mm diameter)
ax.annotate("Input\n2mm", xy=(0, 0), fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

# Lens 1
lens1_x = 5
ax.plot([lens1_x, lens1_x], [-10, 10], 'b-', linewidth=2.5)
ax.annotate("Lens1\nf=25mm", xy=(lens1_x, -13), fontsize=8, ha='center', color='blue')

# Beam expanding through lens to fourier plane
# Input 2mm -> Fourier window 19.4mm
beam_in_half = 1.0  # 1mm half-width (2mm beam)
fourier_half = 9.7  # 9.7mm half-width (19.4mm window)

# Beam envelope: input to masks
mask_center_x = 30  # center of mask region (25mm from lens)
ax.fill([lens1_x, mask_center_x-10, mask_center_x-10, lens1_x],
        [beam_in_half, fourier_half, -fourier_half, -beam_in_half],
        alpha=0.08, color='orange')
ax.plot([lens1_x, mask_center_x-10], [beam_in_half, fourier_half], 'orange', alpha=0.4, linewidth=1)
ax.plot([lens1_x, mask_center_x-10], [-beam_in_half, -fourier_half], 'orange', alpha=0.4, linewidth=1)

# 5 phase masks in fourier plane (5mm spacing, centered at 30mm)
mask_positions = [20, 25, 30, 35, 40]  # lens_f=25, then 5mm spacing
mask_height = fourier_half * 1.05
colors_mask = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#9b59b6']
for i, (mx, mc) in enumerate(zip(mask_positions, colors_mask)):
    ax.add_patch(patches.Rectangle((mx-0.3, -mask_height), 0.6, 2*mask_height,
                                    facecolor=mc, alpha=0.6, edgecolor='black', linewidth=0.5))
    ax.text(mx, mask_height + 1.5, f'M{i}', fontsize=7, ha='center', fontweight='bold')

# Label: beam hits only tiny fraction
ax.annotate("Beam spot: 24.7um\n= 1.3 pixels\n(0.0002% of mask!)",
            xy=(25, fourier_half-2), fontsize=7,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffcccc', alpha=0.9),
            ha='center')

# Fourier plane label
ax.annotate("Fourier Plane\n19.4mm, dx=18.9um",
            xy=(30, -15), fontsize=8, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ccffcc', alpha=0.7))

# Beam converging from masks to Lens2
lens2_x = 55  # 25mm after last mask at 40
ax.fill([mask_center_x+10, lens2_x, lens2_x, mask_center_x+10],
        [fourier_half, beam_in_half, -beam_in_half, -fourier_half],
        alpha=0.08, color='orange')
ax.plot([mask_center_x+10, lens2_x], [fourier_half, beam_in_half], 'orange', alpha=0.4, linewidth=1)
ax.plot([mask_center_x+10, lens2_x], [-fourier_half, -beam_in_half], 'orange', alpha=0.4, linewidth=1)

# Lens 2
ax.plot([lens2_x, lens2_x], [-10, 10], 'b-', linewidth=2.5)
ax.annotate("Lens2\nf=25mm", xy=(lens2_x, -13), fontsize=8, ha='center', color='blue')

# Focus lens and APD
focus_x = 65
ax.plot([focus_x, focus_x], [-5, 5], 'purple', linewidth=2)
ax.annotate("Focus\nf=4.5mm", xy=(focus_x, -13), fontsize=8, ha='center', color='purple')

# APD
apd_x = 75
ax.add_patch(patches.Rectangle((apd_x-1, -3), 2, 6, facecolor='gray', alpha=0.5, edgecolor='black'))
ax.text(apd_x, 5, "APD/MMF", fontsize=8, ha='center')

# Dimension arrows
ax.annotate('', xy=(lens1_x, 14), xytext=(lens2_x, 14),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
ax.text((lens1_x+lens2_x)/2, 15, "50mm (2f + masks + 2f)", fontsize=8, ha='center')

ax.annotate('', xy=(0, 16), xytext=(75, 16),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=0.8))
ax.text(37.5, 17, "~70mm total", fontsize=8, ha='center', color='gray')

# Diffraction spread annotation
ax.annotate("Spread per layer:\n21.7 px (409.6um)",
            xy=(35, -8), fontsize=7, ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))

ax.axhline(y=0, color='gray', linewidth=0.3, linestyle='--')
ax.set_yticks([])

# ════════════════════════════════════════════════════════════════
# Panel 2: D2NN (Free-space, beam reducer 15cm)
# ════════════════════════════════════════════════════════════════
ax = axes[1]
# D2NN is much longer (250mm) so we scale differently
ax.set_xlim(-5, 280)
ax.set_ylim(-12, 12)
ax.set_aspect('equal')
ax.set_title("D2NN (Free-space D2NN) — Beam Reducer 15cm\n5 masks, 50mm spacing, Angular Spectrum, Total=250mm",
             fontsize=11, fontweight='bold')
ax.set_xlabel("Optical axis [mm]", fontsize=10)

# Input beam
ax.annotate("Input\n10mm\naperture", xy=(0, 0), fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

# 5 phase masks at 50mm spacing
d2nn_mask_pos = [10, 60, 110, 160, 210]
beam_half = 5.0  # ~10mm aperture / 2 (scaled for visibility)

# Grid window is only 5.12mm but aperture is 10mm -> beam bigger than grid!
grid_half = 2.56  # 5.12mm / 2

for i, mx in enumerate(d2nn_mask_pos):
    ax.add_patch(patches.Rectangle((mx-0.8, -beam_half), 1.6, 2*beam_half,
                                    facecolor=colors_mask[i], alpha=0.6, edgecolor='black', linewidth=0.5))
    ax.text(mx, beam_half + 1.5, f'M{i}', fontsize=7, ha='center', fontweight='bold')

# Beam envelope (roughly constant due to angular spectrum with large spread)
# But spread is 3100px >> 1024px, so beam spills everywhere
beam_xs = [10, 60, 110, 160, 210, 260]
beam_top = [beam_half, beam_half*1.1, beam_half*1.15, beam_half*1.15, beam_half*1.1, beam_half]
ax.fill(beam_xs + beam_xs[::-1],
        [b for b in beam_top] + [-b for b in beam_top[::-1]],
        alpha=0.08, color='orange')
for j in range(len(beam_xs)-1):
    ax.plot([beam_xs[j], beam_xs[j+1]], [beam_top[j], beam_top[j+1]], 'orange', alpha=0.4, linewidth=1)
    ax.plot([beam_xs[j], beam_xs[j+1]], [-beam_top[j], -beam_top[j+1]], 'orange', alpha=0.4, linewidth=1)

# Grid window annotation (smaller than beam!)
ax.add_patch(patches.Rectangle((55, -grid_half), 10, 2*grid_half,
                                facecolor='none', edgecolor='red', linewidth=1.5, linestyle='--'))
ax.annotate("Grid: 5.12mm\n(beam 10mm\nOVERFLOWS!)",
            xy=(60, -grid_half-2), fontsize=7, ha='center', color='red',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffcccc', alpha=0.8))

# Real-space label
ax.annotate("Real-space\n5.12mm, dx=5.0um",
            xy=(110, -9), fontsize=8, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ccffcc', alpha=0.7))

# Detector
det_x = 260  # 50mm after last mask
ax.add_patch(patches.Rectangle((det_x-1, -beam_half), 2, 2*beam_half,
                                facecolor='gray', alpha=0.5, edgecolor='black'))
ax.text(det_x, beam_half+1.5, "Detector\n(direct, no focus lens)", fontsize=8, ha='center')

# Dimension arrows
ax.annotate('', xy=(10, 8), xytext=(60, 8),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
ax.text(35, 9, "50mm", fontsize=8, ha='center')

ax.annotate('', xy=(0, 10.5), xytext=(260, 10.5),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=0.8))
ax.text(130, 11.5, "~250mm total (4x50mm + 50mm det)", fontsize=8, ha='center', color='gray')

# Spread annotation
ax.annotate("Spread per layer:\n3100 px >> 1024 grid\n(MASSIVE overflow)",
            xy=(160, -8), fontsize=7, ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))

ax.axhline(y=0, color='gray', linewidth=0.3, linestyle='--')
ax.set_yticks([])

# ════════════════════════════════════════════════════════════════
# Parameter Comparison Table
# ════════════════════════════════════════════════════════════════
table_data = [
    ["Propagation",      "2f Lens Fourier transform",    "Angular spectrum diffraction"],
    ["Lens",             "AC254-025-C x2",               "None"],
    ["Mask location",    "Fourier plane (19.4mm)",        "Real-space (5.12mm)"],
    ["Mask pixel",       "18.92 um",                      "5.00 um"],
    ["Grid",             "1024x1024",                     "1024x1024"],
    ["Input beam",       "2mm (receiver window)",         "10mm (aperture)"],
    ["Beam utilization", "0.0002% (1.3 px spot)",         "~100% (fills grid)"],
    ["Layer spacing",    "5 mm",                          "50 mm"],
    ["Diff. spread/layer", "21.7 px",                    "3100 px (overflow!)"],
    ["Total length",     "70 mm",                         "250 mm"],
    ["Focus lens",       "f=4.5mm → APD/MMF",            "None (direct detection)"],
    ["Parameters",       "5.2M (<<1% used)",              "5.2M (all used)"],
]

table = fig.add_axes([0.08, -0.02, 0.84, 0.14])
table.axis('off')
tbl = table.table(cellText=table_data,
                   colLabels=["Parameter", "FD2NN (Fourier)", "D2NN (Free-space, br15cm)"],
                   cellLoc='center', loc='center',
                   colWidths=[0.22, 0.39, 0.39])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1.0, 1.4)

# Color header
for j in range(3):
    tbl[0, j].set_facecolor('#d5d5d5')
    tbl[0, j].set_text_props(fontweight='bold')

# Highlight problem rows
for i, row in enumerate(table_data, start=1):
    if "overflow" in str(row) or "0.0002" in str(row):
        for j in range(3):
            tbl[i, j].set_facecolor('#fff3cd')

plt.tight_layout(rect=[0, 0.12, 1, 0.96])
outpath = "/root/dj/D2NN/kim2026/runs/figures_fd2nn_vs_d2nn_layout.png"
fig.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {outpath}")
