"""Generate side-view optical system geometry diagram for Section 3."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(figsize=(14, 5))

# System parameters (mm)
z_object = 0.0
z_diffuser = 40.0
z_layer1 = 42.0
z_layer2 = 44.0
z_layer3 = 46.0
z_layer4 = 48.0
z_output = 55.0

planes = [
    (z_object, "Object\nPlane", "#2196F3", 30),
    (z_diffuser, "Random\nDiffuser", "#FF9800", 28),
    (z_layer1, "L1", "#4CAF50", 24),
    (z_layer2, "L2", "#4CAF50", 24),
    (z_layer3, "L3", "#4CAF50", 24),
    (z_layer4, "L4", "#4CAF50", 24),
    (z_output, "Output\n(Detector)", "#F44336", 26),
]

# Draw optical axis
ax.axhline(0, color="#888888", linewidth=1, linestyle="-", zorder=0)

# Draw planes as vertical rectangles
for z, label, color, half_h in planes:
    rect = mpatches.FancyBboxPatch(
        (z - 0.4, -half_h), 0.8, 2 * half_h,
        boxstyle="round,pad=0.1",
        facecolor=color, edgecolor="black", linewidth=1.5, alpha=0.75, zorder=2
    )
    ax.add_patch(rect)
    ax.text(z, half_h + 4, label, ha="center", va="bottom", fontsize=9,
            fontweight="bold", color=color, zorder=3)

# Draw distance annotations with arrows
distances = [
    (z_object, z_diffuser, "40 mm", -20),
    (z_diffuser, z_layer1, "2 mm", -26),
    (z_layer1, z_layer2, "2 mm", -32),
    (z_layer4, z_output, "7 mm", -26),
]

for z1, z2, text, y_pos in distances:
    ax.annotate(
        "", xy=(z2, y_pos), xytext=(z1, y_pos),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.5),
    )
    ax.text((z1 + z2) / 2, y_pos - 3.5, text, ha="center", va="top",
            fontsize=9, fontweight="bold", color="#333333",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.9))

# Bracket for layer-to-layer region
ax.annotate(
    "", xy=(z_layer4, -38), xytext=(z_layer1, -38),
    arrowprops=dict(arrowstyle="<->", color="#4CAF50", lw=1.5),
)
ax.text((z_layer1 + z_layer4) / 2, -41, "4 layers, 2 mm spacing\n(D2NN trainable phases)",
        ha="center", va="top", fontsize=8, color="#4CAF50", style="italic")

# Grid size annotation at object plane
ax.text(z_object, -36, "240 x 240 px\npitch = 0.3 mm\n= 72 x 72 mm",
        ha="center", va="top", fontsize=7.5, color="#2196F3",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", edgecolor="#2196F3", alpha=0.8))

# Wavelength annotation
ax.text(27.5, 36, r"$\lambda$ = 0.75 mm (400 GHz THz)",
        ha="center", va="top", fontsize=10, fontweight="bold", color="#333333",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF9C4", edgecolor="#F9A825", alpha=0.9))

# Total distance
ax.annotate(
    "", xy=(z_output, 38), xytext=(z_object, 38),
    arrowprops=dict(arrowstyle="<->", color="#666666", lw=1.2),
)
ax.text((z_object + z_output) / 2, 42, "Total path: 55 mm (~73.3$\\lambda$)",
        ha="center", va="bottom", fontsize=9, color="#666666")

# Draw diverging beam (schematic)
beam_y_top = np.array([15, 22, 23, 23, 23, 23, 24])
beam_y_bot = -beam_y_top
beam_x = np.array([z_object, z_diffuser, z_layer1, z_layer2, z_layer3, z_layer4, z_output])
ax.fill_between(beam_x, beam_y_bot, beam_y_top, alpha=0.06, color="#FF5722", zorder=0)

ax.set_xlim(-5, 60)
ax.set_ylim(-48, 50)
ax.set_xlabel("Propagation distance z (mm)", fontsize=11)
ax.set_title("D2NN System Geometry (Side View) — Luo et al. 2022", fontsize=13, fontweight="bold", pad=15)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig("/root/dj/D2NN/docs/images/fig_system_geometry.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved fig_system_geometry.png")
