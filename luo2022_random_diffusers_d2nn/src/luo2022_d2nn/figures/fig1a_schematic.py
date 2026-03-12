"""Fig 1a — D2NN training schematic diagram.

Generates a publication-quality diagram of the optical system layout:
object plane -> diffuser -> 4 phase layers -> output/detector plane.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from luo2022_d2nn.utils.viz import save_figure


def _draw_plane(ax, x: float, label: str, color: str = "steelblue",
                height: float = 1.6, y_center: float = 0.0,
                linewidth: float = 2.5, label_offset: float = -1.15):
    """Draw a vertical plane (rectangle) at position x."""
    rect = mpatches.FancyBboxPatch(
        (x - 0.08, y_center - height / 2),
        0.16, height,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor="black",
        linewidth=linewidth,
        alpha=0.85,
    )
    ax.add_patch(rect)
    ax.text(x, y_center + label_offset, label, ha="center", va="top",
            fontsize=8, fontweight="bold")


def _draw_distance_arrow(ax, x_start: float, x_end: float, y: float,
                         label: str):
    """Draw a double-headed arrow with distance label."""
    ax.annotate(
        "", xy=(x_end, y), xytext=(x_start, y),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.2),
    )
    ax.text(
        (x_start + x_end) / 2, y + 0.08, label,
        ha="center", va="bottom", fontsize=7, color="black",
    )


def _draw_diffuser_texture(ax, x: float, y_center: float = 0.0,
                           height: float = 1.6):
    """Overlay a random-phase texture on the diffuser plane."""
    rng = np.random.default_rng(42)
    n_pts = 60
    y_vals = np.linspace(y_center - height / 2 + 0.05,
                         y_center + height / 2 - 0.05, n_pts)
    offsets = rng.normal(0, 0.03, size=n_pts)
    ax.plot(x + offsets, y_vals, color="darkblue", linewidth=0.8, alpha=0.6)


def make_fig1a(save_path: Optional[str] = None):
    """Generate Fig 1a: D2NN training schematic.

    Shows the full optical path:
    - Object plane (left)
    - 40 mm propagation to random diffuser
    - 2 mm to first D2NN layer
    - 4 phase layers with 2 mm spacing
    - 7 mm to output/detector plane

    All distances are labelled.  Uses matplotlib patches and arrows.

    Parameters
    ----------
    save_path : str or None
        If given, saves the figure (PNG) and raw layout data (.npy).
    """
    fig, ax = plt.subplots(figsize=(12, 3.5))

    # --- Layout positions (arbitrary horizontal units, proportional to mm) ---
    # Scale: 1 unit ~ 5 mm for readability
    x_obj = 0.0
    x_diff = 8.0       # 40 mm
    x_layer1 = 8.4      # +2 mm
    x_layer2 = 8.8      # +2 mm
    x_layer3 = 9.2      # +2 mm
    x_layer4 = 9.6      # +2 mm
    x_output = 11.0      # +7 mm

    y_center = 0.0
    plane_h = 1.6
    layer_h = 1.3

    # Object plane
    _draw_plane(ax, x_obj, "Object\nplane", color="#5dade2", height=plane_h)

    # Diffuser
    _draw_plane(ax, x_diff, "Random\ndiffuser", color="#f39c12", height=plane_h)
    _draw_diffuser_texture(ax, x_diff, y_center, plane_h)

    # Phase layers (D2NN)
    layer_colors = ["#2ecc71", "#27ae60", "#1abc9c", "#16a085"]
    for i, (x_l, c) in enumerate(zip(
            [x_layer1, x_layer2, x_layer3, x_layer4], layer_colors)):
        _draw_plane(ax, x_l, f"Layer {i + 1}", color=c, height=layer_h,
                    label_offset=-0.95)

    # Output / detector plane
    _draw_plane(ax, x_output, "Output\nplane", color="#e74c3c", height=plane_h)

    # --- Propagation arrows (beam path) ---
    arrow_y = y_center
    ax.annotate(
        "", xy=(x_output + 0.15, arrow_y), xytext=(x_obj + 0.15, arrow_y),
        arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.5, alpha=0.4),
    )

    # --- Distance annotations ---
    ann_y = -plane_h / 2 - 0.35
    _draw_distance_arrow(ax, x_obj, x_diff, ann_y, "40 mm")
    _draw_distance_arrow(ax, x_diff, x_layer1, ann_y - 0.35, "2 mm")

    # Inter-layer: brace-style single label
    _draw_distance_arrow(ax, x_layer1, x_layer2, ann_y - 0.35, "2 mm")
    ax.text(
        (x_layer2 + x_layer3) / 2, ann_y - 0.28, "2 mm",
        ha="center", va="top", fontsize=6, color="gray",
    )
    ax.text(
        (x_layer3 + x_layer4) / 2, ann_y - 0.28, "2 mm",
        ha="center", va="top", fontsize=6, color="gray",
    )

    _draw_distance_arrow(ax, x_layer4, x_output, ann_y, "7 mm")

    # --- Title ---
    ax.set_title("Fig 1a: D2NN Training Schematic (Luo et al. 2022)",
                 fontsize=11, fontweight="bold", pad=12)

    # --- Formatting ---
    ax.set_xlim(-1.0, x_output + 1.5)
    ax.set_ylim(-2.2, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()

    if save_path is not None:
        save_figure(fig, save_path)
        # Save layout metadata
        npy_path = Path(save_path).with_suffix(".npy")
        layout = {
            "x_obj": x_obj, "x_diff": x_diff,
            "x_layers": [x_layer1, x_layer2, x_layer3, x_layer4],
            "x_output": x_output,
            "distances_mm": [40.0, 2.0, 2.0, 2.0, 2.0, 7.0],
        }
        np.save(str(npy_path), layout)

    plt.close(fig)
    return fig
