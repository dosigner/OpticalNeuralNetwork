"""Draw D2NN configuration schematics for Figure 4a."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle

# Colors matching existing visualize_fourier_5l_geometry.py
COLOR_PHASE = "#ff2a78"
COLOR_PHASE_EDGE = "#cc0d57"
COLOR_SBN = "#f39c34"
COLOR_SBN_EDGE = "#d27d17"
COLOR_LENS = "#9ec9e8"
COLOR_LENS_EDGE = "#6a9fc3"
COLOR_AXIS = "#4a4a4a"
COLOR_INPUT = "#222222"
COLOR_SENSOR = "#111111"
COLOR_FOURIER_BG = "#e8f0f8"
COLOR_LABEL = "#111111"


def _draw_input(ax: plt.Axes, x: float, y: float, h: float) -> None:
    ax.plot([x, x], [y - h / 2, y + h / 2], lw=2.5, color=COLOR_INPUT, solid_capstyle="round")


def _draw_sensor(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    ax.add_patch(Rectangle((x, y - h / 2), w, h, facecolor=COLOR_SENSOR, edgecolor=COLOR_SENSOR))


def _draw_phase_layer(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor=COLOR_PHASE, edgecolor=COLOR_PHASE_EDGE, lw=0.8))


def _draw_sbn(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    # SBN drawn as hatched orange block
    ax.add_patch(
        Rectangle((x - w / 2, y - h / 2), w, h, facecolor=COLOR_SBN, edgecolor=COLOR_SBN_EDGE, lw=0.8, hatch="|||")
    )


def _draw_lens(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    ax.add_patch(Ellipse((x, y), w, h, facecolor=COLOR_LENS, edgecolor=COLOR_LENS_EDGE, lw=1.2))


def _draw_axis_line(ax: plt.Axes, x0: float, x1: float, y: float) -> None:
    ax.plot([x0, x1], [y, y], lw=1.0, color=COLOR_AXIS)


def _spacing_annotation(ax: plt.Axes, x0: float, x1: float, y: float, text: str, *, fontsize: int = 7) -> None:
    ax.annotate(
        "", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="<->", lw=0.8, color="#333333")
    )
    ax.text((x0 + x1) / 2, y + 0.025, text, ha="center", va="bottom", fontsize=fontsize, color="#333333")


def _draw_linear_real(ax: plt.Axes) -> None:
    """Linear Real Space (5 Layers), 3mm spacing."""
    y = 0.5
    h = 0.50
    lw = 0.022
    lh = h * 0.85

    x_in = 0.04
    x_layers = [0.18, 0.32, 0.46, 0.60, 0.74]
    x_sensor = 0.92

    _draw_axis_line(ax, x_in, x_sensor + 0.02, y)
    _draw_input(ax, x_in, y, h * 0.7)
    _draw_sensor(ax, x_sensor, y, 0.02, h * 0.7)

    for x in x_layers:
        _draw_phase_layer(ax, x, y, lw, lh)

    # Spacing annotation
    _spacing_annotation(ax, x_layers[-2], x_layers[-1], y - 0.30, "3mm", fontsize=8)

    ax.text(0.50, 0.97, "Linear Real Space (5 Layers)", ha="center", va="top", fontsize=9, fontweight="bold",
            color=COLOR_LABEL)


def _draw_nonlinear_real(ax: plt.Axes) -> None:
    """Nonlinear Real Space (5 Layers), 3mm spacing, SBN per layer."""
    y = 0.5
    h = 0.50
    lw = 0.018
    lh = h * 0.85
    sbn_w = 0.016
    sbn_h = h * 0.75

    x_in = 0.04
    # Phase + SBN pairs, compressed to fit
    x_pairs = [(0.14, 0.17), (0.27, 0.30), (0.40, 0.43), (0.53, 0.56), (0.66, 0.69)]
    x_sensor = 0.92

    _draw_axis_line(ax, x_in, x_sensor + 0.02, y)
    _draw_input(ax, x_in, y, h * 0.7)
    _draw_sensor(ax, x_sensor, y, 0.02, h * 0.7)

    for x_l, x_s in x_pairs:
        _draw_phase_layer(ax, x_l, y, lw, lh)
        _draw_sbn(ax, x_s, y, sbn_w, sbn_h)

    # Spacing annotation
    _spacing_annotation(ax, x_pairs[-2][1], x_pairs[-1][0], y - 0.30, "3mm", fontsize=8)

    ax.text(0.50, 0.97, "Nonlinear Real Space (5 Layers)", ha="center", va="top", fontsize=9, fontweight="bold",
            color=COLOR_LABEL)


def _draw_linear_fourier(ax: plt.Axes) -> None:
    """Linear Fourier Space (5 Layers), 2f system f1=f2=1mm."""
    y = 0.45
    h = 0.45
    lw = 0.018
    lh = h * 0.80
    lens_w = 0.035
    lens_h = h * 0.70

    x_in = 0.02
    x_lens1 = 0.12
    x_layers = [0.28, 0.39, 0.50, 0.61, 0.72]
    x_lens2 = 0.88
    x_sensor = 0.96

    # Fourier domain background
    ax.add_patch(
        FancyBboxPatch(
            (x_lens1 + 0.03, y - h * 0.48), x_lens2 - x_lens1 - 0.06, h * 0.96,
            boxstyle="round,pad=0.01", facecolor=COLOR_FOURIER_BG, edgecolor="none", alpha=0.6,
        )
    )

    _draw_axis_line(ax, x_in, x_sensor + 0.02, y)
    _draw_input(ax, x_in, y, h * 0.6)
    _draw_sensor(ax, x_sensor, y, 0.018, h * 0.6)
    _draw_lens(ax, x_lens1, y, lens_w, lens_h)
    _draw_lens(ax, x_lens2, y, lens_w, lens_h)

    for x in x_layers:
        _draw_phase_layer(ax, x, y, lw, lh)

    # Focal length labels below lenses
    ax.text(x_lens1, y - 0.32, "$f_1$=1mm", ha="center", va="top", fontsize=7, color="#24536f")
    ax.text(x_lens2, y - 0.32, "$f_2$=1mm", ha="center", va="top", fontsize=7, color="#24536f")

    # 2f system span arrows (between title and elements)
    arr_y = y + 0.33
    ax.annotate(
        "", xy=(0.50, arr_y), xytext=(x_in, arr_y),
        arrowprops=dict(arrowstyle="<->", lw=0.8, color="#24536f"),
    )
    ax.text((x_in + 0.50) / 2, arr_y + 0.02, "2$f$ System, $f_1$=1mm", ha="center", va="bottom", fontsize=6.5,
            color="#24536f")
    ax.annotate(
        "", xy=(x_sensor + 0.01, arr_y), xytext=(0.50, arr_y),
        arrowprops=dict(arrowstyle="<->", lw=0.8, color="#24536f"),
    )
    ax.text((0.50 + x_sensor) / 2, arr_y + 0.02, "2$f$ System, $f_2$=1mm", ha="center", va="bottom", fontsize=6.5,
            color="#24536f")

    ax.text(0.50, 0.97, "Linear Fourier Space (5 Layers)", ha="center", va="top", fontsize=9, fontweight="bold",
            color=COLOR_LABEL)


def _draw_nonlinear_fourier(ax: plt.Axes) -> None:
    """Nonlinear Fourier Space (5 Layers), 2f system f1=f2=4mm, SBN rear."""
    y = 0.45
    h = 0.45
    lw = 0.016
    lh = h * 0.78
    sbn_w = 0.018
    sbn_h = h * 0.70
    lens_w = 0.035
    lens_h = h * 0.70

    x_in = 0.02
    x_lens1 = 0.12
    x_layers = [0.26, 0.36, 0.46, 0.56, 0.66]
    x_sbn = 0.73
    x_lens2 = 0.88
    x_sensor = 0.96

    # Fourier domain background
    ax.add_patch(
        FancyBboxPatch(
            (x_lens1 + 0.03, y - h * 0.48), x_lens2 - x_lens1 - 0.06, h * 0.96,
            boxstyle="round,pad=0.01", facecolor=COLOR_FOURIER_BG, edgecolor="none", alpha=0.6,
        )
    )

    _draw_axis_line(ax, x_in, x_sensor + 0.02, y)
    _draw_input(ax, x_in, y, h * 0.6)
    _draw_sensor(ax, x_sensor, y, 0.018, h * 0.6)
    _draw_lens(ax, x_lens1, y, lens_w, lens_h)
    _draw_lens(ax, x_lens2, y, lens_w, lens_h)

    for x in x_layers:
        _draw_phase_layer(ax, x, y, lw, lh)
    _draw_sbn(ax, x_sbn, y, sbn_w, sbn_h)

    # Focal length labels below lenses
    ax.text(x_lens1, y - 0.32, "$f_1$=4mm", ha="center", va="top", fontsize=7, color="#24536f")
    ax.text(x_lens2, y - 0.32, "$f_2$=4mm", ha="center", va="top", fontsize=7, color="#24536f")

    # 2f system span arrows (between title and elements)
    arr_y = y + 0.33
    mid = (x_layers[2] + x_sbn) / 2
    ax.annotate(
        "", xy=(mid, arr_y), xytext=(x_in, arr_y),
        arrowprops=dict(arrowstyle="<->", lw=0.8, color="#24536f"),
    )
    ax.text((x_in + mid) / 2, arr_y + 0.02, "2$f$ System, $f_1$=4mm", ha="center", va="bottom", fontsize=6.5,
            color="#24536f")
    ax.annotate(
        "", xy=(x_sensor + 0.01, arr_y), xytext=(mid, arr_y),
        arrowprops=dict(arrowstyle="<->", lw=0.8, color="#24536f"),
    )
    ax.text((mid + x_sensor) / 2, arr_y + 0.02, "2$f$ System, $f_2$=4mm", ha="center", va="bottom", fontsize=6.5,
            color="#24536f")

    ax.text(0.50, 0.97, "Nonlinear Fourier Space (5 Layers)", ha="center", va="top", fontsize=9, fontweight="bold",
            color=COLOR_LABEL)


_DRAWERS = {
    "linear_real": _draw_linear_real,
    "nonlinear_real": _draw_nonlinear_real,
    "linear_fourier": _draw_linear_fourier,
    "nonlinear_fourier": _draw_nonlinear_fourier,
}


def draw_d2nn_config_schematic(ax: plt.Axes, config_type: str, *, title: str | None = None) -> None:
    """Draw a D2NN configuration schematic on the given axes.

    Parameters
    ----------
    ax : matplotlib Axes
    config_type : one of 'linear_real', 'nonlinear_real', 'linear_fourier', 'nonlinear_fourier'
    title : optional override title
    """
    drawer = _DRAWERS.get(config_type)
    if drawer is None:
        raise ValueError(f"Unknown config_type '{config_type}'. Choose from: {sorted(_DRAWERS)}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")
    drawer(ax)
    if title is not None:
        ax.text(0.50, 0.97, title, ha="center", va="top", fontsize=9, fontweight="bold", color=COLOR_LABEL)
