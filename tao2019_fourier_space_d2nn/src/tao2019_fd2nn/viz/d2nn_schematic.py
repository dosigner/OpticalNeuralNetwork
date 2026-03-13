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


# ── S7 Schematic Drawers ──────────────────────────────────────────


def _s7_layer_positions(num_layers: int, x_start: float = 0.22, x_end: float = 0.78) -> list[float]:
    """Evenly space N layer positions between x_start and x_end."""
    if num_layers == 1:
        return [(x_start + x_end) / 2]
    step = (x_end - x_start) / (num_layers - 1)
    return [x_start + i * step for i in range(num_layers)]


def _draw_s7_linear_fourier(ax: plt.Axes, num_layers: int) -> None:
    """Linear Fourier: Lens1 -> N phase masks -> Lens2."""
    y, h = 0.5, 0.50
    lw, lh = 0.018, h * 0.78
    lens_w, lens_h = 0.035, h * 0.70
    x_in, x_lens1, x_lens2, x_sensor = 0.02, 0.12, 0.88, 0.96
    x_layers = _s7_layer_positions(num_layers, 0.24, 0.76)

    ax.add_patch(FancyBboxPatch(
        (x_lens1 + 0.03, y - h * 0.48), x_lens2 - x_lens1 - 0.06, h * 0.96,
        boxstyle="round,pad=0.01", facecolor=COLOR_FOURIER_BG, edgecolor="none", alpha=0.6))
    _draw_axis_line(ax, x_in, x_sensor + 0.02, y)
    _draw_input(ax, x_in, y, h * 0.6)
    _draw_sensor(ax, x_sensor, y, 0.018, h * 0.6)
    _draw_lens(ax, x_lens1, y, lens_w, lens_h)
    _draw_lens(ax, x_lens2, y, lens_w, lens_h)
    for x in x_layers:
        _draw_phase_layer(ax, x, y, lw, lh)
    if len(x_layers) >= 2 and num_layers <= 5:
        _spacing_annotation(ax, x_layers[-2], x_layers[-1], y - 0.30, "", fontsize=7)
    ax.text(0.50, 0.97, "Linear Fourier", ha="center", va="top", fontsize=8, fontweight="bold", color=COLOR_LABEL)


def _draw_s7_nonlinear_fourier_rear_base(ax: plt.Axes, num_layers: int, *, title: str) -> None:
    """Nonlinear Fourier with a rear SBN: Lens1 -> N phase masks -> SBN -> Lens2."""
    y, h = 0.5, 0.50
    lw, lh = 0.016, h * 0.76
    sbn_w, sbn_h = 0.018, h * 0.68
    lens_w, lens_h = 0.035, h * 0.70
    x_in, x_lens1, x_lens2, x_sensor = 0.02, 0.12, 0.88, 0.96
    x_layers = _s7_layer_positions(num_layers, 0.24, 0.70)
    x_sbn = 0.78

    ax.add_patch(FancyBboxPatch(
        (x_lens1 + 0.03, y - h * 0.48), x_lens2 - x_lens1 - 0.06, h * 0.96,
        boxstyle="round,pad=0.01", facecolor=COLOR_FOURIER_BG, edgecolor="none", alpha=0.6))
    _draw_axis_line(ax, x_in, x_sensor + 0.02, y)
    _draw_input(ax, x_in, y, h * 0.6)
    _draw_sensor(ax, x_sensor, y, 0.018, h * 0.6)
    _draw_lens(ax, x_lens1, y, lens_w, lens_h)
    _draw_lens(ax, x_lens2, y, lens_w, lens_h)
    for x in x_layers:
        _draw_phase_layer(ax, x, y, lw, lh)
    _draw_sbn(ax, x_sbn, y, sbn_w, sbn_h)
    ax.text(0.50, 0.97, title, ha="center", va="top", fontsize=8, fontweight="bold", color=COLOR_LABEL)


def _draw_s7_nonlinear_fourier_single_sbn(ax: plt.Axes, num_layers: int) -> None:
    _draw_s7_nonlinear_fourier_rear_base(ax, num_layers, title="Nonlinear Fourier, Single SBN")


def _draw_s7_nonlinear_fourier_rear(ax: plt.Axes, num_layers: int) -> None:
    _draw_s7_nonlinear_fourier_rear_base(ax, num_layers, title="Nonlinear Fourier, SBN Rear")


def _draw_s7_nonlinear_fourier_per_layer(ax: plt.Axes, num_layers: int) -> None:
    """Nonlinear Fourier, Multi-SBN: Lens1 -> (phase + SBN) x N -> Lens2."""
    y, h = 0.5, 0.50
    lw, lh = 0.014, h * 0.76
    sbn_w, sbn_h = 0.014, h * 0.66
    lens_w, lens_h = 0.035, h * 0.70
    x_in, x_lens1, x_lens2, x_sensor = 0.02, 0.12, 0.88, 0.96
    pair_span = (0.76 - 0.24) / max(num_layers, 1)
    x_pairs = []
    for i in range(num_layers):
        cx = 0.24 + pair_span * (i + 0.5)
        x_pairs.append((cx - 0.012, cx + 0.012))

    ax.add_patch(FancyBboxPatch(
        (x_lens1 + 0.03, y - h * 0.48), x_lens2 - x_lens1 - 0.06, h * 0.96,
        boxstyle="round,pad=0.01", facecolor=COLOR_FOURIER_BG, edgecolor="none", alpha=0.6))
    _draw_axis_line(ax, x_in, x_sensor + 0.02, y)
    _draw_input(ax, x_in, y, h * 0.6)
    _draw_sensor(ax, x_sensor, y, 0.018, h * 0.6)
    _draw_lens(ax, x_lens1, y, lens_w, lens_h)
    _draw_lens(ax, x_lens2, y, lens_w, lens_h)
    for x_l, x_s in x_pairs:
        _draw_phase_layer(ax, x_l, y, lw, lh)
        _draw_sbn(ax, x_s, y, sbn_w, sbn_h)
    ax.text(0.50, 0.97, "Nonlinear Fourier, Multi-SBN",
            ha="center", va="top", fontsize=8, fontweight="bold", color=COLOR_LABEL)


def _draw_s7_nonlinear_fourier_front(ax: plt.Axes, num_layers: int) -> None:
    """Nonlinear Fourier, SBN Front: Lens1 -> SBN -> N phase masks -> Lens2."""
    y, h = 0.5, 0.50
    lw, lh = 0.016, h * 0.76
    sbn_w, sbn_h = 0.018, h * 0.68
    lens_w, lens_h = 0.035, h * 0.70
    x_in, x_lens1, x_lens2, x_sensor = 0.02, 0.12, 0.88, 0.96
    x_sbn = 0.22
    x_layers = _s7_layer_positions(num_layers, 0.30, 0.78)

    ax.add_patch(FancyBboxPatch(
        (x_lens1 + 0.03, y - h * 0.48), x_lens2 - x_lens1 - 0.06, h * 0.96,
        boxstyle="round,pad=0.01", facecolor=COLOR_FOURIER_BG, edgecolor="none", alpha=0.6))
    _draw_axis_line(ax, x_in, x_sensor + 0.02, y)
    _draw_input(ax, x_in, y, h * 0.6)
    _draw_sensor(ax, x_sensor, y, 0.018, h * 0.6)
    _draw_lens(ax, x_lens1, y, lens_w, lens_h)
    _draw_lens(ax, x_lens2, y, lens_w, lens_h)
    _draw_sbn(ax, x_sbn, y, sbn_w, sbn_h)
    for x in x_layers:
        _draw_phase_layer(ax, x, y, lw, lh)
    ax.text(0.50, 0.97, "Nonlinear Fourier, SBN Front",
            ha="center", va="top", fontsize=8, fontweight="bold", color=COLOR_LABEL)


def _draw_s7_hybrid_front(ax: plt.Axes, num_layers: int) -> None:
    """Nonlinear Fourier&Real, SBN Front: SBN -> alternating F/R layers with lenses."""
    y, h = 0.5, 0.50
    lw, lh = 0.012, h * 0.70
    sbn_w, sbn_h = 0.014, h * 0.62
    lens_w, lens_h = 0.025, h * 0.55
    x_in, x_sensor = 0.02, 0.96
    x_sbn = 0.10
    # Alternate: lens-phase-lens pairs, compressed
    total_w = 0.80
    step = total_w / (num_layers + 1)
    elements: list[tuple[str, float]] = []
    for i in range(num_layers):
        cx = 0.16 + step * (i + 0.5)
        if i % 2 == 0:  # fourier layer: lens-phase-lens
            elements.append(("lens", cx - 0.025))
            elements.append(("phase", cx))
            elements.append(("lens", cx + 0.025))
        else:  # real layer: just phase
            elements.append(("phase", cx))

    _draw_axis_line(ax, x_in, x_sensor + 0.02, y)
    _draw_input(ax, x_in, y, h * 0.5)
    _draw_sensor(ax, x_sensor, y, 0.015, h * 0.5)
    _draw_sbn(ax, x_sbn, y, sbn_w, sbn_h)
    for etype, ex in elements:
        if etype == "lens":
            _draw_lens(ax, ex, y, lens_w, lens_h)
        else:
            _draw_phase_layer(ax, ex, y, lw, lh)
    ax.text(0.50, 0.97, "Nonlinear Fourier&Real, SBN Front",
            ha="center", va="top", fontsize=7, fontweight="bold", color=COLOR_LABEL)


def _draw_s7_hybrid_rear(ax: plt.Axes, num_layers: int) -> None:
    """Nonlinear Fourier&Real, SBN Rear: alternating F/R layers with lenses -> SBN."""
    y, h = 0.5, 0.50
    lw, lh = 0.012, h * 0.70
    sbn_w, sbn_h = 0.014, h * 0.62
    lens_w, lens_h = 0.025, h * 0.55
    x_in, x_sensor = 0.02, 0.96
    x_sbn = 0.90
    total_w = 0.78
    step = total_w / (num_layers + 1)
    elements: list[tuple[str, float]] = []
    for i in range(num_layers):
        cx = 0.10 + step * (i + 0.5)
        if i % 2 == 0:
            elements.append(("lens", cx - 0.025))
            elements.append(("phase", cx))
            elements.append(("lens", cx + 0.025))
        else:
            elements.append(("phase", cx))

    _draw_axis_line(ax, x_in, x_sensor + 0.02, y)
    _draw_input(ax, x_in, y, h * 0.5)
    _draw_sensor(ax, x_sensor, y, 0.015, h * 0.5)
    for etype, ex in elements:
        if etype == "lens":
            _draw_lens(ax, ex, y, lens_w, lens_h)
        else:
            _draw_phase_layer(ax, ex, y, lw, lh)
    _draw_sbn(ax, x_sbn, y, sbn_w, sbn_h)
    ax.text(0.50, 0.97, "Nonlinear Fourier&Real, SBN Rear",
            ha="center", va="top", fontsize=7, fontweight="bold", color=COLOR_LABEL)


_S7_DRAWERS = {
    "linear_fourier": _draw_s7_linear_fourier,
    "nonlinear_fourier_single_sbn": _draw_s7_nonlinear_fourier_single_sbn,
    "nonlinear_fourier_multi_sbn": _draw_s7_nonlinear_fourier_per_layer,
    "nonlinear_fourier_sbn_front": _draw_s7_nonlinear_fourier_front,
    "nonlinear_fourier_sbn_rear": _draw_s7_nonlinear_fourier_rear,
    "hybrid_sbn_front": _draw_s7_hybrid_front,
    "hybrid_sbn_rear": _draw_s7_hybrid_rear,
}


def draw_s7_schematic(ax: plt.Axes, config_key: str, *, num_layers: int = 5) -> None:
    """Draw an S7-style D2NN schematic on the given axes."""
    drawer = _S7_DRAWERS.get(config_key)
    if drawer is None:
        raise ValueError(f"Unknown S7 config_key '{config_key}'. Choose from: {sorted(_S7_DRAWERS)}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")
    drawer(ax, num_layers)


def _paper_s7_terminal(ax: plt.Axes, x: float, *, y0: float = 0.34, y1: float = 0.74) -> None:
    ax.plot([x, x], [y0, y1], lw=1.6, color=COLOR_INPUT, solid_capstyle="butt")


def _paper_s7_phase_cluster_positions(num_layers: int, *, center: float = 0.50, span: float = 0.08) -> list[float]:
    if num_layers <= 1:
        return [center]
    step = span / (num_layers - 1)
    start = center - span / 2
    return [start + i * step for i in range(num_layers)]


def _paper_s7_draw_spacing(ax: plt.Axes, x0: float, x1: float, y: float = 0.52) -> None:
    ax.annotate(
        "",
        xy=(x1, y),
        xytext=(x0, y),
        arrowprops=dict(arrowstyle="<|-|>", lw=0.8, color="#111111", shrinkA=0, shrinkB=0),
    )


def _paper_s7_linear_fourier(ax: plt.Axes, num_layers: int) -> None:
    _paper_s7_terminal(ax, 0.16)
    _paper_s7_terminal(ax, 0.84)
    _draw_lens(ax, 0.32, 0.52, 0.040, 0.34)
    _draw_lens(ax, 0.68, 0.52, 0.040, 0.34)
    phases = _paper_s7_phase_cluster_positions(num_layers, center=0.50, span=0.10)
    for x in phases:
        _draw_phase_layer(ax, x, 0.52, 0.010, 0.32)
    if phases:
        _paper_s7_draw_spacing(ax, 0.18, phases[0] - 0.02)
        _paper_s7_draw_spacing(ax, phases[-1] + 0.02, 0.82)


def _paper_s7_nonlinear_fourier_single(ax: plt.Axes, num_layers: int) -> None:
    _paper_s7_terminal(ax, 0.16)
    _paper_s7_terminal(ax, 0.84)
    _draw_lens(ax, 0.32, 0.52, 0.040, 0.34)
    _draw_lens(ax, 0.68, 0.52, 0.040, 0.34)
    phases = _paper_s7_phase_cluster_positions(num_layers, center=0.48, span=0.10)
    for x in phases:
        _draw_phase_layer(ax, x, 0.52, 0.010, 0.32)
    x_sbn = 0.56 if num_layers > 1 else 0.52
    _draw_sbn(ax, x_sbn, 0.52, 0.018, 0.30)
    if phases:
        _paper_s7_draw_spacing(ax, 0.18, phases[0] - 0.02)
    _paper_s7_draw_spacing(ax, x_sbn + 0.02, 0.82)


def _paper_s7_nonlinear_fourier_multi(ax: plt.Axes, num_layers: int) -> None:
    _paper_s7_terminal(ax, 0.16)
    _paper_s7_terminal(ax, 0.84)
    _draw_lens(ax, 0.32, 0.52, 0.040, 0.34)
    _draw_lens(ax, 0.68, 0.52, 0.040, 0.34)
    centers = _paper_s7_phase_cluster_positions(num_layers, center=0.50, span=0.16)
    for x in centers:
        _draw_sbn(ax, x, 0.52, 0.022, 0.30)
        _draw_phase_layer(ax, x, 0.52, 0.010, 0.32)
    if centers:
        _paper_s7_draw_spacing(ax, 0.18, centers[0] - 0.025)
        _paper_s7_draw_spacing(ax, centers[-1] + 0.025, 0.82)


_S7_PAPER_DRAWERS = {
    "linear_fourier": _paper_s7_linear_fourier,
    "nonlinear_fourier": _paper_s7_nonlinear_fourier_single,
    "nonlinear_fourier_single_sbn": _paper_s7_nonlinear_fourier_single,
    "nonlinear_fourier_multi_sbn": _paper_s7_nonlinear_fourier_multi,
}


def draw_s7_paper_inset(
    ax: plt.Axes,
    config_key: str,
    *,
    num_layers: int,
    border_color: str,
    label: str,
) -> None:
    """Draw the compact schematic inset used by Supplementary Fig. S7(c)(d)."""

    drawer = _S7_PAPER_DRAWERS.get(config_key)
    if drawer is None:
        raise ValueError(f"Unknown paper S7 config_key '{config_key}'. Choose from: {sorted(_S7_PAPER_DRAWERS)}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        Rectangle(
            (0.02, 0.24),
            0.96,
            0.62,
            fill=False,
            edgecolor=border_color,
            lw=1.6,
            linestyle=(0, (1.0, 1.2)),
        )
    )
    drawer(ax, num_layers)
    ax.text(0.50, 0.06, label, ha="center", va="bottom", fontsize=8, color=COLOR_LABEL)
