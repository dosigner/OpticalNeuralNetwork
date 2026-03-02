"""Visualize per-component propagation distances for Fourier-space 5-layer models."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = PROJECT_ROOT / "src" / "tao2019_fd2nn" / "config"

LINEAR_CFG = CFG_DIR / "cls_mnist_linear_fourier_5l_f1mm.yaml"
NONLINEAR_CFG = CFG_DIR / "cls_mnist_nonlinear_fourier_5l_f4mm.yaml"
OUTPUT = PROJECT_ROOT / "fig_fourier_5l_component_distances.png"


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be mapping: {path}")
    return data


def _get_params(cfg: dict) -> tuple[float, float, float, float]:
    dual = cfg["optics"]["dual_2f"]
    f1_mm = float(dual["f1_m"]) * 1e3
    f2_mm = float(dual["f2_m"]) * 1e3
    z_layer_um = float(cfg["optics"]["propagation"]["layer_spacing_m"]) * 1e6
    # Current model uses same value for z_out as layer_spacing_m in build_model.
    z_out_um = float(cfg["optics"]["propagation"]["layer_spacing_m"]) * 1e6
    return f1_mm, f2_mm, z_layer_um, z_out_um


def _double_arrow(ax: plt.Axes, x0: float, x1: float, y: float, text: str, *, fs: int = 11) -> None:
    ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="<->", lw=1.6, color="#222222"))
    ax.text((x0 + x1) * 0.5, y + 0.022, text, ha="center", va="bottom", fontsize=fs, color="#111111")


def _vertical_marker(ax: plt.Axes, x: float, y: float, h: float, *, text: str, color: str = "#666666") -> None:
    ax.plot([x, x], [y - h / 2, y + h / 2], lw=1.0, ls="--", color=color, alpha=0.8)
    ax.text(x, y + h / 2 + 0.015, text, ha="center", va="bottom", fontsize=8, color=color)


def _segment_labels(ax: plt.Axes, y: float, segments: Iterable[tuple[float, float, str]]) -> None:
    base_y = y - 0.165
    for i, (x0, x1, text) in enumerate(segments):
        row = i % 2
        yy = base_y - row * 0.04
        _double_arrow(ax, x0, x1, yy, text, fs=8)


def _draw_row(
    ax: plt.Axes,
    *,
    y: float,
    title: str,
    f1_mm: float,
    f2_mm: float,
    z_layer_um: float,
    z_out_um: float,
    nonlinear: bool,
) -> None:
    x_in = 0.04
    x_lens1 = 0.14
    x_f0 = 0.24  # Fourier-plane reference right after first 2f conversion
    x_layers = [0.34, 0.43, 0.52, 0.61, 0.70]
    x_sbn = 0.76
    x_f2in = 0.82  # Plane fed into inverse 2f
    x_lens2 = 0.90
    x_sensor = 0.97
    layer_w = 0.012
    layer_h = 0.18

    # Optical axis
    ax.plot([x_in, x_sensor + 0.012], [y, y], lw=1.4, color="#4a4a4a")

    # Input and sensor planes
    ax.plot([x_in, x_in], [y - 0.09, y + 0.09], lw=3.0, color="#222222")
    ax.add_patch(Rectangle((x_sensor, y - 0.09), 0.012, 0.18, facecolor="#111111", edgecolor="#111111"))
    ax.text(x_in, y - 0.11, "Input", ha="center", va="top", fontsize=9, color="#111111")
    ax.text(x_sensor + 0.006, y - 0.11, "Sensor", ha="center", va="top", fontsize=9, color="#111111")

    # Lenses
    lens_color = "#9ec9e8"
    ax.add_patch(Ellipse((x_lens1, y), 0.03, 0.16, facecolor=lens_color, edgecolor="#6a9fc3", lw=1.2))
    ax.add_patch(Ellipse((x_lens2, y), 0.03, 0.16, facecolor=lens_color, edgecolor="#6a9fc3", lw=1.2))
    ax.text(x_lens1, y - 0.11, "Lens 1", ha="center", va="top", fontsize=9, color="#24536f")
    ax.text(x_lens2, y - 0.11, "Lens 2", ha="center", va="top", fontsize=9, color="#24536f")

    _vertical_marker(ax, x_f0, y, 0.16, text="F0")
    _vertical_marker(ax, x_f2in, y, 0.16, text="F2-in")

    # Fourier-space phase layers
    for x in x_layers:
        ax.add_patch(
            Rectangle((x - layer_w / 2, y - layer_h / 2), layer_w, layer_h, facecolor="#ff2a78", edgecolor="#cc0d57", lw=0.8)
        )
    for i, x in enumerate(x_layers, start=1):
        ax.text(x, y + 0.105, f"L{i}", ha="center", va="bottom", fontsize=8, color="#a10f46")

    # Rear SBN for nonlinear Fourier model
    if nonlinear:
        sbn_w = 0.02
        ax.add_patch(Rectangle((x_sbn - sbn_w / 2, y - layer_h / 2), sbn_w, layer_h, facecolor="#f39c34", edgecolor="#d27d17", lw=1.0))
        ax.text(x_sbn, y - 0.11, "SBN", ha="center", va="top", fontsize=9, color="#7a3f00")

    # 2f system spans
    end_first_2f = x_sbn if nonlinear else x_layers[-1]
    _double_arrow(ax, x_in, end_first_2f, y + 0.16, f"2f System 1 ($f_1={f1_mm:.0f}$ mm)")
    _double_arrow(ax, end_first_2f, x_sensor + 0.012, y + 0.16, f"2f System 2 ($f_2={f2_mm:.0f}$ mm)")

    # Per-component segment distances (model-equivalent)
    segments = [
        (x_in, x_lens1, f"in→lens1: {f1_mm:.0f} mm"),
        (x_lens1, x_f0, f"lens1→F0: {f1_mm:.0f} mm"),
        (x_f0, x_layers[0], f"F0→L1: {z_layer_um:.0f} µm"),
        (x_layers[0], x_layers[1], f"L1→L2: {z_layer_um:.0f} µm"),
        (x_layers[1], x_layers[2], f"L2→L3: {z_layer_um:.0f} µm"),
        (x_layers[2], x_layers[3], f"L3→L4: {z_layer_um:.0f} µm"),
        (x_layers[3], x_layers[4], f"L4→L5: {z_layer_um:.0f} µm"),
    ]
    if nonlinear:
        segments.append((x_layers[4], x_sbn, "L5→SBN: 0 µm (rear)"))
        segments.append((x_sbn, x_f2in, f"SBN→F2-in: {z_out_um:.0f} µm"))
    else:
        segments.append((x_layers[4], x_f2in, f"L5→F2-in: {z_out_um:.0f} µm"))
    segments.append((x_f2in, x_lens2, f"F2-in→lens2: {f2_mm:.0f} mm"))
    segments.append((x_lens2, x_sensor + 0.006, f"lens2→sensor: {f2_mm:.0f} mm"))
    _segment_labels(ax, y, segments)

    # Row title and distance note
    ax.text(0.50, y + 0.235, title, ha="center", va="bottom", fontsize=14, fontweight="bold", color="#111111")
    ax.text(
        0.50,
        y + 0.205,
        f"Layer spacing={z_layer_um:.0f} µm, output gap={z_out_um:.0f} µm, nonlinear rear SBN={'ON' if nonlinear else 'OFF'}",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#333333",
    )


def main() -> None:
    linear_cfg = _load_yaml(LINEAR_CFG)
    nonlinear_cfg = _load_yaml(NONLINEAR_CFG)

    f1_lin, f2_lin, z_lin, z_out_lin = _get_params(linear_cfg)
    f1_non, f2_non, z_non, z_out_non = _get_params(nonlinear_cfg)

    fig, ax = plt.subplots(figsize=(18, 9))
    fig.patch.set_facecolor("#f3f3f3")
    ax.set_facecolor("#f8f8f8")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    _draw_row(
        ax,
        y=0.74,
        title="Linear Fourier Space (5 Layers): Component-to-Component Distances",
        f1_mm=f1_lin,
        f2_mm=f2_lin,
        z_layer_um=z_lin,
        z_out_um=z_out_lin,
        nonlinear=False,
    )
    _draw_row(
        ax,
        y=0.30,
        title="Nonlinear Fourier Space (5 Layers): Component-to-Component Distances",
        f1_mm=f1_non,
        f2_mm=f2_non,
        z_layer_um=z_non,
        z_out_um=z_out_non,
        nonlinear=True,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(str(OUTPUT))


if __name__ == "__main__":
    main()
