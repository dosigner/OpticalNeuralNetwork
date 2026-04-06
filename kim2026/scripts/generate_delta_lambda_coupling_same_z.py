#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kim2026.viz.mpl_fonts import configure_matplotlib_fonts, ensure_output_dir

WAVELENGTH_M = 1.55e-6
DEFAULT_Z_MM = 10.0


@dataclass(frozen=True)
class CouplingCase:
    title: str
    subtitle: str
    delta_m: float
    panel_fill: str
    accent: str
    status_text: str
    status_color: str
    interpretation: str


CASES = (
    CouplingCase(
        title="δ=2 mm (거시적)",
        subtitle="physical-scale pixel",
        delta_m=2.0e-3,
        panel_fill="#f8dfe6",
        accent="#dc2626",
        status_text="X 결합 거의 없음",
        status_color="#dc2626",
        interpretation="ray-like; 사실상 pixel-wise phase shift",
    ),
    CouplingCase(
        title="δ=150 μm (물리 수신면)",
        subtitle="telescope/receiver plane",
        delta_m=150.0e-6,
        panel_fill="#fdeccf",
        accent="#ea580c",
        status_text="WARN 결합 매우 약함",
        status_color="#ea580c",
        interpretation="sub-pixel coupling only",
    ),
    CouplingCase(
        title="δ=10 μm (메타표면)",
        subtitle="metasurface-scale pixel",
        delta_m=10.0e-6,
        panel_fill="#e4f3d5",
        accent="#ef4444",
        status_text="OK 강한 회절 혼합",
        status_color="#5b8a3c",
        interpretation="layer-wide diffractive mixing",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a same-z delta/lambda coupling conceptual diagram."
    )
    parser.add_argument(
        "--z-mm",
        type=float,
        default=DEFAULT_Z_MM,
        help="Common inter-layer spacing in mm for all panels.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "docs" / "04-report" / "diagrams",
        help="Output directory for PNG and SVG assets.",
    )
    return parser.parse_args()


def delta_over_lambda(delta_m: float, wavelength_m: float = WAVELENGTH_M) -> float:
    return delta_m / wavelength_m


def coupling_pixels(delta_m: float, z_m: float, wavelength_m: float = WAVELENGTH_M) -> float:
    return wavelength_m * z_m / (2.0 * delta_m * delta_m)


def format_px(value: float) -> str:
    if value < 0.01:
        return f"{value:.3f}"
    if value < 0.1:
        return f"{value:.2f}"
    if value < 10:
        return f"{value:.2f}"
    return f"{value:.1f}"


def add_grid(ax: plt.Axes, x0: float, y0: float, size: float) -> tuple[float, float, float]:
    cols = rows = 7
    pitch = size / cols
    colors = ("#4a76c2", "#abc0e6")
    for r in range(rows):
        for c in range(cols):
            color = colors[(r + c) % 2]
            ax.add_patch(
                Rectangle(
                    (x0 + c * pitch, y0 + (rows - 1 - r) * pitch),
                    pitch * 0.92,
                    pitch * 0.92,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.3,
                )
            )
    center_x = x0 + (cols // 2 + 0.46) * pitch
    top_y = y0 + size
    return center_x, top_y, pitch


def add_coupling_fan(
    ax: plt.Axes,
    center_x: float,
    top_y: float,
    panel_x: float,
    panel_w: float,
    case: CouplingCase,
    coupling_px: float,
) -> None:
    y_start = top_y - 0.02
    y_end = top_y + 0.22
    if coupling_px < 0.05:
        x_targets = [center_x]
    elif coupling_px < 1.0:
        x_targets = [center_x - 0.15, center_x, center_x + 0.15]
    elif coupling_px < 10.0:
        x_targets = [center_x - 0.5, center_x - 0.2, center_x, center_x + 0.2, center_x + 0.5]
    else:
        span = panel_w * 0.33
        x_targets = [
            center_x - span,
            center_x - span * 0.65,
            center_x - span * 0.35,
            center_x,
            center_x + span * 0.35,
            center_x + span * 0.65,
            center_x + span,
        ]
    for x_t in x_targets:
        ax.add_patch(
            FancyArrowPatch(
                (center_x, y_start),
                (x_t, y_end),
                arrowstyle="->",
                mutation_scale=10,
                linewidth=1.5,
                color=case.accent,
                alpha=0.7,
            )
        )

def draw_panel(ax: plt.Axes, x0: float, y0: float, w: float, h: float, case: CouplingCase, z_mm: float) -> None:
    z_m = z_mm * 1e-3
    ratio = delta_over_lambda(case.delta_m)
    coupling_px = coupling_pixels(case.delta_m, z_m)

    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=case.panel_fill,
            edgecolor="#8f8f8f",
            linewidth=1.2,
        )
    )
    ax.text(x0 + w / 2, y0 + h - 0.18, case.title, ha="center", va="center", fontsize=15, weight="bold")
    ax.text(x0 + w / 2, y0 + h - 0.44, f"δ/λ={ratio:.1f}", ha="center", va="center", fontsize=14)
    ax.text(
        x0 + w / 2,
        y0 + h - 0.70,
        f"N_coupling ≈ {format_px(coupling_px)} px",
        ha="center",
        va="center",
        fontsize=12,
        color=case.accent,
        weight="bold",
    )

    grid_size = min(w * 0.5, h * 0.58)
    grid_x = x0 + (w - grid_size) / 2
    grid_y = y0 + 0.92
    center_x, top_y, _ = add_grid(ax, grid_x, grid_y, grid_size)

    ax.text(center_x, top_y - 0.22, "Layer 2", ha="center", va="center", fontsize=11, color="#6b7280")
    ax.text(center_x, grid_y - 0.22, "Layer 1 (위상 마스크)", ha="center", va="top", fontsize=11, color="#2f2f2f")
    add_coupling_fan(ax, center_x, top_y, x0, w, case, coupling_px)

    ax.text(
        x0 + w / 2,
        y0 + 0.40,
        case.interpretation,
        ha="center",
        va="center",
        fontsize=10.5,
        color="#4b5563",
        style="italic",
    )
    ax.text(
        x0 + w / 2,
        y0 + 0.12,
        case.status_text,
        ha="center",
        va="center",
        fontsize=14,
        color=case.status_color,
        weight="bold",
    )


def main() -> None:
    args = parse_args()
    out_dir = ensure_output_dir(args.out_dir)
    preferred_font = Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
    if preferred_font.exists():
        font_manager.fontManager.addfont(str(preferred_font))
        matplotlib.rcParams["font.family"] = "NanumGothic"
        matplotlib.rcParams["axes.unicode_minus"] = False
        font_family = "NanumGothic"
    else:
        font_family = configure_matplotlib_fonts()
    print(f"Using font: {font_family}")

    fig, ax = plt.subplots(figsize=(17.9, 6.2), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    title = "동일한 z=10 mm에서 δ만 바꾸면 층간 회절 결합이 어떻게 달라지는가"
    subtitle = (
        "Assumption: λ = 1.55 μm, same inter-layer gap z = 10 mm, "
        "N_coupling = λz / (2δ²) ∝ 1/δ²"
    )
    ax.text(9, 5.92, title, ha="center", va="bottom", fontsize=20, weight="bold")
    ax.text(9, 5.70, subtitle, ha="center", va="bottom", fontsize=12, color="#5f6b7a")

    panel_w = 5.6
    gap = 0.5
    left = 0.2
    y0 = 0.35
    h = 4.9
    for idx, case in enumerate(CASES):
        draw_panel(ax, left + idx * (panel_w + gap), y0, panel_w, h, case, args.z_mm)

    ratio = (CASES[0].delta_m / CASES[-1].delta_m) ** 2
    ax.text(
        9,
        0.03,
        f"At fixed z, reducing δ from 2 mm to 10 μm increases N_coupling by {ratio:,.0f}×. "
        "The same geometry changes from near-transparent propagation to strong diffractive mixing.",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#374151",
    )

    png_path = out_dir / "fig3_delta_lambda_coupling_same_z10mm.png"
    svg_path = out_dir / "fig3_delta_lambda_coupling_same_z10mm.svg"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()
