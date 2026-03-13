"""Classifier visualization helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from d2nn.detectors.layout import DetectorLayout
from d2nn.viz.style import apply_style, extent_mm as _extent_mm


def plot_output_with_detectors(
    intensity: np.ndarray,
    layout: DetectorLayout,
    *,
    dx: float,
    title: str = "Output intensity with detectors",
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot output intensity and detector boxes."""

    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.patches import Rectangle

    apply_style()
    arr = np.asarray(intensity)

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    im = ax.imshow(np.log10(arr + 1e-8), cmap="gray", extent=_extent_mm(arr.shape[0], dx), origin="lower")

    for idx, region in enumerate(layout.regions):
        cx, cy = region.center_xy
        w, h = region.size_xy
        rect = Rectangle(
            ((cx - w / 2.0) * 1e3, (cy - h / 2.0) * 1e3),
            w * 1e3,
            h * 1e3,
            fill=False,
            edgecolor="#ff2d2d",
            linewidth=1.8,
            linestyle=(0, (4, 3)),
        )
        ax.add_patch(rect)
        txt = ax.text(
            cx * 1e3,
            (cy + h / 2.0) * 1e3 + 1.2,
            str(idx),
            color="white",
            ha="center",
            va="bottom",
            fontsize=8,
            weight="bold",
        )
        txt.set_path_effects([pe.withStroke(linewidth=1.5, foreground="black")])

    ax.set_title(title)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(I)")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig, ax


def plot_confusion_matrix(
    cm: np.ndarray,
    *,
    normalize: bool = False,
    annotate: bool = True,
    title: str = "Confusion matrix",
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot confusion matrix with optional row normalization."""

    import matplotlib.pyplot as plt

    apply_style()
    arr_counts = cm.astype(np.float64)
    arr = arr_counts.copy()
    if normalize:
        denom = arr.sum(axis=1, keepdims=True)
        arr = np.divide(arr, np.maximum(denom, 1.0))

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(arr, cmap="Blues", origin="upper")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)

    if annotate:
        threshold = float(arr.max()) * 0.55 if arr.size > 0 else 0.0
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if normalize:
                    txt = f"{arr[i, j] * 100:.1f}%"
                else:
                    txt = f"{int(arr_counts[i, j])}"
                color = "white" if arr[i, j] >= threshold else "black"
                ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=8)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig, ax


def plot_energy_distribution_heatmap(
    energy_matrix: np.ndarray,
    *,
    annotate: bool = True,
    title: str = "Detector energy distribution",
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot class-vs-detector energy matrix heatmap."""

    import matplotlib.pyplot as plt

    apply_style()
    arr = np.asarray(energy_matrix, dtype=np.float64)
    row_sum = arr.sum(axis=1, keepdims=True)
    arr_pct = np.divide(arr, np.maximum(row_sum, 1e-12)) * 100.0

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(arr_pct, cmap="magma", aspect="auto", origin="upper")
    ax.set_title(title)
    ax.set_xlabel("Detector index")
    ax.set_ylabel("Class / sample")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("energy [%]")

    if annotate:
        threshold = float(arr_pct.max()) * 0.6 if arr_pct.size > 0 else 0.0
        for i in range(arr_pct.shape[0]):
            for j in range(arr_pct.shape[1]):
                color = "white" if arr_pct[i, j] >= threshold else "black"
                ax.text(j, i, f"{arr_pct[i, j]:.1f}%", ha="center", va="center", color=color, fontsize=7)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig, ax


def plot_inference_summary(
    input_map: np.ndarray,
    output_intensity: np.ndarray,
    layout: DetectorLayout,
    energies: np.ndarray,
    *,
    dx: float,
    input_title: str = "Input Digit",
    pred_label: int | None = None,
    true_label: int | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot 3-panel inference summary:
    input image, output distribution with detectors, and detector energy percentages.
    """

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    apply_style()

    inp = np.asarray(input_map, dtype=np.float64)
    out = np.asarray(output_intensity, dtype=np.float64)
    en = np.asarray(energies, dtype=np.float64).reshape(-1)
    en_pct = en / max(float(en.sum()), 1e-12) * 100.0
    max_idx = int(np.argmax(en_pct))
    pred = int(pred_label) if pred_label is not None else max_idx

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), gridspec_kw={"width_ratios": [1.0, 1.0, 1.35]})

    # Panel 1: input
    ax0 = axes[0]
    im0 = ax0.imshow(inp, cmap="gray", origin="lower")
    ax0.set_title(input_title)
    ax0.set_xticks([])
    ax0.set_yticks([])
    cbar0 = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.03)
    cbar0.set_label("Amp. (a.u.)")

    # Panel 2: output + detectors
    ax1 = axes[1]
    out_disp = np.log10(out + 1e-8)
    im1 = ax1.imshow(out_disp, cmap="gray", extent=_extent_mm(out.shape[0], dx), origin="lower")
    ax1.set_title("Output Distribution")
    ax1.set_xticks([])
    ax1.set_yticks([])
    for idx, region in enumerate(layout.regions):
        cx, cy = region.center_xy
        w, h = region.size_xy
        rect = Rectangle(
            ((cx - w / 2.0) * 1e3, (cy - h / 2.0) * 1e3),
            w * 1e3,
            h * 1e3,
            fill=False,
            edgecolor="#ff2d2d",
            linewidth=1.8,
            linestyle=(0, (4, 3)),
        )
        ax1.add_patch(rect)
        ax1.text(cx * 1e3, (cy + h / 2.0) * 1e3 + 1.2, str(idx), color="white", ha="center", va="bottom", fontsize=8, weight="bold")
    if true_label is not None:
        status_text = f"Pred: {pred} / True: {int(true_label)}"
    else:
        status_text = f"Pred: {pred}"
    ax1.text(
        0.02,
        0.98,
        status_text,
        transform=ax1.transAxes,
        color="white",
        fontsize=12,
        ha="left",
        va="top",
        bbox={"facecolor": "black", "alpha": 0.35, "pad": 2, "edgecolor": "none"},
    )
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)

    # Panel 3: energy bar chart
    ax2 = axes[2]
    x = np.arange(en_pct.shape[0], dtype=np.int64)
    bars = ax2.bar(x, en_pct, color="#312782", edgecolor="#1e1a58")
    bars[max_idx].set_color("#1d5fd0")
    ax2.set_title("Energy Distribution\n(Percentage)")
    ax2.set_xlabel("Detector index")
    ax2.set_ylabel("%")
    ax2.set_xticks(x)
    ax2.set_ylim(0.0, max(5.0, float(en_pct.max()) * 1.12))

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig, axes
