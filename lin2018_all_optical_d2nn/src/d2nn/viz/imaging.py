"""Imaging visualization and metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from d2nn.viz.style import apply_style


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute SSIM score.

    Falls back to NaN if scikit-image is unavailable.
    """

    try:
        from skimage.metrics import structural_similarity
    except Exception:
        return float("nan")

    pred64 = pred.astype(np.float64)
    tgt64 = target.astype(np.float64)
    data_range = max(float(tgt64.max() - tgt64.min()), 1e-8)
    return float(structural_similarity(pred64, tgt64, data_range=data_range))


def plot_imaging_comparison(
    input_image: np.ndarray,
    d2nn_output: np.ndarray,
    free_space_output: np.ndarray,
    *,
    ssim_d2nn: float | None = None,
    ssim_free: float | None = None,
    title: str = "Imaging comparison",
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot input vs D2NN output vs free-space output."""

    import matplotlib.pyplot as plt

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))

    axes[0].imshow(input_image, cmap="gray", origin="lower")
    axes[0].set_title("Input")
    axes[1].imshow(d2nn_output, cmap="gray", origin="lower")
    axes[1].set_title(f"D2NN (SSIM={ssim_d2nn:.3f})" if ssim_d2nn is not None else "D2NN")
    axes[2].imshow(free_space_output, cmap="gray", origin="lower")
    axes[2].set_title(f"Free-space (SSIM={ssim_free:.3f})" if ssim_free is not None else "Free-space")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig, axes
