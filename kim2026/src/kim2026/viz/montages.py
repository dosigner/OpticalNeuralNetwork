"""Montage helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def save_sequence_montage(path: str | Path, images: list[torch.Tensor], *, cols: int = 4, title: str | None = None) -> None:
    """Save a simple intensity montage."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cols = max(int(cols), 1)
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax, image in zip(axes, images):
        ax.imshow(image.numpy(), cmap="magma")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(images):]:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
