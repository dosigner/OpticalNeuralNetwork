"""
utils.py
========
Visualisation and diagnostic utilities for D²NN / Fourier D²NN experiments.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch


# ---------------------------------------------------------------------------
# Field visualisation
# ---------------------------------------------------------------------------

def plot_field(
    field: torch.Tensor,
    title: str = "Optical field",
    save_path: str | None = None,
) -> plt.Figure:
    """Visualise a complex optical field (amplitude + phase side by side).

    Parameters
    ----------
    field:
        Complex tensor of shape ``(H, W)`` or ``(1, H, W)``.
    title:
        Figure title.
    save_path:
        If given, the figure is saved to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if field.dim() == 3:
        field = field.squeeze(0)
    amp = field.abs().detach().cpu().numpy()
    phase = field.angle().detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle(title)

    im0 = axes[0].imshow(amp, cmap="hot", origin="upper")
    axes[0].set_title("Amplitude")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(phase, cmap="hsv", vmin=-np.pi, vmax=np.pi, origin="upper")
    axes[1].set_title("Phase (rad)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_layer_phases(
    model: torch.nn.Module,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot the learnt phase masks of every diffractive layer.

    Parameters
    ----------
    model:
        A ``D2NN`` or ``FourierD2NN`` instance (must have a ``.layers``
        attribute with individual layers exposing a ``.phase`` parameter).
    save_path:
        Optional path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    layers = list(model.layers)
    n = len(layers)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    fig.suptitle("Learnt phase masks")
    for i, layer in enumerate(layers):
        phase = layer.phase.detach().cpu().numpy()
        ax = axes[i]
        im = ax.imshow(phase % (2 * np.pi), cmap="hsv", vmin=0, vmax=2 * np.pi)
        ax.set_title(f"Layer {i + 1}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_output_intensity(
    field: torch.Tensor,
    masks: torch.Tensor,
    predicted: int | None = None,
    true_label: int | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Visualise the output intensity field with detector regions overlaid.

    Parameters
    ----------
    field:
        Output complex field ``(H, W)`` or ``(1, H, W)``.
    masks:
        Detector masks ``(num_classes, H, W)`` boolean tensor.
    predicted:
        Predicted class index (optional, for annotation).
    true_label:
        Ground-truth label (optional, for annotation).
    save_path:
        Optional save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if field.dim() == 3:
        field = field.squeeze(0)
    intensity = field.abs().detach().cpu().numpy() ** 2
    num_classes = masks.shape[0]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(intensity, cmap="hot", origin="upper")
    # Draw detector boundaries
    for k in range(num_classes):
        m = masks[k].cpu().numpy()
        rows, cols = np.where(m)
        if len(rows) == 0:
            continue
        r0, r1 = rows.min(), rows.max() + 1
        c0, c1 = cols.min(), cols.max() + 1
        rect = plt.Rectangle(
            (c0 - 0.5, r0 - 0.5),
            c1 - c0,
            r1 - r0,
            linewidth=1,
            edgecolor="cyan",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            (c0 + c1) / 2,
            (r0 + r1) / 2,
            str(k),
            ha="center",
            va="center",
            color="cyan",
            fontsize=8,
        )

    title = "Output intensity"
    if true_label is not None:
        title += f"  |  true={true_label}"
    if predicted is not None:
        colour = "lime" if predicted == true_label else "red"
        title += f"  pred={predicted}"
        ax.set_title(title, color=colour)
    else:
        ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Training metrics
# ---------------------------------------------------------------------------

def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
    save_path: str | None = None,
) -> plt.Figure:
    """Plot loss and accuracy curves over epochs.

    Parameters
    ----------
    train_losses, val_losses:
        Per-epoch loss values.
    train_accs, val_accs:
        Per-epoch accuracy values (fractions in ``[0, 1]``).
    save_path:
        Optional save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, train_losses, label="Train")
    ax1.plot(epochs, val_losses, label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Cross-entropy loss")
    ax1.legend()

    ax2.plot(epochs, [a * 100 for a in train_accs], label="Train")
    ax2.plot(epochs, [a * 100 for a in val_accs], label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Classification accuracy")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
