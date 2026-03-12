"""Visualization utilities for D2NN figure reproduction.

Provides contrast enhancement, scale bars, and standardised figure saving.
All helpers are display-only — NEVER use contrast_enhance for metrics computation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def contrast_enhance(
    image: np.ndarray | torch.Tensor,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> np.ndarray:
    """Percentile-based contrast stretching for display.

    NEVER use this for metrics computation — display only.

    Parameters
    ----------
    image : numpy array or torch tensor, shape (H, W).
    lower_percentile, upper_percentile : float
        Percentile bounds for clipping.

    Returns
    -------
    numpy array, clipped and scaled to [0, 1].
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image, dtype=np.float64)

    lo = np.percentile(image, lower_percentile)
    hi = np.percentile(image, upper_percentile)

    if hi - lo < 1e-12:
        # Constant image — return zeros
        return np.zeros_like(image, dtype=np.float64)

    out = (image - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def add_scale_bar(
    ax,
    dx_mm: float,
    bar_length_mm: float = 10.0,
    color: str = "white",
    fontsize: int = 8,
) -> None:
    """Add a scale bar to a matplotlib axis.

    Places the bar in the lower-right corner with a label such as "10 mm".

    Parameters
    ----------
    ax : matplotlib Axes
    dx_mm : float
        Pixel pitch in mm (used to convert mm to pixels).
    bar_length_mm : float
        Physical length of the scale bar in mm.
    color : str
        Bar and text colour.
    fontsize : int
        Font size for the label.
    """
    bar_length_px = bar_length_mm / dx_mm

    # Determine image extent from axes limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    img_w = abs(xlim[1] - xlim[0])
    img_h = abs(ylim[1] - ylim[0])

    # Position in lower-right with some margin
    margin_frac = 0.05
    x_end = xlim[1] - margin_frac * img_w
    x_start = x_end - bar_length_px
    # y increases downward for imshow
    y_pos = ylim[1] - margin_frac * img_h if ylim[1] > ylim[0] else ylim[0] + margin_frac * img_h

    ax.plot([x_start, x_end], [y_pos, y_pos], color=color, linewidth=2, solid_capstyle="butt")

    label = f"{bar_length_mm:.0f} mm" if bar_length_mm == int(bar_length_mm) else f"{bar_length_mm:.1f} mm"
    ax.text(
        (x_start + x_end) / 2,
        y_pos - 0.02 * img_h,
        label,
        color=color,
        fontsize=fontsize,
        ha="center",
        va="bottom",
    )


def save_figure(fig, path: str | Path, dpi: int = 300, tight: bool = True) -> None:
    """Save matplotlib figure with standard settings.

    Creates parent directories if needed.

    Parameters
    ----------
    fig : matplotlib Figure
    path : str or Path
        Output file path.
    dpi : int
        Resolution.
    tight : bool
        Whether to use tight bounding box.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"dpi": dpi}
    if tight:
        kwargs["bbox_inches"] = "tight"
    fig.savefig(str(path), **kwargs)
