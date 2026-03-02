"""Field and mask plotting functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from d2nn.viz.style import apply_style


def _extent_mm(N: int, dx: float) -> list[float]:
    L_mm = N * dx * 1e3
    return [-L_mm / 2.0, L_mm / 2.0, -L_mm / 2.0, L_mm / 2.0]


def plot_phase_mask(
    phase: np.ndarray,
    *,
    dx: float,
    phase_max: float = 2.0 * np.pi,
    shift_to_display_max: bool = False,
    wrap_to_display_max: bool = False,
    stretch_to_display_max: bool = False,
    title: str = "Phase mask",
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot phase map in radians.

    Args:
        shift_to_display_max:
            If True, add phase_max/2 before display. This is useful to map
            symmetric training ranges (e.g., -pi~pi) to 0~2pi by a fixed shift.
        wrap_to_display_max:
            If True, phase is wrapped with modulo phase_max for visualization.
            Useful when training phase is represented in a symmetric range
            (e.g., -pi~pi) but display should be 0~2pi.
        stretch_to_display_max:
            If True, linearly stretch phase values to [0, phase_max] for
            visualization only. Useful when trained phase range is narrower
            than display range (e.g., trained 0~pi, display 0~2pi).
    """

    import matplotlib.pyplot as plt

    apply_style()
    arr = np.asarray(phase, dtype=np.float64)
    vmax = float(phase_max)
    if vmax <= 0:
        vmax = 2.0 * np.pi
    if shift_to_display_max:
        arr = arr + 0.5 * vmax
    if wrap_to_display_max:
        arr = np.mod(arr, vmax)
    else:
        arr = np.clip(arr, 0.0, vmax)
    if stretch_to_display_max:
        a_min = float(arr.min())
        a_max = float(arr.max())
        if a_max > a_min + 1e-12:
            arr = (arr - a_min) / (a_max - a_min)
            arr = arr * vmax

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(
        arr,
        cmap="jet",
        vmin=0.0,
        vmax=vmax,
        extent=_extent_mm(arr.shape[0], dx),
        origin="lower",
    )
    ax.set_title(title)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("phase [rad]")
    cbar.set_ticks([0.0, vmax])
    if np.isclose(vmax, 2.0 * np.pi, atol=1e-3):
        cbar.set_ticklabels(["0", "2π"])
    elif np.isclose(vmax, np.pi, atol=1e-3):
        cbar.set_ticklabels(["0", "π"])
    else:
        cbar.set_ticklabels(["0", f"{vmax:.2f}"])

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig, ax


def plot_intensity(
    intensity: np.ndarray,
    *,
    dx: float,
    title: str = "Intensity",
    log_scale: bool = False,
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot intensity map."""

    import matplotlib.pyplot as plt

    apply_style()
    arr = np.asarray(intensity)
    if log_scale:
        arr = np.log10(arr + 1e-8)

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(arr, cmap="inferno", extent=_extent_mm(arr.shape[0], dx), origin="lower")
    ax.set_title(title)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(I)" if log_scale else "intensity [a.u.]")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig, ax
