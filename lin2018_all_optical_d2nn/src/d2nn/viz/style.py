"""Deterministic matplotlib style helpers."""

from __future__ import annotations


def apply_style() -> None:
    """Apply deterministic plotting style."""

    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 120,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.family": "DejaVu Sans",
            "axes.grid": False,
        }
    )
