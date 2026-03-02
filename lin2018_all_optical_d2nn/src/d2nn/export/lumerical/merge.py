"""Helpers for Lumerical layer merge bookkeeping."""

from __future__ import annotations

from pathlib import Path


def collect_layer_files(temp_dir: str | Path) -> list[Path]:
    """Collect layer .fsp files in sorted order."""

    p = Path(temp_dir)
    return sorted(p.glob("layer_*.fsp"))
