from __future__ import annotations

from pathlib import Path

import matplotlib

from kim2026.viz.mpl_fonts import (
    configure_matplotlib_fonts,
    ensure_output_dir,
    pick_font_family,
)


def test_pick_font_family_prefers_first_available() -> None:
    chosen = pick_font_family(
        ["Missing A", "Preferred B", "Fallback C"],
        available=lambda name: name == "Preferred B",
    )
    assert chosen == "Preferred B"


def test_pick_font_family_falls_back_to_dejavu() -> None:
    chosen = pick_font_family(
        ["Missing A", "Missing B"],
        available=lambda _name: False,
    )
    assert chosen == "DejaVu Sans"


def test_configure_matplotlib_fonts_updates_rcparams() -> None:
    original_family = matplotlib.rcParams["font.family"]
    original_minus = matplotlib.rcParams["axes.unicode_minus"]
    try:
        chosen = configure_matplotlib_fonts(
            ["Missing A", "Preferred B"],
            available=lambda name: name == "Preferred B",
        )
        assert chosen == "Preferred B"
        assert matplotlib.rcParams["font.family"][0] == "Preferred B"
        assert matplotlib.rcParams["axes.unicode_minus"] is False
    finally:
        matplotlib.rcParams["font.family"] = original_family
        matplotlib.rcParams["axes.unicode_minus"] = original_minus


def test_ensure_output_dir_creates_parents(tmp_path: Path) -> None:
    out_dir = tmp_path / "nested" / "figures"
    returned = ensure_output_dir(out_dir)
    assert returned == out_dir
    assert out_dir.is_dir()
