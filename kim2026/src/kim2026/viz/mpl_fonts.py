from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import matplotlib
from matplotlib import font_manager

DEFAULT_FONT_CANDIDATES = (
    "Noto Sans CJK KR",
    "NanumGothic",
    "Malgun Gothic",
    "DejaVu Sans",
)


def pick_font_family(
    candidates: Iterable[str] = DEFAULT_FONT_CANDIDATES,
    available: Callable[[str], bool] | None = None,
) -> str:
    """Return the first installed font family from the preferred list."""

    if available is None:
        def available(name: str) -> bool:
            try:
                font_manager.findfont(name, fallback_to_default=False)
            except Exception:
                return False
            return True

    for candidate in candidates:
        if available(candidate):
            return candidate
    return "DejaVu Sans"


def configure_matplotlib_fonts(
    candidates: Iterable[str] = DEFAULT_FONT_CANDIDATES,
    available: Callable[[str], bool] | None = None,
) -> str:
    """Configure matplotlib to use a Hangul-capable font if present."""

    family = pick_font_family(candidates, available=available)
    matplotlib.rcParams["font.family"] = family
    matplotlib.rcParams["axes.unicode_minus"] = False
    return family


def ensure_output_dir(path_like: str | Path) -> Path:
    out_dir = Path(path_like)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
