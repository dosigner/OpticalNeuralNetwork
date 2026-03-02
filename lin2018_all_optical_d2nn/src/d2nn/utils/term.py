"""Terminal color helpers with graceful fallback."""

from __future__ import annotations

try:
    from colorama import Fore, Style, init

    init(autoreset=True)
    _HAS_COLOR = True
except Exception:  # pragma: no cover - optional runtime dependency
    _HAS_COLOR = False

    class _Dummy:
        RESET_ALL = ""
        BLACK = ""
        RED = ""
        GREEN = ""
        YELLOW = ""
        BLUE = ""
        MAGENTA = ""
        CYAN = ""
        WHITE = ""

    Fore = _Dummy()  # type: ignore
    Style = _Dummy()  # type: ignore


_COLOR_MAP = {
    "black": Fore.BLACK,
    "red": Fore.RED,
    "green": Fore.GREEN,
    "yellow": Fore.YELLOW,
    "blue": Fore.BLUE,
    "magenta": Fore.MAGENTA,
    "cyan": Fore.CYAN,
    "white": Fore.WHITE,
}


def paint(text: str, color: str = "white", bold: bool = False) -> str:
    """Return ANSI colored text if colorama is available."""

    if not _HAS_COLOR:
        return text

    color_code = _COLOR_MAP.get(color.lower(), "")
    bold_code = Style.BRIGHT if bold else ""
    return f"{bold_code}{color_code}{text}{Style.RESET_ALL}"
