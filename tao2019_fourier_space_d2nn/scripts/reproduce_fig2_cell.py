"""Reproduce Fig.2-style cell saliency run."""

from __future__ import annotations

from pathlib import Path

from _common import run_with_overrides


def main() -> None:
    base = Path(__file__).resolve().parents[1] / "src" / "tao2019_fd2nn" / "config" / "saliency_cell.yaml"
    run_with_overrides(base, overrides={}, task="saliency")


if __name__ == "__main__":
    main()
