"""Supplementary S9: saliency sensitivity to fabrication blur proxy."""

from __future__ import annotations

from pathlib import Path

from _common import run_with_overrides


def main() -> None:
    base = Path(__file__).resolve().parents[1] / "src" / "tao2019_fd2nn" / "config" / "saliency_cell.yaml"
    blur_sigmas = [0.0, 0.5, 1.0, 2.0]
    for sigma in blur_sigmas:
        run_with_overrides(
            base,
            overrides={
                "experiment": {"name": f"supp_s9_sal_blur_{str(sigma).replace('.', 'p')}"},
                "model": {"fabrication_blur_sigma_px": sigma},
            },
            task="saliency",
        )


if __name__ == "__main__":
    main()
