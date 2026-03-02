"""Supplementary S6: phase range ablation (0~pi vs 0~2pi)."""

from __future__ import annotations

from pathlib import Path

from _common import run_with_overrides


def main() -> None:
    base = Path(__file__).resolve().parents[1] / "src" / "tao2019_fd2nn" / "config" / "cls_mnist_linear_fourier_5l_f1mm.yaml"
    runs = [
        {"experiment": {"name": "supp_s6_phase_0_pi"}, "model": {"modulation": {"phase_max_rad": 3.141592653589793}}},
        {"experiment": {"name": "supp_s6_phase_0_2pi"}, "model": {"modulation": {"phase_max_rad": 6.283185307179586}}},
    ]
    for overrides in runs:
        run_with_overrides(base, overrides=overrides, task="classification")


if __name__ == "__main__":
    main()
