"""Supplementary S2: dual 2f / 4f proxy and SBN placement variants."""

from __future__ import annotations

from pathlib import Path

from _common import run_with_overrides


def main() -> None:
    base = Path(__file__).resolve().parents[1] / "src" / "tao2019_fd2nn" / "config" / "cls_mnist_nonlinear_fourier_5l_f4mm.yaml"
    variants = [
        {"experiment": {"name": "supp_s2_single_sbn_rear"}, "model": {"nonlinearity": {"enabled": True, "position": "rear"}}},
        {
            "experiment": {"name": "supp_s2_multi_sbn"},
            "model": {"nonlinearity": {"enabled": True, "position": "per_layer"}},
        },
        {"experiment": {"name": "supp_s2_linear_baseline"}, "model": {"nonlinearity": {"enabled": False}}},
    ]
    for overrides in variants:
        run_with_overrides(base, overrides=overrides, task="classification")


if __name__ == "__main__":
    main()
