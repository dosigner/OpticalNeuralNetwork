"""Supplementary S7: layer count and SBN position ablation."""

from __future__ import annotations

from pathlib import Path

from _common import run_with_overrides


def main() -> None:
    base = Path(__file__).resolve().parents[1] / "src" / "tao2019_fd2nn" / "config" / "cls_mnist_nonlinear_real_5l.yaml"
    variants = [
        {"experiment": {"name": "supp_s7_3layers"}, "model": {"num_layers": 3, "nonlinearity": {"enabled": True, "position": "rear"}}},
        {"experiment": {"name": "supp_s7_5layers"}, "model": {"num_layers": 5, "nonlinearity": {"enabled": True, "position": "rear"}}},
        {"experiment": {"name": "supp_s7_7layers"}, "model": {"num_layers": 7, "nonlinearity": {"enabled": True, "position": "rear"}}},
        {"experiment": {"name": "supp_s7_sbn_per_layer"}, "model": {"num_layers": 5, "nonlinearity": {"enabled": True, "position": "per_layer"}}},
    ]
    for overrides in variants:
        run_with_overrides(base, overrides=overrides, task="classification")


if __name__ == "__main__":
    main()
