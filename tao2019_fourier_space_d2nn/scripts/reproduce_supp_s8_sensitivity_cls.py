"""Supplementary S8: classification sensitivity to alignment error."""

from __future__ import annotations

from pathlib import Path

from _common import run_with_overrides


def main() -> None:
    base = Path(__file__).resolve().parents[1] / "src" / "tao2019_fd2nn" / "config" / "cls_mnist_linear_real_5l.yaml"
    shifts = [0.0, 2.0, 4.0, 6.0]
    for shift_um in shifts:
        run_with_overrides(
            base,
            overrides={
                "experiment": {"name": f"supp_s8_cls_shift_{int(shift_um)}um"},
                "optics": {"alignment_shift_um": shift_um},
            },
            task="classification",
        )


if __name__ == "__main__":
    main()
