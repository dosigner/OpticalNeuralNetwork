"""Reproduce Fig 7: Depth advantage.

Usage:
    python scripts/reproduce_fig7.py \
        --d2-n1 runs/d2_n1/model.pt --d2-n10 runs/d2_n10/model.pt \
        --d2-n15 runs/d2_n15/model.pt --d2-n20 runs/d2_n20/model.pt \
        --d4-n1 runs/d4_n1/model.pt --d4-n10 runs/d4_n10/model.pt \
        --d4-n15 runs/d4_n15/model.pt --d4-n20 runs/d4_n20/model.pt \
        --d5-n1 runs/d5_n1/model.pt --d5-n10 runs/d5_n10/model.pt \
        --d5-n15 runs/d5_n15/model.pt --d5-n20 runs/d5_n20/model.pt \
        [--config configs/baseline.yaml] \
        [--output figures/fig7.png]
"""

import argparse

from luo2022_d2nn.figures.fig7_depth import make_fig7


_DEPTHS = [2, 4, 5]
_N_VALUES = [1, 10, 15, 20]


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Fig 7: depth advantage."
    )

    # Add one argument per (depth, n) combination
    for d in _DEPTHS:
        for n in _N_VALUES:
            parser.add_argument(
                f"--d{d}-n{n}",
                required=True,
                help=f"Checkpoint for {d}-layer model trained with n={n}.",
            )

    parser.add_argument("--config", default="configs/baseline.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--output", default="figures/fig7.png",
                        help="Output image path.")
    args = parser.parse_args()

    checkpoint_paths: dict[tuple[int, int], str] = {}
    for d in _DEPTHS:
        for n in _N_VALUES:
            attr_name = f"d{d}_n{n}"
            checkpoint_paths[(d, n)] = getattr(args, attr_name)

    make_fig7(checkpoint_paths, args.config, save_path=args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
