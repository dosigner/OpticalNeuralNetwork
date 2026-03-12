"""Reproduce Fig 6: Known vs New vs No-diffuser comparison.

Usage:
    python scripts/reproduce_fig6.py \
        --n1 runs/n1/model.pt \
        --n10 runs/n10/model.pt \
        --n15 runs/n15/model.pt \
        --n20 runs/n20/model.pt \
        [--config configs/baseline.yaml] \
        [--output figures/fig6.png]
"""

import argparse

from luo2022_d2nn.figures.fig6_conditions import make_fig6


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Fig 6: known vs new vs no-diffuser."
    )
    parser.add_argument("--n1", required=True,
                        help="Checkpoint for n=1 model.")
    parser.add_argument("--n10", required=True,
                        help="Checkpoint for n=10 model.")
    parser.add_argument("--n15", required=True,
                        help="Checkpoint for n=15 model.")
    parser.add_argument("--n20", required=True,
                        help="Checkpoint for n=20 model.")
    parser.add_argument("--config", default="configs/baseline.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--output", default="figures/fig6.png",
                        help="Output image path.")
    args = parser.parse_args()

    checkpoint_paths = {
        1: args.n1,
        10: args.n10,
        15: args.n15,
        20: args.n20,
    }

    make_fig6(checkpoint_paths, args.config, save_path=args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
