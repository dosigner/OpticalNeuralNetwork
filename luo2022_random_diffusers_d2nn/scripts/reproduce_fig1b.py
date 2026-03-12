"""Reproduce Fig 1b: Object distortion and reconstruction.

Usage:
    python scripts/reproduce_fig1b.py --checkpoint runs/.../model.pt [--config configs/baseline.yaml]
"""

import argparse

from luo2022_d2nn.figures.fig1b_distortion import make_fig1b


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Fig 1b: distortion and reconstruction comparison."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained D2NN model checkpoint.")
    parser.add_argument("--config", default="configs/baseline.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--output", default="figures/fig1b.png",
                        help="Output image path.")
    parser.add_argument("--digits", nargs="+", type=int, default=[2, 5, 6],
                        help="MNIST digit labels to display.")
    args = parser.parse_args()

    make_fig1b(
        args.checkpoint,
        args.config,
        save_path=args.output,
        digits=args.digits,
    )
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
