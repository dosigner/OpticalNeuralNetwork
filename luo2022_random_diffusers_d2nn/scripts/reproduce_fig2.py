"""Reproduce Fig 2: Known vs New diffuser evaluation.

Usage:
    python scripts/reproduce_fig2.py --checkpoint runs/.../model.pt [--config configs/baseline.yaml]
"""

import argparse

from luo2022_d2nn.figures.fig2_known_new import make_fig2


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Fig 2: known vs new diffuser reconstruction."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained D2NN model checkpoint.")
    parser.add_argument("--config", default="configs/baseline.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--output", default="figures/fig2.png",
                        help="Output image path.")
    args = parser.parse_args()

    make_fig2(
        args.checkpoint,
        args.config,
        save_path=args.output,
    )
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
