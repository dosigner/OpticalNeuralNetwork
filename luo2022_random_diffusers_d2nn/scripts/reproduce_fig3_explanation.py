"""Generate an explanation figure for Fig. 3 axis semantics."""

from __future__ import annotations

import argparse

from luo2022_d2nn.figures.fig3_period_explanation import make_fig3_explanation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the Fig. 3 explanation figure."
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint for the explanation figure.")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Path to YAML config file.")
    parser.add_argument(
        "--output",
        default="figures/fig3_period_explanation.png",
        help="Output image path.",
    )
    args = parser.parse_args()

    make_fig3_explanation(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        save_path=args.output,
    )
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
