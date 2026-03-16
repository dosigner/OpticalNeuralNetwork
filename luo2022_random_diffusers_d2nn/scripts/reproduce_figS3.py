"""Reproduce Supplementary Fig. S3: overlap map of phase islands.

Usage:
    python scripts/reproduce_figS3.py \
        --checkpoint runs/n20_L4/model.pt \
        [--config configs/baseline.yaml] \
        [--output figures/figS3_overlap_map.png]
"""

from __future__ import annotations

import argparse

from luo2022_d2nn.figures.figs3_overlap_map import make_figs3


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce Supplementary Fig. S3: overlap map of phase islands."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint path for the 4-layer n=20 model.",
    )
    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--output",
        default="figures/figS3_overlap_map.png",
        help="Output image path.",
    )
    args = parser.parse_args()

    make_figs3(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        save_path=args.output,
    )
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
