"""Reproduce Supplementary Fig. S4: pruning-condition comparison.

Usage:
    python scripts/reproduce_figS4.py \
        --checkpoint runs/n20_L4/model.pt \
        [--config configs/baseline.yaml] \
        [--output figures/figS4_pruning.png]
"""

from __future__ import annotations

import argparse

from luo2022_d2nn.figures.figs4_pruning import make_figs4


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce Supplementary Fig. S4: pruning-condition comparison."
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
        default="figures/figS4_pruning.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--ood-image",
        default=None,
        help="Optional grayscale image path for the OOD object.",
    )
    args = parser.parse_args()

    make_figs4(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        save_path=args.output,
        ood_image_path=args.ood_image,
    )
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
