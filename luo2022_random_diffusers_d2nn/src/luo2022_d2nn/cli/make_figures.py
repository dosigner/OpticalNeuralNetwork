"""CLI entry point for generating paper figures.

Usage:
    luo2022-make-figures --figure fig3 --checkpoint-n1 runs/n1/model.pt ...
    luo2022-make-figures --figure fig7 --d2-n1 runs/d2_n1/model.pt ...

Dispatches to the appropriate figure function based on --figure argument.
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate paper figures for Luo et al. 2022."
    )
    parser.add_argument(
        "--figure",
        required=True,
        choices=["fig3", "fig5", "fig6", "fig7"],
        help="Which figure to generate.",
    )
    parser.add_argument("--config", default="configs/baseline.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--output", default=None,
                        help="Output image path (default: figures/<figure>.png).")

    # Checkpoint arguments for fig3, fig5, fig6 (n-based)
    for n in [1, 10, 15, 20]:
        parser.add_argument(
            f"--n{n}", default=None,
            help=f"Checkpoint path for n={n} model.",
        )

    # Checkpoint arguments for fig7 (depth x n)
    for d in [2, 4, 5]:
        for n in [1, 10, 15, 20]:
            parser.add_argument(
                f"--d{d}-n{n}", default=None,
                help=f"Checkpoint for {d}-layer n={n} model.",
            )

    return parser


def _collect_n_checkpoints(args: argparse.Namespace) -> dict[int, str]:
    """Gather --n1, --n10, --n15, --n20 into a dict."""
    paths: dict[int, str] = {}
    for n in [1, 10, 15, 20]:
        val = getattr(args, f"n{n}", None)
        if val is not None:
            paths[n] = val
    if not paths:
        print("ERROR: at least one --nN checkpoint is required.", file=sys.stderr)
        sys.exit(1)
    return paths


def _collect_depth_checkpoints(args: argparse.Namespace) -> dict[tuple[int, int], str]:
    """Gather --d{D}-n{N} arguments into a dict."""
    paths: dict[tuple[int, int], str] = {}
    for d in [2, 4, 5]:
        for n in [1, 10, 15, 20]:
            val = getattr(args, f"d{d}_n{n}", None)
            if val is not None:
                paths[(d, n)] = val
    if not paths:
        print("ERROR: at least one --dD-nN checkpoint is required.", file=sys.stderr)
        sys.exit(1)
    return paths


def main() -> None:
    """CLI entry point for generating all figures."""
    parser = _build_parser()
    args = parser.parse_args()

    output = args.output or f"figures/{args.figure}.png"

    if args.figure == "fig3":
        from luo2022_d2nn.figures.fig3_period_sweep import make_fig3
        ckpts = _collect_n_checkpoints(args)
        make_fig3(ckpts, args.config, save_path=output)

    elif args.figure == "fig5":
        from luo2022_d2nn.figures.fig5_memory import make_fig5
        ckpts = _collect_n_checkpoints(args)
        make_fig5(ckpts, args.config, save_path=output)

    elif args.figure == "fig6":
        from luo2022_d2nn.figures.fig6_conditions import make_fig6
        ckpts = _collect_n_checkpoints(args)
        make_fig6(ckpts, args.config, save_path=output)

    elif args.figure == "fig7":
        from luo2022_d2nn.figures.fig7_depth import make_fig7
        ckpts = _collect_depth_checkpoints(args)
        make_fig7(ckpts, args.config, save_path=output)

    print(f"Saved {args.figure} to {output}")


if __name__ == "__main__":
    main()
