#!/usr/bin/env python
"""Generate detailed comparison figures for the FD2NN metasurface sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

from kim2026.viz.fd2nn_sweep import (
    build_field_columns,
    build_field_row_specs,
    generate_figures,
    get_runs,
)

DEFAULT_SWEEP_DIR = Path(__file__).resolve().parent.parent / "runs" / "fd2nn_metasurface_sweep_dual2f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--fig-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = generate_figures(sweep_dir=args.sweep_dir, fig_dir=args.fig_dir)
    print(f"Generated {len(output_paths)} figures:")
    for path in output_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
