#!/usr/bin/env python
"""Generate figures for the phase-first dual-2f FD2NN study."""

from __future__ import annotations

import argparse
from pathlib import Path

from kim2026.viz.fd2nn_phase_restore import generate_figures


DEFAULT_STUDY_DIR = Path(__file__).resolve().parent.parent / "runs" / "fd2nn_phase_restore_dual2f_codex"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--fig-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = generate_figures(study_dir=args.study_dir, fig_dir=args.fig_dir)
    print(f"Generated {len(output_paths)} figures:")
    for path in output_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
