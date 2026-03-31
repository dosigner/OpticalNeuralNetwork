"""Generate D2NN beam-reducer sweep visualization figures."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from kim2026.viz.d2nn_beamreducer_sweep import generate_figures

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STAGE1_DIR = PROJECT_ROOT / "runs" / "06_d2nn_beamreducer_distance-sweep_pitch-3um_codex"
DEFAULT_STAGE2_DIR = PROJECT_ROOT / "runs" / "07_d2nn_beamreducer_pitch-sweep_codex"
DEFAULT_FIG_DIR = PROJECT_ROOT / "runs" / "d2nn_beamreducer_figures"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-dir", type=Path, default=DEFAULT_STAGE1_DIR)
    parser.add_argument("--stage2-dir", type=Path, default=DEFAULT_STAGE2_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    args = parser.parse_args()

    outputs = generate_figures(stage1_dir=args.stage1_dir, stage2_dir=args.stage2_dir, fig_dir=args.fig_dir)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
