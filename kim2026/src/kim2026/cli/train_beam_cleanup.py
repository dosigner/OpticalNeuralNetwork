"""Train a beam-cleanup D2NN from cached NPZ pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

from kim2026.cli.common import apply_runtime_environment, load_config
from kim2026.training.trainer import train_model
from kim2026.utils.seed import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--resume", default=None, help="Optional checkpoint path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_runtime_environment(cfg["runtime"])
    set_global_seed(int(cfg["runtime"]["seed"]), strict_reproducibility=bool(cfg["runtime"]["strict_reproducibility"]))
    train_model(cfg, run_dir=Path(cfg["experiment"]["save_dir"]), resume_path=args.resume)


if __name__ == "__main__":
    main()
