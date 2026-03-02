"""Regenerate standard figures from an existing run directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tao2019_fd2nn.viz.figure_factory import FigureFactory


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate tao2019 figures from checkpoint history")
    parser.add_argument("--run-dir", required=True, help="Run directory path")
    parser.add_argument("--task", choices=["classification", "saliency"], default="classification")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / "checkpoints" / "final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    history = ckpt.get("history", {})
    factory = FigureFactory(run_dir / "figures")
    if args.task == "classification":
        factory.plot_convergence(history, left_key="val_loss", right_key="val_acc", name="convergence_cls.png")
    else:
        factory.plot_convergence(history, left_key="val_loss", right_key="val_fmax", name="convergence_saliency.png")
    print(str(run_dir / "figures"))


if __name__ == "__main__":
    main()
