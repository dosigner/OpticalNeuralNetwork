"""CLI entry point for D2NN training."""

from __future__ import annotations

import argparse
import logging
import sys

import torch

from luo2022_d2nn.cli.common import load_config, select_device, setup_run
from luo2022_d2nn.models.d2nn import D2NN
from luo2022_d2nn.training.trainer import Trainer
from luo2022_d2nn.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a D2NN model (Luo et al. 2022 random diffuser setup)."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file."
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of epochs."
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, dest="batch_size",
        help="Override batch size.",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Override device (cuda / cpu)."
    )
    return parser.parse_args(argv)


def _build_model(cfg: dict, device: torch.device) -> D2NN:
    """Build D2NN model from config."""
    geom = cfg["geometry"]
    grid = cfg["grid"]
    optics = cfg["optics"]
    model_cfg = cfg["model"]

    model = D2NN(
        num_layers=int(geom["num_layers"]),
        grid_size=int(grid["nx"]),
        dx_mm=float(grid["pitch_mm"]),
        wavelength_mm=float(optics["wavelength_mm"]),
        diffuser_to_layer1_mm=float(geom["diffuser_to_layer1_mm"]),
        layer_to_layer_mm=float(geom["layer_to_layer_mm"]),
        last_layer_to_output_mm=float(geom["last_layer_to_output_mm"]),
        pad_factor=int(grid.get("pad_factor", 2)),
        init_phase_dist=str(model_cfg.get("init_phase_distribution", "uniform_0_2pi")),
    )
    return model.to(device)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for training.

    Usage: luo2022-train --config configs/baseline.yaml [--epochs N] [--device cuda]
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    args = _parse_args(argv)

    # 1. Load & validate config
    cfg = load_config(args)

    # Apply device override from CLI
    if args.device is not None:
        cfg.setdefault("experiment", {})["device"] = args.device

    # 2. Select device
    device = select_device(cfg)
    logger.info("Using device: %s", device)

    # 3. Set global seed
    seed = int(cfg["experiment"]["seed"])
    set_global_seed(seed, deterministic=False)

    # 4. Setup run directory
    run_dir = setup_run(cfg)

    # 5. Build dataset + dataloader
    # Lazy import to avoid hard dependency on dataset module at import time
    from luo2022_d2nn.data.mnist import MNISTAmplitude

    ds_cfg = cfg["dataset"]
    train_ds = MNISTAmplitude(
        split="train",
        resize_to=int(ds_cfg.get("resize_to_px", 160)),
        final_size=int(ds_cfg.get("final_resolution_px", 240)),
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size_objects"]),
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # 6. Build D2NN model
    model = _build_model(cfg, device)
    logger.info("Model: %d layers, grid %d", model.num_layers, model.grid_size)

    # 7. Create Trainer
    trainer = Trainer(model, cfg, device)

    # 8. Training loop
    epochs = int(cfg["training"]["epochs"])
    for epoch in range(1, epochs + 1):
        metrics = trainer.train_epoch(train_loader, epoch)
        logger.info(
            "Epoch %d/%d — loss: %.6f, pcc: %.4f",
            epoch, epochs, metrics["avg_loss"], metrics["avg_pcc"],
        )

    # 9. Save final checkpoint
    ckpt_path = run_dir / "model.pt"
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "config": cfg,
        },
        ckpt_path,
    )
    logger.info("Saved checkpoint: %s", ckpt_path)


if __name__ == "__main__":
    main()
