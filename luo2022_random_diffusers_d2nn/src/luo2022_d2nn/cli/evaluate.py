"""CLI entry point for D2NN evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from luo2022_d2nn.cli.common import load_config, select_device
from luo2022_d2nn.models.d2nn import D2NN
from luo2022_d2nn.training.trainer import Trainer
from luo2022_d2nn.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained D2NN model."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)."
    )
    parser.add_argument(
        "--split", type=str, default="test",
        help="Dataset split to evaluate on (default: test).",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Override device (cuda / cpu)."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save results JSON.",
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
    """CLI entry point for evaluation.

    Usage: luo2022-evaluate --config configs/baseline.yaml --checkpoint runs/.../model.pt [--split test]
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    args = _parse_args(argv)

    # 1. Load config
    cfg = load_config(args)

    if args.device is not None:
        cfg.setdefault("experiment", {})["device"] = args.device

    device = select_device(cfg)
    logger.info("Using device: %s", device)

    # Set seed for reproducible evaluation
    seed = int(cfg["experiment"]["seed"])
    set_global_seed(seed, deterministic=False)

    # 2. Build model and load checkpoint
    model = _build_model(cfg, device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Loaded checkpoint from epoch %d", checkpoint.get("epoch", -1))

    # 3. Build dataset + dataloader
    from luo2022_d2nn.data.mnist import MNISTAmplitude

    ds_cfg = cfg["dataset"]
    eval_ds = MNISTAmplitude(
        split=args.split,
        resize_to=int(ds_cfg.get("resize_to_px", 160)),
        final_size=int(ds_cfg.get("final_resolution_px", 240)),
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_ds,
        batch_size=int(cfg["training"]["batch_size_objects"]),
        shuffle=False,
        num_workers=0,
    )

    # 4. Create Trainer for evaluation
    trainer = Trainer(model, cfg, device)

    # 5. Evaluate with known diffusers (last epoch's diffusers)
    eval_cfg = cfg["evaluation"]
    last_epoch = int(cfg["training"]["epochs"])
    known_seed = int(cfg["experiment"]["seed"]) + last_epoch
    known_n = int(eval_cfg.get("known_diffuser_count_from_last_epoch", 20))
    known_diffusers = trainer._generate_epoch_diffusers(known_n, known_seed)

    logger.info("Evaluating with %d known diffusers...", known_n)
    known_results = trainer.evaluate(eval_loader, diffusers=known_diffusers)
    logger.info("Known diffusers — mean PCC: %.4f", known_results["mean_pcc"])

    # 6. Evaluate with new (blind) diffusers
    blind_n = int(eval_cfg.get("blind_test_new_diffuser_count", 20))
    blind_seed = int(cfg["experiment"]["seed"]) + last_epoch + 10000
    blind_diffusers = trainer._generate_epoch_diffusers(blind_n, blind_seed)

    logger.info("Evaluating with %d new (blind) diffusers...", blind_n)
    blind_results = trainer.evaluate(eval_loader, diffusers=blind_diffusers)
    logger.info("Blind diffusers — mean PCC: %.4f", blind_results["mean_pcc"])

    # 7. Print summary
    results = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "known_diffusers": {
            "count": known_n,
            "mean_pcc": known_results["mean_pcc"],
        },
        "blind_diffusers": {
            "count": blind_n,
            "mean_pcc": blind_results["mean_pcc"],
        },
    }

    print("\n=== Evaluation Results ===")
    print(f"Split: {args.split}")
    print(f"Known diffusers (n={known_n}): PCC = {known_results['mean_pcc']:.4f}")
    print(f"Blind diffusers (n={blind_n}): PCC = {blind_results['mean_pcc']:.4f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
