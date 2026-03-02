"""Evaluate trained D2NN models."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from d2nn.cli.common import build_dataloaders, build_detector_tensors, build_model_from_config, choose_device, load_checkpoint, load_config
from d2nn.training.loops import run_classifier_epoch, run_imager_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate D2NN model")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    task = cfg.get("experiment", {}).get("task", "classifier")

    runtime = cfg.get("runtime", {})
    device = choose_device(runtime)

    model = build_model_from_config(cfg).to(device)
    load_checkpoint(args.checkpoint, model)

    train_loader, val_loader = build_dataloaders(cfg)

    if task == "classifier":
        detector_masks, _ = build_detector_tensors(cfg)
        out = run_classifier_epoch(
            model,
            val_loader,
            optimizer=None,
            detector_masks=detector_masks,
            device=device,
            leakage_weight=float(cfg.get("loss", {}).get("leakage_weight", 0.1)),
            temperature=float(cfg.get("loss", {}).get("temperature", 1.0)),
            max_steps=args.max_steps,
        )
        print({"loss": out.loss, "acc": out.acc})
    else:
        out = run_imager_epoch(model, val_loader, optimizer=None, device=device, max_steps=args.max_steps)
        print({"loss": out.loss})


if __name__ == "__main__":
    main()
