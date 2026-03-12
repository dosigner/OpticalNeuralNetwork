"""Shared CLI helpers: config loading, device selection, run setup."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch

from luo2022_d2nn.config.schema import load_and_validate_config
from luo2022_d2nn.utils.io import dump_yaml, resolve_run_dir, save_repro_metadata

logger = logging.getLogger(__name__)


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    """Load and validate config from CLI args.

    Expects args.config to be a path to a YAML config file.
    Applies any CLI overrides (epochs, device, etc.).
    """
    cfg = load_and_validate_config(args.config)

    # Apply CLI overrides
    if hasattr(args, "epochs") and args.epochs is not None:
        cfg["training"]["epochs"] = int(args.epochs)
    if hasattr(args, "batch_size") and args.batch_size is not None:
        cfg["training"]["batch_size_objects"] = int(args.batch_size)

    return cfg


def select_device(cfg: dict[str, Any]) -> torch.device:
    """Select device (cuda if available, else cpu).

    Respects cfg["experiment"].get("device") if set.
    """
    dev_str = cfg.get("experiment", {}).get("device", None)
    if dev_str is not None:
        return torch.device(dev_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_run(cfg: dict[str, Any]) -> Path:
    """Create run directory, save config and reproducibility metadata.

    Returns the created run directory path.
    """
    exp = cfg["experiment"]
    save_dir = exp.get("save_dir", f"runs/{exp['id']}")
    run_dir = resolve_run_dir(save_dir, "")

    # Save resolved config
    dump_yaml(run_dir / "config.yaml", cfg)

    # Save reproducibility metadata
    save_repro_metadata(run_dir)

    logger.info("Run directory: %s", run_dir)
    return run_dir
