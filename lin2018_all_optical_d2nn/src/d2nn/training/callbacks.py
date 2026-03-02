"""Training callbacks and artifact logging utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from d2nn.utils.io import save_json, save_yaml


def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None, epoch: int, extra: dict[str, Any] | None = None) -> None:
    """Save model checkpoint."""

    payload = {
        "model": model.state_dict(),
        "epoch": int(epoch),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload["extra"] = extra

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, p)


def save_metrics(path: str | Path, metrics: dict[str, Any]) -> None:
    """Persist metrics as stable JSON."""

    save_json(path, metrics)


def save_resolved_config(path: str | Path, cfg: dict[str, Any]) -> None:
    """Persist resolved YAML config."""

    save_yaml(path, cfg)
