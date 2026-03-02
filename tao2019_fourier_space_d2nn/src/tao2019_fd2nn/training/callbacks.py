"""Training callbacks and persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from tao2019_fd2nn.utils.io import dump_json, dump_yaml


def save_checkpoint(path: str | Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None, epoch: int, extra: dict[str, Any] | None = None) -> None:
    """Save model checkpoint."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"model": model.state_dict(), "epoch": int(epoch)}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, p)


def save_metrics(path: str | Path, metrics: dict[str, Any]) -> None:
    """Persist metrics.json."""

    dump_json(path, metrics)


def save_resolved_config(path: str | Path, cfg: dict[str, Any]) -> None:
    """Persist merged config snapshot."""

    dump_yaml(path, cfg)
