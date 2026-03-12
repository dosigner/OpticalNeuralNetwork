"""LR schedule builders for D2NN training (Luo et al. 2022)."""

from __future__ import annotations

import torch
from torch.optim.lr_scheduler import ExponentialLR, _LRScheduler


def build_scheduler(
    optimizer: torch.optim.Optimizer, schedule_cfg: dict
) -> _LRScheduler:
    """Build LR scheduler from config.

    For type="epoch_multiplicative" with gamma=0.99:
      LR = initial_lr * gamma^epoch
    Uses ExponentialLR with gamma.
    """
    stype = schedule_cfg.get("type", "epoch_multiplicative")
    gamma = schedule_cfg.get("gamma", 0.99)

    if stype == "epoch_multiplicative":
        return ExponentialLR(optimizer, gamma=gamma)
    else:
        raise ValueError(f"Unknown schedule type: {stype}")
