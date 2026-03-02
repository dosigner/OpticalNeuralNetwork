"""Classification metrics."""

from __future__ import annotations

import torch


def accuracy(energies: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy on detector energies."""

    if energies.numel() == 0:
        return 0.0
    pred = torch.argmax(energies, dim=1)
    return float((pred == labels).to(torch.float32).mean().item())
