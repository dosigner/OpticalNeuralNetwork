"""Math helpers for complex field tensors."""

from __future__ import annotations

import torch


def intensity(field: torch.Tensor) -> torch.Tensor:
    """Return optical intensity |u|^2."""

    return torch.abs(field) ** 2


