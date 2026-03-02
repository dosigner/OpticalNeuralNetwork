"""Math helpers for complex fields and normalization."""

from __future__ import annotations

import torch


def intensity(field: torch.Tensor) -> torch.Tensor:
    """Return optical intensity |E|^2.

    Args:
        field: complex tensor, shape (..., N, N)
    Returns:
        float tensor, shape (..., N, N)
    """

    return torch.abs(field) ** 2


def normalize_minmax(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize tensor to [0, 1] along full tensor domain."""

    x_min = torch.min(x)
    x_max = torch.max(x)
    return (x - x_min) / (x_max - x_min + eps)
