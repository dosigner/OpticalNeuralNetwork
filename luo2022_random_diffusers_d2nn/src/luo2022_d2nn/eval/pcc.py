"""Pearson correlation coefficient evaluation metrics."""

from __future__ import annotations

import torch
from torch import Tensor


def compute_pcc(output: Tensor, target: Tensor) -> Tensor:
    """Compute per-sample PCC. Returns tensor of shape (B,).

    Same formula as pearson_correlation but returns per-sample values, not mean.
    """
    eps = 1e-8
    o = output.reshape(output.shape[0], -1)
    g = target.reshape(target.shape[0], -1)

    o_mean = o.mean(dim=1, keepdim=True)
    g_mean = g.mean(dim=1, keepdim=True)

    o_centered = o - o_mean
    g_centered = g - g_mean

    num = (o_centered * g_centered).sum(dim=1)
    den = torch.sqrt(
        (o_centered ** 2).sum(dim=1) * (g_centered ** 2).sum(dim=1) + eps
    )

    return num / den  # (B,)


def compute_mean_pcc(output: Tensor, target: Tensor) -> float:
    """Return scalar mean PCC across batch."""
    return compute_pcc(output, target).mean().item()
