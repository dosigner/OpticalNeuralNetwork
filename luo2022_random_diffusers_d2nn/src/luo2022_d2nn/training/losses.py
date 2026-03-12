"""PCC + energy penalty loss for D2NN training (Luo et al. 2022)."""

from __future__ import annotations

import torch
from torch import Tensor


def pearson_correlation(output: Tensor, target: Tensor) -> Tensor:
    """Pearson correlation coefficient between output and target.

    Both are (B, N, N) or (B, 1, N, N) real tensors.
    Computes per-sample PCC and returns mean over batch.

    P = Σ(O - Ō)(G - Ḡ) / sqrt(Σ(O - Ō)² · Σ(G - Ḡ)²)
    """
    eps = 1e-8
    # Flatten spatial dims: (B, -1)
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

    pcc = num / den  # (B,)
    return pcc.mean()


def energy_penalty(
    output: Tensor, mask: Tensor, alpha: float = 1.0, beta: float = 0.5
) -> Tensor:
    """Energy penalty loss.

    E = [Σ(α(1-mask)*output - β*mask*output)] / Σ(mask)

    mask is the binary support mask (1 where target > 0).
    Penalizes energy outside support (α term) and rewards energy inside (β term).
    Returns mean over batch.
    """
    eps = 1e-8
    o = output.reshape(output.shape[0], -1)
    m = mask.reshape(mask.shape[0], -1)

    outside = alpha * ((1.0 - m) * o).sum(dim=1)
    inside = beta * (m * o).sum(dim=1)
    norm = m.sum(dim=1).clamp(min=eps)

    penalty = (outside - inside) / norm  # (B,)
    return penalty.mean()


def pcc_energy_loss(
    output: Tensor,
    target: Tensor,
    mask: Tensor,
    alpha: float = 1.0,
    beta: float = 0.5,
) -> Tensor:
    """Combined loss: -PCC + energy_penalty, averaged over batch."""
    return -pearson_correlation(output, target) + energy_penalty(
        output, mask, alpha, beta
    )
