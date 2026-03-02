"""Loss functions for D2NN training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def classification_loss(
    energies: torch.Tensor,
    labels: torch.Tensor,
    *,
    leakage_energy: torch.Tensor | None = None,
    leakage_weight: float = 0.1,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Cross-entropy over detector energies with optional leakage penalty.

    Args:
        energies: detector energies, shape (B, K)
        labels: class labels (B,) or onehot (B, K)
        leakage_energy: optional leakage ratio or leakage energy, shape (B,)
        leakage_weight: leakage penalty multiplier
        temperature: logit scaling factor
    """

    logits = energies / max(temperature, 1e-8)
    if labels.ndim == 2:
        hard_labels = torch.argmax(labels, dim=-1)
    else:
        hard_labels = labels

    ce = F.cross_entropy(logits, hard_labels.long())
    if leakage_energy is None:
        return ce

    leakage = leakage_energy.mean()
    return ce + leakage_weight * leakage


def imaging_mse_loss(output_intensity: torch.Tensor, target_intensity: torch.Tensor) -> torch.Tensor:
    """MSE loss for imaging target.

    Args:
        output_intensity: shape (B, N, N), scaled to [0, 1]
        target_intensity: shape (B, N, N), scaled to [0, 1]
    """

    return F.mse_loss(output_intensity, target_intensity)
