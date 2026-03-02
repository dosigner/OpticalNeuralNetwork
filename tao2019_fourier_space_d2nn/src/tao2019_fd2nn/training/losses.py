"""Losses for classification and saliency."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tao2019_fd2nn.optics.fft2c import gamma_flip2d


def classification_loss(
    energies: torch.Tensor,
    labels: torch.Tensor,
    *,
    loss_mode: str = "cross_entropy",
    leakage_ratio: torch.Tensor | None = None,
    leakage_weight: float = 0.1,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Classification loss on detector energies with optional leakage penalty."""

    logits = energies / max(float(temperature), 1e-8)
    if loss_mode == "mse_onehot":
        onehot = F.one_hot(labels.long(), num_classes=energies.shape[1]).to(energies.dtype)
        probs = logits / logits.sum(dim=1, keepdim=True).clamp_min(1e-8)
        loss = F.mse_loss(probs, onehot)
    else:
        loss = F.cross_entropy(logits, labels.long())
    if leakage_ratio is not None:
        loss = loss + float(leakage_weight) * leakage_ratio.mean()
    return loss


def saliency_mse_loss(
    pred_intensity: torch.Tensor,
    gt_intensity: torch.Tensor,
    *,
    gamma_flip: bool = True,
) -> torch.Tensor:
    """MSE with optional Γ flip alignment for target map."""

    target = gamma_flip2d(gt_intensity) if gamma_flip else gt_intensity
    return F.mse_loss(pred_intensity, target)
