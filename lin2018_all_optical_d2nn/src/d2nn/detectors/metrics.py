"""Classifier metrics from detector energies."""

from __future__ import annotations

import numpy as np
import torch


def predict_from_energies(energies: torch.Tensor) -> torch.Tensor:
    """Return predicted class indices from detector energies."""

    return torch.argmax(energies, dim=-1)


def accuracy(energies: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute top-1 accuracy."""

    preds = predict_from_energies(energies)
    return float((preds == labels).float().mean().item())


def confusion_matrix(energies: torch.Tensor, labels: torch.Tensor, num_classes: int) -> np.ndarray:
    """Build confusion matrix as integer counts."""

    preds = predict_from_energies(energies).detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y.astype(np.intp), preds.astype(np.intp)), 1)
    return cm


def normalize_energies(energies: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize detector energies to per-sample distributions."""

    denom = torch.sum(energies, dim=-1, keepdim=True).clamp_min(eps)
    return energies / denom
