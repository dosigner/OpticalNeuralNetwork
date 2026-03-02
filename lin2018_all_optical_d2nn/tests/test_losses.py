from __future__ import annotations

import torch

from d2nn.training.losses import classification_loss


def test_classification_loss_with_leakage() -> None:
    energies = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    leakage = torch.tensor([0.1, 0.2], dtype=torch.float32)

    loss = classification_loss(energies, labels, leakage_energy=leakage, leakage_weight=0.5, temperature=1.0)
    assert torch.isfinite(loss)
    assert float(loss.item()) > 0.0
