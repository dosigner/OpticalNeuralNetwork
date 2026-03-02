"""Seed and deterministic runtime helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Set global random seeds.

    Args:
        seed: integer seed value.
        deterministic: when True, enable deterministic torch behavior when possible.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_torch_generator(seed: int) -> torch.Generator:
    """Return a seeded torch generator for reproducible DataLoader shuffling."""

    g = torch.Generator()
    g.manual_seed(seed)
    return g
