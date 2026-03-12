"""Seed helpers for deterministic experiments."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    """Set random seeds for python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def worker_init_fn(worker_id: int) -> None:
    """Dataloader worker seed hook."""
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def make_generator(seed: int) -> torch.Generator:
    """Create torch generator with fixed seed."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def runtime_seed_state(seed: int, deterministic: bool) -> dict[str, Any]:
    """Serializable runtime seed metadata."""
    return {"seed": int(seed), "deterministic": bool(deterministic)}
