"""Diffuser registry — tracks unique diffuser realisations.

Two diffusers are "distinct" when the average absolute phase
difference (after mean-subtraction) exceeds *min_delta_phi*.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import torch


class DiffuserRegistry:
    """Registry that stores unique diffusers."""

    def __init__(self, min_delta_phi: float = math.pi / 2) -> None:
        self.min_delta_phi = float(min_delta_phi)
        self._diffusers: List[Dict[str, Any]] = []
        self._phases: List[torch.Tensor] = []  # mean-subtracted phase maps

    # ------------------------------------------------------------------
    def _normalise(self, phase_map: torch.Tensor) -> torch.Tensor:
        p = phase_map.to(torch.float64)
        return p - p.mean()

    def is_unique(self, phase_map: torch.Tensor) -> bool:
        """Return True if *phase_map* is distinct from all registered."""
        p_new = self._normalise(phase_map)
        for p_old in self._phases:
            avg_delta = (p_new - p_old).abs().mean().item()
            if avg_delta <= self.min_delta_phi:
                return False
        return True

    def register(self, diffuser_dict: Dict[str, Any]) -> bool:
        """Register a diffuser if it is unique; return success flag."""
        phase = diffuser_dict["phase_map"]
        if not self.is_unique(phase):
            return False
        self._phases.append(self._normalise(phase))
        self._diffusers.append(diffuser_dict)
        return True

    def __len__(self) -> int:
        return len(self._diffusers)

    def get(self, index: int) -> Dict[str, Any]:
        """Retrieve a registered diffuser by index."""
        return self._diffusers[index]
