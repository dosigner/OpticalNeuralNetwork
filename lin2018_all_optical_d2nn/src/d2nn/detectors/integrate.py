"""Region-based detector energy integration."""

from __future__ import annotations

import numpy as np
import torch



def integrate_regions(intensity: torch.Tensor | np.ndarray, masks: torch.Tensor | np.ndarray, reduction: str = "sum") -> torch.Tensor | np.ndarray:
    """Integrate intensity inside detector masks.

    Args:
        intensity: shape (B, N, N) or (N, N)
        masks: shape (K, N, N)
        reduction: "sum" or "mean"

    Returns:
        energies: shape (B, K) or (K,)
    """

    if reduction not in {"sum", "mean"}:
        raise ValueError("reduction must be 'sum' or 'mean'")

    if isinstance(intensity, np.ndarray):
        i = intensity
        m = masks.astype(np.float32) if isinstance(masks, np.ndarray) else masks.detach().cpu().numpy().astype(np.float32)
        squeezed = False
        if i.ndim == 2:
            i = i[None, ...]
            squeezed = True
        e = (i[:, None, :, :] * m[None, :, :, :]).sum(axis=(-1, -2))
        if reduction == "mean":
            denom = m.sum(axis=(-1, -2))
            e = e / np.maximum(denom[None, :], 1.0)
        return e[0] if squeezed else e

    i = intensity
    m = masks
    if not torch.is_tensor(m):
        m = torch.as_tensor(m, device=i.device)
    m = m.to(dtype=i.real.dtype, device=i.device)

    squeezed_t = False
    if i.ndim == 2:
        i = i.unsqueeze(0)
        squeezed_t = True

    e_t = (i.unsqueeze(1) * m.unsqueeze(0)).sum(dim=(-1, -2))
    if reduction == "mean":
        denom_t = m.sum(dim=(-1, -2)).clamp_min(1.0)
        e_t = e_t / denom_t.unsqueeze(0)

    return e_t[0] if squeezed_t else e_t
