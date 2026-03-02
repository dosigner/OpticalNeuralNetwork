"""Saliency metrics: PR curve and max F-measure."""

from __future__ import annotations

import numpy as np
import torch


def _pr_single(pred: np.ndarray, gt: np.ndarray, thresholds: int) -> tuple[np.ndarray, np.ndarray]:
    p = pred.astype(np.float32)
    g = (gt.astype(np.float32) > 0.5).astype(np.uint8)
    eps = 1e-8
    tvals = np.linspace(0.0, 1.0, thresholds, dtype=np.float32)
    prec = np.empty_like(tvals)
    rec = np.empty_like(tvals)
    pos = max(float(g.sum()), eps)
    for i, t in enumerate(tvals):
        m = (p >= float(t)).astype(np.uint8)
        tp = float((m * g).sum())
        fp = float((m * (1 - g)).sum())
        prec[i] = tp / max(tp + fp, eps)
        rec[i] = tp / pos
    return prec, rec


def pr_curve(pred: torch.Tensor, gt: torch.Tensor, *, thresholds: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean PR curve over batch."""

    p_np = pred.detach().cpu().numpy()
    g_np = gt.detach().cpu().numpy()
    all_p = []
    all_r = []
    for i in range(p_np.shape[0]):
        p, r = _pr_single(p_np[i], g_np[i], thresholds=thresholds)
        all_p.append(p)
        all_r.append(r)
    return np.mean(np.stack(all_p, axis=0), axis=0), np.mean(np.stack(all_r, axis=0), axis=0)


def max_f_measure(
    pred: torch.Tensor,
    gt: torch.Tensor,
    *,
    thresholds: int = 256,
    beta2: float = 0.3,
) -> float:
    """Compute max F-measure from PR curve."""

    p, r = pr_curve(pred, gt, thresholds=thresholds)
    f = (1.0 + beta2) * p * r / np.maximum(beta2 * p + r, 1e-8)
    return float(np.nanmax(f))
