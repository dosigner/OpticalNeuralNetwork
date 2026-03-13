from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np


def _as_mask_array(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("mask must be a 2D array")
    return np.clip(arr, 0.0, 1.0)


def _component_sizes(binary: np.ndarray) -> list[int]:
    h, w = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    sizes: list[int] = []
    for y in range(h):
        for x in range(w):
            if not binary[y, x] or visited[y, x]:
                continue
            q: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            size = 0
            while q:
                cy, cx = q.popleft()
                size += 1
                for ny in range(max(0, cy - 1), min(h, cy + 2)):
                    for nx in range(max(0, cx - 1), min(w, cx + 2)):
                        if binary[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            q.append((ny, nx))
            sizes.append(size)
    return sizes


def _edge_density(mask: np.ndarray) -> float:
    gy = np.zeros_like(mask)
    gx = np.zeros_like(mask)
    gy[1:, :] = np.abs(mask[1:, :] - mask[:-1, :])
    gx[:, 1:] = np.abs(mask[:, 1:] - mask[:, :-1])
    grad = gx + gy
    return float(np.mean(grad))


def _center_offset_norm(mask: np.ndarray) -> float:
    total = float(mask.sum())
    if total <= 1e-8:
        return 0.0
    h, w = mask.shape
    ys, xs = np.indices((h, w), dtype=np.float32)
    cy = float((ys * mask).sum() / total)
    cx = float((xs * mask).sum() / total)
    center_y = 0.5 * (h - 1)
    center_x = 0.5 * (w - 1)
    dy = cy - center_y
    dx = cx - center_x
    max_dist = float(np.hypot(center_y, center_x))
    if max_dist <= 1e-8:
        return 0.0
    return float(np.hypot(dy, dx) / max_dist)


def compute_mask_metrics(mask: np.ndarray, *, binary_threshold: float = 0.5) -> dict[str, float]:
    arr = _as_mask_array(mask)
    binary = arr >= float(binary_threshold)
    component_sizes = _component_sizes(binary)
    fg_pixels = int(binary.sum())
    largest = max(component_sizes, default=0)
    largest_ratio = float(largest / fg_pixels) if fg_pixels > 0 else 0.0
    return {
        "foreground_ratio": float(arr.mean()),
        "center_offset_norm": _center_offset_norm(arr),
        "edge_density": _edge_density(arr),
        "component_count": float(len(component_sizes)),
        "largest_component_ratio": largest_ratio,
        "peak_value": float(arr.max(initial=0.0)),
    }


def summarize_mask_metrics(metrics: Iterable[dict[str, float]]) -> dict[str, dict[str, float] | int]:
    rows = list(metrics)
    if not rows:
        return {"count": 0, "mean": {}, "p10": {}, "p50": {}, "p90": {}}
    keys = sorted(rows[0].keys())
    data = {key: np.asarray([row[key] for row in rows], dtype=np.float32) for key in keys}

    def _summary(fn) -> dict[str, float]:
        return {key: float(fn(values)) for key, values in data.items()}

    return {
        "count": len(rows),
        "mean": _summary(np.mean),
        "p10": _summary(lambda x: np.percentile(x, 10)),
        "p50": _summary(np.median),
        "p90": _summary(lambda x: np.percentile(x, 90)),
    }
