from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def _as_mask_array(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("mask must be a 2D array")
    return np.clip(arr, 0.0, 1.0)


def _otsu_threshold(mask: np.ndarray) -> float:
    hist, bin_edges = np.histogram(mask.ravel(), bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    prob = hist / max(hist.sum(), 1.0)
    omega = np.cumsum(prob)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0.0] = np.nan
    sigma_b2 = ((mu_t * omega - mu) ** 2) / denom
    idx = int(np.nanargmax(sigma_b2))
    return float(centers[idx])


def _sharpened(mask: np.ndarray) -> np.ndarray:
    img = Image.fromarray(np.clip(mask * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    blurred = img.filter(ImageFilter.GaussianBlur(radius=1.0))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    blur_arr = np.asarray(blurred, dtype=np.float32) / 255.0
    sharpened = np.clip(arr + 1.5 * (arr - blur_arr), 0.0, 1.0)
    return sharpened.astype(np.float32)


def apply_gt_variant(mask: np.ndarray, variant: str) -> np.ndarray:
    arr = _as_mask_array(mask)
    key = str(variant).lower()
    if key == "raw":
        return arr.copy()
    if key == "binary":
        threshold = _otsu_threshold(arr)
        return (arr >= threshold).astype(np.float32)
    if key == "sharpened":
        return _sharpened(arr)
    raise ValueError(f"unsupported GT variant: {variant}")
