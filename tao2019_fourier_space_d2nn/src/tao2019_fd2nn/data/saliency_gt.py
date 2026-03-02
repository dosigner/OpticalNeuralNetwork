"""Saliency GT generation adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from skimage.color import rgb2lab
from skimage.filters import gaussian


def _normalize01(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.float32)
    y = y - y.min()
    denom = max(float(y.max()), 1e-8)
    return y / denom


def _to_gray01(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
    return _normalize01(arr)


def _to_rgb01(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], repeats=3, axis=2)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _ft_saliency(image: np.ndarray, *, smooth_sigma: float = 1.0) -> np.ndarray:
    """Frequency-tuned saliency map (Achanta-style global Lab contrast)."""

    rgb = _to_rgb01(image)
    lab = rgb2lab(rgb)
    mean_lab = lab.reshape(-1, 3).mean(axis=0)
    diff = lab - mean_lab
    sal = np.sqrt(np.sum(diff * diff, axis=-1))
    if smooth_sigma > 0:
        sal = gaussian(sal, sigma=float(smooth_sigma), preserve_range=True)
    return _normalize01(sal)


def _spectral_residual_saliency(
    image: np.ndarray,
    *,
    log_smooth_sigma: float = 3.0,
    map_smooth_sigma: float = 2.0,
) -> np.ndarray:
    """Spectral residual saliency approximation."""

    gray = _to_gray01(image)
    F = np.fft.fft2(gray)
    log_amp = np.log(np.abs(F) + 1e-8)
    phase = np.angle(F)
    avg_log_amp = gaussian(log_amp, sigma=float(log_smooth_sigma), preserve_range=True)
    residual = log_amp - avg_log_amp
    inv = np.fft.ifft2(np.exp(residual + 1j * phase))
    sal = np.abs(inv) ** 2
    if map_smooth_sigma > 0:
        sal = gaussian(sal, sigma=float(map_smooth_sigma), preserve_range=True)
    return _normalize01(sal)


@dataclass
class SaliencyGtBuilder:
    """Generate saliency target maps from image and config."""

    source: str = "dataset"
    params: dict[str, Any] = field(default_factory=dict)
    foreground_class: int | None = None

    def _class_gate(self, sal: np.ndarray, label: int | None) -> np.ndarray:
        class_gate = bool(self.params.get("class_gate", False))
        if not class_gate:
            return sal
        if self.foreground_class is None or label is None:
            return sal
        if int(label) != int(self.foreground_class):
            return np.zeros_like(sal, dtype=np.float32)
        return sal

    def build(
        self,
        *,
        image: np.ndarray,
        label: int | None = None,
        dataset_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return normalized saliency map in [0, 1]."""

        src = str(self.source)
        if src == "dataset":
            if dataset_mask is None:
                raise ValueError("gt_source='dataset' requires dataset_mask")
            sal = _normalize01(dataset_mask)
        elif src == "intensity":
            sal = _to_gray01(image)
        elif src == "class_conditioned_intensity":
            sal = _to_gray01(image)
            if self.foreground_class is not None and label is not None and int(label) != int(self.foreground_class):
                sal = np.zeros_like(sal, dtype=np.float32)
        elif src == "ft":
            sal = _ft_saliency(image, smooth_sigma=float(self.params.get("smooth_sigma", 1.0)))
            sal = self._class_gate(sal, label)
        elif src == "spectral_residual":
            sal = _spectral_residual_saliency(
                image,
                log_smooth_sigma=float(self.params.get("log_smooth_sigma", 3.0)),
                map_smooth_sigma=float(self.params.get("map_smooth_sigma", 2.0)),
            )
            sal = self._class_gate(sal, label)
        else:
            raise ValueError(f"unsupported gt_source: {src}")
        return _normalize01(sal)
