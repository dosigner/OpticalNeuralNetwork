"""Preprocessing utilities for optical field encoding."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def resize_and_pad_square(image: torch.Tensor, out_size: int, object_size: int) -> torch.Tensor:
    """Resize into object_size then center-pad to out_size."""

    if image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.ndim == 3:
        image = image.unsqueeze(0)
    else:
        raise ValueError("image tensor must be 2D or 3D")

    resized = F.interpolate(image, size=(object_size, object_size), mode="bilinear", align_corners=False)
    out = torch.zeros((1, 1, out_size, out_size), dtype=resized.dtype)
    y0 = (out_size - object_size) // 2
    x0 = (out_size - object_size) // 2
    out[:, :, y0 : y0 + object_size, x0 : x0 + object_size] = resized
    return out.squeeze(0).squeeze(0)


def to_complex_field(amplitude: torch.Tensor) -> torch.Tensor:
    """Amplitude encoding to complex field."""

    amp = amplitude.clamp(0.0, 1.0).to(torch.float32)
    return torch.complex(amp, torch.zeros_like(amp))


def numpy_grayscale01(image: np.ndarray) -> np.ndarray:
    """Convert image array to grayscale float [0,1]."""

    arr = image.astype(np.float32)
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
    arr = arr - arr.min()
    denom = max(float(arr.max()), 1e-8)
    return arr / denom
