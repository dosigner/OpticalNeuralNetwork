"""Preprocessing utilities for amplitude/phase encoding."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from d2nn.physics.apertures import center_pad_2d


def resize_to_square(image: torch.Tensor, out_size: int, mode: str = "nearest") -> torch.Tensor:
    """Resize a 2D image tensor to out_size x out_size.

    Args:
        image: tensor, shape (H, W), value range [0, 1]
        out_size: target side length [pixels]
    """

    if image.ndim != 2:
        raise ValueError("image must be 2D")
    x = image.unsqueeze(0).unsqueeze(0)
    y = F.interpolate(x, size=(out_size, out_size), mode=mode)
    return y.squeeze(0).squeeze(0)


def amplitude_encode(image: torch.Tensor, *, binarize: bool = False, threshold: float = 0.5) -> torch.Tensor:
    """Encode grayscale image into amplitude transmittance in [0, 1]."""

    amp = image.clamp(0.0, 1.0)
    if binarize:
        amp = (amp >= threshold).to(amp.dtype)
    return amp


def phase_encode(image: torch.Tensor, *, max_phase: float = 2.0 * torch.pi) -> torch.Tensor:
    """Encode grayscale image into phase [rad]."""

    return image.clamp(0.0, 1.0) * max_phase


def to_input_field(
    image: torch.Tensor,
    *,
    encoding: str,
    N: int,
    object_size: int,
    binarize: bool = False,
    phase_max: float = 2.0 * torch.pi,
) -> torch.Tensor:
    """Convert image to complex input field.

    Args:
        image: tensor, shape (H, W), value range [0, 1]
        encoding: "amplitude" or "phase"
        N: simulation grid size [pixels]
        object_size: object support size before padding [pixels]

    Returns:
        complex tensor, shape (N, N)
    """

    resized = resize_to_square(image, object_size, mode="nearest")
    padded = center_pad_2d(resized, target_N=N)

    if encoding == "amplitude":
        amp = amplitude_encode(padded, binarize=binarize)
        return amp.to(torch.complex64)
    elif encoding == "phase":
        phase = phase_encode(padded, max_phase=phase_max)
        return torch.exp(1j * phase.to(torch.complex64))
    else:
        raise ValueError("encoding must be 'amplitude' or 'phase'")
