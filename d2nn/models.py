"""
models.py
=========
End-to-end optical neural network models.

D2NN
    Diffractive Deep Neural Network with free-space propagation between
    layers (Lin et al., Science 2018).

FourierD2NN
    Fourier Diffractive Deep Neural Network where each diffractive layer
    operates inside a 4-f Fourier optical system, enabling spatial-frequency
    domain processing.

Both models accept a real-valued intensity image (or batch of images) as
input, encode it as an optical field, process it through stacked diffractive
layers, and read out predictions from a set of non-overlapping *detector
regions* at the output plane.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import DiffractiveLayer, FourierDiffractiveLayer


# ---------------------------------------------------------------------------
# Helper: detector readout
# ---------------------------------------------------------------------------

def _build_detector_masks(
    field_size: tuple[int, int],
    num_classes: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Create non-overlapping rectangular detector masks.

    The output plane is divided into a grid of ``num_classes`` regions.
    The intensity integrated over each region is the raw score for that
    class.

    Parameters
    ----------
    field_size:
        Spatial resolution ``(H, W)`` of the output field.
    num_classes:
        Number of classes (detector regions).
    device:
        Target device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Boolean mask of shape ``(num_classes, H, W)``.
    """
    H, W = field_size
    import math

    # Place detectors in a grid (as square as possible)
    cols = math.ceil(math.sqrt(num_classes))
    rows = math.ceil(num_classes / cols)

    masks = torch.zeros(num_classes, H, W, dtype=torch.bool, device=device)
    for k in range(num_classes):
        r = k // cols
        c = k % cols
        h_start = r * (H // rows)
        h_end = (r + 1) * (H // rows) if r < rows - 1 else H
        w_start = c * (W // cols)
        w_end = (c + 1) * (W // cols) if c < cols - 1 else W
        masks[k, h_start:h_end, w_start:w_end] = True

    return masks


def _detector_readout(
    field: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Sum intensities over each detector region.

    Parameters
    ----------
    field:
        Output complex field ``(N, H, W)``.
    masks:
        Boolean masks ``(num_classes, H, W)``.

    Returns
    -------
    torch.Tensor
        Raw logits ``(N, num_classes)``.
    """
    intensity = field.abs() ** 2  # (N, H, W)
    # masks: (C, H, W) → broadcast over batch
    scores = (intensity.unsqueeze(1) * masks.unsqueeze(0).float()).sum(dim=(-2, -1))
    return scores  # (N, C)


# ---------------------------------------------------------------------------
# D²NN
# ---------------------------------------------------------------------------

class D2NN(nn.Module):
    """Diffractive Deep Neural Network (free-space propagation).

    Architecture::

        input image  →  [amplitude encoding]
                     →  DiffractiveLayer 1  →  DiffractiveLayer 2  →  …
                     →  [detector readout]  →  class scores

    Parameters
    ----------
    num_layers:
        Number of diffractive planes.
    field_size:
        Spatial resolution ``(H, W)`` of every diffractive plane and the
        input/output field.
    num_classes:
        Number of output classes.
    wavelength:
        Optical wavelength (metres).
    z:
        Free-space distance between consecutive layers (metres).
    dx:
        Physical pixel pitch (metres).
    dy:
        Physical pixel pitch in *y* (metres).  Defaults to *dx*.
    complex_modulation:
        Enable amplitude + phase modulation per layer.
    """

    def __init__(
        self,
        num_layers: int = 5,
        field_size: tuple[int, int] = (28, 28),
        num_classes: int = 10,
        wavelength: float = 532e-9,
        z: float = 0.1,
        dx: float = 8e-6,
        dy: float | None = None,
        complex_modulation: bool = False,
    ) -> None:
        super().__init__()
        self.field_size = field_size
        self.num_classes = num_classes

        self.layers = nn.ModuleList(
            [
                DiffractiveLayer(
                    size=field_size,
                    wavelength=wavelength,
                    z=z,
                    dx=dx,
                    dy=dy,
                    complex_modulation=complex_modulation,
                )
                for _ in range(num_layers)
            ]
        )

        # Detector masks are buffers (not parameters) – created lazily
        self.register_buffer(
            "_detector_masks",
            _build_detector_masks(field_size, num_classes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a real-valued image as an amplitude-only optical field.

        Parameters
        ----------
        x:
            Real tensor ``(N, 1, H, W)`` or ``(N, H, W)`` with values in
            ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Complex field ``(N, H, W)``.
        """
        if x.dim() == 4:
            x = x.squeeze(1)  # (N, H, W)
        # Amplitude encoding: field amplitude ∝ √pixel_value
        amplitude = x.clamp(0.0, 1.0).sqrt()
        return torch.polar(amplitude, torch.zeros_like(amplitude))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Input image batch ``(N, 1, H, W)`` or ``(N, H, W)``,
            values in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Class logits ``(N, num_classes)``.
        """
        field = self.encode(x)
        for layer in self.layers:
            field = layer(field)
        masks = self._detector_masks.to(field.device)
        return _detector_readout(field, masks)


# ---------------------------------------------------------------------------
# Fourier D²NN
# ---------------------------------------------------------------------------

class FourierD2NN(nn.Module):
    """Fourier Diffractive Deep Neural Network (4-f Fourier-plane processing).

    Architecture::

        input image  →  [amplitude encoding]
                     →  FourierDiffractiveLayer 1  →  …
                     →  [detector readout]  →  class scores

    Each ``FourierDiffractiveLayer`` performs a full 4-f optical processing
    step: forward FFT → Fourier-plane phase mask → inverse FFT.

    Parameters
    ----------
    num_layers:
        Number of Fourier diffractive planes.
    field_size:
        Spatial resolution ``(H, W)`` of every plane.
    num_classes:
        Number of output classes.
    complex_modulation:
        Enable amplitude + phase modulation in the Fourier plane.
    """

    def __init__(
        self,
        num_layers: int = 5,
        field_size: tuple[int, int] = (28, 28),
        num_classes: int = 10,
        complex_modulation: bool = False,
    ) -> None:
        super().__init__()
        self.field_size = field_size
        self.num_classes = num_classes

        self.layers = nn.ModuleList(
            [
                FourierDiffractiveLayer(
                    size=field_size,
                    complex_modulation=complex_modulation,
                )
                for _ in range(num_layers)
            ]
        )

        self.register_buffer(
            "_detector_masks",
            _build_detector_masks(field_size, num_classes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a real image as an amplitude optical field."""
        if x.dim() == 4:
            x = x.squeeze(1)
        amplitude = x.clamp(0.0, 1.0).sqrt()
        return torch.polar(amplitude, torch.zeros_like(amplitude))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Input image batch ``(N, 1, H, W)`` or ``(N, H, W)``,
            values in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Class logits ``(N, num_classes)``.
        """
        field = self.encode(x)
        for layer in self.layers:
            field = layer(field)
        masks = self._detector_masks.to(field.device)
        return _detector_readout(field, masks)
