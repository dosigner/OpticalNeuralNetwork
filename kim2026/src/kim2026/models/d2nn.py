"""Phase-only D2NN model for beam cleanup."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from kim2026.optics import MIN_PAD_FACTOR, propagate_padded_same_window


class PhaseOnlyLayer(nn.Module):
    """A learnable phase-only modulation layer."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.phase = nn.Parameter(torch.zeros(n, n, dtype=torch.float32))

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        phase = torch.remainder(self.phase, 2.0 * math.pi)
        transmittance = torch.exp(1j * phase).to(field.device).to(field.dtype)
        return field * transmittance


class BeamCleanupD2NN(nn.Module):
    """Receiver-side phase-only D2NN."""

    def __init__(
        self,
        *,
        n: int,
        wavelength_m: float,
        window_m: float,
        num_layers: int,
        layer_spacing_m: float,
        detector_distance_m: float,
        propagation_pad_factor: int = MIN_PAD_FACTOR,
    ) -> None:
        super().__init__()
        self.n = int(n)
        self.wavelength_m = float(wavelength_m)
        self.window_m = float(window_m)
        self.num_layers = int(num_layers)
        self.layer_spacing_m = float(layer_spacing_m)
        self.detector_distance_m = float(detector_distance_m)
        self.propagation_pad_factor = int(propagation_pad_factor)
        self.layers = nn.ModuleList([PhaseOnlyLayer(self.n) for _ in range(self.num_layers)])

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        output = field
        for idx, layer in enumerate(self.layers):
            output = layer(output)
            if idx < self.num_layers - 1:
                output = propagate_padded_same_window(
                    output,
                    wavelength_m=self.wavelength_m,
                    window_m=self.window_m,
                    z_m=self.layer_spacing_m,
                    pad_factor=self.propagation_pad_factor,
                )
        output = propagate_padded_same_window(
            output,
            wavelength_m=self.wavelength_m,
            window_m=self.window_m,
            z_m=self.detector_distance_m,
            pad_factor=self.propagation_pad_factor,
        )
        return output.to(torch.complex64)
