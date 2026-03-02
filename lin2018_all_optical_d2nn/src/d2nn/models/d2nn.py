"""Composable D2NN model."""

from __future__ import annotations

import torch
from torch import nn

from d2nn.models.layers import DiffractionLayer, PropagationLayer


class D2NNModel(nn.Module):
    """Diffractive deep neural network with optional output plane propagation."""

    def __init__(
        self,
        layers: list[DiffractionLayer],
        output_layer: PropagationLayer | None,
        *,
        max_misalignment_m: float = 0.0,
        dx: float = 1.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.output_layer = output_layer
        self.max_misalignment_m = float(max_misalignment_m)
        self.dx = float(dx)

    def _sample_misalignment_pixels(self, device: torch.device) -> tuple[int, int]:
        if self.max_misalignment_m <= 0.0:
            return 0, 0
        max_px = max(int(round(self.max_misalignment_m / self.dx)), 0)
        if max_px == 0:
            return 0, 0
        shifts = torch.randint(low=-max_px, high=max_px + 1, size=(2,), device=device)
        return int(shifts[0].item()), int(shifts[1].item())

    def forward(self, field: torch.Tensor, return_intermediates: bool = False) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Run D2NN forward pass.

        Args:
            field: complex tensor, shape (B, N, N)
            return_intermediates: if True, return per-layer complex fields
        """

        intermediates: list[torch.Tensor] = []
        out = field
        for layer in self.layers:
            shift = self._sample_misalignment_pixels(out.device) if self.training else (0, 0)
            out = layer(out, shift_pixels=shift)
            if return_intermediates:
                intermediates.append(out)

        if self.output_layer is not None:
            out = self.output_layer(out)
            if return_intermediates:
                intermediates.append(out)

        if return_intermediates:
            return out, intermediates
        return out


def build_d2nn_model(
    *,
    N: int,
    dx: float,
    wavelength: float,
    num_layers: int,
    z_layer: float,
    z_out: float,
    phase_max: float,
    phase_constraint_mode: str = "sigmoid",
    phase_init: str = "zeros",
    train_amplitude: bool = False,
    amplitude_range: tuple[float, float] = (0.0, 1.0),
    use_absorption: bool = False,
    absorption_alpha: float | None = None,
    bandlimit: bool = True,
    fftshifted: bool = False,
    dtype: str = "complex64",
    max_misalignment_m: float = 0.0,
) -> D2NNModel:
    """Build a standard D2NN stack from scalar config values."""

    layers = [
        DiffractionLayer(
            N=N,
            dx=dx,
            wavelength=wavelength,
            z=z_layer,
            phase_max=phase_max,
            phase_constraint_mode=phase_constraint_mode,
            phase_init=phase_init,
            train_amplitude=train_amplitude,
            amplitude_range=amplitude_range,
            use_absorption=use_absorption,
            absorption_alpha=absorption_alpha,
            bandlimit=bandlimit,
            fftshifted=fftshifted,
            dtype=dtype,
            name=f"L{idx + 1}",
        )
        for idx in range(num_layers)
    ]

    output_layer = PropagationLayer(
        N=N,
        dx=dx,
        wavelength=wavelength,
        z=z_out,
        bandlimit=bandlimit,
        fftshifted=fftshifted,
        dtype=dtype,
        name="output",
    )

    return D2NNModel(layers=layers, output_layer=output_layer, max_misalignment_m=max_misalignment_m, dx=dx)
