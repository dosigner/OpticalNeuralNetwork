"""4-layer phase-only D2NN model (Luo et al. 2022)."""

from __future__ import annotations

import torch
import torch.nn as nn

from luo2022_d2nn.models.phase_layer import PhaseLayer
from luo2022_d2nn.optics.bl_asm import bl_asm_transfer_function, bl_asm_propagate


class D2NN(nn.Module):
    """Diffractive Deep Neural Network built from learnable phase layers.

    Parameters
    ----------
    num_layers : int
        Number of phase-only diffractive layers.
    grid_size : int
        Side length N of the computational grid.
    dx_mm : float
        Pixel pitch in mm.
    wavelength_mm : float
        Operating wavelength in mm.
    diffuser_to_layer1_mm : float
        Propagation distance from input (diffuser) to the first phase layer.
    layer_to_layer_mm : float
        Propagation distance between consecutive phase layers.
    last_layer_to_output_mm : float
        Propagation distance from the last phase layer to the output plane.
    pad_factor : int
        Zero-padding multiplier for BL-ASM propagation.
    init_phase_dist : str
        Phase initialization distribution for each layer.
    """

    def __init__(
        self,
        num_layers: int = 4,
        grid_size: int = 240,
        dx_mm: float = 0.3,
        wavelength_mm: float = 0.75,
        diffuser_to_layer1_mm: float = 2.0,
        layer_to_layer_mm: float = 2.0,
        last_layer_to_output_mm: float = 7.0,
        pad_factor: int = 2,
        init_phase_dist: str = "uniform_0_2pi",
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.grid_size = grid_size
        self.dx_mm = dx_mm
        self.wavelength_mm = wavelength_mm
        self.pad_factor = pad_factor

        # Build phase layers
        self.layers = nn.ModuleList(
            [PhaseLayer(grid_size, init_phase_dist) for _ in range(num_layers)]
        )

        # Pre-compute transfer functions for each propagation segment.
        # Segment 0: diffuser → layer 1
        # Segments 1..num_layers-1: layer i → layer i+1
        # Segment num_layers: last layer → output
        distances = [diffuser_to_layer1_mm]
        for _ in range(num_layers - 1):
            distances.append(layer_to_layer_mm)
        distances.append(last_layer_to_output_mm)

        # Store as plain list (not nn.ParameterList — these are fixed)
        self._transfer_functions: list[torch.Tensor] = []
        for z in distances:
            H = bl_asm_transfer_function(
                grid_size,
                dx_mm,
                wavelength_mm,
                z,
                pad_factor=pad_factor,
            )
            self._transfer_functions.append(H)

        # Register as buffers so they travel with .to(device)
        for i, H in enumerate(self._transfer_functions):
            self.register_buffer(f"_H_{i}", H)

    def _get_H(self, idx: int) -> torch.Tensor:
        return getattr(self, f"_H_{idx}")

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """Forward pass through the D2NN.

        Parameters
        ----------
        field : Tensor, shape (B, N, N) or (B*n, N, N), complex
            Input complex field after the diffuser plane.

        Returns
        -------
        Tensor, same shape — complex field at the output plane.
        """
        # Propagate diffuser → layer 1
        u = bl_asm_propagate(field, self._get_H(0), pad_factor=self.pad_factor)

        # Apply phase layers with inter-layer propagation
        for i, layer in enumerate(self.layers):
            u = layer(u)
            # Propagate to next layer (or to output after last layer)
            u = bl_asm_propagate(u, self._get_H(i + 1), pad_factor=self.pad_factor)

        return u
