"""Core D2NN model layers."""

from __future__ import annotations

import torch
from torch import nn

from d2nn.models.constraints import AmplitudeConstraint, PhaseConstraint
from d2nn.physics.asm import asm_propagate, asm_transfer_function
from d2nn.physics.materials import apply_absorption


class DiffractionLayer(nn.Module):
    """Single diffractive layer: propagate then apply complex modulation.

    Forward path:
        E_in -> ASM(z) -> E_plane -> modulator -> E_out
    """

    def __init__(
        self,
        N: int,
        dx: float,
        wavelength: float,
        z: float,
        *,
        phase_max: float = 2.0 * torch.pi,
        phase_constraint_mode: str = "sigmoid",
        phase_init: str = "zeros",
        train_amplitude: bool = False,
        amplitude_range: tuple[float, float] = (0.0, 1.0),
        use_absorption: bool = False,
        absorption_alpha: float | None = None,
        bandlimit: bool = True,
        fftshifted: bool = False,
        dtype: str = "complex64",
        name: str | None = None,
    ):
        super().__init__()
        self.N = int(N)
        self.dx = float(dx)
        self.wavelength = float(wavelength)
        self.z = float(z)
        self.phase_max = float(phase_max)
        self.phase_init = str(phase_init).lower()
        self.phase_constraint = PhaseConstraint(self.phase_max, mode=str(phase_constraint_mode).lower())
        self.amp_constraint = AmplitudeConstraint(amplitude_range)
        self.train_amplitude = bool(train_amplitude)
        self.use_absorption = bool(use_absorption)
        self.absorption_alpha = absorption_alpha
        self.bandlimit = bool(bandlimit)
        self.fftshifted = bool(fftshifted)
        self.dtype = dtype
        self.name = name or "diffraction_layer"

        self.raw_phase = nn.Parameter(torch.empty(self.N, self.N))
        if self.phase_init == "zeros":
            nn.init.zeros_(self.raw_phase)
        elif self.phase_init == "uniform":
            nn.init.uniform_(self.raw_phase, 0.0, self.phase_max)
        else:
            raise ValueError("phase_init must be one of: zeros, uniform")
        if self.train_amplitude:
            self.raw_amplitude = nn.Parameter(torch.zeros(self.N, self.N))
        else:
            self.register_parameter("raw_amplitude", None)

    def forward(self, field: torch.Tensor, shift_pixels: tuple[int, int] | None = None) -> torch.Tensor:
        """Apply propagation and modulation.

        Args:
            field: complex tensor, shape (..., N, N)
            shift_pixels: optional (dy, dx) integer shift to model layer misalignment
        """

        H = asm_transfer_function(
            N=self.N,
            dx=self.dx,
            wavelength=self.wavelength,
            z=self.z,
            bandlimit=self.bandlimit,
            fftshifted=self.fftshifted,
            dtype=self.dtype,
            device=field.device,
        )
        propagated = asm_propagate(field, H, fftshifted=self.fftshifted)
        if shift_pixels is not None:
            propagated = torch.roll(propagated, shifts=shift_pixels, dims=(-2, -1))

        phi = self.phase_constraint(self.raw_phase).to(device=propagated.device, dtype=propagated.real.dtype)
        phase_term = torch.exp(1j * phi)

        if self.train_amplitude and self.raw_amplitude is not None:
            amp = self.amp_constraint(self.raw_amplitude).to(device=propagated.device, dtype=propagated.real.dtype)
            mod = amp * phase_term
        else:
            mod = phase_term
        out = propagated * mod

        if self.use_absorption:
            out = apply_absorption(out, self.absorption_alpha)

        return out


class PropagationLayer(nn.Module):
    """Output propagation layer without trainable modulation."""

    def __init__(
        self,
        N: int,
        dx: float,
        wavelength: float,
        z: float,
        *,
        bandlimit: bool = True,
        fftshifted: bool = False,
        dtype: str = "complex64",
        name: str | None = None,
    ):
        super().__init__()
        self.N = int(N)
        self.dx = float(dx)
        self.wavelength = float(wavelength)
        self.z = float(z)
        self.bandlimit = bool(bandlimit)
        self.fftshifted = bool(fftshifted)
        self.dtype = dtype
        self.name = name or "propagation_layer"

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        H = asm_transfer_function(
            N=self.N,
            dx=self.dx,
            wavelength=self.wavelength,
            z=self.z,
            bandlimit=self.bandlimit,
            fftshifted=self.fftshifted,
            dtype=self.dtype,
            device=field.device,
        )
        return asm_propagate(field, H, fftshifted=self.fftshifted)
