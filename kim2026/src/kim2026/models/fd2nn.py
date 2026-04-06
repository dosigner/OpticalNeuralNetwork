"""Multi-layer Fourier-domain D2NN inside a single 4f optical system.

Architecture:
    Input → Lens1(f) → Mask1 → ASM(z) → Mask2 → ... → MaskN → Lens2(f) → Output

    - Mask1 sits at the focal plane of Lens1 (exact Fourier transform of input)
    - MaskN sits at the focal plane of Lens2
    - ASM propagation between masks creates genuine multi-layer diffraction
    - Full N×N masks (no crop); pixels outside the beam receive zero gradient
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from kim2026.optics.lens_2f import lens_2f_forward, lens_2f_inverse, fourier_plane_pitch
from kim2026.optics.angular_spectrum import propagate_same_window
from kim2026.optics.padded_angular_spectrum import MIN_PAD_FACTOR


class FourierPhaseLayer(nn.Module):
    """Learnable phase-only layer in the Fourier plane."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.phase = nn.Parameter(torch.zeros(n, n, dtype=torch.float32))

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        phase = torch.remainder(self.phase, 2.0 * math.pi)
        transmittance = torch.exp(1j * phase).to(field.device).to(field.dtype)
        return field * transmittance


class MultiLayerFD2NN(nn.Module):
    """FD2NN with ASM propagation between phase masks inside a single 4f system.

    Parameters
    ----------
    n : int
        Grid size (e.g. 1024).
    wavelength_m : float
        Optical wavelength in meters.
    window_m : float
        Spatial-domain field window in meters (N × dx).
    num_layers : int
        Number of phase masks.
    f_m : float
        Focal length of the 2f lenses (Lens1 and Lens2).
    layer_spacing_m : float
        Physical distance between adjacent masks in the Fourier plane.
    na : float or None
        Numerical aperture for the 2f lenses. None = no clipping.
    pad_asm : bool
        If True, use 2× padded ASM. If False, use unpadded ASM
        (safe when beam is small relative to grid, e.g. Fourier plane).
    """

    def __init__(
        self,
        *,
        n: int,
        wavelength_m: float,
        window_m: float,
        num_layers: int,
        f_m: float,
        layer_spacing_m: float,
        na: float | None = None,
        pad_asm: bool = False,
    ) -> None:
        super().__init__()
        self.n = int(n)
        self.wavelength_m = float(wavelength_m)
        self.window_m = float(window_m)
        self.num_layers = int(num_layers)
        self.f_m = float(f_m)
        self.layer_spacing_m = float(layer_spacing_m)
        self.na = float(na) if na is not None else None
        self.pad_asm = bool(pad_asm)

        # Precompute Fourier-plane sampling
        dx_in = self.window_m / self.n
        self.dx_fourier_m = fourier_plane_pitch(
            dx_in_m=dx_in, wavelength_m=self.wavelength_m,
            f_m=self.f_m, n=self.n,
        )
        self.fourier_window_m = self.n * self.dx_fourier_m

        self.layers = nn.ModuleList(
            [FourierPhaseLayer(self.n) for _ in range(self.num_layers)]
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        dx_in = self.window_m / self.n

        # Spatial → Fourier via Lens1
        fourier, dx_f = lens_2f_forward(
            field,
            dx_in_m=dx_in,
            wavelength_m=self.wavelength_m,
            f_m=self.f_m,
            na=self.na,
            apply_scaling=False,
        )

        # Multi-layer: phase mask → ASM propagation
        for idx, layer in enumerate(self.layers):
            fourier = layer(fourier)
            if idx < self.num_layers - 1 and self.layer_spacing_m > 0:
                if self.pad_asm:
                    from kim2026.optics.padded_angular_spectrum import propagate_padded_same_window
                    fourier = propagate_padded_same_window(
                        fourier,
                        wavelength_m=self.wavelength_m,
                        window_m=self.fourier_window_m,
                        z_m=self.layer_spacing_m,
                        pad_factor=MIN_PAD_FACTOR,
                    )
                else:
                    fourier = propagate_same_window(
                        fourier,
                        wavelength_m=self.wavelength_m,
                        window_m=self.fourier_window_m,
                        z_m=self.layer_spacing_m,
                    )

        # Fourier → Spatial via Lens2
        output, _dx_out = lens_2f_inverse(
            fourier,
            dx_fourier_m=dx_f,
            wavelength_m=self.wavelength_m,
            f_m=self.f_m,
            na=self.na,
            apply_scaling=False,
        )
        return output.to(torch.complex64)
