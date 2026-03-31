"""Fourier-space FD2NN built from an ideal dual-2f optical train."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from kim2026.optics.angular_spectrum import propagate_same_window
from kim2026.optics.lens_2f import lens_2f_forward, lens_2f_inverse


class FourierPhaseMask(nn.Module):
    """Trainable phase-only mask with symmetric tanh constraint."""

    def __init__(
        self,
        n: int,
        *,
        phase_max: float = math.pi,
        constraint: str = "symmetric_tanh",
        init_mode: str = "uniform",
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.n = int(n)
        self.phase_max = float(phase_max)
        self.constraint = constraint
        self.raw = nn.Parameter(torch.empty(self.n, self.n))
        if init_mode == "zeros":
            nn.init.zeros_(self.raw)
        elif init_mode == "uniform":
            nn.init.uniform_(self.raw, -float(init_scale), float(init_scale))
        else:
            raise ValueError(f"init_mode must be 'zeros' or 'uniform', got '{init_mode}'")

    def phase(self) -> torch.Tensor:
        """Return constrained phase map."""
        if self.constraint == "symmetric_tanh":
            return self.phase_max * torch.tanh(self.raw)
        if self.constraint == "sigmoid":
            return self.phase_max * torch.sigmoid(self.raw)
        if self.constraint == "unconstrained":
            return self.raw
        raise ValueError(f"unknown constraint: {self.constraint}")

    def wrapped_phase(self) -> torch.Tensor:
        """Return fabrication-view phase wrapped to [0, 2pi)."""
        phase = self.phase()
        return torch.remainder(phase, 2.0 * math.pi)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        phi = self.phase().to(device=field.device, dtype=field.real.dtype)
        return field * torch.exp(1j * phi)


class CroppedFourierD2NN(nn.Module):
    """FD2NN with low-resolution Fourier masks (crop/pad in Fourier plane).

    Input(n_input) → Lens1 → FFT(n_input) → crop(n_mask) → masks(n_mask²) → pad(n_input) → IFFT → Lens2

    Solves Fourier under-resolution: beam spot ~7px uses 10.9% of 64² mask (vs 0.015% of 1024²).
    """

    def __init__(
        self,
        *,
        n_input: int,
        n_mask: int,
        wavelength_m: float,
        window_m: float,
        num_layers: int,
        layer_spacing_m: float = 0.0,
        phase_max: float = math.pi,
        phase_constraint: str = "unconstrained",
        phase_init: str = "uniform",
        phase_init_scale: float = 0.1,
        dual_2f_f1_m: float,
        dual_2f_f2_m: float,
        dual_2f_na1: float | None = None,
        dual_2f_na2: float | None = None,
        dual_2f_apply_scaling: bool = False,
    ) -> None:
        super().__init__()
        self.n_input = int(n_input)
        self.n_mask = int(n_mask)
        self.wavelength_m = float(wavelength_m)
        self.window_m = float(window_m)
        self.layer_spacing_m = float(layer_spacing_m)
        self.num_layers = int(num_layers)
        self.dual_2f_f1_m = float(dual_2f_f1_m)
        self.dual_2f_f2_m = float(dual_2f_f2_m)
        self.dual_2f_na1 = None if dual_2f_na1 is None else float(dual_2f_na1)
        self.dual_2f_na2 = None if dual_2f_na2 is None else float(dual_2f_na2)
        self.dual_2f_apply_scaling = bool(dual_2f_apply_scaling)

        # Masks are n_mask × n_mask (small)
        self.layers = nn.ModuleList([
            FourierPhaseMask(
                self.n_mask,
                phase_max=phase_max,
                constraint=phase_constraint,
                init_mode=phase_init,
                init_scale=phase_init_scale,
            )
            for _ in range(self.num_layers)
        ])

    def _crop_center(self, x: torch.Tensor) -> torch.Tensor:
        """Crop central n_mask×n_mask from n_input×n_input."""
        c = self.n_input // 2
        h = self.n_mask // 2
        return x[..., c - h:c + h, c - h:c + h]

    def _pad_center(self, x: torch.Tensor) -> torch.Tensor:
        """Pad n_mask×n_mask back to n_input×n_input with zeros."""
        pad_total = self.n_input - self.n_mask
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        return torch.nn.functional.pad(x, (pad_before, pad_after, pad_before, pad_after))

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        dx_m = self.window_m / self.n_input
        # Lens1: to Fourier plane (full n_input grid)
        out, dx_fourier_m = lens_2f_forward(
            field.to(torch.complex64),
            dx_in_m=dx_m,
            wavelength_m=self.wavelength_m,
            f_m=self.dual_2f_f1_m,
            na=self.dual_2f_na1,
            apply_scaling=self.dual_2f_apply_scaling,
        )
        # Crop to central n_mask × n_mask
        out_crop = self._crop_center(out)

        # Apply masks (in cropped Fourier plane)
        fourier_window_crop_m = dx_fourier_m * self.n_mask
        for idx, layer in enumerate(self.layers):
            if idx > 0 and self.layer_spacing_m > 0.0:
                out_crop = propagate_same_window(
                    out_crop,
                    wavelength_m=self.wavelength_m,
                    window_m=fourier_window_crop_m,
                    z_m=self.layer_spacing_m,
                )
            out_crop = layer(out_crop)

        # Pad back to n_input × n_input
        out = self._pad_center(out_crop)

        # Lens2: back to image plane
        out, _ = lens_2f_inverse(
            out,
            dx_fourier_m=dx_fourier_m,
            wavelength_m=self.wavelength_m,
            f_m=self.dual_2f_f2_m,
            na=self.dual_2f_na2,
            apply_scaling=self.dual_2f_apply_scaling,
        )
        return out.to(torch.complex64)


class BeamCleanupFD2NN(nn.Module):
    """Pure Fourier-space FD2NN using an ideal dual-2f optical train."""

    def __init__(
        self,
        *,
        n: int,
        wavelength_m: float,
        window_m: float,
        num_layers: int,
        layer_spacing_m: float = 0.0,
        phase_max: float = math.pi,
        phase_constraint: str = "unconstrained",
        phase_init: str = "uniform",
        phase_init_scale: float = 0.1,
        dual_2f_f1_m: float,
        dual_2f_f2_m: float,
        dual_2f_na1: float | None = None,
        dual_2f_na2: float | None = None,
        dual_2f_apply_scaling: bool = False,
    ) -> None:
        super().__init__()
        self.n = int(n)
        self.wavelength_m = float(wavelength_m)
        self.window_m = float(window_m)
        self.layer_spacing_m = float(layer_spacing_m)
        self.num_layers = int(num_layers)
        self.dual_2f_f1_m = float(dual_2f_f1_m)
        self.dual_2f_f2_m = float(dual_2f_f2_m)
        self.dual_2f_na1 = None if dual_2f_na1 is None else float(dual_2f_na1)
        self.dual_2f_na2 = None if dual_2f_na2 is None else float(dual_2f_na2)
        self.dual_2f_apply_scaling = bool(dual_2f_apply_scaling)

        if self.n <= 0:
            raise ValueError("n must be > 0")
        if self.wavelength_m <= 0.0:
            raise ValueError("wavelength_m must be > 0")
        if self.window_m <= 0.0:
            raise ValueError("window_m must be > 0")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        if self.layer_spacing_m < 0.0:
            raise ValueError("layer_spacing_m must be >= 0")
        if self.dual_2f_f1_m <= 0.0:
            raise ValueError("dual_2f_f1_m must be > 0")
        if self.dual_2f_f2_m <= 0.0:
            raise ValueError("dual_2f_f2_m must be > 0")

        self.layers = nn.ModuleList([
            FourierPhaseMask(
                self.n,
                phase_max=phase_max,
                constraint=phase_constraint,
                init_mode=phase_init,
                init_scale=phase_init_scale,
            )
            for _ in range(self.num_layers)
        ])

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """Map a receiver-plane input field through a dual-2f Fourier D2NN."""
        dx_m = self.window_m / self.n
        out, dx_fourier_m = lens_2f_forward(
            field.to(torch.complex64),
            dx_in_m=dx_m,
            wavelength_m=self.wavelength_m,
            f_m=self.dual_2f_f1_m,
            na=self.dual_2f_na1,
            apply_scaling=self.dual_2f_apply_scaling,
        )

        fourier_window_m = dx_fourier_m * self.n
        for idx, layer in enumerate(self.layers):
            if idx > 0 and self.layer_spacing_m > 0.0:
                out = propagate_same_window(
                    out,
                    wavelength_m=self.wavelength_m,
                    window_m=fourier_window_m,
                    z_m=self.layer_spacing_m,
                )
            out = layer(out)

        out, _ = lens_2f_inverse(
            out,
            dx_fourier_m=dx_fourier_m,
            wavelength_m=self.wavelength_m,
            f_m=self.dual_2f_f2_m,
            na=self.dual_2f_na2,
            apply_scaling=self.dual_2f_apply_scaling,
        )
        return out.to(torch.complex64)
