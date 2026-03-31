"""
layers.py
=========
Trainable diffractive layers for D²NN and Fourier D²NN models.

DiffractiveLayer
    A single diffractive plane whose neurons independently modulate the
    *phase* (and optionally the *amplitude*) of the incoming optical field.
    Free-space propagation between consecutive layers is performed with the
    Angular Spectrum Method.

FourierDiffractiveLayer
    A diffractive plane embedded inside a 4-f Fourier optical system.  The
    field is first Fourier-transformed (forward lens), modulated in the
    Fourier plane, then inverse-Fourier-transformed (backward lens) to
    yield the output field at the next layer's input plane.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .propagation import angular_spectrum_propagation, fourier_lens_propagation


class DiffractiveLayer(nn.Module):
    """Single free-space diffractive layer.

    Each neuron at position ``(i, j)`` has a learnable phase shift
    ``φ(i, j) ∈ [0, 2π)`` and, when *complex_modulation* is ``True``,
    also a learnable amplitude coefficient ``a(i, j) ∈ (0, 1]``.

    The transmission function is::

        t(i, j) = a(i, j) · exp(j·φ(i, j))

    After element-wise multiplication the field is propagated by *z* metres
    via the Angular Spectrum Method.

    Parameters
    ----------
    size:
        Spatial resolution ``(H, W)`` of the diffractive plane.
    wavelength:
        Optical wavelength (metres).
    z:
        Free-space propagation distance to the *next* layer (metres).
    dx:
        Physical pixel pitch in the *x* direction (metres).
    dy:
        Physical pixel pitch in the *y* direction (metres).  Defaults to
        *dx*.
    complex_modulation:
        When ``True`` the layer learns both phase and amplitude; when
        ``False`` only phase is learnable (physically more realistic).
    """

    def __init__(
        self,
        size: tuple[int, int],
        wavelength: float,
        z: float,
        dx: float,
        dy: float | None = None,
        complex_modulation: bool = False,
    ) -> None:
        super().__init__()
        H, W = size
        self.wavelength = wavelength
        self.z = z
        self.dx = dx
        self.dy = dy if dy is not None else dx
        self.complex_modulation = complex_modulation

        # Learnable phase: initialise uniformly in [0, 2π)
        self.phase = nn.Parameter(
            torch.empty(H, W).uniform_(0.0, 2.0 * math.pi)
        )

        if complex_modulation:
            # Learnable amplitude: initialise close to 1 (transparent)
            self.log_amplitude = nn.Parameter(torch.zeros(H, W))

    def transmission(self) -> torch.Tensor:
        """Return the complex transmission function ``t = a·exp(j·φ)``."""
        phi = self.phase  # (H, W)  – unbounded; periodicity is OK
        if self.complex_modulation:
            amplitude = torch.sigmoid(self.log_amplitude)  # (H, W) in (0,1)
        else:
            amplitude = torch.ones_like(phi)
        return torch.polar(amplitude, phi)  # complex (H, W)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        field:
            Incoming complex field ``(..., H, W)``.

        Returns
        -------
        torch.Tensor
            Field after modulation and free-space propagation.
        """
        t = self.transmission()  # (H, W)
        modulated = field * t
        propagated = angular_spectrum_propagation(
            modulated, self.wavelength, self.z, self.dx, self.dy
        )
        return propagated


class FourierDiffractiveLayer(nn.Module):
    """Diffractive layer operating in the Fourier (frequency) plane.

    Architecture (4-f system)::

        input field  →  [forward Fourier lens]  →  [phase mask]
                     →  [inverse Fourier lens]  →  output field

    The learnable *phase mask* modulates the spatial-frequency components of
    the field.  When stacked, consecutive 4-f blocks realise a deep
    convolutional-like processing chain entirely in the optical domain.

    Parameters
    ----------
    size:
        Spatial resolution ``(H, W)`` of the diffractive plane / mask.
    complex_modulation:
        When ``True`` the layer learns both Fourier-plane phase and
        amplitude; when ``False`` phase-only modulation is used.
    """

    def __init__(
        self,
        size: tuple[int, int],
        complex_modulation: bool = False,
    ) -> None:
        super().__init__()
        H, W = size
        self.complex_modulation = complex_modulation

        self.phase = nn.Parameter(
            torch.empty(H, W).uniform_(0.0, 2.0 * math.pi)
        )
        if complex_modulation:
            self.log_amplitude = nn.Parameter(torch.zeros(H, W))

    def transmission(self) -> torch.Tensor:
        """Return the complex Fourier-plane transmission function."""
        phi = self.phase
        if self.complex_modulation:
            amplitude = torch.sigmoid(self.log_amplitude)
        else:
            amplitude = torch.ones_like(phi)
        return torch.polar(amplitude, phi)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        field:
            Incoming complex field ``(..., H, W)``.

        Returns
        -------
        torch.Tensor
            Field after 4-f Fourier-plane processing.
        """
        # Forward lens: spatial domain → Fourier domain
        spectrum = fourier_lens_propagation(field, forward=True)
        # Modulate in Fourier plane
        t = self.transmission()
        modulated_spectrum = spectrum * t
        # Backward lens: Fourier domain → spatial domain
        output = fourier_lens_propagation(modulated_spectrum, forward=False)
        return output
