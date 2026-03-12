"""Tests for the Band-Limited ASM propagator."""

import math

import pytest
import torch

from luo2022_d2nn.optics.bl_asm import (
    bl_asm_propagate,
    bl_asm_transfer_function,
    clear_transfer_cache,
)
from luo2022_d2nn.optics.aperture import circular_aperture

# Canonical parameters
N = 64
DX = 0.3  # mm
WAVELENGTH = 0.75  # mm
Z = 2.0  # mm
PAD = 2


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_transfer_cache()
    yield
    clear_transfer_cache()


def test_transfer_function_shape():
    H = bl_asm_transfer_function(N, DX, WAVELENGTH, Z, pad_factor=PAD)
    assert H.shape == (N * PAD, N * PAD)
    assert H.is_complex()


def test_evanescent_masking():
    """High-frequency components (beyond 1/λ) should be zero."""
    H = bl_asm_transfer_function(N, DX, WAVELENGTH, Z, pad_factor=PAD)
    N_pad = N * PAD
    fx = torch.fft.fftfreq(N_pad, d=DX, dtype=torch.float64)
    FX, FY = torch.meshgrid(fx, fx, indexing="xy")
    evanescent = (FX ** 2 + FY ** 2) >= 1.0 / WAVELENGTH ** 2
    assert (H[evanescent].abs() < 1e-6).all()


def test_plane_wave_propagation():
    """A uniform plane wave should remain approximately uniform."""
    field = torch.ones(N, N, dtype=torch.complex64)
    H = bl_asm_transfer_function(N, DX, WAVELENGTH, Z, pad_factor=PAD)
    out = bl_asm_propagate(field, H, pad_factor=PAD)
    intensity = out.abs() ** 2
    # Central region should be close to 1
    centre = intensity[N // 4: 3 * N // 4, N // 4: 3 * N // 4]
    assert centre.mean().item() == pytest.approx(1.0, abs=0.05)


def test_energy_conservation():
    """Total energy should be conserved for a compact (band-limited) input.

    A spatially compact field (circular aperture) has most energy in low
    frequencies that are within the propagating band, so energy is conserved.
    Random white-noise fields lose energy to evanescent masking — that's
    physically correct, not a bug.
    """
    ap = circular_aperture(N, DX, radius_mm=5.0).to(torch.complex64)
    H = bl_asm_transfer_function(N, DX, WAVELENGTH, Z, pad_factor=PAD)
    out = bl_asm_propagate(ap, H, pad_factor=PAD)
    e_in = (ap.abs() ** 2).sum().item()
    e_out = (out.abs() ** 2).sum().item()
    rel = abs(e_out - e_in) / e_in
    assert rel < 0.02, f"Energy not conserved: relative error {rel:.4f}"


def test_cache_reuse():
    """Calling twice with same args should return cached tensor."""
    H1 = bl_asm_transfer_function(N, DX, WAVELENGTH, Z)
    H2 = bl_asm_transfer_function(N, DX, WAVELENGTH, Z)
    assert H1 is H2
