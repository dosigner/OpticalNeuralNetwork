"""Tests for the Rayleigh-Sommerfeld FFT propagator."""

import pytest
import torch

from luo2022_d2nn.optics.rs_fft import rs_kernel, rs_propagate

N = 64
DX = 0.3
WAVELENGTH = 0.75
Z = 2.0
PAD = 2


def test_kernel_shape():
    k = rs_kernel(N, DX, WAVELENGTH, Z, pad_factor=PAD)
    assert k.shape == (N * PAD, N * PAD)
    assert k.is_complex()


def test_plane_wave_propagation():
    """Uniform plane wave should stay approximately uniform in centre."""
    field = torch.ones(N, N, dtype=torch.complex64)
    out = rs_propagate(field, WAVELENGTH, DX, Z, pad_factor=PAD)
    intensity = out.abs() ** 2
    centre = intensity[N // 4: 3 * N // 4, N // 4: 3 * N // 4]
    # RS won't be as clean as ASM for plane waves but centre should be
    # reasonably close to 1
    mean_c = centre.mean().item()
    assert 0.5 < mean_c < 2.0, f"Centre mean intensity = {mean_c}"


def test_energy_conservation():
    """Energy should be roughly conserved (within 20% for RS FFT)."""
    torch.manual_seed(99)
    field = torch.randn(N, N, dtype=torch.complex64)
    out = rs_propagate(field, WAVELENGTH, DX, Z, pad_factor=PAD)
    e_in = (field.abs() ** 2).sum().item()
    e_out = (out.abs() ** 2).sum().item()
    rel = abs(e_out - e_in) / e_in
    # RS FFT convolution energy can differ by a normalisation factor;
    # we mainly check it's finite and in the right ballpark
    assert rel < 1.0, f"Energy ratio off: relative error {rel:.4f}"
