"""Cross-validation: BL-ASM vs RS-FFT propagation.

Four canonical test cases comparing normalised intensity patterns.
Uses short propagation distances where both methods agree well on small grids.
"""

import math

import pytest
import torch

from luo2022_d2nn.optics.bl_asm import (
    bl_asm_propagate,
    bl_asm_transfer_function,
    clear_transfer_cache,
)
from luo2022_d2nn.optics.rs_fft import rs_propagate

N = 64
DX = 0.3
WAVELENGTH = 0.75
PAD = 2


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_transfer_cache()
    yield
    clear_transfer_cache()


def _normalised_intensity(field: torch.Tensor) -> torch.Tensor:
    I = field.abs() ** 2
    I = I / I.max().clamp(min=1e-12)
    return I


def _max_relative_error(a: torch.Tensor, b: torch.Tensor, *, region: float = 0.5) -> float:
    """Max relative error in the central *region* fraction of the grid."""
    N = a.shape[-1]
    lo = int(N * (1 - region) / 2)
    hi = N - lo
    a_c = a[lo:hi, lo:hi]
    b_c = b[lo:hi, lo:hi]
    denom = torch.max(a_c.abs().max(), b_c.abs().max()).clamp(min=1e-12)
    return ((a_c - b_c).abs() / denom).max().item()


def _correlation(a: torch.Tensor, b: torch.Tensor, *, region: float = 0.5) -> float:
    """Pearson correlation in the central region."""
    N = a.shape[-1]
    lo = int(N * (1 - region) / 2)
    hi = N - lo
    a_c = a[lo:hi, lo:hi].flatten()
    b_c = b[lo:hi, lo:hi].flatten()
    return torch.corrcoef(torch.stack([a_c, b_c]))[0, 1].item()


def _random_phase_screen(N: int, seed: int = 0) -> torch.Tensor:
    """Simple random phase screen for testing."""
    gen = torch.Generator().manual_seed(seed)
    phase = torch.randn(N, N, generator=gen) * math.pi
    return torch.exp(1j * phase).to(torch.complex64)


# ------------------------------------------------------------------
# Case 1: Free-space propagation z=2mm
# ------------------------------------------------------------------
def test_case1_free_space():
    z = 2.0
    field = torch.ones(N, N, dtype=torch.complex64)

    H = bl_asm_transfer_function(N, DX, WAVELENGTH, z, pad_factor=PAD)
    out_asm = bl_asm_propagate(field, H, pad_factor=PAD)
    out_rs = rs_propagate(field, WAVELENGTH, DX, z, pad_factor=PAD)

    I_asm = _normalised_intensity(out_asm)
    I_rs = _normalised_intensity(out_rs)
    err = _max_relative_error(I_asm, I_rs, region=0.5)
    assert err < 0.05, f"Case 1 max relative error = {err:.4f}"


# ------------------------------------------------------------------
# Case 2: Through diffuser, z=5mm (moderate distance for small grid)
# ------------------------------------------------------------------
def test_case2_diffuser_propagation():
    z = 5.0
    diffuser = _random_phase_screen(N, seed=7)
    field = diffuser

    H = bl_asm_transfer_function(N, DX, WAVELENGTH, z, pad_factor=PAD)
    out_asm = bl_asm_propagate(field, H, pad_factor=PAD)
    out_rs = rs_propagate(field, WAVELENGTH, DX, z, pad_factor=PAD)

    I_asm = _normalised_intensity(out_asm)
    I_rs = _normalised_intensity(out_rs)
    # For moderate z with scattering, use correlation as primary metric
    corr = _correlation(I_asm, I_rs, region=0.5)
    assert corr > 0.98, f"Case 2 correlation = {corr:.4f}"


# ------------------------------------------------------------------
# Case 3: Single phase layer + propagation z=2mm
# ------------------------------------------------------------------
def test_case3_single_layer():
    z = 2.0
    phase_layer = _random_phase_screen(N, seed=42)
    field = torch.ones(N, N, dtype=torch.complex64) * phase_layer

    H = bl_asm_transfer_function(N, DX, WAVELENGTH, z, pad_factor=PAD)
    out_asm = bl_asm_propagate(field, H, pad_factor=PAD)
    out_rs = rs_propagate(field, WAVELENGTH, DX, z, pad_factor=PAD)

    I_asm = _normalised_intensity(out_asm)
    I_rs = _normalised_intensity(out_rs)
    err = _max_relative_error(I_asm, I_rs, region=0.5)
    assert err < 0.06, f"Case 3 max relative error = {err:.4f}"


# ------------------------------------------------------------------
# Case 4: 4-layer random init
# ------------------------------------------------------------------
def test_case4_four_layers():
    z = 2.0
    field = torch.ones(N, N, dtype=torch.complex64)

    H = bl_asm_transfer_function(N, DX, WAVELENGTH, z, pad_factor=PAD)

    field_asm = field.clone()
    field_rs = field.clone()

    for layer_seed in [10, 20, 30, 40]:
        phase = _random_phase_screen(N, seed=layer_seed)
        field_asm = field_asm * phase
        field_rs = field_rs * phase
        field_asm = bl_asm_propagate(field_asm, H, pad_factor=PAD)
        field_rs = rs_propagate(field_rs, WAVELENGTH, DX, z, pad_factor=PAD)

    I_asm = _normalised_intensity(field_asm)
    I_rs = _normalised_intensity(field_rs)
    # After 4 layers, small numerical differences accumulate;
    # check correlation rather than pointwise max error
    corr = _correlation(I_asm, I_rs, region=0.5)
    assert corr > 0.95, f"Case 4 correlation = {corr:.4f}"
