"""Tests for the D2NN model forward pass."""

import torch
import pytest

from luo2022_d2nn.models.phase_layer import PhaseLayer
from luo2022_d2nn.models.d2nn import D2NN

GRID = 64
DX = 0.3
WAVELENGTH = 0.75


def _rand_complex(shape):
    return torch.randn(shape) + 1j * torch.randn(shape)


# --------------------------------------------------------------------------- #
# PhaseLayer tests
# --------------------------------------------------------------------------- #

def test_phase_layer_shape():
    layer = PhaseLayer(GRID)
    field = _rand_complex((2, GRID, GRID))
    out = layer(field)
    assert out.shape == (2, GRID, GRID)


def test_phase_layer_unit_magnitude():
    layer = PhaseLayer(GRID)
    field = _rand_complex((2, GRID, GRID))
    out = layer(field)
    # Phase-only modulation preserves magnitude
    torch.testing.assert_close(out.abs(), field.abs(), atol=1e-6, rtol=1e-5)


# --------------------------------------------------------------------------- #
# D2NN tests
# --------------------------------------------------------------------------- #

def test_d2nn_forward_shape():
    model = D2NN(num_layers=2, grid_size=GRID, dx_mm=DX, wavelength_mm=WAVELENGTH)
    field = _rand_complex((2, GRID, GRID))
    out = model(field)
    assert out.shape == (2, GRID, GRID)


def test_d2nn_output_is_complex():
    model = D2NN(num_layers=2, grid_size=GRID, dx_mm=DX, wavelength_mm=WAVELENGTH)
    field = _rand_complex((2, GRID, GRID))
    out = model(field)
    assert out.is_complex()


def test_d2nn_gradients_flow():
    model = D2NN(num_layers=2, grid_size=GRID, dx_mm=DX, wavelength_mm=WAVELENGTH)
    field = _rand_complex((2, GRID, GRID))
    out = model(field)
    loss = (out.abs() ** 2).sum()
    loss.backward()
    for i, layer in enumerate(model.layers):
        assert layer.phase.grad is not None, f"Layer {i} has no gradient"
        assert layer.phase.grad.abs().sum() > 0, f"Layer {i} gradient is all zeros"


def test_d2nn_4layer_default():
    model = D2NN(num_layers=4, grid_size=GRID, dx_mm=DX, wavelength_mm=WAVELENGTH)
    field = _rand_complex((2, GRID, GRID))
    out = model(field)
    assert out.shape == (2, GRID, GRID)
    assert out.is_complex()
