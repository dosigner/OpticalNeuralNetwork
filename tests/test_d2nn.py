"""
tests/test_d2nn.py
==================
Unit tests for the d2nn package.

Run with::

    pytest tests/test_d2nn.py -v
"""

from __future__ import annotations

import math

import pytest
import torch

from d2nn.propagation import angular_spectrum_propagation, fourier_lens_propagation
from d2nn.layers import DiffractiveLayer, FourierDiffractiveLayer
from d2nn.models import D2NN, FourierD2NN, _build_detector_masks, _detector_readout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIELD_SIZE = (28, 28)
BATCH = 2
NUM_CLASSES = 10
WAVELENGTH = 532e-9  # green light (metres)
Z = 0.1              # 10 cm propagation
DX = 8e-6            # 8 µm pixel pitch
ENERGY_CONSERVATION_TOL = 0.01  # 1 % relative energy change after propagation


def _random_complex(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.randn(*shape) + 1j * torch.randn(*shape)


def _random_real_images(N: int = BATCH) -> torch.Tensor:
    """Random images in [0, 1] with shape (N, 1, H, W)."""
    return torch.rand(N, 1, *FIELD_SIZE)


# ---------------------------------------------------------------------------
# propagation.py
# ---------------------------------------------------------------------------

class TestAngularSpectrumPropagation:
    def test_output_shape(self):
        field = _random_complex(FIELD_SIZE)
        out = angular_spectrum_propagation(field, WAVELENGTH, Z, DX)
        assert out.shape == field.shape

    def test_output_shape_batched(self):
        field = _random_complex((BATCH, *FIELD_SIZE))
        out = angular_spectrum_propagation(field, WAVELENGTH, Z, DX)
        assert out.shape == field.shape

    def test_output_is_complex(self):
        field = _random_complex(FIELD_SIZE)
        out = angular_spectrum_propagation(field, WAVELENGTH, Z, DX)
        assert out.is_complex()

    def test_zero_distance_identity(self):
        """Propagation over z=0 should (approximately) return the input field."""
        field = _random_complex(FIELD_SIZE)
        out = angular_spectrum_propagation(field, WAVELENGTH, 0.0, DX)
        assert torch.allclose(out, field, atol=1e-5)

    def test_energy_near_conservation(self):
        """Optical energy (intensity integral) should be approx. conserved."""
        field = _random_complex(FIELD_SIZE)
        energy_in = (field.abs() ** 2).sum()
        out = angular_spectrum_propagation(field, WAVELENGTH, Z, DX)
        energy_out = (out.abs() ** 2).sum()
        # Allow up to 1% relative change (evanescent-mode suppression removes some energy)
        assert abs(energy_out.item() - energy_in.item()) / energy_in.item() < ENERGY_CONSERVATION_TOL

    def test_asymmetric_pixel_pitch(self):
        field = _random_complex(FIELD_SIZE)
        out = angular_spectrum_propagation(field, WAVELENGTH, Z, DX, dy=DX * 2)
        assert out.shape == field.shape


class TestFourierLensPropagation:
    def test_output_shape(self):
        field = _random_complex(FIELD_SIZE)
        out = fourier_lens_propagation(field)
        assert out.shape == field.shape

    def test_forward_inverse_roundtrip(self):
        """Forward then inverse Fourier lens should recover the original field."""
        field = _random_complex(FIELD_SIZE)
        out = fourier_lens_propagation(fourier_lens_propagation(field, forward=True), forward=False)
        assert torch.allclose(out, field, atol=1e-5)

    def test_batched(self):
        field = _random_complex((BATCH, *FIELD_SIZE))
        out = fourier_lens_propagation(field)
        assert out.shape == field.shape


# ---------------------------------------------------------------------------
# layers.py
# ---------------------------------------------------------------------------

class TestDiffractiveLayer:
    def _make_layer(self, complex_mod: bool = False) -> DiffractiveLayer:
        return DiffractiveLayer(
            size=FIELD_SIZE,
            wavelength=WAVELENGTH,
            z=Z,
            dx=DX,
            complex_modulation=complex_mod,
        )

    def test_transmission_shape(self):
        layer = self._make_layer()
        t = layer.transmission()
        assert t.shape == torch.Size(FIELD_SIZE)
        assert t.is_complex()

    def test_transmission_unit_modulus_phase_only(self):
        layer = self._make_layer(complex_mod=False)
        t = layer.transmission()
        # Phase-only: |t| == 1 everywhere
        assert torch.allclose(t.abs(), torch.ones(*FIELD_SIZE), atol=1e-6)

    def test_transmission_bounded_amplitude_complex(self):
        layer = self._make_layer(complex_mod=True)
        t = layer.transmission()
        assert (t.abs() > 0).all()
        assert (t.abs() <= 1 + 1e-6).all()

    def test_forward_output_shape(self):
        layer = self._make_layer()
        field = _random_complex((BATCH, *FIELD_SIZE))
        out = layer(field)
        assert out.shape == field.shape

    def test_forward_is_differentiable(self):
        layer = self._make_layer()
        field = _random_complex((BATCH, *FIELD_SIZE))
        out = layer(field)
        loss = out.abs().sum()
        loss.backward()
        assert layer.phase.grad is not None

    def test_parameters_contain_phase(self):
        layer = self._make_layer()
        param_names = [n for n, _ in layer.named_parameters()]
        assert "phase" in param_names

    def test_complex_modulation_has_amplitude_param(self):
        layer = self._make_layer(complex_mod=True)
        param_names = [n for n, _ in layer.named_parameters()]
        assert "log_amplitude" in param_names


class TestFourierDiffractiveLayer:
    def _make_layer(self, complex_mod: bool = False) -> FourierDiffractiveLayer:
        return FourierDiffractiveLayer(size=FIELD_SIZE, complex_modulation=complex_mod)

    def test_forward_output_shape(self):
        layer = self._make_layer()
        field = _random_complex((BATCH, *FIELD_SIZE))
        out = layer(field)
        assert out.shape == field.shape

    def test_forward_is_differentiable(self):
        layer = self._make_layer()
        field = _random_complex((BATCH, *FIELD_SIZE))
        out = layer(field)
        loss = out.abs().sum()
        loss.backward()
        assert layer.phase.grad is not None

    def test_transmission_unit_modulus_phase_only(self):
        layer = self._make_layer(complex_mod=False)
        t = layer.transmission()
        assert torch.allclose(t.abs(), torch.ones(*FIELD_SIZE), atol=1e-6)


# ---------------------------------------------------------------------------
# models.py
# ---------------------------------------------------------------------------

class TestDetectorHelpers:
    def test_masks_shape(self):
        masks = _build_detector_masks(FIELD_SIZE, NUM_CLASSES)
        assert masks.shape == torch.Size([NUM_CLASSES, *FIELD_SIZE])

    def test_masks_bool(self):
        masks = _build_detector_masks(FIELD_SIZE, NUM_CLASSES)
        assert masks.dtype == torch.bool

    def test_masks_non_overlapping(self):
        masks = _build_detector_masks(FIELD_SIZE, NUM_CLASSES)
        overlap = masks.int().sum(dim=0)  # sum over classes
        assert (overlap <= 1).all(), "Detector regions must not overlap"

    def test_detector_readout_shape(self):
        masks = _build_detector_masks(FIELD_SIZE, NUM_CLASSES)
        field = _random_complex((BATCH, *FIELD_SIZE))
        scores = _detector_readout(field, masks)
        assert scores.shape == torch.Size([BATCH, NUM_CLASSES])

    def test_detector_readout_non_negative(self):
        masks = _build_detector_masks(FIELD_SIZE, NUM_CLASSES)
        field = _random_complex((BATCH, *FIELD_SIZE))
        scores = _detector_readout(field, masks)
        assert (scores >= 0).all()


class TestD2NN:
    def _make_model(self, num_layers: int = 2) -> D2NN:
        return D2NN(
            num_layers=num_layers,
            field_size=FIELD_SIZE,
            num_classes=NUM_CLASSES,
            wavelength=WAVELENGTH,
            z=Z,
            dx=DX,
        )

    def test_forward_output_shape(self):
        model = self._make_model()
        x = _random_real_images()
        logits = model(x)
        assert logits.shape == torch.Size([BATCH, NUM_CLASSES])

    def test_forward_nchw_input(self):
        model = self._make_model()
        x = _random_real_images()
        logits = model(x)  # (N,1,H,W) input
        assert logits.shape == torch.Size([BATCH, NUM_CLASSES])

    def test_forward_nhw_input(self):
        model = self._make_model()
        x = torch.rand(BATCH, *FIELD_SIZE)  # (N,H,W) input
        logits = model(x)
        assert logits.shape == torch.Size([BATCH, NUM_CLASSES])

    def test_backward(self):
        model = self._make_model()
        x = _random_real_images()
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_encode_output_is_complex(self):
        model = self._make_model()
        x = _random_real_images()
        field = model.encode(x)
        assert field.is_complex()

    def test_encode_amplitude_in_range(self):
        model = self._make_model()
        x = _random_real_images()
        field = model.encode(x)
        # Amplitude = sqrt(pixel) ∈ [0, 1] for pixels in [0, 1]
        assert (field.abs() >= 0).all()
        assert (field.abs() <= 1 + 1e-6).all()

    def test_num_layers(self):
        for n in [1, 3, 5]:
            model = D2NN(num_layers=n, field_size=FIELD_SIZE, num_classes=NUM_CLASSES,
                         wavelength=WAVELENGTH, z=Z, dx=DX)
            assert len(model.layers) == n

    def test_detector_masks_registered_as_buffer(self):
        model = self._make_model()
        assert "_detector_masks" in dict(model.named_buffers())

    def test_complex_modulation_mode(self):
        model = D2NN(
            num_layers=2, field_size=FIELD_SIZE, num_classes=NUM_CLASSES,
            wavelength=WAVELENGTH, z=Z, dx=DX, complex_modulation=True,
        )
        x = _random_real_images()
        logits = model(x)
        assert logits.shape == torch.Size([BATCH, NUM_CLASSES])


class TestFourierD2NN:
    def _make_model(self, num_layers: int = 2) -> FourierD2NN:
        return FourierD2NN(
            num_layers=num_layers,
            field_size=FIELD_SIZE,
            num_classes=NUM_CLASSES,
        )

    def test_forward_output_shape(self):
        model = self._make_model()
        x = _random_real_images()
        logits = model(x)
        assert logits.shape == torch.Size([BATCH, NUM_CLASSES])

    def test_backward(self):
        model = self._make_model()
        x = _random_real_images()
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_num_layers(self):
        for n in [1, 3, 5]:
            model = FourierD2NN(num_layers=n, field_size=FIELD_SIZE, num_classes=NUM_CLASSES)
            assert len(model.layers) == n

    def test_detector_masks_registered_as_buffer(self):
        model = self._make_model()
        assert "_detector_masks" in dict(model.named_buffers())

    def test_complex_modulation_mode(self):
        model = FourierD2NN(
            num_layers=2, field_size=FIELD_SIZE, num_classes=NUM_CLASSES,
            complex_modulation=True,
        )
        x = _random_real_images()
        logits = model(x)
        assert logits.shape == torch.Size([BATCH, NUM_CLASSES])
