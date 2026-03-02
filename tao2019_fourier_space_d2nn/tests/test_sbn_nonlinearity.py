from __future__ import annotations

import torch
import pytest

from tao2019_fd2nn.models.nonlinearity_sbn import SBNNonlinearity


def test_sbn_forward_shape_and_finite() -> None:
    layer = SBNNonlinearity(phi_max=float(torch.pi), background_intensity=0.0, saturation_intensity=1.0)
    x = torch.randn(2, 32, 32) + 1j * torch.randn(2, 32, 32)
    y = layer(x)
    assert y.shape == x.shape
    assert torch.isfinite(y.real).all()
    assert torch.isfinite(y.imag).all()


def test_sbn_background_perturbation_clamps_negative() -> None:
    layer = SBNNonlinearity(
        phi_max=float(torch.pi),
        norm_mode="background_perturbation",
        background_intensity=1.0,
        saturation_intensity=1.0,
        clamp_negative_perturbation=True,
    )
    x = torch.ones(1, 4, 4, dtype=torch.complex64)  # I=1.0 -> eta=0 after clamp
    y = layer(x)
    assert torch.allclose(y, x)


def test_sbn_physical_phi_scale_calibration() -> None:
    layer = SBNNonlinearity(
        phi_max=1.0,
        norm_mode="background_perturbation",
        background_intensity=0.0,
        saturation_intensity=1.0,
        voltage_v=972.0,
        electrode_gap_m=1.0e-3,
        thickness_m=1.0e-3,
        wavelength_m=5.32e-7,
        kappa_m_per_v=2.7366255e-10,
    )
    assert layer.phi_scale_rad == pytest.approx(float(torch.pi), rel=1e-5, abs=1e-5)


def test_sbn_physical_phi_scale_scales_with_thickness() -> None:
    full = SBNNonlinearity(
        phi_max=1.0,
        norm_mode="background_perturbation",
        background_intensity=0.0,
        saturation_intensity=1.0,
        voltage_v=972.0,
        electrode_gap_m=1.0e-3,
        thickness_m=1.0e-3,
        wavelength_m=5.32e-7,
        kappa_m_per_v=2.7366255e-10,
    )
    half = SBNNonlinearity(
        phi_max=1.0,
        norm_mode="background_perturbation",
        background_intensity=0.0,
        saturation_intensity=1.0,
        voltage_v=972.0,
        electrode_gap_m=1.0e-3,
        thickness_m=0.5e-3,
        wavelength_m=5.32e-7,
        kappa_m_per_v=2.7366255e-10,
    )
    assert half.phi_scale_rad == pytest.approx(0.5 * full.phi_scale_rad, rel=1e-6, abs=1e-6)
