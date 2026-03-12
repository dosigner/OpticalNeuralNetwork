"""Tests for diffuser generation, correlation, and registry."""

import math

import pytest
import torch

from luo2022_d2nn.diffuser.random_phase import generate_diffuser
from luo2022_d2nn.diffuser.registry import DiffuserRegistry

N = 80
DX = 0.3
WAVELENGTH = 0.75


def test_diffuser_shapes():
    d = generate_diffuser(N, DX, WAVELENGTH, seed=0)
    assert d["height_map"].shape == (N, N)
    assert d["phase_map"].shape == (N, N)
    assert d["transmittance"].shape == (N, N)
    assert d["transmittance"].is_complex()
    assert isinstance(d["correlation_length_mm"], float)
    assert isinstance(d["seed"], int)


def test_transmittance_unit_modulus():
    """Transmittance of a phase-only element should have |t| = 1."""
    d = generate_diffuser(N, DX, WAVELENGTH, seed=1)
    mag = d["transmittance"].abs()
    assert torch.allclose(mag, torch.ones_like(mag), atol=1e-5)


def test_correlation_length_positive():
    """Correlation length should be positive and in a reasonable range.

    The exact value depends on smoothing_sigma, grid size, and random
    realisation; we just check it's physically sensible.
    """
    d = generate_diffuser(128, DX, WAVELENGTH, seed=2)
    L = d["correlation_length_mm"]
    assert L > 0, f"Correlation length should be positive, got {L}"
    # Should be at least a pixel and less than the grid
    assert L > DX, f"Correlation length {L:.3f} mm < pixel pitch"
    assert L < 128 * DX, f"Correlation length {L:.3f} mm exceeds grid"


def test_deterministic_seed():
    """Same seed should produce identical diffusers."""
    d1 = generate_diffuser(N, DX, WAVELENGTH, seed=42)
    d2 = generate_diffuser(N, DX, WAVELENGTH, seed=42)
    assert torch.allclose(d1["phase_map"], d2["phase_map"])
    assert torch.allclose(d1["transmittance"], d2["transmittance"])


def test_different_seeds_distinct():
    """Two diffusers from different seeds should produce different phase maps."""
    d1 = generate_diffuser(N, DX, WAVELENGTH, seed=100)
    d2 = generate_diffuser(N, DX, WAVELENGTH, seed=200)
    # Phase maps should NOT be identical
    assert not torch.allclose(d1["phase_map"], d2["phase_map"])
    # Phase difference should be significant
    p1 = d1["phase_map"].to(torch.float64)
    p2 = d2["phase_map"].to(torch.float64)
    avg_delta = (p1 - p2).abs().mean().item()
    assert avg_delta > 0.5, f"avg |Δφ| = {avg_delta:.3f} is too small"


def test_phase_map_mean():
    """Phase map mean should roughly equal 2π Δn μ / λ."""
    d = generate_diffuser(N, DX, WAVELENGTH, seed=3,
                          height_mean_lambda=25.0, height_std_lambda=8.0)
    expected_mean = 2.0 * math.pi * 0.74 * 25.0  # Δn * μ in λ-units → radians
    actual = d["phase_map"].to(torch.float64).mean().item()
    # Allow 30% tolerance (smoothing + finite grid)
    assert abs(actual - expected_mean) / expected_mean < 0.3, (
        f"Phase mean {actual:.2f}, expected ~{expected_mean:.2f}"
    )


# ------------------------------------------------------------------
# Registry tests
# ------------------------------------------------------------------

def test_registry_register_unique():
    """Diffusers from different seeds with high height_std should be unique."""
    reg = DiffuserRegistry()
    # Use higher height_std to ensure phase variance is large enough
    d1 = generate_diffuser(N, DX, WAVELENGTH, seed=10, height_std_lambda=12.0)
    d2 = generate_diffuser(N, DX, WAVELENGTH, seed=20, height_std_lambda=12.0)
    assert reg.register(d1) is True
    assert reg.register(d2) is True
    assert len(reg) == 2


def test_registry_reject_duplicate():
    reg = DiffuserRegistry()
    d1 = generate_diffuser(N, DX, WAVELENGTH, seed=10)
    d2 = generate_diffuser(N, DX, WAVELENGTH, seed=10)  # identical
    assert reg.register(d1) is True
    assert reg.register(d2) is False
    assert len(reg) == 1


def test_registry_get():
    reg = DiffuserRegistry()
    d = generate_diffuser(N, DX, WAVELENGTH, seed=5)
    reg.register(d)
    assert reg.get(0) is d
