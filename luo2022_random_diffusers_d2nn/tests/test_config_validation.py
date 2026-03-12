"""Tests for config schema validation."""

from __future__ import annotations

import pytest

from luo2022_d2nn.config.schema import validate_config


def test_valid_baseline_config(baseline_config):
    """Valid baseline config should pass validation."""
    result = validate_config(baseline_config)
    assert result["experiment"]["id"] == "test_run"
    assert result["optics"]["frequency_ghz"] == 400.0
    assert "wavelength_mm" in result["optics"]
    # Check auto-computed wavelength: c / f = 299792.458 mm/s / 400e9 Hz
    wl = result["optics"]["wavelength_mm"]
    assert abs(wl - 0.749481) < 0.001


def test_missing_required_top_key(baseline_config):
    """Missing top-level key should raise KeyError."""
    del baseline_config["optics"]
    with pytest.raises(KeyError, match="optics"):
        validate_config(baseline_config)


def test_missing_experiment_id(baseline_config):
    """Missing experiment.id should raise KeyError."""
    del baseline_config["experiment"]["id"]
    with pytest.raises(KeyError, match="experiment.id"):
        validate_config(baseline_config)


def test_negative_frequency(baseline_config):
    """Negative frequency should raise ValueError."""
    baseline_config["optics"]["frequency_ghz"] = -1.0
    with pytest.raises(ValueError, match="must be > 0"):
        validate_config(baseline_config)


def test_zero_grid_size(baseline_config):
    """Zero grid dimension should raise ValueError."""
    baseline_config["grid"]["nx"] = 0
    with pytest.raises(ValueError, match="must be > 0"):
        validate_config(baseline_config)


def test_defaults_applied(baseline_config):
    """Default values should be filled in."""
    result = validate_config(baseline_config)
    assert result["experiment"]["seed"] == 42
    assert result["training"]["epochs"] == 100
    assert result["training"]["batch_size_objects"] == 4
    assert result["training"]["diffusers_per_epoch"] == 20
    assert result["training"]["learning_rate_initial"] == 1e-3
    assert result["diffuser"]["delta_n"] == 0.74
    assert result["evaluation"]["metrics"] == ["pcc"]


def test_wavelength_auto_computed(baseline_config):
    """Wavelength should be auto-computed from frequency."""
    del baseline_config["optics"]
    baseline_config["optics"] = {"frequency_ghz": 400.0}
    result = validate_config(baseline_config)
    assert "wavelength_mm" in result["optics"]


def test_wavelength_preserved_if_provided(baseline_config):
    """Explicitly provided wavelength should be preserved."""
    baseline_config["optics"]["wavelength_mm"] = 0.75
    result = validate_config(baseline_config)
    assert result["optics"]["wavelength_mm"] == 0.75
