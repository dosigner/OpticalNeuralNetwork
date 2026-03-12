"""Shared fixtures for luo2022 D2NN tests."""

from __future__ import annotations

import pytest
import torch

from luo2022_d2nn.utils.seed import set_global_seed


@pytest.fixture(autouse=True)
def deterministic_seed():
    """Reset seeds before each test."""
    set_global_seed(42, deterministic=False)


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def baseline_config():
    """Minimal valid baseline config dict."""
    return {
        "experiment": {"id": "test_run", "seed": 42},
        "optics": {"frequency_ghz": 400.0},
        "grid": {"nx": 240, "ny": 240, "pitch_mm": 0.3},
        "geometry": {
            "object_to_diffuser_mm": 40.0,
            "diffuser_to_layer1_mm": 2.0,
            "layer_to_layer_mm": 2.0,
            "num_layers": 4,
            "last_layer_to_output_mm": 7.0,
        },
        "dataset": {"name": "mnist"},
        "diffuser": {"type": "thin_random_phase"},
        "model": {"type": "d2nn_phase_only"},
        "training": {},
        "evaluation": {},
        "visualization": {},
    }
