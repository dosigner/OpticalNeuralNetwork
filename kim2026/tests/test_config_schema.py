from __future__ import annotations

import pytest

from kim2026.config.schema import validate_config


def _base_config() -> dict:
    return {
        "experiment": {"id": "pilot"},
        "optics": {
            "lambda_m": 1.55e-6,
            "half_angle_rad": 3.0e-4,
            "m2": 1.0,
        },
        "grid": {
            "n": 512,
            "source_window_m": 0.03,
            "receiver_window_m": 0.72,
        },
        "channel": {
            "path_length_m": 1000.0,
            "cn2": 2.0e-14,
            "outer_scale_m": 30.0,
            "inner_scale_m": 5.0e-3,
            "num_screens": 8,
            "frozen_flow": {
                "wind_speed_mps": 10.0,
                "wind_direction_mode": "per_episode_random",
                "dt_s": 5.0e-4,
                "frames_per_episode": 32,
                "screen_canvas_scale": 2.0,
            },
        },
        "receiver": {"aperture_diameter_m": 0.6},
        "model": {
            "num_layers": 4,
            "layer_spacing_m": 0.15,
            "detector_distance_m": 0.30,
        },
        "training": {
            "epochs": 2,
            "batch_size": 16,
            "loss": {
                "weights": {
                    "overlap": 1.0,
                    "radius": 0.25,
                    "encircled": 0.25,
                }
            },
        },
        "data": {
            "cache_dir": "data/cache",
            "split_manifest_path": "data/cache/split_manifest.json",
        },
        "evaluation": {"metrics": ["overlap", "strehl", "beam_radius"]},
        "visualization": {"save_raw": True},
        "runtime": {
            "seed": 20260316,
            "strict_reproducibility": True,
            "allow_tf32": False,
            "deterministic_algorithms": True,
        },
    }


def test_validate_config_fills_reproducible_defaults() -> None:
    cfg = _base_config()
    result = validate_config(cfg)

    assert result["optics"]["dtype"] == "complex64"
    assert result["runtime"]["cublas_workspace_config"] == ":4096:8"
    assert result["training"]["pair_generation_batch_size"] == 64
    assert result["training"]["eval_batch_size"] == 32
    assert result["runtime"]["device"] == "cuda"


def test_validate_config_rejects_missing_required_sections() -> None:
    cfg = _base_config()
    del cfg["channel"]

    with pytest.raises(KeyError):
        validate_config(cfg)


def test_validate_config_rejects_non_positive_values() -> None:
    cfg = _base_config()
    cfg["receiver"]["aperture_diameter_m"] = 0.0

    with pytest.raises(ValueError):
        validate_config(cfg)


def test_validate_config_requires_dual_2f_for_fd2nn() -> None:
    cfg = _base_config()
    cfg["model"] = {
        "type": "fd2nn",
        "num_layers": 5,
        "layer_spacing_m": 0.0,
        "phase_max": 3.14159265,
        "phase_constraint": "unconstrained",
        "phase_init": "uniform",
        "phase_init_scale": 0.1,
    }

    with pytest.raises(KeyError):
        validate_config(cfg)


def test_validate_config_rejects_domain_sequence_for_fd2nn() -> None:
    cfg = _base_config()
    cfg["optics"]["dual_2f"] = {
        "enabled": True,
        "f1_m": 1.0e-3,
        "f2_m": 1.0e-3,
        "na1": 0.16,
        "na2": 0.16,
        "apply_scaling": False,
    }
    cfg["model"] = {
        "type": "fd2nn",
        "num_layers": 5,
        "layer_spacing_m": 1.0e-4,
        "domain_sequence": ["fourier"] * 5,
        "phase_max": 3.14159265,
        "phase_constraint": "unconstrained",
        "phase_init": "uniform",
        "phase_init_scale": 0.1,
    }

    with pytest.raises(ValueError, match="domain_sequence"):
        validate_config(cfg)


def test_validate_config_accepts_dual_2f_fd2nn() -> None:
    cfg = _base_config()
    cfg["optics"]["dual_2f"] = {
        "enabled": True,
        "f1_m": 1.0e-3,
        "f2_m": 1.0e-3,
        "na1": 0.16,
        "na2": 0.16,
        "apply_scaling": False,
    }
    cfg["model"] = {
        "type": "fd2nn",
        "num_layers": 5,
        "layer_spacing_m": 1.0e-4,
        "phase_max": 3.14159265,
        "phase_init": "uniform",
        "phase_init_scale": 0.1,
    }

    result = validate_config(cfg)

    assert result["optics"]["dual_2f"]["enabled"] is True
    assert result["optics"]["dual_2f"]["f1_m"] == pytest.approx(1.0e-3)
    assert result["model"]["phase_constraint"] == "unconstrained"


def test_validate_config_fd2nn_complex_mode_defaults_to_phase_first_weights() -> None:
    cfg = _base_config()
    cfg["optics"]["dual_2f"] = {
        "enabled": True,
        "f1_m": 1.0e-3,
        "f2_m": 1.0e-3,
        "na1": 0.16,
        "na2": 0.16,
        "apply_scaling": False,
    }
    cfg["model"] = {
        "type": "fd2nn",
        "num_layers": 5,
        "layer_spacing_m": 1.0e-4,
    }
    cfg["training"]["loss"]["mode"] = "complex"

    result = validate_config(cfg)

    assert result["training"]["loss"]["complex_weights"]["soft_phasor"] == pytest.approx(1.0)
    assert result["training"]["loss"]["complex_weights"]["amplitude_mse"] == pytest.approx(0.05)
    assert result["training"]["loss"]["complex_weights"]["leakage"] == pytest.approx(0.1)
    assert result["training"]["loss"]["complex_weights"]["support_gamma"] == pytest.approx(2.0)
    assert result["training"]["loss"]["complex_weights"]["full_field_phase"] == pytest.approx(0.0)
    assert result["training"]["loss"]["complex_weights"]["full_field_phase_gamma"] == pytest.approx(1.0)
    assert result["training"]["loss"]["complex_weights"]["full_field_phase_threshold"] == pytest.approx(0.05)
