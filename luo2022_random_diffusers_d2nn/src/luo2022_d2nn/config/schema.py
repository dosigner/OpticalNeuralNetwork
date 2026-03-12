"""Config schema validation for luo2022 D2NN experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from luo2022_d2nn.utils.io import load_yaml

_REQ_TOP = ("experiment", "optics", "grid", "geometry", "dataset", "diffuser", "model", "training", "evaluation", "visualization")


def _require(cfg: dict[str, Any], key: str, context: str = "") -> Any:
    if key not in cfg:
        prefix = f"{context}." if context else ""
        raise KeyError(f"missing required key: {prefix}{key}")
    return cfg[key]


def _positive(v: Any, name: str) -> float:
    x = float(v)
    if x <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return x


def _non_negative(v: Any, name: str) -> float:
    x = float(v)
    if x < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return x


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize luo2022-style config."""
    out = deepcopy(cfg)
    for k in _REQ_TOP:
        _require(out, k)

    # experiment
    exp = out["experiment"]
    _require(exp, "id", "experiment")
    exp.setdefault("mode", "simulate")
    exp.setdefault("seed", 20220126)
    exp.setdefault("save_dir", f"runs/{exp['id']}")

    # optics
    optics = out["optics"]
    _positive(_require(optics, "frequency_ghz", "optics"), "optics.frequency_ghz")
    optics.setdefault("coherent", True)
    optics.setdefault("scalar_model", True)
    optics.setdefault("detector_type", "intensity")
    # Compute wavelength if not provided
    if "wavelength_mm" not in optics:
        c_mm_per_s = 299792458.0e3  # mm/s
        optics["wavelength_mm"] = c_mm_per_s / (optics["frequency_ghz"] * 1e9)

    # grid
    grid = out["grid"]
    nx = int(_require(grid, "nx", "grid"))
    ny = int(_require(grid, "ny", "grid"))
    if nx <= 0 or ny <= 0:
        raise ValueError("grid.nx and grid.ny must be > 0")
    _positive(_require(grid, "pitch_mm", "grid"), "grid.pitch_mm")
    grid.setdefault("pad_factor", 2)
    grid.setdefault("crop_after_propagation", True)

    # geometry
    geom = out["geometry"]
    _positive(_require(geom, "object_to_diffuser_mm", "geometry"), "geometry.object_to_diffuser_mm")
    _positive(_require(geom, "diffuser_to_layer1_mm", "geometry"), "geometry.diffuser_to_layer1_mm")
    _positive(_require(geom, "layer_to_layer_mm", "geometry"), "geometry.layer_to_layer_mm")
    num_layers = int(_require(geom, "num_layers", "geometry"))
    if num_layers <= 0:
        raise ValueError("geometry.num_layers must be > 0")
    _positive(_require(geom, "last_layer_to_output_mm", "geometry"), "geometry.last_layer_to_output_mm")

    # dataset
    ds = out["dataset"]
    _require(ds, "name", "dataset")
    ds.setdefault("train_count", 50000)
    ds.setdefault("val_count", 10000)
    ds.setdefault("test_count", 10000)
    ds.setdefault("source_resolution_px", 28)
    ds.setdefault("resize_to_px", 160)
    ds.setdefault("final_resolution_px", 240)
    ds.setdefault("resize_mode", "bilinear")
    ds.setdefault("amplitude_encoding", "grayscale")
    ds.setdefault("mask_strategy", "positive_pixels")
    ds.setdefault("deterministic_split", "first_50000_train_last_10000_val")

    # diffuser
    diff = out["diffuser"]
    _require(diff, "type", "diffuser")
    diff.setdefault("delta_n", 0.74)
    diff.setdefault("height_mean_lambda", 25.0)
    diff.setdefault("height_std_lambda", 8.0)
    diff.setdefault("smoothing_sigma_lambda", 4.0)
    diff.setdefault("target_correlation_length_lambda", 10.0)
    diff.setdefault("uniqueness_delta_phi_min_rad", 1.5707963267948966)

    # model
    model = out["model"]
    _require(model, "type", "model")
    model.setdefault("num_layers", geom["num_layers"])
    model.setdefault("phase_parameterization", "wrapped_phase")
    model.setdefault("init_phase_distribution", "uniform_0_2pi")

    # training
    tr = out["training"]
    _positive(tr.get("epochs", 100), "training.epochs")
    tr.setdefault("epochs", 100)
    tr.setdefault("batch_size_objects", 4)
    tr.setdefault("diffusers_per_epoch", 20)
    tr.setdefault("optimizer", "adam")
    tr.setdefault("learning_rate_initial", 1e-3)
    if "learning_rate_schedule" not in tr:
        tr["learning_rate_schedule"] = {"type": "epoch_multiplicative", "gamma": 0.99}
    if "loss" not in tr:
        tr["loss"] = {"type": "pcc_plus_energy", "alpha": 1.0, "beta": 0.5}

    # evaluation
    ev = out["evaluation"]
    ev.setdefault("metrics", ["pcc"])
    ev.setdefault("use_raw_images_for_metrics", True)
    ev.setdefault("known_diffuser_count_from_last_epoch", 20)
    ev.setdefault("blind_test_new_diffuser_count", 20)

    # visualization
    viz = out["visualization"]
    viz.setdefault("paper_style", True)
    viz.setdefault("save_raw", True)
    viz.setdefault("save_display", True)
    if "contrast_enhancement" not in viz:
        viz["contrast_enhancement"] = {
            "type": "percentile_stretch",
            "lower_percentile": 1.0,
            "upper_percentile": 99.0,
        }

    return out


def load_and_validate_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and validate as luo2022-style config."""
    cfg = load_yaml(path)
    return validate_config(cfg)
