"""Config schema validation for kim2026 experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

_REQ_TOP = (
    "experiment",
    "optics",
    "grid",
    "channel",
    "receiver",
    "model",
    "training",
    "data",
    "evaluation",
    "visualization",
    "runtime",
)


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
    """Validate and normalize a kim2026 config."""
    out = deepcopy(cfg)
    for key in _REQ_TOP:
        _require(out, key)

    exp = out["experiment"]
    _require(exp, "id", "experiment")
    exp.setdefault("save_dir", f"runs/{exp['id']}")

    optics = out["optics"]
    _positive(_require(optics, "lambda_m", "optics"), "optics.lambda_m")
    _positive(_require(optics, "half_angle_rad", "optics"), "optics.half_angle_rad")
    _positive(_require(optics, "m2", "optics"), "optics.m2")
    optics.setdefault("dtype", "complex64")

    grid = out["grid"]
    n = int(_require(grid, "n", "grid"))
    if n <= 0:
        raise ValueError("grid.n must be > 0")
    _positive(_require(grid, "source_window_m", "grid"), "grid.source_window_m")
    _positive(_require(grid, "receiver_window_m", "grid"), "grid.receiver_window_m")

    channel = out["channel"]
    _positive(_require(channel, "path_length_m", "channel"), "channel.path_length_m")
    _non_negative(_require(channel, "cn2", "channel"), "channel.cn2")
    _positive(_require(channel, "outer_scale_m", "channel"), "channel.outer_scale_m")
    _positive(_require(channel, "inner_scale_m", "channel"), "channel.inner_scale_m")
    num_screens = int(_require(channel, "num_screens", "channel"))
    if num_screens <= 0:
        raise ValueError("channel.num_screens must be > 0")
    channel_mode = str(channel.get("mode", "frozen_flow"))
    if channel_mode not in ("frozen_flow", "static"):
        raise ValueError(f"channel.mode must be 'frozen_flow' or 'static', got '{channel_mode}'")
    channel["mode"] = channel_mode
    if channel_mode == "static":
        num_realizations = int(channel.get("num_realizations", 200))
        if num_realizations <= 0:
            raise ValueError("channel.num_realizations must be > 0")
        channel["num_realizations"] = num_realizations
    else:
        frozen = _require(channel, "frozen_flow", "channel")
        _positive(_require(frozen, "wind_speed_mps", "channel.frozen_flow"), "channel.frozen_flow.wind_speed_mps")
        _require(frozen, "wind_direction_mode", "channel.frozen_flow")
        _positive(_require(frozen, "dt_s", "channel.frozen_flow"), "channel.frozen_flow.dt_s")
        frames_per_episode = int(_require(frozen, "frames_per_episode", "channel.frozen_flow"))
        if frames_per_episode <= 0:
            raise ValueError("channel.frozen_flow.frames_per_episode must be > 0")
        _positive(_require(frozen, "screen_canvas_scale", "channel.frozen_flow"), "channel.frozen_flow.screen_canvas_scale")

    receiver = out["receiver"]
    _positive(_require(receiver, "aperture_diameter_m", "receiver"), "receiver.aperture_diameter_m")

    model = out["model"]
    model_type = str(model.get("type", "d2nn"))
    if model_type not in ("d2nn", "fd2nn"):
        raise ValueError(f"model.type must be 'd2nn' or 'fd2nn', got '{model_type}'")
    model["type"] = model_type
    num_layers = int(_require(model, "num_layers", "model"))
    if num_layers <= 0:
        raise ValueError("model.num_layers must be > 0")
    if model_type == "fd2nn":
        if "domain_sequence" in model:
            raise ValueError("model.domain_sequence is not supported for dual-2f fd2nn")
        _non_negative(_require(model, "layer_spacing_m", "model"), "model.layer_spacing_m")
        model.setdefault("phase_max", 3.14159265)
        model.setdefault("phase_constraint", "unconstrained")
        if str(model["phase_constraint"]) not in ("unconstrained", "symmetric_tanh", "sigmoid"):
            raise ValueError("model.phase_constraint must be one of: unconstrained, symmetric_tanh, sigmoid")
        model.setdefault("phase_init", "uniform")
        model.setdefault("phase_init_scale", 0.1)
        dual_2f = _require(optics, "dual_2f", "optics")
        enabled = bool(_require(dual_2f, "enabled", "optics.dual_2f"))
        if not enabled:
            raise ValueError("optics.dual_2f.enabled must be true for model.type='fd2nn'")
        _positive(_require(dual_2f, "f1_m", "optics.dual_2f"), "optics.dual_2f.f1_m")
        _positive(_require(dual_2f, "f2_m", "optics.dual_2f"), "optics.dual_2f.f2_m")
        dual_2f["na1"] = None if dual_2f.get("na1") is None else _positive(dual_2f["na1"], "optics.dual_2f.na1")
        dual_2f["na2"] = None if dual_2f.get("na2") is None else _positive(dual_2f["na2"], "optics.dual_2f.na2")
        dual_2f["apply_scaling"] = bool(_require(dual_2f, "apply_scaling", "optics.dual_2f"))
    else:
        _positive(_require(model, "layer_spacing_m", "model"), "model.layer_spacing_m")
        _positive(_require(model, "detector_distance_m", "model"), "model.detector_distance_m")
        model.setdefault("phase_wrap", True)

    training = out["training"]
    training.setdefault("epochs", 1)
    _positive(_require(training, "batch_size", "training"), "training.batch_size")
    training.setdefault("pair_generation_batch_size", 64)
    training.setdefault("eval_batch_size", 32)
    training.setdefault("learning_rate", 1e-3)
    loss = _require(training, "loss", "training")
    loss_mode = str(loss.get("mode", "intensity"))
    if loss_mode not in ("intensity", "complex", "roi_complex"):
        raise ValueError(f"training.loss.mode must be 'intensity', 'complex', or 'roi_complex', got '{loss_mode}'")
    loss["mode"] = loss_mode
    if loss_mode == "roi_complex":
        roi_t = float(loss.get("roi_threshold", 0.5))
        if not (0.0 < roi_t <= 1.0):
            raise ValueError("training.loss.roi_threshold must be in (0, 1]")
        loss["roi_threshold"] = roi_t
        iw = float(loss.get("intensity_weight", 1.0))
        _non_negative(iw, "training.loss.intensity_weight")
        loss["intensity_weight"] = iw
        pw = float(loss.get("phase_weight", 0.2))
        _non_negative(pw, "training.loss.phase_weight")
        loss["phase_weight"] = pw
        lw = float(loss.get("leakage_weight", 0.3))
        _non_negative(lw, "training.loss.leakage_weight")
        loss["leakage_weight"] = lw
        lt = float(loss.get("leakage_threshold", 0.15))
        if not (0.0 <= lt <= 1.0):
            raise ValueError("training.loss.leakage_threshold must be in [0, 1]")
        loss["leakage_threshold"] = lt
        pg = float(loss.get("phase_gamma", 2.0))
        _non_negative(pg, "training.loss.phase_gamma")
        loss["phase_gamma"] = pg
        ffpw = float(loss.get("full_field_phase_weight", 0.0))
        _non_negative(ffpw, "training.loss.full_field_phase_weight")
        loss["full_field_phase_weight"] = ffpw
        ffpg = float(loss.get("full_field_phase_gamma", 1.0))
        _non_negative(ffpg, "training.loss.full_field_phase_gamma")
        loss["full_field_phase_gamma"] = ffpg
        ffpt = float(loss.get("full_field_phase_threshold", 0.05))
        if not (0.0 <= ffpt <= 1.0):
            raise ValueError("training.loss.full_field_phase_threshold must be in [0, 1]")
        loss["full_field_phase_threshold"] = ffpt
        loss.setdefault("weights", {"overlap": 0.0, "radius": 0.0, "encircled": 0.0})
    elif loss_mode == "complex":
        if model_type == "fd2nn":
            complex_weights = loss.get(
                "complex_weights",
                {
                    "soft_phasor": 1.0,
                    "amplitude_mse": 0.05,
                    "leakage": 0.1,
                    "support_gamma": 2.0,
                    "full_field_phase": 0.0,
                    "full_field_phase_gamma": 1.0,
                    "full_field_phase_threshold": 0.05,
                },
            )
            _non_negative(complex_weights.get("soft_phasor", 1.0), "training.loss.complex_weights.soft_phasor")
            _non_negative(complex_weights.get("amplitude_mse", 0.05), "training.loss.complex_weights.amplitude_mse")
            _non_negative(complex_weights.get("leakage", 0.1), "training.loss.complex_weights.leakage")
            _non_negative(complex_weights.get("support_gamma", 2.0), "training.loss.complex_weights.support_gamma")
            _non_negative(complex_weights.get("full_field_phase", 0.0), "training.loss.complex_weights.full_field_phase")
            _non_negative(
                complex_weights.get("full_field_phase_gamma", 1.0),
                "training.loss.complex_weights.full_field_phase_gamma",
            )
            ffpt = float(complex_weights.get("full_field_phase_threshold", 0.05))
            if not (0.0 <= ffpt <= 1.0):
                raise ValueError("training.loss.complex_weights.full_field_phase_threshold must be in [0, 1]")
            complex_weights["full_field_phase_threshold"] = ffpt
            complex_weights.setdefault("full_field_phase", 0.0)
            complex_weights.setdefault("full_field_phase_gamma", 1.0)
        else:
            complex_weights = loss.get("complex_weights", {"complex_overlap": 1.0, "amplitude_mse": 0.5})
            _non_negative(complex_weights.get("complex_overlap", 1.0), "training.loss.complex_weights.complex_overlap")
            _non_negative(complex_weights.get("amplitude_mse", 0.5), "training.loss.complex_weights.amplitude_mse")
        loss["complex_weights"] = complex_weights
    weights = loss.get("weights", {"overlap": 1.0, "radius": 0.25, "encircled": 0.25})
    loss["weights"] = weights
    _non_negative(float(weights.get("overlap", 0.0)), "training.loss.weights.overlap")
    _non_negative(float(weights.get("radius", 0.0)), "training.loss.weights.radius")
    _non_negative(float(weights.get("encircled", 0.0)), "training.loss.weights.encircled")

    data = out["data"]
    _require(data, "cache_dir", "data")
    _require(data, "split_manifest_path", "data")
    data.setdefault("episode_manifest_path", str(Path(data["cache_dir"]) / "episodes.json"))
    data.setdefault("split_episode_counts", {"train": 100, "val": 20, "test": 20})

    evaluation = out["evaluation"]
    evaluation.setdefault("metrics", ["overlap", "strehl", "beam_radius", "encircled_energy"])
    evaluation.setdefault("split", "val")
    evaluation.setdefault("save_json", True)

    viz = out["visualization"]
    viz.setdefault("save_raw", True)
    viz.setdefault("save_plots", True)
    viz.setdefault("output_dir", str(Path(exp["save_dir"]) / "figures"))

    runtime = out["runtime"]
    runtime.setdefault("seed", 20260316)
    runtime.setdefault("strict_reproducibility", True)
    runtime.setdefault("allow_tf32", False)
    runtime.setdefault("deterministic_algorithms", True)
    runtime.setdefault("cublas_workspace_config", ":4096:8")
    runtime.setdefault("device", "cuda")
    runtime.setdefault("num_workers", 4)
    runtime.setdefault("pin_memory", True)
    runtime.setdefault("persistent_workers", False)
    runtime.setdefault("prefetch_factor", 2)
    runtime.setdefault("fft_warmup_iters", 3)

    return out


def load_and_validate_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and validate it."""
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return validate_config(data)
