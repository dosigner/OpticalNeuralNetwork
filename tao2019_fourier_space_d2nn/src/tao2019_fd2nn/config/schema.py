"""Config schema validation for spec.md-style templates."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from tao2019_fd2nn.utils.io import load_yaml

_REQ_TOP = ("experiment", "optics", "model", "task", "data", "training", "eval", "viz")
_TASKS = {"classification", "saliency"}
_MODEL_TYPES = {"fd2nn", "real_d2nn", "hybrid_d2nn"}
_EVANESCENT = {"mask", "decay", "keep"}
_INTENSITY_NORM = {"none", "background_perturbation", "per_sample_minmax", "per_minmax"}
_PERTURBATION_MODES = {"always", "test_only"}
_SALIENCY_LOSS_NORMALIZATION = {"pred_only", "pred_and_target"}
_SALIENCY_LOSS_SCOPE = {"crop", "full"}


def _require(out: dict[str, Any], key: str) -> Any:
    if key not in out:
        raise KeyError(f"missing required key: {key}")
    return out[key]


def _positive(v: Any, name: str) -> float:
    x = float(v)
    if x <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return x


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize spec-style config."""

    out = deepcopy(cfg)
    for k in _REQ_TOP:
        _require(out, k)

    exp = out["experiment"]
    _require(exp, "name")
    exp.setdefault("seed", 42)
    exp.setdefault("device", "auto")
    exp.setdefault("dtype", "float32")
    exp.setdefault("deterministic", True)
    exp.setdefault("save_dir", "runs")

    optics = out["optics"]
    _positive(_require(optics, "wavelength_m"), "optics.wavelength_m")
    _positive(optics.get("n", 1.0), "optics.n")
    optics.setdefault("alignment_shift_um", 0.0)
    float(optics["alignment_shift_um"])
    optics.setdefault("alignment_shift_um", 0.0)
    if float(optics["alignment_shift_um"]) < 0.0:
        raise ValueError("optics.alignment_shift_um must be >= 0")

    grid = _require(optics, "grid")
    nx = int(_require(grid, "nx"))
    ny = int(_require(grid, "ny"))
    if nx <= 0 or ny <= 0:
        raise ValueError("optics.grid.{nx,ny} must be > 0")
    _positive(_require(grid, "dx_m"), "optics.grid.dx_m")
    _positive(_require(grid, "dy_m"), "optics.grid.dy_m")

    prop = _require(optics, "propagation")
    _require(prop, "method")
    _ls = float(_require(prop, "layer_spacing_m"))
    if _ls < 0.0:
        raise ValueError("optics.propagation.layer_spacing_m must be >= 0")
    prop.setdefault("bandlimit", True)
    ev = str(prop.get("evanescent", "mask"))
    if ev not in _EVANESCENT:
        raise ValueError(f"optics.propagation.evanescent must be one of {_EVANESCENT}")

    dual_2f = optics.get("dual_2f", {})
    if dual_2f.get("enabled", False):
        _positive(_require(dual_2f, "f1_m"), "optics.dual_2f.f1_m")
        _positive(_require(dual_2f, "f2_m"), "optics.dual_2f.f2_m")
        if "na1" in dual_2f:
            _positive(dual_2f["na1"], "optics.dual_2f.na1")
        if "na2" in dual_2f:
            _positive(dual_2f["na2"], "optics.dual_2f.na2")
        dual_2f.setdefault("apply_scaling", False)
        if not isinstance(dual_2f["apply_scaling"], bool):
            raise ValueError("optics.dual_2f.apply_scaling must be bool")

    hybrid_2f = optics.get("hybrid_2f", {})
    if hybrid_2f.get("enabled", False):
        _positive(_require(hybrid_2f, "f_m"), "optics.hybrid_2f.f_m")
        _positive(_require(hybrid_2f, "na"), "optics.hybrid_2f.na")
        if int(_require(hybrid_2f, "num_2f_systems")) <= 0:
            raise ValueError("optics.hybrid_2f.num_2f_systems must be > 0")

    model = out["model"]
    model_type = str(_require(model, "type"))
    if model_type not in _MODEL_TYPES:
        raise ValueError(f"model.type must be one of {_MODEL_TYPES}")
    if int(_require(model, "num_layers")) <= 0:
        raise ValueError("model.num_layers must be > 0")
    model.setdefault("fabrication_blur_sigma_px", 0.0)
    model.setdefault("fabrication_blur_kernel_size", 3)
    if float(model["fabrication_blur_sigma_px"]) < 0.0:
        raise ValueError("model.fabrication_blur_sigma_px must be >= 0")
    kernel_size = int(model["fabrication_blur_kernel_size"])
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("model.fabrication_blur_kernel_size must be a positive odd integer")

    modulation = _require(model, "modulation")
    _require(modulation, "kind")
    _require(modulation, "phase_constraint")
    _positive(_require(modulation, "phase_max_rad"), "model.modulation.phase_max_rad")
    modulation.setdefault("init", "uniform")
    modulation.setdefault("init_scale", 0.1)
    model.setdefault("fabrication_blur_sigma_px", 0.0)
    model.setdefault("fabrication_blur_kernel_size", 3)
    if float(model["fabrication_blur_sigma_px"]) < 0.0:
        raise ValueError("model.fabrication_blur_sigma_px must be >= 0")
    kernel_size = int(model["fabrication_blur_kernel_size"])
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("model.fabrication_blur_kernel_size must be a positive odd integer")

    nonlin = model.get("nonlinearity", {})
    if nonlin.get("enabled", False):
        nonlin.setdefault("type", "sbn60")
        nonlin.setdefault("position", "rear")
        _positive(nonlin.get("phi_max_rad", 3.141592653589793), "model.nonlinearity.phi_max_rad")
        nonlin.setdefault("background_intensity", 0.0)
        nonlin.setdefault("saturation_intensity", 1.0)
        nonlin.setdefault("clamp_negative_perturbation", True)
        if float(nonlin["saturation_intensity"]) <= 0.0:
            raise ValueError("model.nonlinearity.saturation_intensity must be > 0")
        if not isinstance(nonlin["clamp_negative_perturbation"], bool):
            raise ValueError("model.nonlinearity.clamp_negative_perturbation must be bool")
        if "voltage_v" in nonlin:
            _positive(nonlin["voltage_v"], "model.nonlinearity.voltage_v")
        if "electrode_gap_m" in nonlin:
            _positive(nonlin["electrode_gap_m"], "model.nonlinearity.electrode_gap_m")
        if "e_app_v_per_m" in nonlin:
            _positive(nonlin["e_app_v_per_m"], "model.nonlinearity.e_app_v_per_m")
        if "kappa_m_per_v" in nonlin:
            _positive(nonlin["kappa_m_per_v"], "model.nonlinearity.kappa_m_per_v")
        if "thickness_m" in nonlin:
            _positive(nonlin["thickness_m"], "model.nonlinearity.thickness_m")
        if "wavelength_m" in nonlin:
            _positive(nonlin["wavelength_m"], "model.nonlinearity.wavelength_m")
        norm = str(nonlin.get("intensity_norm", "background_perturbation"))
        if norm not in _INTENSITY_NORM:
            raise ValueError(f"model.nonlinearity.intensity_norm must be one of {_INTENSITY_NORM}")

    task = out["task"]
    task_name = str(_require(task, "name"))
    if task_name not in _TASKS:
        raise ValueError(f"task.name must be one of {_TASKS}")
    if task_name == "classification":
        detector = _require(task, "detector")
        _positive(_require(detector, "width_um"), "task.detector.width_um")
    else:
        task.setdefault("gamma_flip", True)

    data = out["data"]
    _require(data, "dataset")
    _require(data, "preprocess")
    prep = data["preprocess"]
    _require(prep, "normalize")
    if "pad_to" in prep:
        if len(prep["pad_to"]) != 2:
            raise ValueError("data.preprocess.pad_to must be [H,W]")
    if task_name == "classification" and data["dataset"] != "mnist":
        raise ValueError("classification task currently requires data.dataset='mnist'")

    training = out["training"]
    _positive(_require(training, "lr"), "training.lr")
    if int(_require(training, "batch_size")) <= 0:
        raise ValueError("training.batch_size must be > 0")
    if int(_require(training, "epochs")) <= 0:
        raise ValueError("training.epochs must be > 0")
    _require(training, "loss")
    if task_name == "saliency":
        loss_name = str(training["loss"]).lower()
        if loss_name != "mse":
            raise ValueError("saliency task currently requires training.loss='mse'")
    training.setdefault("log_interval_steps", 20)
    training.setdefault("color_logs", True)
    training.setdefault("show_cuda_memory", True)
    if int(training["log_interval_steps"]) <= 0:
        raise ValueError("training.log_interval_steps must be > 0")
    if task_name == "saliency":
        training.setdefault("compute_train_fmax", False)
        training.setdefault("eval_interval_epochs", 5)
        training.setdefault("loss_normalization", "pred_only")
        training.setdefault("loss_scope", "crop")
        if int(training["eval_interval_epochs"]) <= 0:
            raise ValueError("training.eval_interval_epochs must be > 0")
        if str(training["loss_normalization"]) not in _SALIENCY_LOSS_NORMALIZATION:
            raise ValueError(f"training.loss_normalization must be one of {_SALIENCY_LOSS_NORMALIZATION}")
        if str(training["loss_scope"]) not in _SALIENCY_LOSS_SCOPE:
            raise ValueError(f"training.loss_scope must be one of {_SALIENCY_LOSS_SCOPE}")

    eval_cfg = out["eval"]
    if not isinstance(eval_cfg, dict) or not eval_cfg:
        raise ValueError("eval must be a non-empty mapping")
    eval_cfg.setdefault("perturbation_mode", "always")
    if str(eval_cfg["perturbation_mode"]) not in _PERTURBATION_MODES:
        raise ValueError(f"eval.perturbation_mode must be one of {_PERTURBATION_MODES}")
    if task_name == "classification":
        _require(eval_cfg, "metric")
    else:
        _require(eval_cfg, "pr_thresholds")

    _require(out["viz"], "enabled")
    return out


def load_and_validate_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and validate as spec-style config."""

    cfg = load_yaml(path)
    return validate_config(cfg)
