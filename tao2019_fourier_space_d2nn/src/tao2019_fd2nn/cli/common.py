"""Shared CLI helpers for spec-style configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, random_split

from tao2019_fd2nn.config import load_and_validate_config
from tao2019_fd2nn.data import (
    CellGdcSaliencyDataset,
    Cifar10SaliencyDataset,
    DavisSaliencyDataset,
    EcssdSaliencyDataset,
    MnistAmplitudeDataset,
    SaliencyPairsDataset,
    VideoFramesSaliencyDataset,
)
from tao2019_fd2nn.models import Fd2nnConfig, Fd2nnModel, make_detector_masks
from tao2019_fd2nn.utils.io import resolve_run_dir, save_repro_metadata
from tao2019_fd2nn.utils.seed import make_generator, worker_init_fn

_CIFAR10_CLASS_TO_IDX = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9,
}


def choose_device(exp_cfg: dict[str, Any]) -> torch.device:
    """Resolve runtime device from experiment.device."""

    requested = str(exp_cfg.get("device", "auto")).lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate YAML config."""

    return load_and_validate_config(path)


def create_run_dir(cfg: dict[str, Any], *, cwd: str | Path | None = None) -> Path:
    """Create run directory from experiment section."""

    exp = cfg["experiment"]
    run_dir = resolve_run_dir(exp.get("save_dir", "runs"), exp["name"])
    save_repro_metadata(run_dir, save_requirements_file=True, cwd=cwd)
    return run_dir


def _phase_init(mod_cfg: dict[str, Any]) -> str:
    init = str(mod_cfg.get("init", "uniform")).lower()
    return "uniform" if init == "uniform" else "zeros"


def _model_na(optics: dict[str, Any]) -> float | None:
    if optics.get("hybrid_2f", {}).get("enabled", False):
        return float(optics["hybrid_2f"].get("na"))
    ap = optics.get("aperture", {})
    if ap.get("enabled", False):
        return float(ap.get("na"))
    return None


def _model_type_to_space(model_type: str) -> str:
    if model_type == "fd2nn":
        return "fd2nn"
    if model_type == "real_d2nn":
        return "real_d2nn"
    if model_type == "hybrid_d2nn":
        return "hybrid_d2nn"
    raise ValueError(f"unsupported model.type: {model_type}")


def build_model(cfg: dict[str, Any]) -> Fd2nnModel:
    """Build model from spec-style config."""

    optics = cfg["optics"]
    model = cfg["model"]
    model_type = str(model["type"])
    mod = model["modulation"]
    nonlin = model.get("nonlinearity", {})
    grid = optics["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    if nx != ny:
        raise ValueError("current implementation requires square grid (nx == ny)")

    exp_dtype = str(cfg["experiment"].get("dtype", "float32")).lower()
    complex_dtype = "complex128" if "64" in exp_dtype else "complex64"
    dual_2f_cfg = dict(optics.get("dual_2f", {}))
    dual_enabled = bool(dual_2f_cfg.get("enabled", False))
    # Hybrid configs define relay-lens parameters in optics.hybrid_2f.
    # Map them into the dual-2f runtime slots so domain switches use lens_2f.
    if model_type == "hybrid_d2nn" and not dual_enabled:
        hybrid_2f_cfg = optics.get("hybrid_2f", {})
        if bool(hybrid_2f_cfg.get("enabled", False)):
            f_m = float(hybrid_2f_cfg["f_m"])
            na = float(hybrid_2f_cfg.get("na")) if "na" in hybrid_2f_cfg else None
            dual_2f_cfg = {
                "enabled": True,
                "f1_m": f_m,
                "f2_m": f_m,
                "apply_scaling": bool(hybrid_2f_cfg.get("apply_scaling", False)),
            }
            if na is not None:
                dual_2f_cfg["na1"] = na
                dual_2f_cfg["na2"] = na
            dual_enabled = True

    fcfg = Fd2nnConfig(
        N=nx,
        dx_m=float(grid["dx_m"]),
        wavelength_m=float(optics["wavelength_m"]),
        z_layer_m=float(optics["propagation"]["layer_spacing_m"]),
        z_out_m=float(optics["propagation"]["layer_spacing_m"]),
        num_layers=int(model["num_layers"]),
        phase_max=float(mod["phase_max_rad"]),
        phase_constraint=str(mod.get("phase_constraint", "sigmoid")),
        phase_init=_phase_init(mod),
        model_type=_model_type_to_space(model_type),
        na=_model_na(optics),
        evanescent=str(optics["propagation"].get("evanescent", "mask")),
        dtype=complex_dtype,
        use_dual_2f=dual_enabled,
        dual_2f_f1_m=(float(dual_2f_cfg["f1_m"]) if dual_enabled and "f1_m" in dual_2f_cfg else None),
        dual_2f_f2_m=(float(dual_2f_cfg["f2_m"]) if dual_enabled and "f2_m" in dual_2f_cfg else None),
        dual_2f_na1=(float(dual_2f_cfg["na1"]) if dual_enabled and "na1" in dual_2f_cfg else None),
        dual_2f_na2=(float(dual_2f_cfg["na2"]) if dual_enabled and "na2" in dual_2f_cfg else None),
        dual_2f_apply_scaling=(bool(dual_2f_cfg.get("apply_scaling", False)) if dual_enabled else False),
        hybrid_sequence=tuple(model.get("hybrid", {}).get("plane_sequence", [])),
        sbn_enabled=bool(nonlin.get("enabled", False)),
        sbn_phi_max=float(nonlin.get("phi_max_rad", torch.pi)),
        sbn_position=str(nonlin.get("position", "rear")),
        sbn_background_intensity=float(nonlin.get("background_intensity", 0.0)),
        sbn_saturation_intensity=float(nonlin.get("saturation_intensity", 1.0)),
        sbn_clamp_negative_perturbation=bool(nonlin.get("clamp_negative_perturbation", True)),
        sbn_learnable_saturation=bool(nonlin.get("learnable_saturation", False)),
        sbn_voltage_v=(float(nonlin["voltage_v"]) if "voltage_v" in nonlin else None),
        sbn_electrode_gap_m=(float(nonlin["electrode_gap_m"]) if "electrode_gap_m" in nonlin else None),
        sbn_e_app_v_per_m=(float(nonlin["e_app_v_per_m"]) if "e_app_v_per_m" in nonlin else None),
        sbn_kappa_m_per_v=(float(nonlin["kappa_m_per_v"]) if "kappa_m_per_v" in nonlin else None),
        sbn_thickness_m=(float(nonlin["thickness_m"]) if "thickness_m" in nonlin else None),
        sbn_wavelength_m=float(nonlin.get("wavelength_m", optics["wavelength_m"])),
    )
    return Fd2nnModel(fcfg)


def _mnist_object_size(prep: dict[str, Any]) -> int:
    if "resize_to" in prep:
        return int(prep["resize_to"][0])
    factor = int(prep.get("upsample_factor", 3))
    return 28 * factor


def _pad_to_size(cfg: dict[str, Any]) -> int:
    prep = cfg["data"]["preprocess"]
    if "pad_to" in prep:
        return int(prep["pad_to"][0])
    return int(cfg["optics"]["grid"]["nx"])


def _build_mnist_datasets(cfg: dict[str, Any]):
    data_cfg = cfg["data"]
    prep = data_cfg["preprocess"]
    N = _pad_to_size(cfg)
    obj_size = _mnist_object_size(prep)
    base_train = MnistAmplitudeDataset(
        root=str(data_cfg.get("root", "data/mnist")),
        train=True,
        download=True,
        N=N,
        object_size=obj_size,
        binarize=False,
    )
    split = data_cfg.get("split", {})
    n_train = int(split.get("train", 55000))
    n_val = int(split.get("val", 5000))
    if n_train + n_val > len(base_train):
        n_train = len(base_train) - n_val
    gen = make_generator(int(cfg["experiment"].get("seed", 42)))
    train_ds, val_ds = random_split(base_train, [n_train, n_val], generator=gen)
    return train_ds, val_ds


def _resolve_saliency_dirs(data_cfg: dict[str, Any], split_name: str | None) -> tuple[Path, Path]:
    root = Path(str(data_cfg.get("root", "")))
    legacy_image_dir = Path(data_cfg.get("image_dir", root / "images"))
    legacy_mask_dir = Path(data_cfg.get("mask_dir", root / "masks"))
    if split_name is None:
        return legacy_image_dir, legacy_mask_dir

    split_image_key = f"{split_name}_image_dir"
    split_mask_key = f"{split_name}_mask_dir"
    if split_image_key in data_cfg or split_mask_key in data_cfg:
        image_dir = Path(data_cfg.get(split_image_key, legacy_image_dir))
        mask_dir = Path(data_cfg.get(split_mask_key, legacy_mask_dir))
        return image_dir, mask_dir

    split_image_dir = root / split_name / "images"
    split_mask_dir = root / split_name / "masks"
    if split_image_dir.exists() and split_mask_dir.exists():
        return split_image_dir, split_mask_dir
    return legacy_image_dir, legacy_mask_dir


def _build_saliency_pairs_from_root(dataset_cls, cfg: dict[str, Any], *, split_name: str | None):
    data_cfg = cfg["data"]
    N = _pad_to_size(cfg)
    prep = data_cfg["preprocess"]
    object_size = int(prep.get("resize_to", [prep.get("raw_patch_px", [80, 80])[0], 0])[0]) if "resize_to" in prep else int(
        prep.get("raw_patch_px", [80, 80])[0] * int(prep.get("upsample_factor", 1))
    )
    image_dir, mask_dir = _resolve_saliency_dirs(data_cfg, split_name=split_name)
    return dataset_cls(image_dir=str(image_dir), mask_dir=str(mask_dir), N=N, object_size=object_size)


def _build_dataset(cfg: dict[str, Any], *, train: bool):
    data_cfg = cfg["data"]
    name = str(data_cfg["dataset"])
    prep = data_cfg["preprocess"]
    N = _pad_to_size(cfg)
    if name == "mnist":
        raise RuntimeError("mnist dataset should be built via _build_mnist_datasets")
    if name == "cifar10":
        target_name = str(data_cfg.get("train_class", "cat" if train else "horse")).lower()
        if not train and "test_sets" in data_cfg and data_cfg["test_sets"]:
            first = data_cfg["test_sets"][0]
            if isinstance(first, dict) and first.get("type") == "cifar10":
                target_name = str(first.get("class", target_name)).lower()
        target_idx = _CIFAR10_CLASS_TO_IDX.get(target_name, 3)
        obj_size = int(prep.get("resize_to", [100, 100])[0])
        return Cifar10SaliencyDataset(
            root=str(data_cfg.get("root", "data/cifar10")),
            train=train,
            download=True,
            N=N,
            object_size=obj_size,
            foreground_class=target_idx,
            gt_source="ft",
            gt_params={"smooth_sigma": 1.0, "class_gate": True},
        )
    split_name = "train" if train else "val"
    if name == "saliency_pairs":
        return _build_saliency_pairs_from_root(SaliencyPairsDataset, cfg, split_name=split_name)
    if name == "cell_gdc":
        return _build_saliency_pairs_from_root(CellGdcSaliencyDataset, cfg, split_name=split_name)
    if name == "davis":
        return _build_saliency_pairs_from_root(DavisSaliencyDataset, cfg, split_name=split_name)
    if name == "ecssd":
        return _build_saliency_pairs_from_root(EcssdSaliencyDataset, cfg, split_name=split_name)
    if name == "video_frames":
        return _build_saliency_pairs_from_root(VideoFramesSaliencyDataset, cfg, split_name=split_name)
    raise ValueError(f"unsupported data.dataset: {name}")


def build_dataloaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    """Build train and validation dataloaders."""

    training = cfg["training"]
    data_cfg = cfg["data"]
    task_name = cfg["task"]["name"]
    seed = int(cfg["experiment"].get("seed", 42))
    gen = make_generator(seed)

    if task_name == "classification" and data_cfg["dataset"] == "mnist":
        train_ds, val_ds = _build_mnist_datasets(cfg)
    else:
        train_ds = _build_dataset(cfg, train=True)
        val_ds = _build_dataset(cfg, train=False)

    batch_size = int(training.get("batch_size", 10))
    num_workers = int(training.get("num_workers", 0))
    pin_memory = bool(training.get("pin_memory", True))
    persistent_workers = bool(training.get("persistent_workers", num_workers > 0))
    prefetch_factor = training.get("prefetch_factor", None)
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["worker_init_fn"] = worker_init_fn
        kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)

    train_loader = DataLoader(train_ds, shuffle=True, generator=gen, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    return train_loader, val_loader


def build_test_loader(cfg: dict[str, Any]) -> DataLoader | None:
    """Build test dataloader when a canonical test split exists."""

    data_cfg = cfg["data"]
    task_name = str(cfg["task"]["name"])
    if task_name != "classification" or str(data_cfg["dataset"]) != "mnist":
        return None

    prep = data_cfg["preprocess"]
    test_ds = MnistAmplitudeDataset(
        root=str(data_cfg.get("root", "data/mnist")),
        train=False,
        download=True,
        N=_pad_to_size(cfg),
        object_size=_mnist_object_size(prep),
        binarize=False,
    )

    training = cfg["training"]
    batch_size = int(training.get("batch_size", 10))
    num_workers = int(training.get("num_workers", 0))
    pin_memory = bool(training.get("pin_memory", True))
    persistent_workers = bool(training.get("persistent_workers", num_workers > 0))
    prefetch_factor = training.get("prefetch_factor", None)
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["worker_init_fn"] = worker_init_fn
        kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(test_ds, shuffle=False, **kwargs)


def build_detector_masks(cfg: dict[str, Any], *, device: torch.device | str | None = None) -> torch.Tensor:
    """Build detector masks from task.detector config."""

    detector = cfg["task"]["detector"]
    grid = cfg["optics"]["grid"]
    return make_detector_masks(
        N=int(grid["nx"]),
        dx_m=float(grid["dx_m"]),
        num_classes=10,
        width_um=float(detector.get("width_um", 12.0)),
        gap_um=float(detector.get("gap_um", 4.0)),
        layout=str(detector.get("layout", "default10")),
        device=device,
    )
