"""Shared CLI utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from d2nn.data import FashionMNISTFieldDataset, ImageFolderFieldDataset, MNISTFieldDataset
from d2nn.detectors.layout import build_region_masks, load_layout
from d2nn.models import build_d2nn_model
from d2nn.utils.io import load_yaml, resolve_run_dir
from d2nn.utils.seed import make_torch_generator


def load_config(path: str | Path) -> dict[str, Any]:
    """Load experiment config from YAML."""

    return load_yaml(path)


def choose_device(runtime_cfg: dict[str, Any]) -> torch.device:
    """Resolve runtime device from config."""

    requested = str(runtime_cfg.get("device", "auto")).lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_run_dir(cfg: dict[str, Any]) -> Path:
    """Create run directory from runtime naming policy."""

    runtime = cfg.get("runtime", {})
    runs_dir = runtime.get("runs_dir", "runs")
    exp_name = cfg.get("experiment", {}).get("name", "experiment")
    seed = int(runtime.get("seed", 0))
    run_id_mode = str(runtime.get("run_id_mode", "timestamp"))
    return resolve_run_dir(runs_dir, exp_name, cfg, seed, run_id_mode=run_id_mode)


def update_latest_symlink(run_dir: Path) -> Path:
    """Update `latest` symlink inside experiment folder to point to current run.

    Falls back silently when symlink creation is unavailable.
    """

    latest = run_dir.parent / "latest"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(run_dir.name)
    except OSError:
        # Keep training robust on filesystems that disallow symlink.
        pass
    return latest


def build_model_from_config(cfg: dict[str, Any]) -> torch.nn.Module:
    """Instantiate D2NN model from config dictionaries."""

    physics = cfg["physics"]
    model_cfg = cfg["model"]
    err = cfg.get("error_model", {})

    return build_d2nn_model(
        N=int(physics["N"]),
        dx=float(physics["dx"]),
        wavelength=float(physics["wavelength"]),
        num_layers=int(model_cfg["num_layers"]),
        z_layer=float(physics["z_layer"]),
        z_out=float(physics["z_out"]),
        phase_max=float(model_cfg["phase_max"]),
        phase_constraint_mode=str(model_cfg.get("phase_constraint_mode", "sigmoid")),
        phase_init=str(model_cfg.get("phase_init", "zeros")),
        train_amplitude=bool(model_cfg.get("train_amplitude", False)),
        amplitude_range=tuple(model_cfg.get("amplitude_range", [0.0, 1.0])),
        use_absorption=float(err.get("absorption_alpha", 0.0)) > 0.0,
        absorption_alpha=err.get("absorption_alpha", None),
        bandlimit=bool(physics.get("bandlimit", True)),
        fftshifted=bool(physics.get("fftshifted", False)),
        dtype=str(model_cfg.get("dtype", "complex64")),
        max_misalignment_m=float(err.get("max_misalignment_m", 0.0)),
    )


def _build_dataset(cfg: dict[str, Any], train: bool):
    data_cfg = cfg["data"]
    physics = cfg["physics"]

    dataset_name = data_cfg["dataset"]
    if dataset_name == "mnist":
        return MNISTFieldDataset(
            root=data_cfg.get("root", "data"),
            train=train,
            download=bool(data_cfg.get("download", True)),
            N=int(physics["N"]),
            object_size=int(data_cfg.get("object_size", 80)),
            binarize=bool(data_cfg.get("binarize", True)),
        )

    if dataset_name == "fashion_mnist":
        return FashionMNISTFieldDataset(
            root=data_cfg.get("root", "data"),
            train=train,
            download=bool(data_cfg.get("download", True)),
            N=int(physics["N"]),
            object_size=int(data_cfg.get("object_size", 80)),
            phase_max=2.0 * torch.pi,
        )

    if dataset_name == "image_folder":
        root_default = data_cfg.get("root", "data/images")
        root_train = data_cfg.get("root_train", root_default)
        root_val = data_cfg.get("root_val", root_default)
        root = root_train if train else root_val
        return ImageFolderFieldDataset(
            root=root,
            N=int(physics["N"]),
            object_size=int(data_cfg.get("object_size", 120)),
        )

    raise ValueError(f"unsupported dataset: {dataset_name}")


def build_dataloaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders."""

    data_cfg = cfg["data"]
    runtime = cfg.get("runtime", {})

    train_ds = _build_dataset(cfg, train=True)
    val_ds = _build_dataset(cfg, train=False)

    batch_size = int(data_cfg.get("batch_size", 8))
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", False))
    persistent_workers = bool(data_cfg.get("persistent_workers", num_workers > 0))
    prefetch_factor = data_cfg.get("prefetch_factor", None)
    seed = int(runtime.get("seed", 0))
    gen = make_torch_generator(seed)

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        generator=gen,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, val_loader


def build_detector_tensors(cfg: dict[str, Any]) -> tuple[torch.Tensor, Any]:
    """Load detector layout and region masks."""

    physics = cfg["physics"]
    layout_path = cfg["detector_layout"]["path"]
    layout = load_layout(layout_path)
    masks_np = build_region_masks(layout, N=int(physics["N"]), dx=float(physics["dx"]))
    masks = torch.from_numpy(masks_np)
    return masks, layout


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> dict[str, Any]:
    """Load checkpoint payload into model (and optimizer if provided)."""

    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    """Detach tensor to CPU numpy array."""

    return x.detach().cpu().numpy()
