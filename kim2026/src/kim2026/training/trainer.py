"""Deterministic trainer for beam-cleanup D2NN."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from kim2026.data.canonical_pupil import enforce_reducer_validation_gate
from kim2026.data.dataset import CachedFieldDataset
from kim2026.models.d2nn import BeamCleanupD2NN
from kim2026.models.fd2nn import MultiLayerFD2NN
from kim2026.optics import MIN_PAD_FACTOR, propagate_padded_same_window
from kim2026.training.losses import beam_cleanup_loss, complex_field_loss, roi_complex_loss
from kim2026.training.targets import apply_receiver_aperture, make_detector_plane_target
from kim2026.utils.seed import capture_rng_state, restore_rng_state, set_global_seed


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "u_vacuum": torch.stack([item["u_vacuum"] for item in batch], dim=0),
        "u_turb": torch.stack([item["u_turb"] for item in batch], dim=0),
        "metadata": [item["metadata"] for item in batch],
    }


def _build_model(cfg: dict[str, Any], n: int) -> nn.Module:
    model_type = str(cfg["model"].get("type", "d2nn"))
    if model_type == "fd2nn":
        dual_2f = cfg["optics"]["dual_2f"]
        na_val = None if dual_2f.get("na1") is None else float(dual_2f["na1"])
        return MultiLayerFD2NN(
            n=n,
            wavelength_m=float(cfg["optics"]["lambda_m"]),
            window_m=float(cfg["grid"]["receiver_window_m"]),
            num_layers=int(cfg["model"]["num_layers"]),
            f_m=float(dual_2f["f1_m"]),
            layer_spacing_m=float(cfg["model"].get("layer_spacing_m", 0.0)),
            na=na_val,
        )
    return BeamCleanupD2NN(
        n=n,
        wavelength_m=float(cfg["optics"]["lambda_m"]),
        window_m=float(cfg["grid"]["receiver_window_m"]),
        num_layers=int(cfg["model"]["num_layers"]),
        layer_spacing_m=float(cfg["model"]["layer_spacing_m"]),
        detector_distance_m=float(cfg["model"]["detector_distance_m"]),
        propagation_pad_factor=int(cfg["model"].get("propagation_pad_factor", MIN_PAD_FACTOR)),
    )


def _warmup_receiver_propagation(
    *,
    n: int,
    wavelength_m: float,
    window_m: float,
    distances_m: list[float],
    iterations: int,
    device: torch.device,
    pad_factor: int,
) -> None:
    if int(iterations) <= 0:
        return
    field = torch.zeros(1, n, n, dtype=torch.complex64, device=device)
    for _ in range(int(iterations)):
        for distance_m in distances_m:
            _ = propagate_padded_same_window(
                field,
                wavelength_m=wavelength_m,
                window_m=window_m,
                z_m=distance_m,
                pad_factor=pad_factor,
            )


def _epoch_pass(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    cfg: dict[str, Any],
    device: torch.device,
) -> float:
    train_mode = optimizer is not None
    model.train(train_mode)
    model_type = str(cfg["model"].get("type", "d2nn"))
    loss_mode = str(cfg["training"]["loss"].get("mode", "intensity"))
    complex_mode = loss_mode in ("complex", "roi_complex")
    propagation_pad_factor = int(cfg["model"].get("propagation_pad_factor", MIN_PAD_FACTOR))

    if model_type == "fd2nn":
        total_distance = 0.0
    else:
        total_distance = (model.num_layers - 1) * model.layer_spacing_m + model.detector_distance_m
    total_loss = 0.0
    count = 0
    weights = cfg["training"]["loss"]["weights"]
    receiver_window_m = float(cfg["grid"]["receiver_window_m"])
    aperture_diameter_m = float(cfg["receiver"]["aperture_diameter_m"])
    wavelength_m = float(cfg["optics"]["lambda_m"])

    for batch in loader:
        u_turb = batch["u_turb"].to(device)
        u_vacuum = batch["u_vacuum"].to(device)
        target = make_detector_plane_target(
            u_vacuum,
            wavelength_m=wavelength_m,
            receiver_window_m=receiver_window_m,
            aperture_diameter_m=aperture_diameter_m,
            total_distance_m=total_distance,
            complex_mode=complex_mode,
            propagation_pad_factor=propagation_pad_factor,
        )
        input_field = apply_receiver_aperture(
            u_turb,
            receiver_window_m=receiver_window_m,
            aperture_diameter_m=aperture_diameter_m,
        )

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        pred_field = model(input_field)

        if loss_mode == "roi_complex":
            loss = roi_complex_loss(
                pred_field,
                target,
                roi_threshold=float(cfg["training"]["loss"].get("roi_threshold", 0.5)),
                intensity_weight=float(cfg["training"]["loss"].get("intensity_weight", 1.0)),
                phase_weight=float(cfg["training"]["loss"].get("phase_weight", 0.2)),
                leakage_weight=float(cfg["training"]["loss"].get("leakage_weight", 0.3)),
                phase_gamma=float(cfg["training"]["loss"].get("phase_gamma", 2.0)),
                full_field_phase_weight=float(cfg["training"]["loss"].get("full_field_phase_weight", 0.0)),
                full_field_phase_gamma=float(cfg["training"]["loss"].get("full_field_phase_gamma", 1.0)),
                full_field_phase_threshold=float(cfg["training"]["loss"].get("full_field_phase_threshold", 0.05)),
                window_m=receiver_window_m,
            )
        elif complex_mode:
            complex_weights = cfg["training"]["loss"].get("complex_weights", None)
            loss = complex_field_loss(
                pred_field,
                target,
                weights=complex_weights,
                window_m=receiver_window_m,
            )
        else:
            pred_intensity = pred_field.abs().square()
            loss = beam_cleanup_loss(pred_intensity, target, window_m=receiver_window_m, weights=weights)

        if train_mode:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        count += 1

    return total_loss / max(count, 1)


def _save_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict[str, float]],
    cfg: dict[str, Any],
) -> None:
    checkpoint = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "config": cfg,
        "rng_state": capture_rng_state(),
    }
    torch.save(checkpoint, path)


def train_model(cfg: dict[str, Any], *, run_dir: str | Path, resume_path: str | Path | None = None) -> dict[str, Any]:
    """Train a beam-cleanup D2NN from cached NPZ pairs."""
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    strict = bool(cfg["runtime"].get("strict_reproducibility", True))
    set_global_seed(int(cfg["runtime"]["seed"]), strict_reproducibility=strict)
    enforce_reducer_validation_gate(cfg["data"])

    train_ds = CachedFieldDataset(
        cache_dir=cfg["data"]["cache_dir"],
        manifest_path=cfg["data"]["split_manifest_path"],
        split="train",
        plane_selector=str(cfg["data"].get("plane_selector", "stored")),
    )
    val_ds = CachedFieldDataset(
        cache_dir=cfg["data"]["cache_dir"],
        manifest_path=cfg["data"]["split_manifest_path"],
        split="val",
        plane_selector=str(cfg["data"].get("plane_selector", "stored")),
    )
    if len(train_ds) == 0:
        raise ValueError("training dataset is empty")

    n = int(train_ds[0]["u_turb"].shape[-1])
    requested_device = str(cfg["runtime"].get("device", "cuda")).lower()
    if requested_device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(cfg, n).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["training"].get("learning_rate", 1e-3)))

    model_type = str(cfg["model"].get("type", "d2nn"))
    if model_type != "fd2nn":
        warmup_distances_m = [float(cfg["model"]["detector_distance_m"])]
        if int(cfg["model"]["num_layers"]) > 1:
            warmup_distances_m.append(float(cfg["model"]["layer_spacing_m"]))
        _warmup_receiver_propagation(
            n=n,
            wavelength_m=float(cfg["optics"]["lambda_m"]),
            window_m=float(cfg["grid"]["receiver_window_m"]),
            distances_m=warmup_distances_m,
            iterations=int(cfg["runtime"].get("fft_warmup_iters", 0)),
            device=device,
            pad_factor=int(cfg["model"].get("propagation_pad_factor", MIN_PAD_FACTOR)),
        )

    start_epoch = 0
    history: list[dict[str, float]] = []
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        history = list(checkpoint["history"])
        restore_rng_state(checkpoint["rng_state"])

    loader_kwargs = {
        "batch_size": int(cfg["training"]["batch_size"]),
        "shuffle": False,
        "num_workers": int(cfg["runtime"].get("num_workers", 0)),
        "pin_memory": bool(cfg["runtime"].get("pin_memory", False)),
        "persistent_workers": bool(cfg["runtime"].get("persistent_workers", False)),
        "collate_fn": _collate,
    }
    if loader_kwargs["num_workers"] == 0:
        loader_kwargs.pop("persistent_workers")
    train_loader = DataLoader(train_ds, **loader_kwargs)
    val_loader = DataLoader(val_ds, **loader_kwargs)

    latest_ckpt = run_path / "checkpoint.pt"
    total_epochs = int(cfg["training"]["epochs"])
    for epoch in range(start_epoch + 1, total_epochs + 1):
        train_loss = _epoch_pass(model, train_loader, optimizer=optimizer, cfg=cfg, device=device)
        val_loss = _epoch_pass(model, val_loader, optimizer=None, cfg=cfg, device=device) if len(val_ds) else train_loss
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})
        _save_checkpoint(
            path=latest_ckpt,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            history=history,
            cfg=cfg,
        )

    return {"model": model, "history": history, "checkpoint_path": latest_ckpt}
