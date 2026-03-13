"""Training loops for classification and saliency."""

from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from tao2019_fd2nn.models.detectors import integrate_detector_energies
from tao2019_fd2nn.optics.fft2c import gamma_flip2d
from tao2019_fd2nn.training.losses import classification_loss, saliency_mse_loss, saliency_structured_loss
from tao2019_fd2nn.training.metrics_classification import accuracy
from tao2019_fd2nn.training.metrics_saliency import max_f_measure
from tao2019_fd2nn.utils.math import intensity


@dataclass
class ClassifierEpochResult:
    loss: float
    acc: float


@dataclass
class SaliencyEpochResult:
    loss: float
    fmax: float


StepCallback = Callable[[dict[str, Any]], None]
EpochCallback = Callable[[dict[str, Any]], None]


def _per_sample_minmax(x: torch.Tensor) -> torch.Tensor:
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min).clamp_min(1e-8)


def _prepare_saliency_loss_maps(
    pred_intensity: torch.Tensor,
    gt_intensity: torch.Tensor,
    *,
    eval_crop_box: tuple[int, int, int, int] | None,
    loss_normalization: str,
    loss_scope: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = pred_intensity
    target = gt_intensity.clamp(0.0, 1.0)
    if loss_scope == "crop":
        if eval_crop_box is not None:
            y0, y1, x0, x1 = eval_crop_box
            pred = pred[..., y0:y1, x0:x1]
            target = target[..., y0:y1, x0:x1]
    elif loss_scope != "full":
        raise ValueError(f"unsupported saliency loss_scope: {loss_scope}")

    pred = _per_sample_minmax(pred)
    if loss_normalization == "pred_and_target":
        target = _per_sample_minmax(target)
    elif loss_normalization != "pred_only":
        raise ValueError(f"unsupported saliency loss_normalization: {loss_normalization}")

    return pred, target


def _total_steps(loader: DataLoader, max_steps: int | None) -> int | None:
    total = len(loader) if hasattr(loader, "__len__") else None
    if max_steps is None:
        return int(total) if total is not None else None
    return int(min(total, max_steps)) if total is not None else int(max_steps)


def _cuda_mem_stats(device: torch.device) -> tuple[float | None, float | None]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None, None
    allocated = float(torch.cuda.memory_allocated(device)) / (1024.0**3)
    peak = float(torch.cuda.max_memory_allocated(device)) / (1024.0**3)
    return allocated, peak


def _detector_leakage_ratio(intensity_map: torch.Tensor, detector_masks: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    union = detector_masks.any(dim=0).to(intensity_map.dtype)
    in_detector = (intensity_map * union.unsqueeze(0)).sum(dim=(-1, -2))
    total = intensity_map.sum(dim=(-1, -2)).clamp_min(eps)
    return 1.0 - (in_detector / total)


def run_classifier_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    detector_masks: torch.Tensor,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    loss_mode: str,
    leakage_weight: float,
    temperature: float,
    max_steps: int | None = None,
    epoch_idx: int = 1,
    total_epochs: int = 1,
    phase: str = "train",
    step_callback: StepCallback | None = None,
) -> ClassifierEpochResult:
    """Run one classification epoch."""

    train_mode = optimizer is not None
    model.train(train_mode)

    masks = detector_masks.to(device=device)
    sum_loss = 0.0
    n_steps = 0
    running_correct = 0
    running_seen = 0
    all_e: list[torch.Tensor] = []
    all_y: list[torch.Tensor] = []
    t0 = perf_counter()
    total_step_count = _total_steps(loader, max_steps)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    for idx, (fields, labels) in enumerate(loader):
        if max_steps is not None and idx >= max_steps:
            break
        fields = fields.to(device)
        labels = labels.to(device)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        out_field = model(fields)
        out_intensity = intensity(out_field)
        energies = integrate_detector_energies(out_intensity, masks)
        leakage = _detector_leakage_ratio(out_intensity, masks)
        loss = classification_loss(
            energies=energies,
            labels=labels,
            loss_mode=loss_mode,
            leakage_ratio=leakage,
            leakage_weight=leakage_weight,
            temperature=temperature,
        )
        if train_mode:
            loss.backward()
            optimizer.step()
        loss_value = float(loss.item())
        sum_loss += loss_value
        n_steps += 1
        preds = energies.argmax(dim=1)
        running_correct += int((preds == labels).sum().item())
        running_seen += int(labels.numel())
        avg_loss = sum_loss / n_steps
        running_acc = float(running_correct) / max(float(running_seen), 1.0)
        elapsed = perf_counter() - t0
        samples_per_sec = float(running_seen) / max(elapsed, 1e-9)
        eta_sec = None
        if total_step_count is not None and n_steps < total_step_count:
            steps_left = total_step_count - n_steps
            eta_sec = float(steps_left) * (elapsed / max(float(n_steps), 1.0))
        mem_gb, mem_peak_gb = _cuda_mem_stats(device)
        if step_callback is not None:
            step_callback(
                {
                    "task": "classification",
                    "phase": phase,
                    "epoch": int(epoch_idx),
                    "total_epochs": int(total_epochs),
                    "step": int(n_steps),
                    "total_steps": total_step_count,
                    "loss": loss_value,
                    "avg_loss": avg_loss,
                    "metric_name": "acc",
                    "metric_value": running_acc,
                    "samples_per_sec": samples_per_sec,
                    "eta_sec": eta_sec,
                    "gpu_mem_gb": mem_gb,
                    "gpu_mem_peak_gb": mem_peak_gb,
                }
            )
        all_e.append(energies.detach())
        all_y.append(labels.detach())

    if n_steps == 0:
        return ClassifierEpochResult(loss=0.0, acc=0.0)

    energies_cat = torch.cat(all_e, dim=0)
    labels_cat = torch.cat(all_y, dim=0)
    return ClassifierEpochResult(loss=sum_loss / n_steps, acc=accuracy(energies_cat, labels_cat))


def run_saliency_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gamma_flip: bool,
    pr_thresholds: int,
    max_steps: int | None = None,
    epoch_idx: int = 1,
    total_epochs: int = 1,
    phase: str = "train",
    compute_fmax: bool = True,
    eval_crop_box: tuple[int, int, int, int] | None = None,
    step_callback: StepCallback | None = None,
    loss_mode: str = "mse",
    loss_weights: dict[str, float] | None = None,
    loss_normalization: str = "pred_only",
    loss_scope: str = "crop",
) -> SaliencyEpochResult:
    """Run one saliency epoch."""

    train_mode = optimizer is not None
    model.train(train_mode)

    sum_loss = 0.0
    n_steps = 0
    running_seen = 0
    preds: list[torch.Tensor] = []
    gts: list[torch.Tensor] = []
    t0 = perf_counter()
    total_step_count = _total_steps(loader, max_steps)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    for idx, (fields, targets) in enumerate(loader):
        if max_steps is not None and idx >= max_steps:
            break
        fields = fields.to(device)
        targets = targets.to(device)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        out_field = model(fields)
        out_i = intensity(out_field)
        out_i_loss, tgt_loss = _prepare_saliency_loss_maps(
            out_i,
            targets,
            eval_crop_box=eval_crop_box,
            loss_normalization=loss_normalization,
            loss_scope=loss_scope,
        )

        # Select loss function based on loss_mode
        if loss_mode == "structured":
            loss = saliency_structured_loss(out_i_loss, tgt_loss, gamma_flip=gamma_flip, loss_weights=loss_weights)
        else:  # default "mse"
            loss = saliency_mse_loss(out_i_loss, tgt_loss, gamma_flip=gamma_flip)
        if train_mode:
            loss.backward()
            optimizer.step()
        loss_value = float(loss.item())
        sum_loss += loss_value
        n_steps += 1
        running_seen += int(fields.shape[0])
        avg_loss = sum_loss / n_steps
        elapsed = perf_counter() - t0
        samples_per_sec = float(running_seen) / max(elapsed, 1e-9)
        eta_sec = None
        if total_step_count is not None and n_steps < total_step_count:
            steps_left = total_step_count - n_steps
            eta_sec = float(steps_left) * (elapsed / max(float(n_steps), 1.0))
        mem_gb, mem_peak_gb = _cuda_mem_stats(device)
        if step_callback is not None:
            step_callback(
                {
                    "task": "saliency",
                    "phase": phase,
                    "epoch": int(epoch_idx),
                    "total_epochs": int(total_epochs),
                    "step": int(n_steps),
                    "total_steps": total_step_count,
                    "loss": loss_value,
                    "avg_loss": avg_loss,
                    "samples_per_sec": samples_per_sec,
                    "eta_sec": eta_sec,
                    "gpu_mem_gb": mem_gb,
                    "gpu_mem_peak_gb": mem_peak_gb,
                }
            )
        if compute_fmax:
            p = out_i.detach()
            g = targets.clamp(0.0, 1.0).detach()
            if eval_crop_box is not None:
                y0, y1, x0, x1 = eval_crop_box
                p = p[..., y0:y1, x0:x1]
                g = g[..., y0:y1, x0:x1]
            p = _per_sample_minmax(p).cpu()
            g = g.cpu()
            if gamma_flip:
                p = gamma_flip2d(p)
            preds.append(p)
            gts.append(g)

    if n_steps == 0:
        return SaliencyEpochResult(loss=0.0, fmax=float("nan"))
    if not compute_fmax:
        return SaliencyEpochResult(loss=sum_loss / n_steps, fmax=float("nan"))
    pred_cat = torch.cat(preds, dim=0)
    gt_cat = torch.cat(gts, dim=0)
    fmax = max_f_measure(pred_cat, gt_cat, thresholds=pr_thresholds)
    return SaliencyEpochResult(loss=sum_loss / n_steps, fmax=fmax)


def train_classifier(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    detector_masks: torch.Tensor,
    *,
    device: torch.device,
    lr: float,
    epochs: int,
    loss_mode: str = "cross_entropy",
    leakage_weight: float = 0.1,
    temperature: float = 1.0,
    test_loader: DataLoader | None = None,
    max_steps_per_epoch: int | None = None,
    step_callback: StepCallback | None = None,
    epoch_callback: EpochCallback | None = None,
) -> dict[str, list[float]]:
    """Classifier training orchestration."""

    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    history: dict[str, list[float]] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    if test_loader is not None:
        history["test_loss"] = []
        history["test_acc"] = []
    total_epochs = int(epochs)
    for epoch_idx in range(total_epochs):
        tr = run_classifier_epoch(
            model,
            train_loader,
            detector_masks,
            device=device,
            optimizer=optimizer,
            loss_mode=loss_mode,
            leakage_weight=leakage_weight,
            temperature=temperature,
            max_steps=max_steps_per_epoch,
            epoch_idx=epoch_idx + 1,
            total_epochs=total_epochs,
            phase="train",
            step_callback=step_callback,
        )
        va = run_classifier_epoch(
            model,
            val_loader,
            detector_masks,
            device=device,
            optimizer=None,
            loss_mode=loss_mode,
            leakage_weight=leakage_weight,
            temperature=temperature,
            max_steps=max_steps_per_epoch,
            epoch_idx=epoch_idx + 1,
            total_epochs=total_epochs,
            phase="val",
            step_callback=step_callback,
        )
        history["train_loss"].append(tr.loss)
        history["train_acc"].append(tr.acc)
        history["val_loss"].append(va.loss)
        history["val_acc"].append(va.acc)
        te: ClassifierEpochResult | None = None
        if test_loader is not None:
            te = run_classifier_epoch(
                model,
                test_loader,
                detector_masks,
                device=device,
                optimizer=None,
                loss_mode=loss_mode,
                leakage_weight=leakage_weight,
                temperature=temperature,
                max_steps=max_steps_per_epoch,
                epoch_idx=epoch_idx + 1,
                total_epochs=total_epochs,
                phase="test",
                step_callback=step_callback,
            )
            history["test_loss"].append(te.loss)
            history["test_acc"].append(te.acc)
        if epoch_callback is not None:
            payload = {
                "task": "classification",
                "epoch": int(epoch_idx + 1),
                "total_epochs": total_epochs,
                "train_loss": tr.loss,
                "train_acc": tr.acc,
                "val_loss": va.loss,
                "val_acc": va.acc,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            if te is not None:
                payload["test_loss"] = te.loss
                payload["test_acc"] = te.acc
            epoch_callback(payload)
    return history


def train_saliency(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: torch.device,
    lr: float,
    epochs: int,
    gamma_flip: bool = True,
    pr_thresholds: int = 256,
    compute_train_fmax: bool = False,
    eval_interval_epochs: int = 5,
    lr_scheduler: str = "none",
    lr_min: float = 1e-5,
    max_steps_per_epoch: int | None = None,
    eval_crop_box: tuple[int, int, int, int] | None = None,
    step_callback: StepCallback | None = None,
    epoch_callback: EpochCallback | None = None,
    best_state_callback: Callable[[dict], None] | None = None,
    loss_mode: str = "mse",
    loss_weights: dict[str, float] | None = None,
    loss_normalization: str = "pred_only",
    loss_scope: str = "crop",
) -> dict[str, list[float]]:
    """Saliency training orchestration."""

    import copy

    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))

    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    if lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(epochs), eta_min=float(lr_min)
        )

    history: dict[str, list[float]] = {"train_loss": [], "train_fmax": [], "val_loss": [], "val_fmax": []}
    total_epochs = int(epochs)
    eval_interval = max(1, int(eval_interval_epochs))
    best_val_fmax = float("-inf")
    best_state: dict | None = None

    for epoch_idx in range(total_epochs):
        epoch_num = epoch_idx + 1
        compute_val_fmax = (epoch_num % eval_interval == 0) or (epoch_num == total_epochs)
        tr = run_saliency_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            gamma_flip=gamma_flip,
            pr_thresholds=pr_thresholds,
            compute_fmax=bool(compute_train_fmax),
            max_steps=max_steps_per_epoch,
            epoch_idx=epoch_num,
            total_epochs=total_epochs,
            phase="train",
            eval_crop_box=eval_crop_box,
            step_callback=step_callback,
            loss_mode=loss_mode,
            loss_weights=loss_weights,
            loss_normalization=loss_normalization,
            loss_scope=loss_scope,
        )
        va = run_saliency_epoch(
            model,
            val_loader,
            device=device,
            optimizer=None,
            gamma_flip=gamma_flip,
            pr_thresholds=pr_thresholds,
            compute_fmax=compute_val_fmax,
            max_steps=max_steps_per_epoch,
            epoch_idx=epoch_num,
            total_epochs=total_epochs,
            phase="val",
            eval_crop_box=eval_crop_box,
            step_callback=step_callback,
            loss_mode=loss_mode,
            loss_weights=loss_weights,
            loss_normalization=loss_normalization,
            loss_scope=loss_scope,
        )

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(tr.loss)
        history["train_fmax"].append(tr.fmax)
        history["val_loss"].append(va.loss)
        history["val_fmax"].append(va.fmax)

        # Track best state during training (fixes best.pt bug)
        if compute_val_fmax and math.isfinite(va.fmax) and va.fmax > best_val_fmax:
            best_val_fmax = va.fmax
            best_state = copy.deepcopy(model.state_dict())
            if best_state_callback is not None:
                best_state_callback({"epoch": epoch_num, "val_fmax": va.fmax, "state_dict": best_state})

        if epoch_callback is not None:
            epoch_callback(
                {
                    "task": "saliency",
                    "epoch": int(epoch_num),
                    "total_epochs": total_epochs,
                    "train_loss": tr.loss,
                    "train_fmax": tr.fmax,
                    "train_fmax_computed": bool(compute_train_fmax),
                    "val_loss": va.loss,
                    "val_fmax": va.fmax,
                    "val_fmax_computed": compute_val_fmax,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "best_val_fmax": best_val_fmax if math.isfinite(best_val_fmax) else None,
                }
            )

    # Restore best weights into model so callers can save them
    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def best_epoch_index(history: dict[str, list[float]], *, metric_name: str, maximize: bool = True) -> int:
    """Return best epoch index by metric."""

    values = history.get(metric_name, [])
    if not values:
        return 0
    finite = [(i, float(v)) for i, v in enumerate(values) if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not finite:
        return max(0, len(values) - 1)
    best_idx, _ = (max(finite, key=lambda kv: kv[1]) if maximize else min(finite, key=lambda kv: kv[1]))
    return int(best_idx)


def summarize_history(history: dict[str, list[float]]) -> dict[str, Any]:
    """Return compact summary fields from history."""

    out: dict[str, Any] = {}
    for k, vals in history.items():
        out[f"{k}_last"] = vals[-1] if vals else None
        finite_vals = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
        if not finite_vals:
            out[f"{k}_best"] = None
        elif "acc" in k or "fmax" in k:
            out[f"{k}_best"] = max(finite_vals)
        else:
            out[f"{k}_best"] = min(finite_vals)
    return out
