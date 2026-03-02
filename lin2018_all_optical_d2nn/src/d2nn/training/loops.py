"""Training and evaluation loops for classification and imaging tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from d2nn.detectors.integrate import integrate_regions
from d2nn.detectors.metrics import accuracy
from d2nn.training.losses import classification_loss, imaging_mse_loss
from d2nn.utils.math import intensity as field_intensity
from d2nn.utils.term import paint


@dataclass
class ClassifierEpochResult:
    loss: float
    acc: float


@dataclass
class ImagerEpochResult:
    loss: float


def _planned_steps(loader: DataLoader, max_steps: int | None) -> int | None:
    """Return expected number of loop steps for progress display."""

    try:
        total = len(loader)
    except TypeError:
        total = None

    if total is None:
        return max_steps
    if max_steps is None:
        return int(total)
    return int(min(total, max_steps))


def _styled_prefix(prefix: str) -> str:
    """Apply lightweight semantic coloring for progress prefixes."""

    if prefix.startswith("[train]"):
        return paint(prefix, color="green", bold=True)
    if prefix.startswith("[val]"):
        return paint(prefix, color="magenta", bold=True)
    return paint(prefix, color="cyan", bold=True)


def _compute_leakage(intensity_map: torch.Tensor, detector_masks: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    union = detector_masks.any(dim=0).to(intensity_map.dtype)
    in_detector = (intensity_map * union.unsqueeze(0)).sum(dim=(-1, -2))
    total = intensity_map.sum(dim=(-1, -2)).clamp_min(eps)
    leakage_ratio = 1.0 - (in_detector / total)
    return leakage_ratio


def run_classifier_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    detector_masks: torch.Tensor,
    *,
    device: torch.device,
    leakage_weight: float,
    temperature: float,
    max_steps: int | None = None,
    progress_prefix: str | None = None,
    log_every_steps: int = 100,
) -> ClassifierEpochResult:
    """Run one epoch for detector-based classifier training/eval."""

    train_mode = optimizer is not None
    model.train(train_mode)

    running_loss = 0.0
    all_energies: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    masks = detector_masks.to(device=device)
    total_steps = _planned_steps(loader, max_steps)
    interval = max(int(log_every_steps), 1)

    for step_idx, (fields, labels) in enumerate(loader):
        if max_steps is not None and step_idx >= max_steps:
            break

        fields = fields.to(device=device)
        labels = labels.to(device=device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        out_field = model(fields)
        out_intensity = field_intensity(out_field)
        energies = integrate_regions(out_intensity, masks, reduction="sum")
        leakage = _compute_leakage(out_intensity, masks)

        loss = classification_loss(
            energies=energies,
            labels=labels,
            leakage_energy=leakage,
            leakage_weight=leakage_weight,
            temperature=temperature,
        )

        if train_mode:
            loss.backward()
            optimizer.step()

        running_loss += float(loss.item())
        all_energies.append(energies.detach())
        all_labels.append(labels.detach())

        if progress_prefix is not None:
            step_no = step_idx + 1
            is_last = (total_steps is not None and step_no >= total_steps)
            if step_no % interval == 0 or is_last:
                total_str = str(total_steps) if total_steps is not None else "?"
                print(
                    f"{_styled_prefix(progress_prefix)} step {step_no}/{total_str} loss={loss.item():.5f}",
                    flush=True,
                )

    if not all_energies:
        return ClassifierEpochResult(loss=0.0, acc=0.0)

    energies_cat = torch.cat(all_energies, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    avg_loss = running_loss / len(all_energies)
    acc = accuracy(energies_cat, labels_cat)
    return ClassifierEpochResult(loss=avg_loss, acc=acc)


def run_imager_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    *,
    device: torch.device,
    max_steps: int | None = None,
    progress_prefix: str | None = None,
    log_every_steps: int = 100,
) -> ImagerEpochResult:
    """Run one epoch for imaging D2NN training/eval."""

    train_mode = optimizer is not None
    model.train(train_mode)

    running_loss = 0.0
    n_steps = 0
    total_steps = _planned_steps(loader, max_steps)
    interval = max(int(log_every_steps), 1)

    for step_idx, (fields, targets) in enumerate(loader):
        if max_steps is not None and step_idx >= max_steps:
            break

        fields = fields.to(device=device)
        targets = targets.to(device=device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        out_field = model(fields)
        out_i = field_intensity(out_field)
        out_i = out_i / out_i.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-8)
        target_i = targets.clamp(0.0, 1.0)

        loss = imaging_mse_loss(out_i, target_i)

        if train_mode:
            loss.backward()
            optimizer.step()

        running_loss += float(loss.item())
        n_steps += 1

        if progress_prefix is not None:
            step_no = step_idx + 1
            is_last = (total_steps is not None and step_no >= total_steps)
            if step_no % interval == 0 or is_last:
                total_str = str(total_steps) if total_steps is not None else "?"
                print(
                    f"{_styled_prefix(progress_prefix)} step {step_no}/{total_str} loss={loss.item():.5f}",
                    flush=True,
                )

    if n_steps == 0:
        return ImagerEpochResult(loss=0.0)
    return ImagerEpochResult(loss=running_loss / n_steps)


def train_classifier(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    detector_masks: torch.Tensor,
    *,
    device: torch.device,
    lr: float,
    epochs: int,
    leakage_weight: float,
    temperature: float,
    max_steps_per_epoch: int | None = None,
    verbose: bool = False,
    log_every_steps: int = 100,
) -> dict[str, Any]:
    """Train classifier and return history dict."""

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: dict[str, list[float]] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch_idx in range(epochs):
        if verbose:
            print(paint(f"[train] epoch {epoch_idx + 1}/{epochs}", color="green", bold=True), flush=True)
        tr = run_classifier_epoch(
            model,
            train_loader,
            optimizer,
            detector_masks,
            device=device,
            leakage_weight=leakage_weight,
            temperature=temperature,
            max_steps=max_steps_per_epoch,
            progress_prefix=(f"[train][epoch {epoch_idx + 1}/{epochs}]" if verbose else None),
            log_every_steps=log_every_steps,
        )
        va = run_classifier_epoch(
            model,
            val_loader,
            None,
            detector_masks,
            device=device,
            leakage_weight=leakage_weight,
            temperature=temperature,
            max_steps=max_steps_per_epoch,
            progress_prefix=(f"[val][epoch {epoch_idx + 1}/{epochs}]" if verbose else None),
            log_every_steps=log_every_steps,
        )
        history["train_loss"].append(tr.loss)
        history["train_acc"].append(tr.acc)
        history["val_loss"].append(va.loss)
        history["val_acc"].append(va.acc)

        if verbose:
            print(
                (
                    f"{paint('[epoch-summary]', color='yellow', bold=True)} {epoch_idx + 1}/{epochs} "
                    f"train_loss={tr.loss:.5f} train_acc={tr.acc:.4f} "
                    f"val_loss={va.loss:.5f} val_acc={va.acc:.4f}"
                ),
                flush=True,
            )

    return history


def train_imager(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: torch.device,
    lr: float,
    epochs: int,
    max_steps_per_epoch: int | None = None,
    verbose: bool = False,
    log_every_steps: int = 100,
) -> dict[str, Any]:
    """Train imaging network and return history dict."""

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch_idx in range(epochs):
        if verbose:
            print(paint(f"[train] epoch {epoch_idx + 1}/{epochs}", color="green", bold=True), flush=True)

        tr = run_imager_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            max_steps=max_steps_per_epoch,
            progress_prefix=(f"[train][epoch {epoch_idx + 1}/{epochs}]" if verbose else None),
            log_every_steps=log_every_steps,
        )
        va = run_imager_epoch(
            model,
            val_loader,
            None,
            device=device,
            max_steps=max_steps_per_epoch,
            progress_prefix=(f"[val][epoch {epoch_idx + 1}/{epochs}]" if verbose else None),
            log_every_steps=log_every_steps,
        )
        history["train_loss"].append(tr.loss)
        history["val_loss"].append(va.loss)

        if verbose:
            print(
                (
                    f"{paint('[epoch-summary]', color='yellow', bold=True)} "
                    f"{epoch_idx + 1}/{epochs} train_loss={tr.loss:.5f} val_loss={va.loss:.5f}"
                ),
                flush=True,
            )

    return history
