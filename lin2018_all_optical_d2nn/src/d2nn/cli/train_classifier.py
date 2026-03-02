"""Train detector-based D2NN classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from d2nn.cli.common import (
    build_dataloaders,
    build_detector_tensors,
    build_model_from_config,
    choose_device,
    create_run_dir,
    load_config,
    tensor_to_numpy,
    update_latest_symlink,
)
from d2nn.detectors.metrics import confusion_matrix, normalize_energies
from d2nn.training.callbacks import save_checkpoint, save_metrics, save_resolved_config
from d2nn.training.loops import train_classifier
from d2nn.utils.seed import set_global_seed
from d2nn.utils.term import paint
from d2nn.viz.classifier import plot_confusion_matrix, plot_energy_distribution_heatmap, plot_inference_summary, plot_output_with_detectors
from d2nn.viz.fields import plot_phase_mask
from d2nn.viz.style import apply_style


def _plot_training_curves(
    history: dict[str, list[float]],
    save_path: Path,
    *,
    num_layers: int,
    neurons_million: float,
    layer_distance_cm: float,
    modulation_title: str,
    blind_test_acc: float | None = None,
) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    blue = "#1f4cff"
    red = "#ff1f1f"
    bg = "#e8e8e8"

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    train_acc = history.get("train_acc", [])
    val_acc = history.get("val_acc", [])
    loss_curve = val_loss if val_loss else train_loss
    acc_curve = val_acc if val_acc else train_acc
    if not loss_curve and not acc_curve:
        return

    epochs = list(range(len(loss_curve))) if loss_curve else list(range(len(acc_curve)))

    fig, ax_loss = plt.subplots(1, 1, figsize=(7.2, 4.6))
    fig.patch.set_facecolor(bg)
    ax_loss.set_facecolor(bg)
    ax_acc = ax_loss.twinx()

    if loss_curve:
        ax_loss.plot(epochs, loss_curve, color=blue, linewidth=1.8)
    if acc_curve:
        ax_acc.plot(epochs, acc_curve, color=red, linewidth=1.8)

    ax_loss.set_xlabel("Epoch Number")
    ax_loss.set_ylabel("Loss", color=blue)
    ax_acc.set_ylabel("Classification Accuracy", color=red)
    ax_loss.tick_params(axis="y", colors=blue)
    ax_acc.tick_params(axis="y", colors=red)
    ax_loss.tick_params(axis="x", colors="#666666")
    ax_acc.set_ylim(0.0, 1.0)

    if loss_curve:
        max_loss = max(loss_curve)
        ax_loss.set_ylim(0.0, max_loss * 1.05 if max_loss > 0.0 else 1.0)

    for spine in ax_loss.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.2)
    for spine in ax_acc.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.2)

    ax_loss.set_title(modulation_title, pad=12)

    left_info = f"{num_layers} layer\n{neurons_million:.2f} Million Neurons\n{layer_distance_cm:g}cm Layer Distance"
    ax_loss.text(
        -0.22,
        0.5,
        left_info,
        transform=ax_loss.transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=11,
        color="#222222",
    )

    if blind_test_acc is not None:
        ax_loss.text(
            0.30,
            0.45,
            f"Blind Testing Accuracy: {blind_test_acc * 100:.2f}%",
            transform=ax_loss.transAxes,
            fontsize=12,
            color="#222222",
        )

    fig.subplots_adjust(left=0.27, right=0.83, top=0.84, bottom=0.18)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def _collect_eval_outputs(model: torch.nn.Module, loader, detector_masks: torch.Tensor, device: torch.device, max_batches: int | None = None):
    from d2nn.detectors.integrate import integrate_regions
    from d2nn.utils.math import intensity

    all_e: list[torch.Tensor] = []
    all_y: list[torch.Tensor] = []
    sample_i = None
    sample_field = None
    sample_energy = None
    sample_true = None
    sample_pred = None

    model.eval()
    masks = detector_masks.to(device=device)
    with torch.no_grad():
        for bidx, (fields, labels) in enumerate(loader):
            if max_batches is not None and bidx >= max_batches:
                break
            fields = fields.to(device)
            labels = labels.to(device)
            out = model(fields)
            out_i = intensity(out)
            energies = integrate_regions(out_i, masks, reduction="sum")
            all_e.append(energies.detach().cpu())
            all_y.append(labels.detach().cpu())
            if sample_i is None:
                sample_i = out_i[0].detach().cpu().numpy()
                sample_field = fields[0].detach().cpu().numpy()
                sample_energy = energies[0].detach().cpu().numpy()
                sample_true = int(labels[0].item())
                sample_pred = int(torch.argmax(energies[0]).item())

    if not all_e:
        return (
            torch.empty(0, 0),
            torch.empty(0, dtype=torch.long),
            np.zeros((detector_masks.shape[-1], detector_masks.shape[-1])),
            None,
            None,
            None,
            None,
        )

    return (
        torch.cat(all_e, dim=0),
        torch.cat(all_y, dim=0),
        sample_i,
        sample_field,
        sample_energy,
        sample_true,
        sample_pred,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train D2NN classifier")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--max-steps-per-epoch", type=int, default=None, help="Debug limiter for training steps")
    parser.add_argument("--log-every-steps", type=int, default=100, help="Print batch progress every N steps")
    parser.add_argument("--quiet", action="store_true", help="Disable progress logs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    runtime = cfg.get("runtime", {})
    training_cfg = cfg.get("training", {})
    loss_cfg = cfg.get("loss", {})
    physics = cfg.get("physics", {})
    model_cfg = cfg.get("model", {})

    seed = int(runtime.get("seed", 1234))
    deterministic = bool(runtime.get("deterministic", True))
    set_global_seed(seed, deterministic=deterministic)

    run_dir = create_run_dir(cfg)
    save_resolved_config(run_dir / "config_resolved.yaml", cfg)

    device = choose_device(runtime)
    train_loader, val_loader = build_dataloaders(cfg)
    detector_masks, layout = build_detector_tensors(cfg)

    model = build_model_from_config(cfg).to(device)
    if not args.quiet:
        print(
            (
                f"{paint('[run]', color='cyan', bold=True)} dir={run_dir} task=classifier device={device} seed={seed} "
                f"epochs={int(training_cfg.get('epochs', 10))} "
                f"train_batches={len(train_loader)} val_batches={len(val_loader)}"
            ),
            flush=True,
        )

    history = train_classifier(
        model,
        train_loader,
        val_loader,
        detector_masks,
        device=device,
        lr=float(training_cfg.get("lr", 1e-3)),
        epochs=int(training_cfg.get("epochs", 10)),
        leakage_weight=float(loss_cfg.get("leakage_weight", 0.1)),
        temperature=float(loss_cfg.get("temperature", 1.0)),
        max_steps_per_epoch=args.max_steps_per_epoch or training_cfg.get("max_steps_per_epoch", None),
        verbose=(not args.quiet),
        log_every_steps=args.log_every_steps,
    )

    checkpoint_path = run_dir / "checkpoints" / "final.pt"
    save_checkpoint(checkpoint_path, model=model, optimizer=None, epoch=int(training_cfg.get("epochs", 10)), extra={"history": history})

    (
        eval_energies,
        eval_labels,
        sample_intensity,
        sample_field,
        sample_energy,
        sample_true,
        sample_pred,
    ) = _collect_eval_outputs(model, val_loader, detector_masks, device)
    num_classes = int(eval_energies.shape[1]) if eval_energies.ndim == 2 and eval_energies.numel() else 10

    if eval_energies.numel() > 0:
        eval_preds = torch.argmax(eval_energies, dim=-1)
        val_acc = float((eval_preds == eval_labels).float().mean().item())
        cm = confusion_matrix(eval_energies, eval_labels, num_classes=num_classes)
        en = normalize_energies(eval_energies)

        class_energy = np.zeros((num_classes, num_classes), dtype=np.float64)
        for c in range(num_classes):
            idx = eval_labels == c
            if torch.any(idx):
                class_energy[c] = en[idx].mean(dim=0).numpy()
    else:
        val_acc = 0.0
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        class_energy = np.zeros((num_classes, num_classes), dtype=np.float64)

    metrics: dict[str, Any] = {
        "train_loss_last": history["train_loss"][-1] if history["train_loss"] else None,
        "train_acc_last": history["train_acc"][-1] if history["train_acc"] else None,
        "val_loss_last": history["val_loss"][-1] if history["val_loss"] else None,
        "val_acc_last": history["val_acc"][-1] if history["val_acc"] else None,
        "val_acc_eval": val_acc,
        "epochs": int(training_cfg.get("epochs", 10)),
        "seed": seed,
        "device": str(device),
    }
    save_metrics(run_dir / "metrics.json", metrics)

    figures = run_dir / "figures"
    num_layers = int(model_cfg.get("num_layers", len(model.layers)))
    N = int(physics.get("N", 0))
    neurons_million = float(N * N) / 1e6 if N > 0 else 0.0
    layer_distance_cm = float(physics.get("z_layer", 0.0)) * 100.0
    modulation_title = "Complex Modulation" if bool(model_cfg.get("train_amplitude", False)) else "Phase Only Modulation"
    _plot_training_curves(
        history,
        figures / "training_curves.png",
        num_layers=num_layers,
        neurons_million=neurons_million,
        layer_distance_cm=layer_distance_cm,
        modulation_title=modulation_title,
        blind_test_acc=val_acc,
    )

    dx = float(physics.get("dx", 1.0))
    viz_cfg = cfg.get("viz", {})
    phase_max = float(viz_cfg.get("phase_display_max", model_cfg.get("phase_max", 2.0 * np.pi)))
    phase_shift = bool(viz_cfg.get("phase_shift_to_display_max", False))
    phase_wrap = bool(viz_cfg.get("phase_wrap_to_display_max", False))
    phase_stretch = bool(viz_cfg.get("phase_stretch_to_display_max", False))
    for idx, layer in enumerate(model.layers):
        phase = tensor_to_numpy(layer.phase_constraint(layer.raw_phase))
        plot_phase_mask(
            phase,
            dx=dx,
            phase_max=phase_max,
            shift_to_display_max=phase_shift,
            wrap_to_display_max=phase_wrap,
            stretch_to_display_max=phase_stretch,
            title=f"Layer {idx + 1} phase",
            save_path=figures / f"phase_layer_{idx + 1}.png",
        )

    if sample_intensity is not None:
        plot_output_with_detectors(sample_intensity, layout, dx=dx, save_path=figures / "sample_output_with_detectors.png")
    if sample_intensity is not None and sample_field is not None and sample_energy is not None:
        sample_amp = np.abs(sample_field)
        sample_phase = np.angle(sample_field)
        if float(sample_amp.std()) < 1e-6:
            sample_map = (sample_phase + np.pi) / (2.0 * np.pi)
            input_title = "Input Phase"
        else:
            sample_map = sample_amp
            input_title = "Input Digit"
        plot_inference_summary(
            sample_map,
            sample_intensity,
            layout,
            sample_energy,
            dx=dx,
            input_title=input_title,
            pred_label=sample_pred,
            true_label=sample_true,
            save_path=figures / "sample_inference_summary.png",
        )

    plot_confusion_matrix(cm, normalize=False, save_path=figures / "confusion_matrix_counts.png")
    plot_confusion_matrix(cm, normalize=True, save_path=figures / "confusion_matrix_normalized.png")
    plot_energy_distribution_heatmap(class_energy, save_path=figures / "energy_distribution_heatmap.png")

    latest = update_latest_symlink(run_dir)
    if not args.quiet:
        print(
            f"{paint('[done]', color='blue', bold=True)} metrics={run_dir / 'metrics.json'} figures={run_dir / 'figures'}",
            flush=True,
        )
        print(f"{paint('[latest]', color='cyan', bold=True)} {latest}", flush=True)
        for fig in sorted(figures.glob("*.png")):
            print(f"{paint('[figure]', color='white')} {fig}", flush=True)
    print(str(run_dir))


if __name__ == "__main__":
    main()
