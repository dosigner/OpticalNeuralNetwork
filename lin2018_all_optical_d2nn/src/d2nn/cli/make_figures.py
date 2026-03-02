"""Regenerate figures from a run directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from d2nn.cli.common import build_dataloaders, build_detector_tensors, build_model_from_config, choose_device, load_checkpoint, load_config, tensor_to_numpy
from d2nn.detectors.metrics import confusion_matrix, normalize_energies
from d2nn.training.loops import run_classifier_epoch
from d2nn.viz.classifier import plot_confusion_matrix, plot_energy_distribution_heatmap, plot_inference_summary
from d2nn.viz.fields import plot_phase_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate run figures")
    parser.add_argument("--run-dir", required=True, help="Path to run directory")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--phase-stretch", action="store_true", help="Stretch phase map to full display range for visualization")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg_path = run_dir / "config_resolved.yaml"
    ckpt_path = run_dir / "checkpoints" / "final.pt"

    cfg = load_config(cfg_path)
    task = cfg.get("experiment", {}).get("task", "classifier")
    device = choose_device(cfg.get("runtime", {}))
    model = build_model_from_config(cfg).to(device)
    load_checkpoint(ckpt_path, model)

    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    dx = float(cfg.get("physics", {}).get("dx", 1.0))
    phase_max = float(cfg.get("viz", {}).get("phase_display_max", cfg.get("model", {}).get("phase_max", 2.0 * torch.pi)))
    phase_shift_cfg = bool(cfg.get("viz", {}).get("phase_shift_to_display_max", False))
    phase_wrap_cfg = bool(cfg.get("viz", {}).get("phase_wrap_to_display_max", False))
    phase_stretch_cfg = bool(cfg.get("viz", {}).get("phase_stretch_to_display_max", False))
    phase_shift = phase_shift_cfg
    phase_wrap = phase_wrap_cfg
    phase_stretch = bool(args.phase_stretch or phase_stretch_cfg)
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

    if task == "classifier":
        # Figure regeneration should be robust in restricted environments.
        # Force single-process loading regardless of training-time dataloader settings.
        cfg_local = dict(cfg)
        data_local = dict(cfg.get("data", {}))
        data_local["num_workers"] = 0
        data_local["persistent_workers"] = False
        data_local["pin_memory"] = False
        cfg_local["data"] = data_local

        _, val_loader = build_dataloaders(cfg_local)
        masks, layout = build_detector_tensors(cfg)
        eval_out = run_classifier_epoch(
            model,
            val_loader,
            optimizer=None,
            detector_masks=masks,
            device=device,
            leakage_weight=float(cfg.get("loss", {}).get("leakage_weight", 0.1)),
            temperature=float(cfg.get("loss", {}).get("temperature", 1.0)),
            max_steps=args.max_steps,
        )

        # Recompute confusion and class energy from a bounded pass.
        all_e = []
        all_y = []
        model.eval()
        masks_d = masks.to(device)
        from d2nn.detectors.integrate import integrate_regions
        from d2nn.utils.math import intensity

        with torch.no_grad():
            sample_field = None
            sample_intensity = None
            sample_energy = None
            sample_true = None
            sample_pred = None
            for bidx, (fields, labels) in enumerate(val_loader):
                if bidx >= args.max_steps:
                    break
                fields = fields.to(device)
                labels = labels.to(device)
                out = model(fields)
                e = integrate_regions(intensity(out), masks_d, reduction="sum")
                if sample_field is None:
                    sample_field = fields[0].cpu().numpy()
                    sample_intensity = intensity(out)[0].cpu().numpy()
                    sample_energy = e[0].cpu().numpy()
                    sample_true = int(labels[0].item())
                    sample_pred = int(torch.argmax(e[0]).item())
                all_e.append(e.cpu())
                all_y.append(labels.cpu())

        if all_e:
            energies = torch.cat(all_e, dim=0)
            labels = torch.cat(all_y, dim=0)
            cm = confusion_matrix(energies, labels, num_classes=energies.shape[1])
            en = normalize_energies(energies)
            class_energy = torch.zeros((energies.shape[1], energies.shape[1]), dtype=torch.float32)
            for c in range(energies.shape[1]):
                idx = labels == c
                if torch.any(idx):
                    class_energy[c] = en[idx].mean(dim=0)

            plot_confusion_matrix(cm, normalize=False, save_path=figures / "confusion_matrix_counts.png")
            plot_confusion_matrix(cm, normalize=True, save_path=figures / "confusion_matrix_normalized.png")
            plot_energy_distribution_heatmap(class_energy.numpy(), save_path=figures / "energy_distribution_heatmap.png")
            if sample_field is not None and sample_intensity is not None and sample_energy is not None:
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

        print({"task": "classifier", "loss": eval_out.loss, "acc": eval_out.acc})
    else:
        print({"task": task, "status": "phase figures regenerated"})


if __name__ == "__main__":
    main()
