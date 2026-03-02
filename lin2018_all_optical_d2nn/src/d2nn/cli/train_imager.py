"""Train D2NN imaging model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from d2nn.cli.common import (
    build_dataloaders,
    build_model_from_config,
    choose_device,
    create_run_dir,
    load_config,
    tensor_to_numpy,
    update_latest_symlink,
)
from d2nn.physics.asm import asm_propagate, asm_transfer_function
from d2nn.training.callbacks import save_checkpoint, save_metrics, save_resolved_config
from d2nn.training.loops import train_imager
from d2nn.utils.math import intensity
from d2nn.utils.seed import set_global_seed
from d2nn.utils.term import paint
from d2nn.viz.fields import plot_phase_mask
from d2nn.viz.imaging import compute_ssim, plot_imaging_comparison
from d2nn.viz.propagation import plot_propagation_stack, plot_xz_cross_section


def _plot_training_curve(history: dict[str, list[float]], save_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="val")
    ax.set_title("Imaging loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def _first_batch(loader):
    for batch in loader:
        return batch
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train D2NN imager")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--max-steps-per-epoch", type=int, default=None, help="Debug limiter for training steps")
    parser.add_argument("--log-every-steps", type=int, default=100, help="Print batch progress every N steps")
    parser.add_argument("--quiet", action="store_true", help="Disable progress logs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    runtime = cfg.get("runtime", {})
    training_cfg = cfg.get("training", {})
    physics = cfg.get("physics", {})
    model_cfg = cfg.get("model", {})

    seed = int(runtime.get("seed", 1234))
    deterministic = bool(runtime.get("deterministic", True))
    set_global_seed(seed, deterministic=deterministic)

    run_dir = create_run_dir(cfg)
    save_resolved_config(run_dir / "config_resolved.yaml", cfg)

    device = choose_device(runtime)
    train_loader, val_loader = build_dataloaders(cfg)

    model = build_model_from_config(cfg).to(device)
    if not args.quiet:
        print(
            (
                f"{paint('[run]', color='cyan', bold=True)} dir={run_dir} task=imager device={device} seed={seed} "
                f"epochs={int(training_cfg.get('epochs', 10))} "
                f"train_batches={len(train_loader)} val_batches={len(val_loader)}"
            ),
            flush=True,
        )

    history = train_imager(
        model,
        train_loader,
        val_loader,
        device=device,
        lr=float(training_cfg.get("lr", 1e-3)),
        epochs=int(training_cfg.get("epochs", 10)),
        max_steps_per_epoch=args.max_steps_per_epoch or training_cfg.get("max_steps_per_epoch", None),
        verbose=(not args.quiet),
        log_every_steps=args.log_every_steps,
    )

    checkpoint_path = run_dir / "checkpoints" / "final.pt"
    save_checkpoint(checkpoint_path, model=model, optimizer=None, epoch=int(training_cfg.get("epochs", 10)), extra={"history": history})

    batch = _first_batch(val_loader)
    ssim_d2nn = float("nan")
    ssim_free = float("nan")

    figures = run_dir / "figures"
    _plot_training_curve(history, figures / "training_curve.png")

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

    if batch is not None:
        fields, target = batch
        fields = fields.to(device)
        target = target.to(device)

        with torch.no_grad():
            out, intermediates = model(fields, return_intermediates=True)
            out_i = intensity(out)
            out_i = out_i / out_i.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-8)

            z_total = float(physics["z_layer"]) * int(cfg["model"]["num_layers"]) + float(physics["z_out"])
            H_free = asm_transfer_function(
                N=int(physics["N"]),
                dx=float(physics["dx"]),
                wavelength=float(physics["wavelength"]),
                z=z_total,
                bandlimit=bool(physics.get("bandlimit", True)),
                fftshifted=bool(physics.get("fftshifted", False)),
                dtype=str(cfg["model"].get("dtype", "complex64")),
                device=device,
            )
            free = asm_propagate(fields, H_free, fftshifted=bool(physics.get("fftshifted", False)))
            free_i = intensity(free)
            free_i = free_i / free_i.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-8)

        tgt_np = tensor_to_numpy(target[0])
        out_np = tensor_to_numpy(out_i[0])
        free_np = tensor_to_numpy(free_i[0])
        ssim_d2nn = compute_ssim(out_np, tgt_np)
        ssim_free = compute_ssim(free_np, tgt_np)

        plot_imaging_comparison(
            input_image=tgt_np,
            d2nn_output=out_np,
            free_space_output=free_np,
            ssim_d2nn=ssim_d2nn,
            ssim_free=ssim_free,
            save_path=figures / "imaging_comparison.png",
        )

        inter_np = [tensor_to_numpy(x[0]) for x in intermediates]
        plot_propagation_stack(inter_np, quantity="amplitude", save_path=figures / "propagation_stack_amp.png")
        plot_propagation_stack(inter_np, quantity="phase", save_path=figures / "propagation_stack_phase.png")
        plot_xz_cross_section(inter_np, x_index=inter_np[0].shape[1] // 2, quantity="amplitude", save_path=figures / "propagation_xz_amp.png")

    metrics: dict[str, Any] = {
        "train_loss_last": history["train_loss"][-1] if history["train_loss"] else None,
        "val_loss_last": history["val_loss"][-1] if history["val_loss"] else None,
        "ssim_d2nn": ssim_d2nn,
        "ssim_free_space": ssim_free,
        "epochs": int(training_cfg.get("epochs", 10)),
        "seed": seed,
        "device": str(device),
    }
    save_metrics(run_dir / "metrics.json", metrics)

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
