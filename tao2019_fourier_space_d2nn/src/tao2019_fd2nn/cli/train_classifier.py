"""Train classification model with spec-style config."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tao2019_fd2nn.cli.common import (
    build_dataloaders,
    build_detector_masks,
    build_model,
    build_test_loader,
    choose_device,
    create_run_dir,
    load_config,
)
from tao2019_fd2nn.models.detectors import integrate_detector_energies
from tao2019_fd2nn.training.callbacks import save_checkpoint, save_metrics, save_resolved_config
from tao2019_fd2nn.training.metrics_classification import accuracy
from tao2019_fd2nn.training.trainer import best_epoch_index, summarize_history, train_classifier
from tao2019_fd2nn.utils.live_log import LiveLogger
from tao2019_fd2nn.utils.math import intensity
from tao2019_fd2nn.utils.seed import set_global_seed
from tao2019_fd2nn.viz.figure_factory import FigureFactory


def _sample_eval(model: torch.nn.Module, loader, detector_masks: torch.Tensor, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_e = []
    all_y = []
    with torch.no_grad():
        for fields, labels in loader:
            fields = fields.to(device)
            labels = labels.to(device)
            energies = integrate_detector_energies(intensity(model(fields)), detector_masks.to(device))
            all_e.append(energies.cpu())
            all_y.append(labels.cpu())
    if not all_e:
        return np.zeros((0, 10), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return torch.cat(all_e, dim=0).numpy(), torch.cat(all_y, dim=0).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tao2019 D2NN classifier")
    parser.add_argument("--config", required=True, help="spec-style YAML config")
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp = cfg["experiment"]
    set_global_seed(int(exp.get("seed", 42)), deterministic=bool(exp.get("deterministic", True)))

    run_dir = create_run_dir(cfg, cwd=Path(__file__).resolve().parents[3])
    save_resolved_config(run_dir / "config_resolved.yaml", cfg)

    device = choose_device(exp)
    log = LiveLogger(
        run_dir=run_dir,
        task="classification",
        total_epochs=int(cfg["training"]["epochs"]),
        log_interval_steps=int(cfg["training"].get("log_interval_steps", 20)),
        use_color=bool(cfg["training"].get("color_logs", True)),
        show_cuda_memory=bool(cfg["training"].get("show_cuda_memory", True)),
    )
    log.start(experiment_name=str(exp["name"]), device=str(device))

    model = build_model(cfg).to(device)
    train_loader, val_loader = build_dataloaders(cfg)
    test_loader = build_test_loader(cfg)
    detector_masks = build_detector_masks(cfg, device=device)

    history = train_classifier(
        model,
        train_loader,
        val_loader,
        detector_masks,
        device=device,
        lr=float(cfg["training"]["lr"]),
        epochs=int(cfg["training"]["epochs"]),
        loss_mode=str(cfg["training"].get("loss", "cross_entropy")),
        leakage_weight=0.0,
        temperature=1.0,
        test_loader=test_loader,
        max_steps_per_epoch=args.max_steps_per_epoch,
        step_callback=log.on_step,
        epoch_callback=log.on_epoch_end,
    )

    final_epoch = int(cfg["training"]["epochs"])
    save_checkpoint(run_dir / "checkpoints" / "final.pt", model=model, optimizer=None, epoch=final_epoch, extra={"history": history})
    best_idx = best_epoch_index(history, metric_name="val_acc", maximize=True)
    best_metric = history["val_acc"][best_idx] if history["val_acc"] else None
    save_checkpoint(run_dir / "checkpoints" / "best.pt", model=model, optimizer=None, epoch=best_idx + 1, extra={"best_metric": best_metric})

    eval_loader = test_loader if test_loader is not None else val_loader
    e_np, y_np = _sample_eval(model, eval_loader, detector_masks, device)
    eval_acc = accuracy(torch.from_numpy(e_np), torch.from_numpy(y_np)) if e_np.size else 0.0

    metrics: dict[str, Any] = summarize_history(history)
    metrics["eval_acc"] = eval_acc
    metrics["best_epoch"] = best_idx + 1
    save_metrics(run_dir / "metrics.json", metrics)

    factory = FigureFactory(run_dir / "figures")
    factory.plot_convergence(history, left_key="val_loss", right_key="val_acc", name="convergence_cls.png")
    phases = [layer.phase().detach().cpu().numpy() for layer in model.layers]
    factory.plot_phase_masks(phases, phase_max=float(cfg["model"]["modulation"]["phase_max_rad"]), name="phase_masks.png")

    log.finish(run_dir=run_dir)
    print(str(run_dir))


if __name__ == "__main__":
    main()
