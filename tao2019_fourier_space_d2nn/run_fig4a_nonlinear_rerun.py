"""Re-run Figure 4(a) nonlinear configs and regenerate plot.

Retrains:
  - Nonlinear Real 5L   (position: per_layer, already correct)
  - Nonlinear Fourier 5L (position: per_layer, fixed from rear)

Reuses existing linear results and regenerates fig4a plot.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR / "src"))

from tao2019_fd2nn.cli.common import (
    build_dataloaders, build_detector_masks, build_model,
    build_test_loader, choose_device, create_run_dir, load_config,
)
from tao2019_fd2nn.models.detectors import integrate_detector_energies
from tao2019_fd2nn.training.callbacks import save_checkpoint, save_metrics, save_resolved_config
from tao2019_fd2nn.training.metrics_classification import accuracy
from tao2019_fd2nn.training.trainer import best_epoch_index, summarize_history, train_classifier
from tao2019_fd2nn.utils.live_log import LiveLogger
from tao2019_fd2nn.utils.math import intensity
from tao2019_fd2nn.utils.seed import set_global_seed

EPOCHS = 30
BATCH_SIZE = 10

NONLINEAR_CONFIGS = [
    {
        "label": "Nonlinear Real",
        "config_path": "src/tao2019_fd2nn/config/cls_mnist_nonlinear_real_5l.yaml",
        "color": "#ff7f0e",
    },
    {
        "label": "Nonlinear Fourier",
        "config_path": "src/tao2019_fd2nn/config/cls_mnist_nonlinear_fourier_5l_f4mm.yaml",
        "color": "#d62728",
    },
]

ALL_CONFIGS = [
    {"label": "Linear Real",      "color": "#1f77b4", "run_dir": "runs/fig4a_cls_mnist_linear_real_5l/260227_094918"},
    {"label": "Nonlinear Real",   "color": "#ff7f0e"},
    {"label": "Linear Fourier",   "color": "#2ca02c", "run_dir": "runs/fig4a_cls_mnist_linear_fourier_5l_f1mm/260227_113511"},
    {"label": "Nonlinear Fourier","color": "#d62728"},
]


def _sample_eval(model, loader, detector_masks, device):
    model.eval()
    all_e, all_y = [], []
    with torch.no_grad():
        for fields, labels in loader:
            fields, labels = fields.to(device), labels.to(device)
            energies = integrate_detector_energies(intensity(model(fields)), detector_masks.to(device))
            all_e.append(energies.cpu())
            all_y.append(labels.cpu())
    if not all_e:
        return torch.zeros(0, 10).numpy(), torch.zeros(0).numpy()
    return torch.cat(all_e, 0).numpy(), torch.cat(all_y, 0).numpy()


def train_single(config_entry, base_dir):
    label = config_entry["label"]
    config_path = base_dir / config_entry["config_path"]

    print(f"\n{'='*70}")
    print(f"  Training: {label}")
    print(f"  Config:   {config_path}")
    print(f"{'='*70}\n")

    cfg = load_config(str(config_path))
    cfg["training"]["batch_size"] = BATCH_SIZE
    cfg["training"]["epochs"] = EPOCHS

    exp = cfg["experiment"]
    set_global_seed(int(exp.get("seed", 42)), deterministic=bool(exp.get("deterministic", True)))

    run_dir = create_run_dir(cfg, cwd=base_dir)
    save_resolved_config(run_dir / "config_resolved.yaml", cfg)

    device = choose_device(exp)
    log = LiveLogger(
        run_dir=run_dir, task="classification", total_epochs=EPOCHS,
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
        model, train_loader, val_loader, detector_masks,
        device=device, lr=float(cfg["training"]["lr"]), epochs=EPOCHS,
        loss_mode=str(cfg["training"].get("loss", "cross_entropy")),
        leakage_weight=0.0, temperature=1.0, test_loader=test_loader,
        max_steps_per_epoch=None, step_callback=log.on_step, epoch_callback=log.on_epoch_end,
    )

    save_checkpoint(run_dir / "checkpoints" / "final.pt", model=model, optimizer=None,
                    epoch=EPOCHS, extra={"history": history})
    best_idx = best_epoch_index(history, metric_name="val_acc", maximize=True)
    save_checkpoint(run_dir / "checkpoints" / "best.pt", model=model, optimizer=None,
                    epoch=best_idx + 1, extra={"best_metric": history["val_acc"][best_idx] if history["val_acc"] else None})

    eval_loader = test_loader if test_loader is not None else val_loader
    e_np, y_np = _sample_eval(model, eval_loader, detector_masks, device)
    import numpy as np
    eval_acc = accuracy(torch.from_numpy(e_np), torch.from_numpy(y_np)) if e_np.size else 0.0

    metrics = summarize_history(history)
    metrics["eval_acc"] = eval_acc
    metrics["best_epoch"] = best_idx + 1
    save_metrics(run_dir / "metrics.json", metrics)
    log.finish(run_dir=run_dir)

    print(f"\n  Result: test_acc={eval_acc*100:.1f}%  best_epoch={best_idx+1}")
    return run_dir, history


def load_history_from_run(run_dir: str) -> dict:
    ckpt_path = _SCRIPT_DIR / run_dir / "checkpoints" / "final.pt"
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    return ckpt.get("history", {})


def plot_fig4a(histories: dict[str, dict], out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("MNIST Dataset Classification", fontsize=14)
    ax.set_xlabel("Epoch Number", fontsize=12)
    ax.set_ylabel("Classification Accuracy", fontsize=12)

    for entry in ALL_CONFIGS:
        h = histories[entry["label"]]
        accs = h.get("test_acc", [])
        if not accs:
            accs = h.get("val_acc", [])
        epochs = [0] + list(range(1, len(accs) + 1))
        accs_plot = [0.0] + [float(v) for v in accs]
        max_acc = max(accs_plot)
        ax.plot(epochs, accs_plot, color=entry["color"], linewidth=2,
                label=f"{entry['label']} ({max_acc*100:.1f}%)")

    ax.set_xlim(0, 30)
    ax.set_ylim(0.8, 1.0)
    ax.legend(loc="lower right", fontsize=9,
              title="D$^2$NN,\nMaximum Testing Accuracy", title_fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}")


def main():
    base_dir = _SCRIPT_DIR
    histories = {}

    # Load existing linear results
    for entry in ALL_CONFIGS:
        if "run_dir" in entry:
            histories[entry["label"]] = load_history_from_run(entry["run_dir"])
            print(f"Loaded existing: {entry['label']}")

    # Retrain nonlinear configs
    for entry in NONLINEAR_CONFIGS:
        run_dir, history = train_single(entry, base_dir)
        histories[entry["label"]] = history

    plot_fig4a(histories, base_dir / "fig4a_mnist_bs10_convergence.png")

    print(f"\n{'='*70}")
    print("  Figure 4(a) Results")
    print(f"{'='*70}")
    for entry in ALL_CONFIGS:
        h = histories[entry["label"]]
        accs = h.get("test_acc", [])
        if accs:
            print(f"  {entry['label']:<22s} max={max(accs)*100:.1f}%  final={accs[-1]*100:.1f}%")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
