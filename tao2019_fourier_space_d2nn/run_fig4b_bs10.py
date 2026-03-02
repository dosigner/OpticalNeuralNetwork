"""Reproduce Tao 2019 Figure 4(b): MNIST classification with batch_size=10, 30 epochs.

Trains 4 D2NN configurations sequentially:
  1. 10 Layers Linear Real
  2. 10 Layers Nonlinear Real
  3.  5 Layers Nonlinear Fourier & Real (hybrid)
  4. 10 Layers Nonlinear Fourier & Real (hybrid)

Outputs:
  - Per-config run directories under runs/
  - fig4b_mnist_bs10_summary.json
  - fig4b_mnist_bs10_convergence.png
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure the local src is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR / "src"))

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

BATCH_SIZE = 10
EPOCHS = 30

CONFIGS = [
    {
        "label": "10 Layers Linear Real",
        "config_path": "src/tao2019_fd2nn/config/cls_mnist_linear_real_10l.yaml",
        "color": "#1f77b4",
    },
    {
        "label": "10 Layers Nonlinear Real",
        "config_path": "src/tao2019_fd2nn/config/cls_mnist_nonlinear_real_10l.yaml",
        "color": "#ff7f0e",
    },
    {
        "label": "5 Layers Nonlinear Fourier&Real",
        "config_path": "src/tao2019_fd2nn/config/cls_mnist_hybrid_5l.yaml",
        "color": "#e6b422",
    },
    {
        "label": "10 Layers Nonlinear Fourier&Real",
        "config_path": "src/tao2019_fd2nn/config/cls_mnist_hybrid_10l.yaml",
        "color": "#9467bd",
    },
]


def _sample_eval(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    detector_masks: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_e: list[torch.Tensor] = []
    all_y: list[torch.Tensor] = []
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


def train_single(config_entry: dict[str, str], base_dir: Path) -> dict[str, Any]:
    label = config_entry["label"]
    config_path = base_dir / config_entry["config_path"]

    print(f"\n{'='*70}")
    print(f"  Training: {label}")
    print(f"  Config:   {config_path}")
    print(f"  Batch:    {BATCH_SIZE}, Epochs: {EPOCHS}")
    print(f"{'='*70}\n")

    cfg = load_config(str(config_path))

    # Override batch_size and epochs
    cfg["training"]["batch_size"] = BATCH_SIZE
    cfg["training"]["epochs"] = EPOCHS

    # Append bs10 tag to experiment name for unique run dirs
    cfg["experiment"]["name"] = cfg["experiment"]["name"] + "_bs10"

    exp = cfg["experiment"]
    set_global_seed(int(exp.get("seed", 42)), deterministic=bool(exp.get("deterministic", True)))

    run_dir = create_run_dir(cfg, cwd=base_dir)
    save_resolved_config(run_dir / "config_resolved.yaml", cfg)

    device = choose_device(exp)
    log = LiveLogger(
        run_dir=run_dir,
        task="classification",
        total_epochs=EPOCHS,
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
        epochs=EPOCHS,
        loss_mode=str(cfg["training"].get("loss", "cross_entropy")),
        leakage_weight=0.0,
        temperature=1.0,
        test_loader=test_loader,
        max_steps_per_epoch=None,
        step_callback=log.on_step,
        epoch_callback=log.on_epoch_end,
    )

    # Save checkpoints
    save_checkpoint(
        run_dir / "checkpoints" / "final.pt",
        model=model,
        optimizer=None,
        epoch=EPOCHS,
        extra={"history": history},
    )
    best_idx = best_epoch_index(history, metric_name="val_acc", maximize=True)
    best_metric = history["val_acc"][best_idx] if history["val_acc"] else None
    save_checkpoint(
        run_dir / "checkpoints" / "best.pt",
        model=model,
        optimizer=None,
        epoch=best_idx + 1,
        extra={"best_metric": best_metric},
    )

    # Final eval
    eval_loader = test_loader if test_loader is not None else val_loader
    e_np, y_np = _sample_eval(model, eval_loader, detector_masks, device)
    eval_acc = accuracy(torch.from_numpy(e_np), torch.from_numpy(y_np)) if e_np.size else 0.0

    metrics: dict[str, Any] = summarize_history(history)
    metrics["eval_acc"] = eval_acc
    metrics["best_epoch"] = best_idx + 1
    save_metrics(run_dir / "metrics.json", metrics)

    log.finish(run_dir=run_dir)

    result = {
        "label": label,
        "config": config_entry["config_path"],
        "run_dir": str(run_dir),
        "epochs_recorded": len(history["val_acc"]),
        "max_val_acc": float(max(history["val_acc"])) if history["val_acc"] else 0.0,
        "final_val_acc": float(history["val_acc"][-1]) if history["val_acc"] else 0.0,
        "best_epoch": best_idx + 1,
        "test_acc": float(eval_acc),
        "test_acc_history": [float(v) for v in history.get("test_acc", [])],
        "val_acc_history": [float(v) for v in history["val_acc"]],
    }

    print(f"\n  Result: max_val_acc={result['max_val_acc']:.4f}, test_acc={result['test_acc']:.4f}, best_epoch={result['best_epoch']}")
    return result


def plot_convergence(results: list[dict[str, Any]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("MNIST Dataset Classification", fontsize=14)
    ax.set_xlabel("Epoch Number", fontsize=12)
    ax.set_ylabel("Classification Accuracy", fontsize=12)

    for i, (entry, res) in enumerate(zip(CONFIGS, results)):
        # Use test_acc_history if available, otherwise val_acc_history
        accs = res.get("test_acc_history", [])
        if not accs:
            accs = res["val_acc_history"]
        epochs = [0] + list(range(1, len(accs) + 1))
        accs = [0.0] + list(accs)
        max_acc = max(accs) if accs else 0.0
        ax.plot(
            epochs,
            accs,
            color=entry["color"],
            linewidth=2,
            label=f"{entry['label']} ({max_acc*100:.1f}%)",
        )

    ax.set_xlim(0, 30)
    ax.set_ylim(0.8, 1.0)
    ax.legend(
        loc="lower right",
        fontsize=9,
        title="D$^2$NN,\nMaximum Testing Accuracy",
        title_fontsize=9,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Convergence plot saved: {out_path}")


def main() -> None:
    base_dir = _SCRIPT_DIR
    all_results: list[dict[str, Any]] = []

    for entry in CONFIGS:
        result = train_single(entry, base_dir)
        all_results.append(result)

    # Save summary JSON
    summary = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": "mnist",
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "configurations": [e["label"] for e in CONFIGS],
        "results": all_results,
    }

    summary_path = base_dir / "fig4b_mnist_bs10_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved: {summary_path}")

    # Plot convergence
    plot_path = base_dir / "fig4b_mnist_bs10_convergence.png"
    plot_convergence(all_results, plot_path)

    # Final summary table
    print(f"\n{'='*70}")
    print("  Figure 4(b) Results Summary (batch_size=10)")
    print(f"{'='*70}")
    for res in all_results:
        print(f"  {res['label']:<42s} test_acc={res['test_acc']*100:.1f}%  (best_epoch={res['best_epoch']})")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
