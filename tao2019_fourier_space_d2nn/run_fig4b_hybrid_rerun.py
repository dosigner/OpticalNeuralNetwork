"""Re-run hybrid configs (3,4) with corrected layer_spacing_m=0.0, both bs1024 and bs10.

Combines with existing results from configs 1,2 to regenerate summary + convergence plot.
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

EPOCHS = 30

HYBRID_CONFIGS = [
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

ALL_CONFIGS = [
    {"label": "10 Layers Linear Real", "color": "#1f77b4"},
    {"label": "10 Layers Nonlinear Real", "color": "#ff7f0e"},
    {"label": "5 Layers Nonlinear Fourier&Real", "color": "#e6b422"},
    {"label": "10 Layers Nonlinear Fourier&Real", "color": "#9467bd"},
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
        return np.zeros((0, 10), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return torch.cat(all_e, 0).numpy(), torch.cat(all_y, 0).numpy()


def train_single(config_entry, base_dir, batch_size, tag):
    label = config_entry["label"]
    config_path = base_dir / config_entry["config_path"]

    print(f"\n{'='*70}")
    print(f"  Training: {label}")
    print(f"  Config:   {config_path}")
    print(f"  Batch:    {batch_size}, Epochs: {EPOCHS}")
    print(f"{'='*70}\n")

    cfg = load_config(str(config_path))
    cfg["training"]["batch_size"] = batch_size
    cfg["training"]["epochs"] = EPOCHS
    cfg["experiment"]["name"] = cfg["experiment"]["name"] + f"_{tag}"

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

    save_checkpoint(run_dir / "checkpoints" / "final.pt", model=model, optimizer=None, epoch=EPOCHS, extra={"history": history})
    best_idx = best_epoch_index(history, metric_name="val_acc", maximize=True)
    best_metric = history["val_acc"][best_idx] if history["val_acc"] else None
    save_checkpoint(run_dir / "checkpoints" / "best.pt", model=model, optimizer=None, epoch=best_idx + 1, extra={"best_metric": best_metric})

    eval_loader = test_loader if test_loader is not None else val_loader
    e_np, y_np = _sample_eval(model, eval_loader, detector_masks, device)
    eval_acc = accuracy(torch.from_numpy(e_np), torch.from_numpy(y_np)) if e_np.size else 0.0

    metrics = summarize_history(history)
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


def load_existing_results(summary_path):
    with open(summary_path) as f:
        return json.load(f)


def plot_convergence(results, out_path, batch_size):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("MNIST Dataset Classification", fontsize=14)
    ax.set_xlabel("Epoch Number", fontsize=12)
    ax.set_ylabel("Classification Accuracy", fontsize=12)

    for entry, res in zip(ALL_CONFIGS, results):
        accs = res.get("test_acc_history", [])
        if not accs:
            accs = res["val_acc_history"]
        epochs = [0] + list(range(1, len(accs) + 1))
        accs = [0.0] + list(accs)
        max_acc = max(accs) if accs else 0.0
        ax.plot(epochs, accs, color=entry["color"], linewidth=2,
                label=f"{entry['label']} ({max_acc*100:.1f}%)")

    ax.set_xlim(0, 30)
    ax.set_ylim(0.8, 1.0)
    ax.legend(loc="lower right", fontsize=9,
              title="D$^2$NN,\nMaximum Testing Accuracy", title_fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Convergence plot saved: {out_path}")


def run_batch(batch_size, tag, existing_summary_path):
    base_dir = _SCRIPT_DIR

    # Load existing results for configs 1,2
    existing = load_existing_results(existing_summary_path)
    old_results = {r["label"]: r for r in existing["results"]}

    # Train hybrid configs 3,4
    new_results = []
    for entry in HYBRID_CONFIGS:
        result = train_single(entry, base_dir, batch_size, tag)
        new_results.append(result)

    # Combine: configs 1,2 from old + configs 3,4 from new
    all_results = [
        old_results["10 Layers Linear Real"],
        old_results["10 Layers Nonlinear Real"],
        new_results[0],  # 5L hybrid
        new_results[1],  # 10L hybrid
    ]

    # Save summary
    summary = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": "mnist",
        "batch_size": batch_size,
        "epochs": EPOCHS,
        "note": "hybrid configs re-run with layer_spacing_m=0.0 (corrected)",
        "configurations": [e["label"] for e in ALL_CONFIGS],
        "results": all_results,
    }
    summary_path = base_dir / f"fig4b_mnist_bs{batch_size}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved: {summary_path}")

    # Plot
    plot_path = base_dir / f"fig4b_mnist_bs{batch_size}_convergence.png"
    plot_convergence(all_results, plot_path, batch_size)

    # Table
    print(f"\n{'='*70}")
    print(f"  Figure 4(b) Results Summary (batch_size={batch_size})")
    print(f"{'='*70}")
    for res in all_results:
        print(f"  {res['label']:<42s} test_acc={res['test_acc']*100:.1f}%  (best_epoch={res['best_epoch']})")
    print(f"{'='*70}\n")


def main():
    base_dir = _SCRIPT_DIR

    # bs1024: reuse existing config 1,2 results
    print("\n" + "█"*70)
    print("  BATCH SIZE = 1024")
    print("█"*70)
    run_batch(1024, "bs1024_v2", base_dir / "fig4b_mnist_bs1024_summary.json")

    # bs10: reuse existing config 1,2 results
    # Wait for bs10 to finish first - check if summary exists
    bs10_summary = base_dir / "fig4b_mnist_bs10_summary.json"
    if bs10_summary.exists():
        print("\n" + "█"*70)
        print("  BATCH SIZE = 10")
        print("█"*70)
        run_batch(10, "bs10_v2", bs10_summary)
    else:
        print(f"\n[SKIP] bs10 summary not found yet: {bs10_summary}")
        print("  Will run bs10 hybrid configs with fresh config 1,2 training")
        # Fall back: just train hybrid configs, save partial results
        new_results = []
        for entry in HYBRID_CONFIGS:
            result = train_single(entry, base_dir, 10, "bs10_v2")
            new_results.append(result)
        # Save just hybrid results for now
        partial = {
            "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "note": "partial - hybrid configs only, layer_spacing_m=0.0",
            "batch_size": 10,
            "results": new_results,
        }
        with open(base_dir / "fig4b_mnist_bs10_hybrid_v2_partial.json", "w") as f:
            json.dump(partial, f, indent=2)
        print("Partial hybrid results saved.")


if __name__ == "__main__":
    main()
