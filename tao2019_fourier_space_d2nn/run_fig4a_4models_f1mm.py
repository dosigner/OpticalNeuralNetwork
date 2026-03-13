"""Rerun 4 models for Fig 4a (Linear Real, Nonlinear Real, Linear Fourier, Nonlinear Fourier f=1mm) with batch_size=10 and epochs=30."""

import sys
from pathlib import Path

# Insert the src directory into sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR / "src"))

import torch
import matplotlib.pyplot as plt

from tao2019_fd2nn.cli.common import (
    build_dataloaders,
    build_detector_masks,
    build_model,
    build_test_loader,
    choose_device,
    create_run_dir,
    load_config,
)
from tao2019_fd2nn.training.callbacks import save_checkpoint, save_metrics, save_resolved_config
from tao2019_fd2nn.training.trainer import best_epoch_index, summarize_history, train_classifier
from tao2019_fd2nn.utils.live_log import LiveLogger
from tao2019_fd2nn.utils.seed import set_global_seed

def parse_config(config_path):
    print(config_path)
    cfg = load_config(config_path)
    cfg["training"]["epochs"] = 30
    cfg["training"]["batch_size"] = 10
    return cfg

CONFIGS_TO_RUN = {
    "Linear Real": _SCRIPT_DIR / "src" / "tao2019_fd2nn" / "config" / "cls_mnist_linear_real_5l.yaml",
    "Nonlinear Real": _SCRIPT_DIR / "src" / "tao2019_fd2nn" / "config" / "cls_mnist_nonlinear_real_5l.yaml",
    "Linear Fourier": _SCRIPT_DIR / "src" / "tao2019_fd2nn" / "config" / "cls_mnist_linear_fourier_5l_f1mm.yaml",
    "Nonlinear Fourier": _SCRIPT_DIR / "src" / "tao2019_fd2nn" / "config" / "cls_mnist_nonlinear_fourier_5l_f1mm.yaml",
}

def train_and_eval(name, config_path):
    print(f"\n{'='*70}\n  Training: {name}\n  Config:   {config_path}\n{'='*70}\n")
    
    cfg = parse_config(config_path)
    exp = cfg["experiment"]
    if name == "Nonlinear Fourier":
        exp["name"] = "fig4a_cls_mnist_nonlinear_fourier_5l_f1mm"
    
    set_global_seed(int(exp.get("seed", 42)), deterministic=bool(exp.get("deterministic", True)))
    
    device = choose_device(exp)
    run_dir = create_run_dir(cfg, cwd=_SCRIPT_DIR)
    save_resolved_config(run_dir / "config_resolved.yaml", cfg)
    
    log = LiveLogger(
        run_dir=run_dir,
        task="classification",
        total_epochs=int(cfg["training"]["epochs"]),
        log_interval_steps=20,
        use_color=True,
        show_cuda_memory=True,
    )
    
    log.start(experiment_name=str(exp["name"]), device=str(device))
    
    model = build_model(cfg).to(device)
    train_loader, val_loader = build_dataloaders(cfg)
    test_loader = build_test_loader(cfg)
    detector_masks = build_detector_masks(cfg, device=device)
    
    history = train_classifier(
        model, train_loader, val_loader, detector_masks,
        device=device,
        lr=float(cfg["training"].get("lr", 0.01)),
        epochs=int(cfg["training"]["epochs"]),
        loss_mode=str(cfg["training"].get("loss", "mse_onehot")),
        leakage_weight=float(cfg["training"].get("leakage_weight", 0.0)),
        temperature=float(cfg["model"].get("logits", {}).get("temperature", 1.0)),
        test_loader=test_loader,
        step_callback=log.on_step,
        epoch_callback=log.on_epoch_end,
    )
    
    save_checkpoint(
        run_dir / "checkpoints" / "final.pt",
        model=model,
        optimizer=None,
        epoch=int(cfg["training"]["epochs"]),
        extra={"history": history},
    )
    
    best_idx = best_epoch_index(history, metric_name="val_acc", maximize=True)
    metrics = summarize_history(history)
    save_metrics(run_dir / "metrics.json", metrics)
    
    log.finish(run_dir=run_dir)
    print(f"\n  Result [{name}]: max_test_acc={max(history['test_acc']):.4f}  final_test_acc={history['test_acc'][-1]:.4f}\n")
    return history

def main():
    results = {}
    for name, path in CONFIGS_TO_RUN.items():
        history = train_and_eval(name, path)
        results[name] = history
    
    # Plot results
    plt.figure(figsize=(8, 6))
    for name, history in results.items():
        plt.plot(range(1, len(history["test_acc"]) + 1), 
                 [acc * 100 for acc in history["test_acc"]], 
                 label=name, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Figure 4(a) Reproduction (MNIST 5-layer, bs=10)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_path = _SCRIPT_DIR / "fig4a_mnist_bs10_reproduction_f1mm.png"
    plt.savefig(plot_path)
    print(f"Plot saved: {plot_path}")
    
    print("\n" + "="*70)
    print("  Figure 4(a) Results")
    print("="*70)
    for name, history in results.items():
        max_acc = max(history["test_acc"]) * 100
        final_acc = history["test_acc"][-1] * 100
        print(f"  {name:<22} max={max_acc:.1f}%  final={final_acc:.1f}%")
    print("="*70)

if __name__ == "__main__":
    main()
