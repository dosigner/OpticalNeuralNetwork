"""Fig 4a final: 4 models with per_sample_minmax intensity norm for nonlinear models.

Linear models are unaffected (no SBN), but retrained for consistency.
Output: runs/fig4a_final_per_sample_minmax/ and comparison plot.
"""

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR / "src"))

import torch
import matplotlib
matplotlib.use("Agg")
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

_CFG_DIR = _SCRIPT_DIR / "src" / "tao2019_fd2nn" / "config"

CONFIGS_TO_RUN = {
    "Linear Real": _CFG_DIR / "cls_mnist_linear_real_5l.yaml",
    "Nonlinear Real": _CFG_DIR / "cls_mnist_nonlinear_real_5l.yaml",
    "Linear Fourier": _CFG_DIR / "cls_mnist_linear_fourier_5l_f1mm.yaml",
    "Nonlinear Fourier": _CFG_DIR / "cls_mnist_nonlinear_fourier_5l_f1mm.yaml",
}

PAPER_TARGETS = {
    "Linear Real": 92.7,
    "Nonlinear Real": 95.4,
    "Linear Fourier": 93.5,
    "Nonlinear Fourier": 97.0,
}

def parse_config(config_path, name):
    cfg = load_config(config_path)
    cfg["training"]["epochs"] = 30
    cfg["training"]["batch_size"] = 10
    # Override intensity_norm to per_sample_minmax for nonlinear models
    nonlin = cfg.get("model", {}).get("nonlinearity", {})
    if nonlin.get("enabled", False):
        nonlin["intensity_norm"] = "per_sample_minmax"
    return cfg


def train_and_eval(name, config_path):
    print(f"\n{'='*70}\n  Training: {name} (per_sample_minmax)\n  Config:   {config_path}\n{'='*70}\n")

    cfg = parse_config(config_path, name)
    exp = cfg["experiment"]

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

    metrics = summarize_history(history)
    save_metrics(run_dir / "metrics.json", metrics)

    log.finish(run_dir=run_dir)
    max_acc = max(history["test_acc"]) * 100
    final_acc = history["test_acc"][-1] * 100
    print(f"\n  Result [{name}]: max_test_acc={max_acc:.1f}%  final_test_acc={final_acc:.1f}%\n")
    return history


def plot_results(results, out_path):
    colors = {
        "Linear Real": "#1f77b4",
        "Nonlinear Real": "#ff7f0e",
        "Linear Fourier": "#2ca02c",
        "Nonlinear Fourier": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, history in results.items():
        epochs = range(1, len(history["test_acc"]) + 1)
        accs = [acc * 100 for acc in history["test_acc"]]
        max_acc = max(accs)
        ax.plot(epochs, accs, label=f"{name} ({max_acc:.1f}%)",
                linewidth=2, color=colors[name])

    # Paper reference lines
    for name, target in PAPER_TARGETS.items():
        ax.axhline(y=target, color=colors[name], linestyle="--", alpha=0.3, linewidth=1)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Figure 4(a) Reproduction (MNIST 5-layer, bs=10, per_sample_minmax)", fontsize=12)
    ax.set_ylim(89, 98)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {out_path}")


def main():
    results = {}
    for name, path in CONFIGS_TO_RUN.items():
        history = train_and_eval(name, path)
        results[name] = history

    out_path = _SCRIPT_DIR / "runs" / "fig4a_final_per_sample_minmax"
    out_path.mkdir(parents=True, exist_ok=True)

    plot_path = out_path / "fig4a_per_sample_minmax_comparison.png"
    plot_results(results, plot_path)

    # Also save to top-level for easy access
    plot_results(results, _SCRIPT_DIR / "fig4a_per_sample_minmax_comparison.png")

    # Print summary table
    print("\n" + "=" * 80)
    print("  Figure 4(a) Results — per_sample_minmax")
    print("=" * 80)
    print(f"  {'Configuration':<22} {'Paper':>8} {'Ours(max)':>10} {'Ours(last)':>11} {'Gap':>6}")
    print("-" * 80)
    for name, history in results.items():
        max_acc = max(history["test_acc"]) * 100
        final_acc = history["test_acc"][-1] * 100
        target = PAPER_TARGETS[name]
        gap = max_acc - target
        print(f"  {name:<22} {target:>7.1f}% {max_acc:>9.1f}% {final_acc:>10.1f}% {gap:>+5.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
