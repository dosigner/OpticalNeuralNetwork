"""Plot Figure 4(a): 5-layer MNIST classification convergence curves."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent

CONFIGS = [
    {
        "label": "Linear Real",
        "run_dir": "runs/fig4a_cls_mnist_linear_real_5l/260227_094918",
        "color": "#1f77b4",
    },
    {
        "label": "Nonlinear Real",
        "run_dir": "runs/fig4a_cls_mnist_nonlinear_real_5l/260227_103345",
        "color": "#ff7f0e",
    },
    {
        "label": "Linear Fourier",
        "run_dir": "runs/fig4a_cls_mnist_linear_fourier_5l_f1mm/260227_113511",
        "color": "#2ca02c",
    },
    {
        "label": "Nonlinear Fourier",
        "run_dir": "runs/fig4a_cls_mnist_nonlinear_fourier_5l_f4mm/260227_122858",
        "color": "#d62728",
    },
]


def load_history(run_dir: str) -> dict:
    ckpt_path = _SCRIPT_DIR / run_dir / "checkpoints" / "final.pt"
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    return ckpt.get("history", {})


def main():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("MNIST Dataset Classification", fontsize=14)
    ax.set_xlabel("Epoch Number", fontsize=12)
    ax.set_ylabel("Classification Accuracy", fontsize=12)

    for entry in CONFIGS:
        history = load_history(entry["run_dir"])
        accs = history.get("test_acc", [])
        if not accs:
            accs = history.get("val_acc", [])

        epochs = [0] + list(range(1, len(accs) + 1))
        accs_plot = [0.0] + [float(v) for v in accs]
        max_acc = max(accs_plot)

        ax.plot(
            epochs, accs_plot,
            color=entry["color"],
            linewidth=2,
            label=f"{entry['label']} ({max_acc*100:.1f}%)",
        )
        print(f"{entry['label']:<20s}  max_test_acc={max_acc*100:.1f}%  final={accs_plot[-1]*100:.1f}%")

    ax.set_xlim(0, 30)
    ax.set_ylim(0.8, 1.0)
    ax.legend(
        loc="lower right", fontsize=9,
        title="D$^2$NN,\nMaximum Testing Accuracy",
        title_fontsize=9,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = _SCRIPT_DIR / "fig4a_mnist_bs10_convergence.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
