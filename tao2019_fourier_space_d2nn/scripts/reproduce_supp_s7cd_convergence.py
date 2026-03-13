"""Supplementary S7(c)(d): paper-style convergence plots from current runs.

This script reuses the current `s7a_*` MNIST classification runs and redraws
Supplementary Fig. S7(c)(d) in the paper's visual style.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import torch

from tao2019_fd2nn.viz.figure_factory import FigureFactory

PANEL_C_SOURCES: list[dict[str, str]] = [
    {"label": "Linear Fourier", "experiment": "s7a_linear_fourier_1l"},
    {
        "label": "Nonlinear Fourier",
        "experiment": "s7a_nonlinear_fourier_single_sbn_1l",
        "note": "For 1-layer Fourier D2NN, single-SBN and multi-SBN runs are visually identical in the current run set.",
    },
]

PANEL_D_SOURCES: list[dict[str, str]] = [
    {"label": "Linear Fourier", "experiment": "s7a_linear_fourier_5l"},
    {"label": "Nonlinear Fourier, Single SBN", "experiment": "s7a_nonlinear_fourier_single_sbn_5l"},
    {"label": "Nonlinear Fourier, Multi-SBN", "experiment": "s7a_nonlinear_fourier_muti-sbn_5l"},
]

S7_COLORS = {
    "Linear Fourier": "#1f77b4",
    "Nonlinear Fourier": "#d95319",
    "Nonlinear Fourier, Single SBN": "#d95319",
    "Nonlinear Fourier, Multi-SBN": "#edb120",
}

PANEL_C_INSETS: list[dict[str, object]] = [
    {
        "bbox": [0.44, 0.32, 0.48, 0.22],
        "config_key": "linear_fourier",
        "num_layers": 1,
        "border_color": S7_COLORS["Linear Fourier"],
        "label": "Linear Fourier",
    },
    {
        "bbox": [0.44, 0.14, 0.48, 0.22],
        "config_key": "nonlinear_fourier",
        "num_layers": 1,
        "border_color": S7_COLORS["Nonlinear Fourier"],
        "label": "Nonlinear Fourier",
    },
]

PANEL_D_INSETS: list[dict[str, object]] = [
    {
        "bbox": [0.37, 0.57, 0.51, 0.20],
        "config_key": "linear_fourier",
        "num_layers": 5,
        "border_color": S7_COLORS["Linear Fourier"],
        "label": "Linear Fourier",
    },
    {
        "bbox": [0.37, 0.40, 0.51, 0.20],
        "config_key": "nonlinear_fourier_single_sbn",
        "num_layers": 5,
        "border_color": S7_COLORS["Nonlinear Fourier, Single SBN"],
        "label": "Nonlinear Fourier, Single SBN",
    },
    {
        "bbox": [0.37, 0.23, 0.51, 0.20],
        "config_key": "nonlinear_fourier_multi_sbn",
        "num_layers": 5,
        "border_color": S7_COLORS["Nonlinear Fourier, Multi-SBN"],
        "label": "Nonlinear Fourier, Multi-SBN",
    },
]


def _latest_run_dir(runs_root: Path, experiment_name: str) -> Path:
    exp_dir = runs_root / experiment_name
    if not exp_dir.exists():
        raise FileNotFoundError(f"missing run directory for experiment '{experiment_name}': {exp_dir}")
    run_dirs = sorted(p for p in exp_dir.iterdir() if p.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"no run subdirectories found for experiment '{experiment_name}' in {exp_dir}")
    return run_dirs[-1]


def _load_val_acc_curve(run_dir: Path, *, label: str) -> list[float]:
    ckpt_path = run_dir / "checkpoints" / "final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"[{label}] missing checkpoint file: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    history = ckpt.get("history")
    if not isinstance(history, dict):
        raise ValueError(f"[{label}] checkpoint history missing or invalid in {ckpt_path}")
    curve = history.get("val_acc")
    if not isinstance(curve, list) or not curve:
        raise ValueError(f"[{label}] history['val_acc'] missing or empty in {ckpt_path}")
    return [float(v) for v in curve]


def _panel_payload(
    runs_root: Path,
    sources: list[dict[str, str]],
    *,
    project_root: Path,
    max_epochs: int,
) -> tuple[dict[str, list[float]], dict[str, float], list[dict[str, Any]]]:
    curves: dict[str, list[float]] = {}
    max_acc: dict[str, float] = {}
    summary: list[dict[str, Any]] = []

    for source in sources:
        label = source["label"]
        experiment = source["experiment"]
        run_dir = _latest_run_dir(runs_root, experiment)
        curve = _load_val_acc_curve(run_dir, label=label)[:max_epochs]
        curves[label] = curve
        max_acc[label] = max(curve)
        summary.append(
            {
                "label": label,
                "experiment": experiment,
                "run_dir": str(run_dir.relative_to(project_root)),
                "epochs_recorded": len(curve),
                "max_val_acc": max(curve),
                "final_val_acc": curve[-1],
                "best_epoch": curve.index(max(curve)) + 1,
                "note": source.get("note"),
            }
        )

    return curves, max_acc, summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supplementary S7(c)(d): current-run paper-style convergence plots")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output-dir", default=None, help="Output directory for figures and summary")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    runs_root = project_root / "runs"
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    curves_c, max_acc_c, summary_c = _panel_payload(
        runs_root,
        PANEL_C_SOURCES,
        project_root=project_root,
        max_epochs=int(args.epochs),
    )
    curves_d, max_acc_d, summary_d = _panel_payload(
        runs_root,
        PANEL_D_SOURCES,
        project_root=project_root,
        max_epochs=int(args.epochs),
    )

    factory = FigureFactory(output_dir)
    path_c = factory.plot_s7cd_panel(
        curves_c,
        max_acc_c,
        ordered_labels=["Linear Fourier", "Nonlinear Fourier"],
        colors=S7_COLORS,
        title="Single Layer",
        panel_label="(c)",
        ylim=(0.4, 0.8),
        insets=PANEL_C_INSETS,
        name="supp_s7c_mnist_1l_convergence.png",
    )
    path_d = factory.plot_s7cd_panel(
        curves_d,
        max_acc_d,
        ordered_labels=[
            "Linear Fourier",
            "Nonlinear Fourier, Single SBN",
            "Nonlinear Fourier, Multi-SBN",
        ],
        colors=S7_COLORS,
        title="Five Layer",
        panel_label="(d)",
        ylim=(0.6, 1.0),
        insets=PANEL_D_INSETS,
        name="supp_s7d_mnist_5l_convergence.png",
    )
    path_cd = factory.plot_s7cd_composite(
        curves_c,
        max_acc_c,
        left_ordered_labels=["Linear Fourier", "Nonlinear Fourier"],
        left_colors=S7_COLORS,
        left_title="Single Layer",
        left_panel_label="(c)",
        left_ylim=(0.4, 0.8),
        left_insets=PANEL_C_INSETS,
        right_curves=curves_d,
        right_max_acc=max_acc_d,
        right_ordered_labels=[
            "Linear Fourier",
            "Nonlinear Fourier, Single SBN",
            "Nonlinear Fourier, Multi-SBN",
        ],
        right_colors=S7_COLORS,
        right_title="Five Layer",
        right_panel_label="(d)",
        right_ylim=(0.6, 1.0),
        right_insets=PANEL_D_INSETS,
        name="supp_s7cd_mnist_convergence.png",
    )

    summary = {
        "created_at_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "dataset": "mnist",
        "metric": "val_acc",
        "epochs": int(args.epochs),
        "data_source": "current existing runs",
        "panel_c": summary_c,
        "panel_d": summary_d,
        "figure_c": str(path_c),
        "figure_d": str(path_d),
        "figure_cd": str(path_cd),
    }
    summary_path = output_dir / "supp_s7cd_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Panel (c): {path_c}")
    print(f"Panel (d): {path_d}")
    print(f"Composite : {path_cd}")
    print(f"Summary   : {summary_path}")


if __name__ == "__main__":
    main()
