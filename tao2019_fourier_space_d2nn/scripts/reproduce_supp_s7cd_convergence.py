"""Supplementary S7(c)(d): convergence plots for 1-layer and 5-layer D2NN.

Correct 4 configurations from the paper S7:
  1. Nonlinear Fourier, Multi SBN  (fd2nn, SBN per_layer)
  2. Nonlinear Fourier, Single SBN (fd2nn, SBN rear)
  3. Linear Fourier                (fd2nn, no SBN)
  4. Nonlinear Real, Multi SBN     (real_d2nn, SBN per_layer)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from _common import run_with_overrides
from tao2019_fd2nn.cli.common import build_detector_masks, build_model, choose_device
from tao2019_fd2nn.data.mnist import MnistAmplitudeDataset
from tao2019_fd2nn.models.detectors import integrate_detector_energies
from tao2019_fd2nn.training.metrics_classification import accuracy
from tao2019_fd2nn.utils.math import intensity
from tao2019_fd2nn.viz.figure_factory import FigureFactory


# ── S7 configurations ──────────────────────────────────────────────
S7_LABELS = [
    "Nonlinear Fourier, Multi SBN",
    "Nonlinear Fourier, Single SBN",
    "Linear Fourier",
    "Nonlinear Real, Multi SBN",
]

S7_COLORS = {
    "Nonlinear Fourier, Multi SBN": "#1f77b4",
    "Nonlinear Fourier, Single SBN": "#d95319",
    "Linear Fourier": "#edb120",
    "Nonlinear Real, Multi SBN": "#7e2f8e",
}

ONE_LAYER_CONFIGS: list[tuple[str, str]] = [
    ("Nonlinear Fourier, Multi SBN", "cls_mnist_nonlinear_fourier_multi_sbn_1l.yaml"),
    ("Nonlinear Fourier, Single SBN", "cls_mnist_nonlinear_fourier_1l_f1mm.yaml"),
    ("Linear Fourier", "cls_mnist_linear_fourier_1l_f1mm.yaml"),
    ("Nonlinear Real, Multi SBN", "cls_mnist_nonlinear_real_1l.yaml"),
]

FIVE_LAYER_CONFIGS: list[tuple[str, str]] = [
    ("Nonlinear Fourier, Multi SBN", "cls_mnist_nonlinear_fourier_multi_sbn_5l.yaml"),
    ("Nonlinear Fourier, Single SBN", "cls_mnist_nonlinear_fourier_5l_f4mm.yaml"),
    ("Linear Fourier", "cls_mnist_linear_fourier_5l_f1mm.yaml"),
    ("Nonlinear Real, Multi SBN", "cls_mnist_nonlinear_real_5l.yaml"),
]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"YAML root must be mapping: {path}")
    return loaded


def _experiment_name(config_path: Path) -> str:
    cfg = _load_yaml(config_path)
    exp = cfg.get("experiment", {})
    if not isinstance(exp, dict) or "name" not in exp:
        raise ValueError(f"missing experiment.name in config: {config_path}")
    return str(exp["name"])


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
    values = [float(v) for v in curve]
    if not all(0.0 <= v <= 1.0 for v in values):
        raise ValueError(f"[{label}] history['val_acc'] contains out-of-range values in {ckpt_path}")
    return values


def _pad_to_size(cfg: dict[str, Any]) -> int:
    prep = cfg["data"]["preprocess"]
    if "pad_to" in prep:
        return int(prep["pad_to"][0])
    return int(cfg["optics"]["grid"]["nx"])


def _mnist_object_size(cfg: dict[str, Any]) -> int:
    prep = cfg["data"]["preprocess"]
    if "resize_to" in prep:
        return int(prep["resize_to"][0])
    return int(prep.get("upsample_factor", 3)) * 28


def _mnist_test_loader(cfg: dict[str, Any], *, batch_size: int = 512) -> DataLoader:
    data_cfg = cfg["data"]
    ds = MnistAmplitudeDataset(
        root=str(data_cfg.get("root", "data/mnist")),
        train=False,
        download=True,
        N=_pad_to_size(cfg),
        object_size=_mnist_object_size(cfg),
        binarize=False,
    )
    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=max(0, int(cfg["training"].get("num_workers", 0))),
        pin_memory=bool(cfg["training"].get("pin_memory", True)),
    )


def _compute_test_acc(run_dir: Path) -> float:
    cfg_path = run_dir / "config_resolved.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing resolved config: {cfg_path}")
    cfg = _load_yaml(cfg_path)
    device = choose_device(cfg["experiment"])
    model = build_model(cfg).to(device)
    ckpt_path = run_dir / "checkpoints" / "final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing checkpoint file: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    detector_masks = build_detector_masks(cfg, device=device)
    loader = _mnist_test_loader(cfg)

    all_e: list[torch.Tensor] = []
    all_y: list[torch.Tensor] = []
    with torch.no_grad():
        for fields, labels in loader:
            fields = fields.to(device)
            labels = labels.to(device)
            energies = integrate_detector_energies(intensity(model(fields)), detector_masks)
            all_e.append(energies.cpu())
            all_y.append(labels.cpu())
    if not all_e:
        return 0.0
    return float(accuracy(torch.cat(all_e, dim=0), torch.cat(all_y, dim=0)))


def _summary_entry(
    label: str,
    config_path: Path,
    run_dir: Path,
    curve: list[float],
    *,
    test_acc: float,
    trained_this_run: bool,
    project_root: Path,
) -> dict[str, Any]:
    max_acc = max(curve)
    best_epoch = curve.index(max_acc) + 1
    try:
        run_ref = str(run_dir.relative_to(project_root))
    except ValueError:
        run_ref = str(run_dir)
    try:
        cfg_ref = str(config_path.relative_to(project_root))
    except ValueError:
        cfg_ref = str(config_path)
    return {
        "label": label,
        "config": cfg_ref,
        "run_dir": run_ref,
        "epochs_recorded": len(curve),
        "max_val_acc": max_acc,
        "final_val_acc": curve[-1],
        "best_epoch": best_epoch,
        "test_acc": float(test_acc),
        "trained_this_run": bool(trained_this_run),
    }


def _process_panel(
    configs: list[tuple[str, str]],
    cfg_dir: Path,
    runs_root: Path,
    *,
    overrides: dict[str, Any],
    train: bool,
    max_epochs: int,
    project_root: Path,
) -> tuple[dict[str, list[float]], dict[str, float], list[dict[str, Any]]]:
    curves: dict[str, list[float]] = {}
    max_acc: dict[str, float] = {}
    results: list[dict[str, Any]] = []

    for label, cfg_name in configs:
        config_path = cfg_dir / cfg_name
        exp_name = _experiment_name(config_path)
        trained = False
        if train:
            run_with_overrides(config_path, overrides=overrides, task="classification")
            trained = True
        run_dir = _latest_run_dir(runs_root, exp_name)
        curve = _load_val_acc_curve(run_dir, label=label)[:max_epochs]
        test_acc = _compute_test_acc(run_dir)
        curves[label] = curve
        max_acc[label] = max(curve)
        results.append(
            _summary_entry(
                label,
                config_path,
                run_dir,
                curve,
                test_acc=test_acc,
                trained_this_run=trained,
                project_root=project_root,
            )
        )

    return curves, max_acc, results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supplementary S7(c)(d): convergence plots")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--skip-train-1l", action="store_true", help="Reuse existing 1-layer runs")
    parser.add_argument("--skip-train-5l", action="store_true", default=True, help="Reuse existing 5-layer runs (default)")
    parser.add_argument("--train-5l", action="store_true", help="Force retrain 5-layer models")
    parser.add_argument("--output-dir", default=None, help="Output directory for figures")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cfg_dir = project_root / "src" / "tao2019_fd2nn" / "config"
    runs_root = project_root / "runs"
    output_dir = Path(args.output_dir) if args.output_dir else project_root
    max_epochs = int(args.epochs)

    overrides = {"training": {"batch_size": int(args.batch_size), "epochs": max_epochs}}

    skip_5l = args.skip_train_5l and not args.train_5l

    # ── Panel (c): 1-layer ──
    print("=== Panel (c): 1-Layer Convergence ===")
    curves_1l, max_acc_1l, results_1l = _process_panel(
        ONE_LAYER_CONFIGS,
        cfg_dir,
        runs_root,
        overrides=overrides,
        train=(not args.skip_train_1l),
        max_epochs=max_epochs,
        project_root=project_root,
    )

    # ── Panel (d): 5-layer ──
    print("=== Panel (d): 5-Layer Convergence ===")
    curves_5l, max_acc_5l, results_5l = _process_panel(
        FIVE_LAYER_CONFIGS,
        cfg_dir,
        runs_root,
        overrides=overrides,
        train=(not skip_5l),
        max_epochs=max_epochs,
        project_root=project_root,
    )

    # ── Plot both panels ──
    factory = FigureFactory(output_dir)

    # Panel (c): dynamic ylim for 1-layer (lower accuracy)
    min_val_1l = min(min(c) for c in curves_1l.values())
    ylim_low = max(0.0, round(min_val_1l - 0.1, 1))
    ylim_1l = (ylim_low, 1.0)

    path_c = factory.plot_comparison(
        curves_1l,
        max_acc_1l,
        ordered_labels=S7_LABELS,
        colors=S7_COLORS,
        name="supp_s7c_mnist_1l_convergence.png",
        legend_title="Single Layer,\nMaximum Validation Accuracy",
        ylim=ylim_1l,
    )
    print(f"Panel (c) saved: {path_c}")

    path_d = factory.plot_comparison(
        curves_5l,
        max_acc_5l,
        ordered_labels=S7_LABELS,
        colors=S7_COLORS,
        name="supp_s7d_mnist_5l_convergence.png",
        legend_title="Five Layers,\nMaximum Validation Accuracy",
    )
    print(f"Panel (d) saved: {path_d}")

    # ── Save summary JSON ──
    summary_path = output_dir / "supp_s7cd_summary.json"
    summary = {
        "created_at_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "dataset": "mnist",
        "splits": {"validation": "5k", "test": "10k"},
        "metric": {"val": "val_acc", "test": "test_acc"},
        "batch_size": int(args.batch_size),
        "epochs": max_epochs,
        "configurations": S7_LABELS,
        "panel_c_1layer": results_1l,
        "panel_d_5layer": results_5l,
        "figure_c": str(path_c),
        "figure_d": str(path_d),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()