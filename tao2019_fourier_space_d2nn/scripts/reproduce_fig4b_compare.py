"""Reproduce Fig.4b MNIST classification comparison (10L real + hybrid)."""

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


EXPERIMENT_CONFIGS: list[tuple[str, str]] = [
    ("10 Layers Linear Real", "cls_mnist_linear_real_10l.yaml"),
    ("10 Layers Nonlinear Real", "cls_mnist_nonlinear_real_10l.yaml"),
    ("5 Layers Nonlinear Fourier&Real", "cls_mnist_hybrid_5l.yaml"),
    ("10 Layers Nonlinear Fourier&Real", "cls_mnist_hybrid_10l.yaml"),
]

ORDERED_LABELS = [label for label, _ in EXPERIMENT_CONFIGS]
COLORS = {
    "10 Layers Linear Real": "#1f77b4",
    "10 Layers Nonlinear Real": "#d95319",
    "5 Layers Nonlinear Fourier&Real": "#edb120",
    "10 Layers Nonlinear Fourier&Real": "#7e2f8e",
}


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


def _load_primary_curve(run_dir: Path, *, label: str) -> tuple[list[float], str]:
    ckpt_path = run_dir / "checkpoints" / "final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"[{label}] missing checkpoint file: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    history = ckpt.get("history")
    if not isinstance(history, dict):
        raise ValueError(f"[{label}] checkpoint history missing or invalid in {ckpt_path}")
    for key in ("test_acc", "val_acc"):
        curve = history.get(key)
        if not isinstance(curve, list) or not curve:
            continue
        values = [float(v) for v in curve]
        if not all(0.0 <= v <= 1.0 for v in values):
            raise ValueError(f"[{label}] history['{key}'] contains out-of-range values in {ckpt_path}")
        return values, key
    raise ValueError(f"[{label}] neither history['test_acc'] nor history['val_acc'] is available in {ckpt_path}")


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
    curve_metric: str,
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
    entry = {
        "label": label,
        "config": cfg_ref,
        "run_dir": run_ref,
        "epochs_recorded": len(curve),
        "curve_metric": curve_metric,
        "max_curve_acc": max_acc,
        "final_curve_acc": curve[-1],
        "best_epoch": best_epoch,
        "test_acc": float(test_acc),
        "trained_this_run": bool(trained_this_run),
    }
    if curve_metric == "val_acc":
        entry["max_val_acc"] = max_acc
        entry["final_val_acc"] = curve[-1]
    elif curve_metric == "test_acc":
        entry["max_test_curve_acc"] = max_acc
        entry["final_test_curve_acc"] = curve[-1]
    return entry


def _parse_train_labels(raw: str) -> set[str]:
    if not raw.strip():
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MNIST Fig.4b comparisons and render summary graph")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", default="fig4b_mnist_epoch30_bs10.png")
    parser.add_argument("--summary-json", default="fig4b_mnist_epoch30_bs10_summary.json")
    parser.add_argument("--train-labels", default="", help="Comma-separated labels to train; others reuse latest run")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cfg_dir = project_root / "src" / "tao2019_fd2nn" / "config"
    runs_root = project_root / "runs"

    curves: dict[str, list[float]] = {}
    curve_metrics: dict[str, str] = {}
    max_acc: dict[str, float] = {}
    results: list[dict[str, Any]] = []
    train_labels = _parse_train_labels(str(args.train_labels))
    all_labels = {label for label, _ in EXPERIMENT_CONFIGS}
    unknown = sorted(train_labels - all_labels)
    if unknown:
        raise ValueError(f"unknown labels in --train-labels: {unknown}")
    overrides = {
        "training": {
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "num_workers": int(args.num_workers),
        }
    }

    for label, cfg_name in EXPERIMENT_CONFIGS:
        config_path = cfg_dir / cfg_name
        exp_name = _experiment_name(config_path)
        should_train = (not train_labels) or (label in train_labels)
        if should_train:
            run_with_overrides(config_path, overrides=overrides, task="classification")
        run_dir = _latest_run_dir(runs_root, exp_name)
        curve, curve_metric = _load_primary_curve(run_dir, label=label)
        test_acc = _compute_test_acc(run_dir)
        curves[label] = curve
        curve_metrics[label] = curve_metric
        max_acc[label] = max(curve)
        results.append(
            _summary_entry(
                label,
                config_path,
                run_dir,
                curve,
                curve_metric=curve_metric,
                test_acc=test_acc,
                trained_this_run=should_train,
                project_root=project_root,
            )
        )

    curve_metric_values = sorted(set(curve_metrics.values()))
    curve_metric = curve_metric_values[0] if len(curve_metric_values) == 1 else "mixed"
    if curve_metric == "test_acc":
        legend_title = "D$^2$NN,\nMaximum Testing Accuracy"
    elif curve_metric == "val_acc":
        legend_title = "D$^2$NN,\nMaximum Validation Accuracy"
    else:
        legend_title = "D$^2$NN,\nMaximum Accuracy"

    factory = FigureFactory(project_root)
    figure_path = factory.plot_comparison(
        curves,
        max_acc,
        ordered_labels=ORDERED_LABELS,
        colors=COLORS,
        name=args.output,
        legend_title=legend_title,
        ylim=(0.8, 1.0),
        title="MNIST Dataset Classification",
    )

    summary_path = Path(args.summary_json)
    if not summary_path.is_absolute():
        summary_path = project_root / summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_at_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "dataset": "mnist",
        "splits": {"validation": "5k", "test": "10k"},
        "metric": {"curve": curve_metric, "validation": "val_acc", "test": "test_acc"},
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "train_labels": sorted(train_labels) if train_labels else "all",
        "figure_path": str(figure_path),
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(str(figure_path))
    print(str(summary_path))


if __name__ == "__main__":
    main()
