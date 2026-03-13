"""Supplementary S7(a)(b): Performance vs Layer Number & SBN Position.

(a) Accuracy vs layer count (1-5) for 3 Fourier-space configurations:
    - Linear Fourier
    - Nonlinear Fourier, Single SBN (rear)
    - Nonlinear Fourier, Multi-SBN (per_layer)

(b) Convergence curves for 4 SBN-position configurations (all 10-layer, per-layer SBN):
    - Nonlinear Fourier, SBN Front (per_layer_front)
    - Nonlinear Fourier, SBN Rear (per_layer)
    - Nonlinear Fourier & Real, SBN Front (per_layer_front)
    - Nonlinear Fourier & Real, SBN Rear (per_layer)
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

# ── S7(a) Configuration ──────────────────────────────────────────

S7A_LAYER_COUNTS = [1, 2, 3, 4, 5]

# (label, base_config_filename, extra_overrides)
S7A_VARIANTS: list[tuple[str, str, dict[str, Any]]] = [
    (
        "Linear Fourier",
        "cls_mnist_linear_fourier_1l_f1mm.yaml",
        {},
    ),
    (
        "Nonlinear Fourier, Single SBN",
        "cls_mnist_nonlinear_fourier_1l_f1mm.yaml",
        {"model": {"nonlinearity": {"intensity_norm": "per_sample_minmax"}}},
    ),
    (
        "Nonlinear Fourier, Muti-SBN",
        "cls_mnist_nonlinear_fourier_multi_sbn_1l.yaml",
        {"model": {"nonlinearity": {"intensity_norm": "per_sample_minmax"}}},
    ),
]

S7A_COLORS = {
    "Linear Fourier": "#1f77b4",
    "Nonlinear Fourier, Single SBN": "#d95319",
    "Nonlinear Fourier, Muti-SBN": "#edb120",
}
S7A_MARKERS = {
    "Linear Fourier": "x",
    "Nonlinear Fourier, Single SBN": "x",
    "Nonlinear Fourier, Muti-SBN": "x",
}
S7A_SCHEMATIC_KEYS = [
    "linear_fourier",
    "nonlinear_fourier_single_sbn",
    "nonlinear_fourier_multi_sbn",
]
S7A_SCHEMATIC_NUM_LAYERS = [10, 10, 10]

# ── S7(b) Configuration ──────────────────────────────────────────

# (label, base_config_filename, extra_overrides)
S7B_CONFIGS: list[tuple[str, str, dict[str, Any]]] = [
    (
        "Nonlinear Fourier, SBN Front",
        "cls_mnist_nonlinear_fourier_1l_f1mm.yaml",
        {
            "model": {
                "num_layers": 10,
                "nonlinearity": {
                    "position": "per_layer_front",
                    "intensity_norm": "per_sample_minmax",
                },
            },
        },
    ),
    (
        "Nonlinear Fourier, SBN Rear",
        "cls_mnist_nonlinear_fourier_1l_f1mm.yaml",
        {
            "model": {
                "num_layers": 10,
                "nonlinearity": {
                    "position": "per_layer",
                    "intensity_norm": "per_sample_minmax",
                },
            },
        },
    ),
    (
        "Nonlinear Fourier & Real, SBN Front",
        "cls_mnist_hybrid_10l.yaml",
        {
            "model": {
                "nonlinearity": {
                    "position": "per_layer_front",
                    "intensity_norm": "per_sample_minmax",
                },
            },
        },
    ),
    (
        "Nonlinear Fourier & Real, SBN Rear",
        "cls_mnist_hybrid_10l.yaml",
        {
            "model": {
                "nonlinearity": {
                    "position": "per_layer",
                    "intensity_norm": "per_sample_minmax",
                },
            },
        },
    ),
]

S7B_COLORS = {
    "Nonlinear Fourier, SBN Front": "#1f77b4",
    "Nonlinear Fourier, SBN Rear": "#d95319",
    "Nonlinear Fourier & Real, SBN Front": "#edb120",
    "Nonlinear Fourier & Real, SBN Rear": "#7e2f8e",
}
S7B_LABELS = [
    "Nonlinear Fourier, SBN Front",
    "Nonlinear Fourier, SBN Rear",
    "Nonlinear Fourier & Real, SBN Front",
    "Nonlinear Fourier & Real, SBN Rear",
]
S7B_SCHEMATIC_KEYS = [
    "nonlinear_fourier_sbn_front",
    "nonlinear_fourier_sbn_rear",
    "hybrid_sbn_front",
    "hybrid_sbn_rear",
]
S7B_SCHEMATIC_NUM_LAYERS = [10, 10, 10, 10]


# ── Helpers ───────────────────────────────────────────────────────


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
    return [float(v) for v in curve]


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


def _make_exp_name(prefix: str, label: str, num_layers: int | None = None) -> str:
    """Create a unique experiment name from label."""
    slug = label.lower().replace(" ", "_").replace(",", "").replace("&", "and")
    if num_layers is not None:
        return f"{prefix}_{slug}_{num_layers}l"
    return f"{prefix}_{slug}"


# ── S7(a) Training & Collection ───────────────────────────────────


def run_s7a(
    cfg_dir: Path,
    runs_root: Path,
    *,
    train: bool,
    batch_size: int,
    epochs: int,
) -> dict[str, dict[int, float]]:
    """Train S7(a) models and return {label: {layer_count: test_accuracy}}."""
    results: dict[str, dict[int, float]] = {}

    for label, cfg_name, extra_overrides in S7A_VARIANTS:
        base_config = cfg_dir / cfg_name
        results[label] = {}

        for n_layers in S7A_LAYER_COUNTS:
            exp_name = _make_exp_name("s7a", label, n_layers)
            overrides: dict[str, Any] = {
                "experiment": {"name": exp_name},
                "model": {"num_layers": n_layers},
                "training": {"batch_size": batch_size, "epochs": epochs},
            }
            # Deep merge extra overrides
            for top_key, top_val in extra_overrides.items():
                if top_key not in overrides:
                    overrides[top_key] = top_val
                elif isinstance(top_val, dict) and isinstance(overrides[top_key], dict):
                    for k, v in top_val.items():
                        if k not in overrides[top_key]:
                            overrides[top_key][k] = v
                        elif isinstance(v, dict) and isinstance(overrides[top_key][k], dict):
                            overrides[top_key][k].update(v)
                        else:
                            overrides[top_key][k] = v

            print(f"\n{'='*60}")
            print(f"S7(a) | {label} | {n_layers} layers")
            print(f"{'='*60}")

            if train:
                run_with_overrides(base_config, overrides=overrides, task="classification")

            try:
                run_dir = _latest_run_dir(runs_root, exp_name)
                test_acc = _compute_test_acc(run_dir)
                results[label][n_layers] = test_acc
                print(f"  -> test_acc = {test_acc * 100:.1f}%")
            except FileNotFoundError as e:
                print(f"  -> SKIP (not found): {e}")

    return results


# ── S7(b) Training & Collection ───────────────────────────────────


def run_s7b(
    cfg_dir: Path,
    runs_root: Path,
    *,
    train: bool,
    batch_size: int,
    epochs: int,
) -> tuple[dict[str, list[float]], dict[str, float]]:
    """Train S7(b) models and return (curves, max_acc)."""
    curves: dict[str, list[float]] = {}
    max_acc: dict[str, float] = {}

    for label, cfg_name, extra_overrides in S7B_CONFIGS:
        base_config = cfg_dir / cfg_name
        exp_name = _make_exp_name("s7b", label)
        overrides: dict[str, Any] = {
            "experiment": {"name": exp_name},
            "training": {"batch_size": batch_size, "epochs": epochs},
        }
        # Deep merge extra overrides
        for top_key, top_val in extra_overrides.items():
            if top_key not in overrides:
                overrides[top_key] = top_val
            elif isinstance(top_val, dict) and isinstance(overrides[top_key], dict):
                for k, v in top_val.items():
                    if k not in overrides[top_key]:
                        overrides[top_key][k] = v
                    elif isinstance(v, dict) and isinstance(overrides[top_key][k], dict):
                        overrides[top_key][k].update(v)
                    else:
                        overrides[top_key][k] = v

        print(f"\n{'='*60}")
        print(f"S7(b) | {label}")
        print(f"{'='*60}")

        if train:
            run_with_overrides(base_config, overrides=overrides, task="classification")

        try:
            run_dir = _latest_run_dir(runs_root, exp_name)
            curve = _load_val_acc_curve(run_dir, label=label)[:epochs]
            test_acc = _compute_test_acc(run_dir)
            curves[label] = curve
            max_acc[label] = test_acc
            print(f"  -> test_acc = {test_acc * 100:.1f}%, max_val_acc = {max(curve) * 100:.1f}%")
        except FileNotFoundError as e:
            print(f"  -> SKIP (not found): {e}")

    return curves, max_acc


# ── Main ──────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supplementary S7(a)(b): layer count & SBN position")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--skip-train-s7a", action="store_true", help="Reuse existing S7(a) runs")
    parser.add_argument("--skip-train-s7b", action="store_true", help="Reuse existing S7(b) runs")
    parser.add_argument("--output-dir", default=None, help="Output directory for figures")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cfg_dir = project_root / "src" / "tao2019_fd2nn" / "config"
    runs_root = project_root / "runs"
    output_dir = Path(args.output_dir) if args.output_dir else project_root
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── S7(a) ──
    print("\n" + "=" * 70)
    print("  S7(a): Performance vs Layer Number")
    print("=" * 70)
    s7a_data = run_s7a(
        cfg_dir, runs_root,
        train=(not args.skip_train_s7a),
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    # ── S7(b) ──
    print("\n" + "=" * 70)
    print("  S7(b): SBN Position")
    print("=" * 70)
    s7b_curves, s7b_max_acc = run_s7b(
        cfg_dir, runs_root,
        train=(not args.skip_train_s7b),
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    # ── Plot S7(a) ──
    factory = FigureFactory(output_dir)

    s7a_labels = [label for label, _, _ in S7A_VARIANTS]
    if all(label in s7a_data and len(s7a_data[label]) == len(S7A_LAYER_COUNTS) for label in s7a_labels):
        path_a = factory.plot_accuracy_vs_layers(
            s7a_data,
            ordered_labels=s7a_labels,
            colors=S7A_COLORS,
            markers=S7A_MARKERS,
            schematic_keys=S7A_SCHEMATIC_KEYS,
            schematic_num_layers=S7A_SCHEMATIC_NUM_LAYERS,
            name="supp_s7a_accuracy_vs_layers.png",
        )
        print(f"\nS7(a) figure saved: {path_a}")
    else:
        print("\nS7(a) figure SKIPPED (incomplete data)")
        path_a = None

    # ── Plot S7(b) ──
    if all(label in s7b_curves for label in S7B_LABELS):
        path_b = factory.plot_s7b_convergence_with_schematics(
            s7b_curves,
            s7b_max_acc,
            ordered_labels=S7B_LABELS,
            colors=S7B_COLORS,
            schematic_keys=S7B_SCHEMATIC_KEYS,
            schematic_num_layers=S7B_SCHEMATIC_NUM_LAYERS,
            name="supp_s7b_sbn_position.png",
        )
        print(f"S7(b) figure saved: {path_b}")
    else:
        print("S7(b) figure SKIPPED (incomplete data)")
        path_b = None

    # ── Save summary ──
    summary = {
        "created_at_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "dataset": "mnist",
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "s7a_accuracy_vs_layers": {
            label: {str(k): v for k, v in acc.items()}
            for label, acc in s7a_data.items()
        },
        "s7b_convergence": {
            label: {
                "max_val_acc": max(curve) if curve else None,
                "final_val_acc": curve[-1] if curve else None,
                "test_acc": s7b_max_acc.get(label),
            }
            for label, curve in s7b_curves.items()
        },
        "figure_a": str(path_a) if path_a else None,
        "figure_b": str(path_b) if path_b else None,
    }
    summary_path = output_dir / "supp_s7ab_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
