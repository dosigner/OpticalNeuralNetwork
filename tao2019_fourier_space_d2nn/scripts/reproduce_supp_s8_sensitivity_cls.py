"""Supplementary Figure S8: classification sensitivity to fabrication blur and alignment shift."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
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


S8_EXPERIMENTS: list[dict[str, Any]] = [
    {
        "label": "Nonlinear Fourier 1L",
        "family": "nonlinear_fourier",
        "num_layers": 1,
        "config": "cls_mnist_nonlinear_fourier_1l_f4mm.yaml",
    },
    {
        "label": "Nonlinear Fourier 2L",
        "family": "nonlinear_fourier",
        "num_layers": 2,
        "config": "cls_mnist_nonlinear_fourier_2l_f4mm.yaml",
    },
    {
        "label": "Nonlinear Fourier 5L",
        "family": "nonlinear_fourier",
        "num_layers": 5,
        "config": "cls_mnist_nonlinear_fourier_5l_f4mm.yaml",
    },
    {
        "label": "Nonlinear Real 1L",
        "family": "nonlinear_real",
        "num_layers": 1,
        "config": "cls_mnist_nonlinear_real_1l.yaml",
    },
    {
        "label": "Nonlinear Real 2L",
        "family": "nonlinear_real",
        "num_layers": 2,
        "config": "cls_mnist_nonlinear_real_2l.yaml",
    },
    {
        "label": "Nonlinear Real 5L",
        "family": "nonlinear_real",
        "num_layers": 5,
        "config": "cls_mnist_nonlinear_real_5l.yaml",
    },
    {
        "label": "Linear Real 1L",
        "family": "linear_real",
        "num_layers": 1,
        "config": "cls_mnist_linear_real_1l_100um.yaml",
    },
    {
        "label": "Linear Real 2L",
        "family": "linear_real",
        "num_layers": 2,
        "config": "cls_mnist_linear_real_2l_100um.yaml",
    },
    {
        "label": "Linear Real 5L",
        "family": "linear_real",
        "num_layers": 5,
        "config": "cls_mnist_linear_real_5l_100um.yaml",
    },
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


def _has_checkpoint(run_dir: Path, checkpoint_name: str) -> bool:
    return (run_dir / "checkpoints" / checkpoint_name).exists()


def _latest_completed_run_dir(runs_root: Path, experiment_name: str, *, checkpoint_name: str) -> Path:
    exp_dir = runs_root / experiment_name
    if not exp_dir.exists():
        raise FileNotFoundError(f"missing run directory for experiment '{experiment_name}': {exp_dir}")
    run_dirs = sorted(p for p in exp_dir.iterdir() if p.is_dir())
    completed = [run_dir for run_dir in run_dirs if _has_checkpoint(run_dir, checkpoint_name)]
    if not completed:
        raise FileNotFoundError(
            f"no completed run with checkpoint '{checkpoint_name}' found for experiment '{experiment_name}' in {exp_dir}"
        )
    return completed[-1]


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


def _mnist_test_loader(cfg: dict[str, Any], *, batch_size: int) -> DataLoader:
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


def _evaluate_accuracy(
    run_dir: Path,
    *,
    eval_batch_size: int,
    num_workers: int,
    checkpoint_name: str,
    fabrication_sigma_px: float = 0.0,
    alignment_shift_um: float = 0.0,
) -> float:
    cfg_path = run_dir / "config_resolved.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing resolved config: {cfg_path}")
    cfg = _load_yaml(cfg_path)
    cfg.setdefault("model", {})
    cfg.setdefault("optics", {})
    cfg.setdefault("eval", {})
    cfg.setdefault("training", {})
    cfg["model"]["fabrication_blur_sigma_px"] = float(fabrication_sigma_px)
    cfg["model"]["fabrication_blur_kernel_size"] = 3
    cfg["optics"]["alignment_shift_um"] = float(alignment_shift_um)
    cfg["eval"]["perturbation_mode"] = "test_only"
    cfg["training"]["num_workers"] = int(num_workers)
    cfg["training"]["pin_memory"] = bool(int(num_workers) > 0)
    cfg["training"]["persistent_workers"] = bool(int(num_workers) > 0)
    if int(num_workers) <= 0:
        cfg["training"]["prefetch_factor"] = None

    device = choose_device(cfg["experiment"])
    model = build_model(cfg).to(device)
    ckpt_path = run_dir / "checkpoints" / checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing checkpoint file: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    detector_masks = build_detector_masks(cfg, device=device)
    loader = _mnist_test_loader(cfg, batch_size=eval_batch_size)

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


def _parse_float_list(raw: str) -> list[float]:
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    if not parts:
        raise ValueError("expected at least one numeric value")
    return [float(item) for item in parts]


def _resolve_output_path(project_root: Path, raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = project_root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _training_overrides(*, train_batch_size: int, epochs: int, num_workers: int) -> dict[str, Any]:
    num_workers = int(num_workers)
    training: dict[str, Any] = {
        "batch_size": int(train_batch_size),
        "epochs": int(epochs),
        "num_workers": num_workers,
        "pin_memory": bool(num_workers > 0),
        "persistent_workers": bool(num_workers > 0),
    }
    if num_workers <= 0:
        training["prefetch_factor"] = None
    return {"training": training}


def _write_temp_config(base_config_path: Path, overrides: dict[str, Any]) -> Path:
    cfg = _load_yaml(base_config_path)
    merged = cfg.copy()
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            section = dict(merged[key])
            section.update(value)
            merged[key] = section
        else:
            merged[key] = value
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    with tmp:
        yaml.safe_dump(merged, tmp, sort_keys=False)
    return Path(tmp.name)


def _launch_parallel_training(
    config_paths: list[Path],
    *,
    project_root: Path,
    train_batch_size: int,
    epochs: int,
    num_workers: int,
    max_parallel_trains: int,
) -> None:
    if not config_paths:
        return
    overrides = _training_overrides(
        train_batch_size=int(train_batch_size),
        epochs=int(epochs),
        num_workers=int(num_workers),
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root / "src")
    env["MPLCONFIGDIR"] = "/tmp/mpl"
    launcher_log_dir = project_root / "reports" / "supp_s8_train_logs"
    launcher_log_dir.mkdir(parents=True, exist_ok=True)

    pending = list(config_paths)
    running: list[tuple[Path, Path, Path, Any, subprocess.Popen[str]]] = []
    limit = max(1, int(max_parallel_trains))

    while pending or running:
        while pending and len(running) < limit:
            base_path = pending.pop(0)
            tmp_cfg = _write_temp_config(base_path, overrides)
            launcher_log_path = launcher_log_dir / f"{base_path.stem}.log"
            launcher_log = launcher_log_path.open("a", encoding="utf-8")
            cmd = [sys.executable, "-m", "tao2019_fd2nn.cli.train_classifier", "--config", str(tmp_cfg)]
            proc = subprocess.Popen(
                cmd,
                cwd=project_root,
                env=env,
                stdout=launcher_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
            running.append((base_path, tmp_cfg, launcher_log_path, launcher_log, proc))
            print(f"[S8] launched baseline training: {base_path.name} log={launcher_log_path.relative_to(project_root)}")

        still_running: list[tuple[Path, Path, Path, Any, subprocess.Popen[str]]] = []
        for base_path, tmp_cfg, launcher_log_path, launcher_log, proc in running:
            rc = proc.poll()
            if rc is None:
                still_running.append((base_path, tmp_cfg, launcher_log_path, launcher_log, proc))
                continue
            launcher_log.close()
            try:
                tmp_cfg.unlink(missing_ok=True)
            except OSError:
                pass
            if rc != 0:
                raise subprocess.CalledProcessError(rc, proc.args, output=f"see log: {launcher_log_path}")
            print(f"[S8] completed baseline training: {base_path.name}")
        running = still_running
        if pending or running:
            time.sleep(2.0)


def _train_baseline_if_needed(
    config_path: Path,
    *,
    train_missing: bool,
    train_batch_size: int,
    epochs: int,
    num_workers: int,
    runs_root: Path,
    checkpoint_name: str,
) -> Path:
    exp_name = _experiment_name(config_path)
    try:
        return _latest_completed_run_dir(runs_root, exp_name, checkpoint_name=checkpoint_name)
    except FileNotFoundError:
        if not train_missing:
            raise
    overrides = _training_overrides(
        train_batch_size=int(train_batch_size),
        epochs=int(epochs),
        num_workers=int(num_workers),
    )
    run_with_overrides(config_path, overrides=overrides, task="classification")
    return _latest_completed_run_dir(runs_root, exp_name, checkpoint_name=checkpoint_name)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce Supplementary Figure S8 classification sensitivity")
    parser.add_argument("--fabrication-sigmas", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--alignment-shifts-um", default="0.0,2.0,4.0,6.0")
    parser.add_argument("--train-missing", action="store_true", help="train baseline checkpoints if a run is missing")
    parser.add_argument("--train-batch-size", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-parallel-trains", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--checkpoint", choices=("final.pt", "best.pt"), default="final.pt")
    parser.add_argument("--output", default="reports/supp_s8_cls.png")
    parser.add_argument("--summary-json", default="reports/supp_s8_cls_summary.json")
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    fabrication_sigmas = _parse_float_list(str(args.fabrication_sigmas))
    alignment_shifts_um = _parse_float_list(str(args.alignment_shifts_um))

    project_root = Path(__file__).resolve().parents[1]
    cfg_dir = project_root / "src" / "tao2019_fd2nn" / "config"
    runs_root = project_root / "runs"
    output_path = _resolve_output_path(project_root, str(args.output))
    summary_path = _resolve_output_path(project_root, str(args.summary_json))

    missing_config_paths: list[Path] = []
    if bool(args.train_missing):
        for spec in S8_EXPERIMENTS:
            config_path = cfg_dir / str(spec["config"])
            try:
                _latest_completed_run_dir(
                    runs_root,
                    _experiment_name(config_path),
                    checkpoint_name=str(args.checkpoint),
                )
            except FileNotFoundError:
                missing_config_paths.append(config_path)
        _launch_parallel_training(
            missing_config_paths,
            project_root=project_root,
            train_batch_size=int(args.train_batch_size),
            epochs=int(args.epochs),
            num_workers=int(args.num_workers),
            max_parallel_trains=int(args.max_parallel_trains),
        )

    fabrication_curves: dict[str, list[float]] = {}
    alignment_curves: dict[str, list[float]] = {}
    rows: list[dict[str, Any]] = []
    experiments: list[dict[str, Any]] = []

    for spec in S8_EXPERIMENTS:
        config_path = cfg_dir / str(spec["config"])
        run_dir = _train_baseline_if_needed(
            config_path,
            train_missing=bool(args.train_missing),
            train_batch_size=int(args.train_batch_size),
            epochs=int(args.epochs),
            num_workers=int(args.num_workers),
            runs_root=runs_root,
            checkpoint_name=str(args.checkpoint),
        )
        label = str(spec["label"])
        baseline_acc = _evaluate_accuracy(
            run_dir,
            eval_batch_size=int(args.eval_batch_size),
            num_workers=int(args.num_workers),
            checkpoint_name=str(args.checkpoint),
        )
        fabrication_values = [
            _evaluate_accuracy(
                run_dir,
                eval_batch_size=int(args.eval_batch_size),
                num_workers=int(args.num_workers),
                checkpoint_name=str(args.checkpoint),
                fabrication_sigma_px=float(sigma),
            )
            for sigma in fabrication_sigmas
        ]
        alignment_values = [
            _evaluate_accuracy(
                run_dir,
                eval_batch_size=int(args.eval_batch_size),
                num_workers=int(args.num_workers),
                checkpoint_name=str(args.checkpoint),
                alignment_shift_um=float(shift_um),
            )
            for shift_um in alignment_shifts_um
        ]

        fabrication_curves[label] = fabrication_values
        alignment_curves[label] = alignment_values
        experiment_entry = {
            "label": label,
            "family": str(spec["family"]),
            "num_layers": int(spec["num_layers"]),
            "config": str(config_path.relative_to(project_root)),
            "run_dir": str(run_dir.relative_to(project_root)),
            "checkpoint": str(args.checkpoint),
            "baseline_test_acc": baseline_acc,
            "fabrication": {"sigma_px": fabrication_sigmas, "accuracy": fabrication_values},
            "alignment": {"shift_um": alignment_shifts_um, "accuracy": alignment_values},
        }
        experiments.append(experiment_entry)
        for sigma, acc in zip(fabrication_sigmas, fabrication_values):
            rows.append(
                {
                    "label": label,
                    "family": str(spec["family"]),
                    "num_layers": int(spec["num_layers"]),
                    "perturbation_type": "fabrication",
                    "perturbation_value": float(sigma),
                    "eval_acc": float(acc),
                }
            )
        for shift_um, acc in zip(alignment_shifts_um, alignment_values):
            rows.append(
                {
                    "label": label,
                    "family": str(spec["family"]),
                    "num_layers": int(spec["num_layers"]),
                    "perturbation_type": "alignment",
                    "perturbation_value": float(shift_um),
                    "eval_acc": float(acc),
                }
            )

    factory = FigureFactory(output_path.parent)
    figure_path = factory.plot_s8_classification_sensitivity(
        fabrication_curves,
        alignment_curves,
        fabrication_x=fabrication_sigmas,
        alignment_x=alignment_shifts_um,
        name=output_path.name,
    )

    summary = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "fabrication_sigmas_px": fabrication_sigmas,
        "alignment_shifts_um": alignment_shifts_um,
        "figure": str(figure_path.relative_to(project_root)),
        "experiments": experiments,
        "rows": rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(str(figure_path))
    print(str(summary_path))


if __name__ == "__main__":
    main()
