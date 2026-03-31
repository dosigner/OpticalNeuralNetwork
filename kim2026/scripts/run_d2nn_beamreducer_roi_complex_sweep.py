#!/usr/bin/env python
"""Run ROI-complex loss sweep for D2NN beam reducer.

Sweeps roi_threshold × leakage_weight while keeping physical parameters
fixed at the best configuration from the previous coarse sweep
(pitch=5µm, layer_spacing=50mm, detector_distance=50mm).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG_PATH = ROOT / "configs" / "fso_1024_train_d2nn_beamreducer_roi_complex_base.yaml"
OUTPUT_DIR = ROOT / "runs" / "08_d2nn_beamreducer_roi-complex-sweep_claude"

ROI_THRESHOLDS = (0.50, 0.70, 0.90)
LEAKAGE_WEIGHTS = (0.5, 1.0, 2.0)
PRIMARY_METRIC = "model.complex_overlap"


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _format_roi(threshold: float) -> str:
    return f"roi{int(threshold * 100)}"


def _format_lw(weight: float) -> str:
    if weight == int(weight):
        return f"lw{int(weight * 10):02d}"
    return f"lw{int(weight * 10):02d}"


def build_runs() -> list[dict]:
    runs: list[dict] = []
    for roi_t in ROI_THRESHOLDS:
        for lw in LEAKAGE_WEIGHTS:
            name = f"{_format_roi(roi_t)}_{_format_lw(lw)}"
            runs.append({
                "name": name,
                "roi_threshold": float(roi_t),
                "leakage_weight": float(lw),
            })
    return runs


def materialize_config(run: dict, *, base_config_path: Path = BASE_CONFIG_PATH, output_dir: Path = OUTPUT_DIR, epochs: int | None = None) -> Path:
    cfg = deepcopy(_load_yaml(base_config_path))
    run_dir = output_dir / run["name"]
    config_path = run_dir / "config.yaml"

    cfg["experiment"]["id"] = str(run["name"])
    cfg["experiment"]["save_dir"] = str(run_dir)
    cfg["training"]["loss"]["roi_threshold"] = float(run["roi_threshold"])
    cfg["training"]["loss"]["leakage_weight"] = float(run["leakage_weight"])
    if epochs is not None:
        cfg["training"]["epochs"] = int(epochs)
    cfg["visualization"]["output_dir"] = str(run_dir / "figures")

    _write_yaml(config_path, cfg)
    return config_path


def execute_runs(*, runs: list[dict], base_config_path: Path = BASE_CONFIG_PATH, output_dir: Path = OUTPUT_DIR, epochs: int | None = None) -> list[dict]:
    records: list[dict] = []
    for i, run in enumerate(runs, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(runs)}] {run['name']}  roi_threshold={run['roi_threshold']:.2f}  leakage_weight={run['leakage_weight']:.1f}")
        print(f"{'='*60}")
        config_path = materialize_config(run, base_config_path=base_config_path, output_dir=output_dir, epochs=epochs)
        train_cmd = [sys.executable, "-m", "kim2026.cli.train_beam_cleanup", "--config", str(config_path)]
        eval_cmd = [sys.executable, "-m", "kim2026.cli.evaluate_beam_cleanup", "--config", str(config_path)]
        subprocess.run(train_cmd, check=True, cwd=str(ROOT))
        subprocess.run(eval_cmd, check=True, cwd=str(ROOT))
        records.append({
            "name": run["name"],
            "roi_threshold": run["roi_threshold"],
            "leakage_weight": run["leakage_weight"],
            "run_dir": str(output_dir / run["name"]),
            "config_path": str(config_path),
        })
    return records


def aggregate_summary(*, runs: list[dict], output_dir: Path = OUTPUT_DIR, primary_metric: str = PRIMARY_METRIC) -> dict:
    ranked: list[dict] = []
    for run in runs:
        eval_path = Path(output_dir) / run["name"] / "evaluation.json"
        evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
        section, metric_name = primary_metric.split(".", maxsplit=1)
        metric_val = float(evaluation[section][metric_name])
        ranked.append({
            **run,
            "metrics": evaluation,
            "primary_metric_value": metric_val,
        })
    ranked.sort(key=lambda x: x["primary_metric_value"], reverse=True)
    summary = {
        "sweep": "roi_complex_loss",
        "primary_metric": primary_metric,
        "best_run_name": ranked[0]["name"],
        "best_roi_threshold": ranked[0]["roi_threshold"],
        "best_leakage_weight": ranked[0]["leakage_weight"],
        "runs": ranked,
    }
    _write_json(output_dir / "sweep_summary.json", summary)
    return summary


def print_dry_run(runs: list[dict], *, epochs: int | None = None) -> None:
    print(f"Planned {len(runs)} ROI-complex loss sweep runs")
    if epochs is not None:
        print(f"Epochs override: {epochs}")
    for run in runs:
        print(f"  {run['name']}: roi_threshold={run['roi_threshold']:.2f}, leakage_weight={run['leakage_weight']:.1f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args(argv)

    if args.apply and args.dry_run:
        parser.error("choose only one of --dry-run or --apply")

    runs = build_runs()

    if not args.apply:
        print_dry_run(runs, epochs=args.epochs)
        return 0

    records = execute_runs(runs=runs, epochs=args.epochs)
    summary = aggregate_summary(runs=records)
    print(f"\nBest run: {summary['best_run_name']} "
          f"(roi_threshold={summary['best_roi_threshold']:.2f}, "
          f"leakage_weight={summary['best_leakage_weight']:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
