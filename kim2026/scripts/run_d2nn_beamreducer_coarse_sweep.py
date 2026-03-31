#!/usr/bin/env python
"""Run the beam-reducer coarse sweep for plain D2NN experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG_PATH = ROOT / "configs" / "fso_1024_train_d2nn_beamreducer_base.yaml"
STAGE1_OUTPUT_DIR = ROOT / "runs" / "06_d2nn_beamreducer_distance-sweep_pitch-3um_codex"
STAGE2_OUTPUT_DIR = ROOT / "runs" / "07_d2nn_beamreducer_pitch-sweep_codex"
GRID_N = 1024
MAX_DISTANCE_MM = 100.0
MIN_PITCH_UM = 2.0
MAX_PITCH_UM = 5.0
STAGE1_PITCH_UM = 3.0
STAGE1_SPACINGS_MM = (10.0, 50.0, 100.0)
STAGE1_DETECTOR_DISTANCES_MM = (10.0, 50.0, 100.0)
STAGE2_PITCHES_UM = (2.0, 3.0, 5.0)
PRIMARY_METRIC = "model.overlap"


def pitch_um_to_window_m(pitch_um: float, n: int) -> float:
    return float(pitch_um) * 1.0e-6 * int(n)


def _format_mm(value_mm: float) -> str:
    return f"{int(value_mm)}mm"


def _validate_distance_mm(value_mm: float, *, label: str) -> float:
    value = float(value_mm)
    if value <= 0.0 or value > MAX_DISTANCE_MM:
        raise ValueError(f"{label} must be > 0 and <= 100 mm")
    return value


def _validate_pitch_um(value_um: float) -> float:
    value = float(value_um)
    if value < MIN_PITCH_UM or value > MAX_PITCH_UM:
        raise ValueError("pitch values must stay within the approved 2~5 um range")
    return value


def _make_run(*, name: str, pitch_um: float, layer_spacing_mm: float, detector_distance_mm: float) -> dict:
    validated_pitch_um = _validate_pitch_um(pitch_um)
    validated_layer_spacing_mm = _validate_distance_mm(layer_spacing_mm, label="layer spacing")
    validated_detector_distance_mm = _validate_distance_mm(detector_distance_mm, label="detector distance")
    return {
        "name": name,
        "pitch_um": validated_pitch_um,
        "receiver_window_m": pitch_um_to_window_m(validated_pitch_um, GRID_N),
        "layer_spacing_m": validated_layer_spacing_mm / 1000.0,
        "detector_distance_m": validated_detector_distance_mm / 1000.0,
    }


def build_stage1_runs(
    *,
    pitch_um: float = STAGE1_PITCH_UM,
    layer_spacing_values_mm: tuple[float, ...] = STAGE1_SPACINGS_MM,
    detector_distance_values_mm: tuple[float, ...] = STAGE1_DETECTOR_DISTANCES_MM,
) -> list[dict]:
    runs: list[dict] = []
    for layer_spacing_mm in layer_spacing_values_mm:
        for detector_distance_mm in detector_distance_values_mm:
            runs.append(
                _make_run(
                    name=f"ls{_format_mm(layer_spacing_mm)}_dd{_format_mm(detector_distance_mm)}",
                    pitch_um=pitch_um,
                    layer_spacing_mm=layer_spacing_mm,
                    detector_distance_mm=detector_distance_mm,
                )
            )
    return runs


def build_stage2_runs(
    best_layer_spacing_m: float,
    best_detector_distance_m: float,
    *,
    pitches_um: tuple[float, ...] = STAGE2_PITCHES_UM,
) -> list[dict]:
    layer_spacing_mm = float(best_layer_spacing_m) * 1000.0
    detector_distance_mm = float(best_detector_distance_m) * 1000.0
    runs: list[dict] = []
    for pitch_um in pitches_um:
        runs.append(
            _make_run(
                name=f"pitch{int(pitch_um)}um",
                pitch_um=pitch_um,
                layer_spacing_mm=layer_spacing_mm,
                detector_distance_mm=detector_distance_mm,
            )
        )
    return runs


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


def _materialize_config(
    run: dict,
    *,
    base_config_path: Path,
    stage_dir: Path,
    epochs: int | None = None,
) -> Path:
    cfg = deepcopy(_load_yaml(Path(base_config_path)))
    run_dir = Path(stage_dir) / run["name"]
    config_path = run_dir / "config.yaml"

    cfg["experiment"]["id"] = str(run["name"])
    cfg["experiment"]["save_dir"] = str(run_dir)
    cfg["grid"]["receiver_window_m"] = float(run["receiver_window_m"])
    cfg["model"]["layer_spacing_m"] = float(run["layer_spacing_m"])
    cfg["model"]["detector_distance_m"] = float(run["detector_distance_m"])
    if epochs is not None:
        cfg["training"]["epochs"] = int(epochs)
    cfg["visualization"]["output_dir"] = str(run_dir / "figures")

    _write_yaml(config_path, cfg)
    return config_path


def materialize_stage1_config(
    run: dict,
    *,
    base_config_path: Path = BASE_CONFIG_PATH,
    stage_dir: Path = STAGE1_OUTPUT_DIR,
    epochs: int | None = None,
) -> Path:
    return _materialize_config(
        run,
        base_config_path=Path(base_config_path),
        stage_dir=Path(stage_dir),
        epochs=epochs,
    )


def materialize_stage2_config(
    run: dict,
    *,
    base_config_path: Path = BASE_CONFIG_PATH,
    stage_dir: Path = STAGE2_OUTPUT_DIR,
    epochs: int | None = None,
) -> Path:
    return _materialize_config(
        run,
        base_config_path=Path(base_config_path),
        stage_dir=Path(stage_dir),
        epochs=epochs,
    )


def plan_stage_runs(
    *,
    stage: str,
    best_layer_spacing_mm: float | None = None,
    best_detector_distance_mm: float | None = None,
    stage2_pitches_um: tuple[float, ...] = STAGE2_PITCHES_UM,
) -> tuple[list[dict], Path]:
    if stage == "stage1":
        return build_stage1_runs(), STAGE1_OUTPUT_DIR
    if stage != "stage2":
        raise ValueError(f"unsupported stage: {stage}")
    if best_layer_spacing_mm is None or best_detector_distance_mm is None:
        raise ValueError("stage2 requires --best-layer-spacing-mm and --best-detector-distance-mm")
    runs = build_stage2_runs(
        _validate_distance_mm(best_layer_spacing_mm, label="layer spacing") / 1000.0,
        _validate_distance_mm(best_detector_distance_mm, label="detector distance") / 1000.0,
        pitches_um=tuple(stage2_pitches_um),
    )
    return runs, STAGE2_OUTPUT_DIR


def _format_run_plan(run: dict, *, stage_dir: Path) -> str:
    config_path = stage_dir / run["name"] / "config.yaml"
    return (
        f"{run['name']}: pitch_um={run['pitch_um']:.0f}, "
        f"receiver_window_m={run['receiver_window_m']:.6f}, "
        f"layer_spacing_m={run['layer_spacing_m']:.6f}, "
        f"detector_distance_m={run['detector_distance_m']:.6f}, "
        f"config:{config_path}"
    )


def print_dry_run(*, stage: str, runs: list[dict], stage_dir: Path, epochs: int | None = None) -> None:
    print(f"Planned {len(runs)} runs for {stage}")
    if epochs is not None:
        print(f"epochs override: {epochs}")
    for run in runs:
        print(_format_run_plan(run, stage_dir=stage_dir))


def end_to_end_dry_run(
    *,
    stage: str,
    best_layer_spacing_mm: float | None = None,
    best_detector_distance_mm: float | None = None,
    epochs: int | None = None,
) -> dict:
    runs, stage_dir = plan_stage_runs(
        stage=stage,
        best_layer_spacing_mm=best_layer_spacing_mm,
        best_detector_distance_mm=best_detector_distance_mm,
    )
    print_dry_run(stage=stage, runs=runs, stage_dir=stage_dir, epochs=epochs)
    return {
        "count": len(runs),
        "windows_m": sorted({float(run["receiver_window_m"]) for run in runs}),
        "layer_spacing_mm": [float(run["layer_spacing_m"]) * 1000.0 for run in runs],
        "detector_distance_mm": [float(run["detector_distance_m"]) * 1000.0 for run in runs],
    }


def execute_stage_runs(
    *,
    stage: str,
    runs: list[dict],
    base_config_path: Path = BASE_CONFIG_PATH,
    stage_dir: Path | None = None,
    epochs: int | None = None,
) -> list[dict]:
    resolved_stage_dir = Path(stage_dir or (STAGE1_OUTPUT_DIR if stage == "stage1" else STAGE2_OUTPUT_DIR))
    materialize = materialize_stage1_config if stage == "stage1" else materialize_stage2_config
    records: list[dict] = []
    for run in runs:
        config_path = materialize(
            run,
            base_config_path=Path(base_config_path),
            stage_dir=resolved_stage_dir,
            epochs=epochs,
        )
        train_cmd = [sys.executable, "-m", "kim2026.cli.train_beam_cleanup", "--config", str(config_path)]
        eval_cmd = [sys.executable, "-m", "kim2026.cli.evaluate_beam_cleanup", "--config", str(config_path)]
        subprocess.run(train_cmd, check=True, cwd=str(ROOT))
        subprocess.run(eval_cmd, check=True, cwd=str(ROOT))
        records.append(
            {
                "name": run["name"],
                "run_dir": resolved_stage_dir / run["name"],
                "config_path": config_path,
            }
        )
    return records


def _load_evaluation_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _primary_metric_value(evaluation: dict, *, primary_metric: str) -> float:
    section, metric_name = primary_metric.split(".", maxsplit=1)
    return float(evaluation[section][metric_name])


def aggregate_stage_summary(
    *,
    stage: str,
    runs: list[dict],
    stage_dir: Path,
    primary_metric: str = PRIMARY_METRIC,
) -> dict:
    resolved_stage_dir = Path(stage_dir)
    ranked_runs: list[dict] = []
    for run in runs:
        evaluation = _load_evaluation_summary(resolved_stage_dir / run["name"] / "evaluation.json")
        ranked_runs.append(
            {
                **run,
                "metrics": evaluation,
                "primary_metric_value": _primary_metric_value(evaluation, primary_metric=primary_metric),
            }
        )
    ranked_runs.sort(key=lambda item: item["primary_metric_value"], reverse=True)
    best_run = ranked_runs[0]
    summary = {
        "stage": stage,
        "primary_metric": primary_metric,
        "best_run_name": best_run["name"],
        "runs": ranked_runs,
    }
    if stage == "stage1":
        summary["best_layer_spacing_m"] = float(best_run["layer_spacing_m"])
        summary["best_detector_distance_m"] = float(best_run["detector_distance_m"])
    elif stage == "stage2":
        summary["best_pitch_um"] = float(best_run["pitch_um"])
    else:
        raise ValueError(f"unsupported stage: {stage}")
    _write_json(resolved_stage_dir / "stage_summary.json", summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("stage1", "stage2"))
    parser.add_argument("--dry-run", action="store_true", help="Print the planned runs without launching training")
    parser.add_argument("--apply", action="store_true", help="Launch training and evaluation for the selected stage")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override written into generated configs")
    parser.add_argument("--best-layer-spacing-mm", type=float, default=None)
    parser.add_argument("--best-detector-distance-mm", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.apply and args.dry_run:
        parser.error("choose only one of --dry-run or --apply")

    runs, stage_dir = plan_stage_runs(
        stage=args.stage,
        best_layer_spacing_mm=args.best_layer_spacing_mm,
        best_detector_distance_mm=args.best_detector_distance_mm,
    )

    if args.dry_run or not args.apply:
        print_dry_run(stage=args.stage, runs=runs, stage_dir=stage_dir, epochs=args.epochs)
        return 0

    execute_stage_runs(
        stage=args.stage,
        runs=runs,
        base_config_path=BASE_CONFIG_PATH,
        stage_dir=stage_dir,
        epochs=args.epochs,
    )
    summary = aggregate_stage_summary(stage=args.stage, runs=runs, stage_dir=stage_dir)

    if args.stage == "stage1":
        print(
            f"Best run: {summary['best_run_name']} "
            f"(layer_spacing_m={summary['best_layer_spacing_m']:.3f}, "
            f"detector_distance_m={summary['best_detector_distance_m']:.3f})"
        )
    else:
        print(f"Best run: {summary['best_run_name']} (pitch_um={summary['best_pitch_um']:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
