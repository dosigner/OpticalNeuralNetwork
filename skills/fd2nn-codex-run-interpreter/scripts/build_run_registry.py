from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


DEFAULT_ROOT = Path("/root/dj/D2NN/kim2026")

OFFICIAL_RUNS = {
    "02": {
        "dir_name": "02_fd2nn_spacing-sweep_loss-old_roi-1024_codex",
        "sweep_axis": "spacing-sweep",
        "loss_family": "old",
        "fixed_conditions": {"roi": 1024},
        "expected_conditions": ["spacing_0mm", "spacing_0p1mm", "spacing_1mm", "spacing_2mm"],
    },
    "03": {
        "dir_name": "03_fd2nn_spacing-sweep_loss-shape_roi-1024_codex",
        "sweep_axis": "spacing-sweep",
        "loss_family": "shape",
        "fixed_conditions": {"roi": 1024},
        "expected_conditions": ["spacing_0mm", "spacing_0p1mm", "spacing_1mm", "spacing_2mm"],
    },
    "04": {
        "dir_name": "04_fd2nn_spacing-sweep_loss-shape_roi-512_codex",
        "sweep_axis": "spacing-sweep",
        "loss_family": "shape",
        "fixed_conditions": {"roi": 512},
        "expected_conditions": ["spacing_0mm", "spacing_0p1mm", "spacing_1mm", "spacing_2mm"],
    },
    "05": {
        "dir_name": "05_fd2nn_roi-sweep_loss-phase-first_spacing-0p1mm_codex",
        "sweep_axis": "roi-sweep",
        "loss_family": "phase-first",
        "fixed_conditions": {"spacing": "0p1mm"},
        "expected_conditions": ["roi512_spacing_0p1mm", "roi1024_spacing_0p1mm"],
    },
}

SPACING_PATTERN = re.compile(r"spacing_(?P<value>\d+(?:p\d+)?)mm$")
ROI_PATTERN = re.compile(r"roi(?P<value>\d+)_spacing_(?P<spacing>.+)$")
LOWER_IS_BETTER = {
    "phase_rmse_rad",
    "full_field_phase_rmse_rad",
    "support_weighted_phase_rmse_rad",
    "out_of_support_energy_fraction",
    "amplitude_rmse",
    "beam_radius_m",
}


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_spacing_key(name: str) -> float:
    match = SPACING_PATTERN.search(name)
    if not match:
        return math.inf
    return float(match.group("value").replace("p", "."))


def parse_roi_key(name: str) -> int:
    match = ROI_PATTERN.match(name)
    if not match:
        return 10**9
    return int(match.group("value"))


def sort_condition_keys(keys: list[str], sweep_axis: str) -> list[str]:
    if sweep_axis == "spacing-sweep":
        return sorted(keys, key=lambda key: (parse_spacing_key(key), key))
    if sweep_axis == "roi-sweep":
        return sorted(keys, key=lambda key: (parse_roi_key(key), key))
    return sorted(keys)


def summarize_condition_metrics(run_dir: Path) -> tuple[str, dict[str, dict], str | None, dict[str, str]]:
    sweep_summary = run_dir / "sweep_summary.json"
    if sweep_summary.exists():
        summary = load_json(sweep_summary)
        return "sweep-summary", summary, str(sweep_summary), {}

    study_summary = run_dir / "study_summary.json"
    if study_summary.exists():
        summary = load_json(study_summary)
        return "study-summary", summary, str(study_summary), {}

    condition_metrics: dict[str, dict] = {}
    metric_paths: dict[str, str] = {}
    for child in sorted(run_dir.iterdir()):
        metrics_path = child / "test_metrics.json"
        if child.is_dir() and metrics_path.exists():
            condition_metrics[child.name] = load_json(metrics_path)
            metric_paths[child.name] = str(metrics_path)
    return "per-condition-test-metrics", condition_metrics, None, metric_paths


def pick_representative_metric(condition_metrics: dict[str, dict]) -> str:
    preferred = (
        "complex_overlap",
        "intensity_overlap",
        "support_weighted_phase_rmse_rad",
        "phase_rmse_rad",
    )
    if not condition_metrics:
        return "complex_overlap"
    available = set()
    for metrics in condition_metrics.values():
        available.update(metrics.keys())
    for key in preferred:
        if key in available:
            return key
    return sorted(available)[0]


def pick_best_condition(
    condition_metrics: dict[str, dict],
    metric_name: str,
) -> tuple[str | None, float | None]:
    if not condition_metrics:
        return None, None

    values = {
        name: float(metrics[metric_name])
        for name, metrics in condition_metrics.items()
        if metric_name in metrics
    }
    if not values:
        return None, None

    if metric_name in LOWER_IS_BETTER:
        best_name = min(values, key=values.get)
    else:
        best_name = max(values, key=values.get)
    return best_name, values[best_name]


def build_conclusion(
    run_id: str,
    sweep_axis: str,
    condition_metrics: dict[str, dict],
    metric_name: str,
    expected_conditions: list[str],
    available_conditions: list[str],
) -> str:
    best_name, best_value = pick_best_condition(condition_metrics, metric_name)
    if best_name is None or best_value is None:
        return f"Run {run_id} is an official {sweep_axis} study."

    coverage = f"{len(available_conditions)}/{len(expected_conditions)}"
    direction_word = "lowest" if metric_name in LOWER_IS_BETTER else "highest"
    return (
        f"Run {run_id} is an official {sweep_axis} study; "
        f"available metrics cover {coverage} expected conditions, and {best_name} gives the {direction_word} "
        f"{metric_name} ({best_value:.3f}) among the available conditions."
    )


def build_run_registry(root: str | Path = DEFAULT_ROOT) -> dict[str, dict]:
    root = Path(root)
    runs_dir = root / "runs"
    registry: dict[str, dict] = {}

    for run_id, meta in OFFICIAL_RUNS.items():
        run_dir = runs_dir / meta["dir_name"]
        if not run_dir.exists():
            expected_conditions = sort_condition_keys(meta["expected_conditions"], meta["sweep_axis"])
            registry[run_id] = {
                "run_name": meta["dir_name"],
                "run_path": str(run_dir),
                "status": "missing-run",
                "sweep_axis": meta["sweep_axis"],
                "loss_family": meta["loss_family"],
                "fixed_conditions": meta["fixed_conditions"],
                "compared_conditions": expected_conditions,
                "available_conditions": [],
                "missing_conditions": expected_conditions,
                "summary_kind": "missing-run",
                "summary_path": None,
                "condition_metric_paths": {},
                "representative_metric": None,
                "best_condition": None,
                "best_value": None,
                "key_metrics": [],
                "condition_metrics": {},
                "official_figure_store": str(root / "figures"),
                "sources": [str(run_dir)],
                "conclusion": (
                    f"Run {run_id} is part of the official evidence base, but its directory is missing at {run_dir}."
                ),
            }
            continue

        summary_kind, condition_metrics, summary_path, metric_paths = summarize_condition_metrics(run_dir)
        expected_conditions = sort_condition_keys(meta["expected_conditions"], meta["sweep_axis"])
        available_conditions = sort_condition_keys(list(condition_metrics.keys()), meta["sweep_axis"])
        missing_conditions = [name for name in expected_conditions if name not in available_conditions]
        representative_metric = pick_representative_metric(condition_metrics)
        best_condition, best_value = pick_best_condition(condition_metrics, representative_metric)
        condition_metrics = {name: condition_metrics[name] for name in available_conditions}
        registry[run_id] = {
            "run_name": meta["dir_name"],
            "run_path": str(run_dir),
            "status": "ok",
            "sweep_axis": meta["sweep_axis"],
            "loss_family": meta["loss_family"],
            "fixed_conditions": meta["fixed_conditions"],
            "compared_conditions": expected_conditions,
            "available_conditions": available_conditions,
            "missing_conditions": missing_conditions,
            "summary_kind": summary_kind,
            "summary_path": summary_path,
            "condition_metric_paths": metric_paths,
            "representative_metric": representative_metric,
            "best_condition": best_condition,
            "best_value": best_value,
            "key_metrics": sorted(
                {
                    key
                    for metrics in condition_metrics.values()
                    for key in metrics.keys()
                }
            ),
            "condition_metrics": condition_metrics,
            "official_figure_store": str(root / "figures"),
            "sources": [
                str(run_dir),
                *( [summary_path] if summary_path else [] ),
                *metric_paths.values(),
            ],
            "conclusion": build_conclusion(
                run_id,
                meta["sweep_axis"],
                condition_metrics,
                representative_metric,
                expected_conditions,
                available_conditions,
            ),
        }

    return registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a structured registry for official kim2026 codex runs.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="kim2026 project root")
    args = parser.parse_args()
    registry = build_run_registry(args.root)
    print(json.dumps(registry, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
