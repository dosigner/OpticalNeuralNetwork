#!/usr/bin/env python3
"""Analyze a simple D2NN registry into summary JSON."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _load_registry(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _selection_key(row: dict) -> tuple[int, float, float]:
    metrics = row.get("metrics", {})
    leak = float(metrics.get("roi50_out_of_support_energy_fraction", metrics.get("out_of_support_energy_fraction", 0.0)))
    intensity = float(metrics.get("roi50_intensity_overlap", metrics.get("intensity_overlap", float("-inf"))))
    phase = float(
        metrics.get(
            "roi50_support_weighted_phase_rmse_rad",
            metrics.get("support_weighted_phase_rmse_rad", metrics.get("phase_rmse_rad", float("inf"))),
        )
    )
    if leak > 0.15:
        return (0, float("-inf"), float("-inf"))
    return (1, intensity, -phase)


def _summarize(rows: list[dict]) -> dict:
    by_family: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_family[row.get("task_family", "unknown")].append(row)

    frontier = {}
    for family, family_rows in by_family.items():
        if not family_rows:
            continue
        frontier[family] = max(family_rows, key=_selection_key)["run_id"]

    label_counts = Counter()
    missing_config = sum(0 if row.get("config_present") else 1 for row in rows)
    missing_history = sum(0 if row.get("history_present") else 1 for row in rows)
    if rows:
        label_counts["optimization_limited"] = max(len(rows) - len(frontier), 0)

    report = "\n".join(
        [
            "## 🧭 한 줄 결론",
            "증거 기준 최상위 run과 누락 artifact를 먼저 함께 확인 필요",
            "",
            "## 📊 Pareto 비교표",
            "| task family | frontier run | 판단 |",
            "|---|---|---|",
            *[f"| {family} | {run_id} | leakage gate 통과 기준 상위 |" for family, run_id in sorted(frontier.items())],
            "",
            "## ⚠️ 해석 한계",
            f"config 부재 {missing_config}건, history 부재 {missing_history}건",
        ]
    )

    return {
        "frontier_ids": sorted(frontier.values()),
        "frontier_by_family": frontier,
        "label_counts": dict(label_counts),
        "missing_config_count": missing_config,
        "missing_history_count": missing_history,
        "report": {"markdown": report},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    registry = _load_registry(args.registry)
    rows = list(registry.get("rows", []))
    payload = {
        "registry_path": str(args.registry),
        "row_count": len(rows),
        **_summarize(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote analysis to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
