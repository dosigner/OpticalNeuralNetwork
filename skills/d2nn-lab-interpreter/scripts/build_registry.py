#!/usr/bin/env python3
"""Build a simple canonical registry from D2NN run folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _infer_metrics(run_dir: Path) -> dict:
    evaluation_path = run_dir / "evaluation.json"
    test_metrics_path = run_dir / "test_metrics.json"
    if evaluation_path.exists():
        payload = _load_json(evaluation_path)
        if isinstance(payload, dict):
            model = payload.get("model")
            if isinstance(model, dict):
                return model
    if test_metrics_path.exists():
        payload = _load_json(test_metrics_path)
        if isinstance(payload, dict):
            return payload
    return {}


def _task_family(metrics: dict) -> str:
    keys = set(metrics)
    if {"complex_overlap", "intensity_overlap"} & keys:
        return "beam_cleanup"
    return "unknown"


def _collect_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for run_dir in sorted(p for p in root.rglob("*") if p.is_dir()):
        if not ((run_dir / "evaluation.json").exists() or (run_dir / "test_metrics.json").exists()):
            continue
        metrics = _infer_metrics(run_dir)
        rows.append(
            {
                "run_id": str(run_dir.relative_to(root)),
                "run_dir": str(run_dir),
                "task_family": _task_family(metrics),
                "metrics": metrics,
                "config_present": (run_dir / "config.yaml").exists(),
                "history_present": (run_dir / "history.json").exists(),
                "notes": [],
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = _collect_rows(args.root)
    payload = {
        "root": str(args.root),
        "row_count": len(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
