#!/usr/bin/env python
"""Analyze FD2NN sweep results and generate summary table.

Usage:
    python scripts/analyze_sweep.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SWEEP_DIR = Path(__file__).resolve().parent.parent / "runs" / "sweep_fd2nn"

CN2_VALUES = [1e-15, 5e-15, 1e-14]
SPACING_VALUES = [1e-6, 3e-6, 5e-6, 10e-6]


def make_run_id(cn2: float, spacing: float) -> str:
    cn2_str = f"cn2_{cn2:.0e}".replace("+", "").replace("-", "m")
    sp_str = f"sp_{spacing * 1e6:.0f}um"
    return f"{cn2_str}_{sp_str}"


def load_evaluation(run_id: str) -> dict | None:
    eval_path = SWEEP_DIR / run_id / "evaluation.json"
    if not eval_path.exists():
        return None
    with open(eval_path) as f:
        return json.load(f)


def main() -> None:
    print("=" * 70)
    print("FD2NN Sweep Analysis: Cn2 x Layer Spacing")
    print("=" * 70)

    # Collect results
    results = {}
    for cn2 in CN2_VALUES:
        for spacing in SPACING_VALUES:
            run_id = make_run_id(cn2, spacing)
            evaluation = load_evaluation(run_id)
            results[(cn2, spacing)] = evaluation

    # Print complex_overlap heatmap
    print("\n--- Complex Overlap (model, higher=better) ---")
    header = f"{'Cn2':>12} |" + "".join(f" {s*1e6:5.0f}um" for s in SPACING_VALUES)
    print(header)
    print("-" * len(header))
    for cn2 in CN2_VALUES:
        row = f"{cn2:>12.0e} |"
        for spacing in SPACING_VALUES:
            ev = results.get((cn2, spacing))
            if ev and "model" in ev:
                val = ev["model"].get("complex_overlap", float("nan"))
                row += f" {val:6.3f}"
            else:
                row += "    N/A"
        print(row)

    # Print phase RMSE heatmap
    print("\n--- Phase RMSE [rad] (model, lower=better) ---")
    print(header)
    print("-" * len(header))
    for cn2 in CN2_VALUES:
        row = f"{cn2:>12.0e} |"
        for spacing in SPACING_VALUES:
            ev = results.get((cn2, spacing))
            if ev and "model" in ev:
                val = ev["model"].get("phase_rmse_rad", float("nan"))
                row += f" {val:6.3f}"
            else:
                row += "    N/A"
        print(row)

    # Print baseline comparison
    print("\n--- Baseline vs Model (complex_overlap, cn2=1e-15) ---")
    for spacing in SPACING_VALUES:
        ev = results.get((1e-15, spacing))
        if ev and "baseline" in ev and "model" in ev:
            bl = ev["baseline"].get("complex_overlap", float("nan"))
            md = ev["model"].get("complex_overlap", float("nan"))
            improvement = md - bl
            print(f"  spacing={spacing*1e6:.0f}um: baseline={bl:.4f} → model={md:.4f} (Δ={improvement:+.4f})")
        else:
            print(f"  spacing={spacing*1e6:.0f}um: N/A")

    # Find best config
    best_overlap = -1.0
    best_config = None
    for (cn2, spacing), ev in results.items():
        if ev and "model" in ev:
            overlap = ev["model"].get("complex_overlap", 0.0)
            if overlap > best_overlap:
                best_overlap = overlap
                best_config = (cn2, spacing)

    if best_config:
        print(f"\n>>> Best config: Cn2={best_config[0]:.0e}, spacing={best_config[1]*1e6:.0f}um")
        print(f"    Complex overlap: {best_overlap:.4f}")

    # Save summary
    summary = {
        "sweep_axes": {
            "cn2": CN2_VALUES,
            "layer_spacing_m": SPACING_VALUES,
        },
        "results": {
            make_run_id(cn2, sp): results.get((cn2, sp))
            for cn2 in CN2_VALUES
            for sp in SPACING_VALUES
        },
        "best": {
            "run_id": make_run_id(*best_config) if best_config else None,
            "complex_overlap": best_overlap,
        },
    }
    summary_path = SWEEP_DIR / "sweep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
