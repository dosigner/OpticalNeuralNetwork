"""Simple parameter sweep runners."""

from __future__ import annotations

from typing import Any



def run_value_sweep(values: list[float | int], eval_fn) -> list[dict[str, Any]]:
    """Run generic sweep over values.

    Args:
        values: list of sweep values
        eval_fn: callable(value) -> dict of metrics
    """

    results: list[dict[str, Any]] = []
    for value in values:
        metrics = eval_fn(value)
        row = {"value": value}
        row.update(metrics)
        results.append(row)
    return results
