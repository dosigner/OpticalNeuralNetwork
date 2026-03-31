"""Visualization helpers for D2NN beam-reducer coarse sweeps."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from kim2026.viz.beam_plots import save_triptych


plt.rcParams.update({
    "font.size": 10,
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})

REQUIRED_RUN_FILES = ("evaluation.json", "sample_fields.npz")
STAGE1_PATTERN = re.compile(r"ls(?P<ls>\d+)mm_dd(?P<dd>\d+)mm")
STAGE2_PATTERN = re.compile(r"pitch(?P<pitch>\d+)um")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _complex_from_npz(npz: np.lib.npyio.NpzFile, prefix: str) -> np.ndarray:
    return npz[f"{prefix}_real"] + 1j * npz[f"{prefix}_imag"]


def _is_complete_run(run_dir: Path) -> bool:
    return all((run_dir / name).exists() for name in REQUIRED_RUN_FILES)


def _stage1_sort_key(name: str) -> tuple[float, float, str]:
    match = STAGE1_PATTERN.fullmatch(name)
    if match is None:
        return (float("inf"), float("inf"), name)
    return (float(match.group("ls")), float(match.group("dd")), name)


def _stage2_sort_key(name: str) -> tuple[float, str]:
    match = STAGE2_PATTERN.fullmatch(name)
    if match is None:
        return (float("inf"), name)
    return (float(match.group("pitch")), name)


def _load_run(run_dir: Path) -> dict:
    npz = np.load(run_dir / "sample_fields.npz")
    return {
        "name": run_dir.name,
        "evaluation": _load_json(run_dir / "evaluation.json"),
        "input": _complex_from_npz(npz, "input"),
        "vacuum": None if "vacuum_real" not in npz else _complex_from_npz(npz, "vacuum"),
        "baseline": None if "baseline_real" not in npz else _complex_from_npz(npz, "baseline"),
        "pred": _complex_from_npz(npz, "pred"),
        "target": _complex_from_npz(npz, "target"),
    }


def get_stage1_runs(stage1_dir: str | Path) -> dict[str, dict]:
    stage1_dir = Path(stage1_dir)
    run_dirs = [
        run_dir
        for run_dir in stage1_dir.iterdir()
        if run_dir.is_dir() and STAGE1_PATTERN.fullmatch(run_dir.name) and _is_complete_run(run_dir)
    ]
    run_dirs.sort(key=lambda run_dir: _stage1_sort_key(run_dir.name))
    return {run_dir.name: _load_run(run_dir) for run_dir in run_dirs}


def get_stage2_runs(stage2_dir: str | Path) -> dict[str, dict]:
    stage2_dir = Path(stage2_dir)
    run_dirs = [
        run_dir
        for run_dir in stage2_dir.iterdir()
        if run_dir.is_dir() and STAGE2_PATTERN.fullmatch(run_dir.name) and _is_complete_run(run_dir)
    ]
    run_dirs.sort(key=lambda run_dir: _stage2_sort_key(run_dir.name))
    return {run_dir.name: _load_run(run_dir) for run_dir in run_dirs}


def _best_run(runs: dict[str, dict]) -> tuple[str, dict]:
    best_name = max(runs, key=lambda name: float(runs[name]["evaluation"]["model"]["overlap"]))
    return best_name, runs[best_name]


def _plot_stage_metrics(runs: dict[str, dict], *, title: str, path: Path) -> Path:
    labels = list(runs)
    overlaps = [float(runs[name]["evaluation"]["model"]["overlap"]) for name in labels]
    strehl = [float(runs[name]["evaluation"]["model"].get("strehl", 0.0)) for name in labels]
    baseline_overlap = [float(runs[name]["evaluation"]["baseline"]["overlap"]) for name in labels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), squeeze=False)
    ax0, ax1 = axes[0]

    ax0.bar(labels, overlaps, color="#1f77b4", edgecolor="black", linewidth=0.6, label="model")
    ax0.plot(labels, baseline_overlap, color="black", linestyle="--", marker="o", linewidth=1.2, label="baseline")
    ax0.set_title("Overlap")
    ax0.set_ylabel("Normalized Overlap")
    ax0.grid(True, axis="y", alpha=0.3)
    ax0.legend(fontsize=8)

    ax1.bar(labels, strehl, color="#ff7f0e", edgecolor="black", linewidth=0.6)
    ax1.set_title("Strehl Ratio")
    ax1.set_ylabel("Strehl")
    ax1.grid(True, axis="y", alpha=0.3)

    for axis in (ax0, ax1):
        axis.tick_params(axis="x", rotation=25)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_best_fields(runs: dict[str, dict], *, title: str, path: Path) -> Path:
    best_name, run = _best_run(runs)
    return save_triptych(
        path,
        input_field=run["input"],
        vacuum_field=run["vacuum"],
        baseline_field=run["baseline"] if run["baseline"] is not None else run["input"],
        pred_field=run["pred"],
        target_field=run["target"],
        title=f"{title}: {best_name}",
    )


def generate_figures(stage1_dir: str | Path, stage2_dir: str | Path, fig_dir: str | Path | None = None) -> list[Path]:
    stage1_dir = Path(stage1_dir)
    stage2_dir = Path(stage2_dir)
    fig_dir = Path(fig_dir) if fig_dir is not None else stage2_dir.parent / "d2nn_beamreducer_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    stage1_runs = get_stage1_runs(stage1_dir)
    stage2_runs = get_stage2_runs(stage2_dir)
    if not stage1_runs:
        raise ValueError(f"no complete stage1 runs found in {stage1_dir}")
    if not stage2_runs:
        raise ValueError(f"no complete stage2 runs found in {stage2_dir}")

    return [
        _plot_stage_metrics(stage1_runs, title="Stage 1 Distance Sweep Metrics", path=fig_dir / "fig1_stage1_test_metrics.png"),
        _plot_best_fields(stage1_runs, title="Stage 1 Best Run Fields", path=fig_dir / "fig2_stage1_best_fields.png"),
        _plot_stage_metrics(stage2_runs, title="Stage 2 Pitch Sweep Metrics", path=fig_dir / "fig3_stage2_test_metrics.png"),
        _plot_best_fields(stage2_runs, title="Stage 2 Best Run Fields", path=fig_dir / "fig4_stage2_best_fields.png"),
    ]
