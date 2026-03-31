"""Visualization helpers for phase-first dual-2f FD2NN studies."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REQUIRED_RUN_FILES = ("history.json", "test_metrics.json", "sample_fields.npz")

plt.rcParams.update({
    "font.size": 10,
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})

ROI_PATTERN = re.compile(r"roi(?P<roi>\d+)_spacing_(?P<spacing>.+)")


def _load_json(path: Path) -> dict | list:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _complex_from_npz(npz: np.lib.npyio.NpzFile, prefix: str) -> np.ndarray:
    return npz[f"{prefix}_real"] + 1j * npz[f"{prefix}_imag"]


def _parse_run_name(name: str) -> tuple[int, str]:
    match = ROI_PATTERN.fullmatch(name)
    if not match:
        return math.inf, name
    return int(match.group("roi")), match.group("spacing")


def format_run_label(name: str) -> str:
    roi_n, spacing = _parse_run_name(name)
    if not math.isfinite(roi_n):
        return name
    return f"ROI {roi_n}, {spacing.replace('p', '.').replace('mm', ' mm')}"


def is_complete_run(run_dir: Path) -> bool:
    return all((run_dir / file_name).exists() for file_name in REQUIRED_RUN_FILES)


def load_run(run_dir: Path) -> dict:
    npz = np.load(run_dir / "sample_fields.npz")
    raw_phase_files = sorted(run_dir.glob("phases_raw_epoch*.npy"))
    wrapped_phase_files = sorted(run_dir.glob("phases_wrapped_epoch*.npy"))
    return {
        "name": run_dir.name,
        "history": _load_json(run_dir / "history.json"),
        "test": _load_json(run_dir / "test_metrics.json"),
        "input": _complex_from_npz(npz, "input"),
        "pred": _complex_from_npz(npz, "pred"),
        "target": _complex_from_npz(npz, "target"),
        "raw_phase_files": raw_phase_files,
        "wrapped_phase_files": wrapped_phase_files,
    }


def get_runs(study_dir: str | Path) -> dict[str, dict]:
    study_dir = Path(study_dir)
    run_dirs = [run_dir for run_dir in study_dir.iterdir() if run_dir.is_dir() and is_complete_run(run_dir)]
    run_dirs.sort(key=lambda run_dir: (_parse_run_name(run_dir.name)[0], run_dir.name))
    return {run_dir.name: load_run(run_dir) for run_dir in run_dirs}


def _normalized_irradiance(field: np.ndarray) -> np.ndarray:
    intensity = np.abs(field) ** 2
    return intensity / max(float(intensity.max()), 1e-12)


def _phase_error(field: np.ndarray, target: np.ndarray) -> np.ndarray:
    diff = np.angle(field * np.conj(target))
    return (diff + np.pi) % (2 * np.pi) - np.pi


def _support_weights(target: np.ndarray, *, gamma: float = 2.0) -> np.ndarray:
    amp = np.abs(target)
    return np.power(amp / max(float(amp.max()), 1e-12), gamma)


def _leakage_map(pred: np.ndarray, target: np.ndarray, *, gamma: float = 2.0) -> np.ndarray:
    pred_i = _normalized_irradiance(pred)
    return pred_i * (1.0 - _support_weights(target, gamma=gamma))


def _save_figure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def plot_epoch_curves(runs: dict[str, dict], path: str | Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    series = [
        ("train_loss", "Training Loss", axes[0, 0]),
        ("complex_overlap", "Validation Complex Overlap", axes[0, 1]),
        ("full_field_phase_rmse_rad", "Full-Field Phase RMSE [rad]", axes[1, 0]),
        ("support_weighted_phase_rmse_rad", "Support-Weighted Phase RMSE [rad]", axes[1, 1]),
    ]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    for color, (name, run) in zip(colors, runs.items()):
        history = run["history"]
        label = format_run_label(name)
        for key, title, axis in series:
            epochs = [entry["epoch"] for entry in history if key in entry]
            values = [entry[key] for entry in history if key in entry]
            if not epochs:
                continue
            axis.plot(epochs, values, marker="o", linewidth=1.6, markersize=4, color=color, label=label)
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.grid(True, alpha=0.3)
            axis.legend(fontsize=8)

    fig.suptitle("Phase-First Dual-2f Study Epoch Curves", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save_figure(Path(path))


def plot_phase_metrics(runs: dict[str, dict], path: str | Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    metrics = [
        ("complex_overlap", "Complex Overlap"),
        ("full_field_phase_rmse_rad", "Full-Field Phase RMSE [rad]"),
        ("support_weighted_phase_rmse_rad", "Support-Weighted Phase RMSE [rad]"),
        ("out_of_support_energy_fraction", "Out-of-Support Energy Fraction"),
    ]
    names = list(runs)
    labels = [format_run_label(name) for name in names]

    for axis, (key, title) in zip(axes.flat, metrics):
        values = [runs[name]["test"][key] for name in names]
        axis.bar(labels, values, color=["#e74c3c", "#3498db"][: len(values)], edgecolor="black", linewidth=0.6)
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.3)
        for idx, value in enumerate(values):
            axis.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Phase-First Dual-2f Study Metrics", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save_figure(Path(path))


def plot_field_comparison(runs: dict[str, dict], path: str | Path) -> Path:
    fig, axes = plt.subplots(4, len(runs), figsize=(4.2 * len(runs), 11), squeeze=False)

    for col_idx, (name, run) in enumerate(runs.items()):
        pred = run["pred"]
        target = run["target"]
        images = [
            (_normalized_irradiance(target), "Target Irradiance", "inferno", 0.0, 1.0),
            (_normalized_irradiance(pred), "Prediction Irradiance", "inferno", 0.0, 1.0),
            (np.angle(pred), "Prediction Phase [rad]", "twilight_shifted", -math.pi, math.pi),
            (_phase_error(pred, target), "Phase Error [rad]", "twilight_shifted", -math.pi, math.pi),
        ]
        for row_idx, (image, title, cmap, vmin, vmax) in enumerate(images):
            im = axes[row_idx, col_idx].imshow(image, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(format_run_label(name))
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(title)
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            fig.colorbar(im, ax=axes[row_idx, col_idx], fraction=0.046, pad=0.02)

    fig.suptitle("Phase-First Dual-2f Field Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save_figure(Path(path))


def plot_support_and_leakage(runs: dict[str, dict], path: str | Path) -> Path:
    fig, axes = plt.subplots(2, len(runs), figsize=(4.2 * len(runs), 6.5), squeeze=False)

    for col_idx, (name, run) in enumerate(runs.items()):
        target = run["target"]
        pred = run["pred"]
        support = _support_weights(target, gamma=2.0)
        leakage = _leakage_map(pred, target, gamma=2.0)
        for row_idx, (image, title) in enumerate(((support, "Target Support Weights"), (leakage, "Out-of-Support Leakage"))):
            im = axes[row_idx, col_idx].imshow(image, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(format_run_label(name))
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(title)
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            fig.colorbar(im, ax=axes[row_idx, col_idx], fraction=0.046, pad=0.02)

    fig.suptitle("Support and Leakage Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save_figure(Path(path))


def plot_phase_masks_raw_vs_wrapped(runs: dict[str, dict], path: str | Path) -> Path:
    fig, axes = plt.subplots(len(runs), 2, figsize=(8.5, 4.0 * len(runs)), squeeze=False)

    for row_idx, (name, run) in enumerate(runs.items()):
        if not run["raw_phase_files"] or not run["wrapped_phase_files"]:
            raise ValueError(f"missing raw/wrapped phase snapshots for {name}")
        raw = np.load(run["raw_phase_files"][-1]).sum(axis=0)
        wrapped = np.load(run["wrapped_phase_files"][-1]).sum(axis=0)
        panels = (
            (raw, "Raw Summed Phase [rad]", "twilight_shifted", None, None),
            (wrapped, "Wrapped Summed Phase [rad]", "hsv", 0.0, 2.0 * math.pi),
        )
        for col_idx, (image, title, cmap, vmin, vmax) in enumerate(panels):
            im = axes[row_idx, col_idx].imshow(image, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row_idx, col_idx].set_title(f"{format_run_label(name)}: {title}")
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            fig.colorbar(im, ax=axes[row_idx, col_idx], fraction=0.046, pad=0.02)

    fig.suptitle("Raw vs Wrapped Learned Phase", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save_figure(Path(path))


def generate_figures(study_dir: str | Path, fig_dir: str | Path | None = None) -> list[Path]:
    study_dir = Path(study_dir)
    fig_dir = Path(fig_dir) if fig_dir is not None else study_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    runs = get_runs(study_dir)
    if not runs:
        raise ValueError(f"no complete study runs found in {study_dir}")

    return [
        plot_epoch_curves(runs, fig_dir / "fig1_epoch_curves.png"),
        plot_phase_metrics(runs, fig_dir / "fig2_phase_metrics.png"),
        plot_field_comparison(runs, fig_dir / "fig3_field_comparison.png"),
        plot_support_and_leakage(runs, fig_dir / "fig4_support_and_leakage.png"),
        plot_phase_masks_raw_vs_wrapped(runs, fig_dir / "fig5_phase_masks_raw_vs_wrapped.png"),
    ]
