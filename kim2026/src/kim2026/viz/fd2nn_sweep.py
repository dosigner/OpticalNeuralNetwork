"""Visualization helpers for FD2NN metasurface sweep outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from kim2026.training.metrics import beam_cleanup_selection_sort_key

REQUIRED_RUN_FILES = ("history.json", "test_metrics.json", "sample_fields.npz")

plt.rcParams.update({
    "font.size": 10,
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})

COLOR_MAP = {
    "input": "#222222",
    "target": "#1f77b4",
    "spacing_0mm": "#888888",
    "spacing_0p1mm": "#e74c3c",
    "spacing_1mm": "#e74c3c",
    "spacing_2mm": "#3498db",
    "spacing_3mm": "#2ecc71",
    "spacing_5mm": "#9b59b6",
    "spacing_6mm": "#3498db",
    "spacing_10mm": "#f39c12",
    "spacing_12mm": "#9b59b6",
    "spacing_25mm": "#f39c12",
    "spacing_50mm": "#e67e22",
}


def _load_json(path: Path) -> dict | list:
    with open(path) as f:
        return json.load(f)


def parse_spacing_mm(name: str) -> float:
    token = name.removeprefix("spacing_")
    if token.endswith("mm"):
        return float(token[:-2].replace("p", "."))
    if token.endswith("um"):
        return float(token[:-2].replace("p", ".")) / 1000.0
    return math.inf


def format_run_label(name: str) -> str:
    spacing_mm = parse_spacing_mm(name)
    if not math.isfinite(spacing_mm):
        return name
    if spacing_mm == 0:
        return "0 (FFT only)"
    return f"{spacing_mm:g} mm"


def is_complete_run(run_dir: Path) -> bool:
    return all((run_dir / file_name).exists() for file_name in REQUIRED_RUN_FILES)


def _complex_from_npz(npz: np.lib.npyio.NpzFile, prefix: str) -> np.ndarray:
    return npz[f"{prefix}_real"] + 1j * npz[f"{prefix}_imag"]


def load_run(run_dir: Path) -> dict:
    npz = np.load(run_dir / "sample_fields.npz")
    return {
        "name": run_dir.name,
        "history": _load_json(run_dir / "history.json"),
        "test": _load_json(run_dir / "test_metrics.json"),
        "input": _complex_from_npz(npz, "input"),
        "pred": _complex_from_npz(npz, "pred"),
        "target": _complex_from_npz(npz, "target"),
        "phase_files": sorted(run_dir.glob("phases_epoch*.npy")),
    }


def get_runs(sweep_dir: str | Path) -> dict[str, dict]:
    sweep_dir = Path(sweep_dir)
    run_dirs = [
        run_dir
        for run_dir in sweep_dir.iterdir()
        if run_dir.is_dir() and run_dir.name.startswith("spacing_") and is_complete_run(run_dir)
    ]
    run_dirs.sort(key=lambda run_dir: (parse_spacing_mm(run_dir.name), run_dir.name))
    return {run_dir.name: load_run(run_dir) for run_dir in run_dirs}


def build_field_columns(runs: dict[str, dict]) -> list[dict]:
    if not runs:
        raise ValueError("runs must not be empty")

    first = next(iter(runs.values()))
    columns = [
        {"name": "input", "label": "Turbulent Input", "field": first["input"], "source": "input"},
        {"name": "target", "label": "Vacuum Target", "field": first["target"], "source": "target"},
    ]
    for name, run in runs.items():
        columns.append({"name": name, "label": format_run_label(name), "field": run["pred"], "source": name})
    return columns


def build_field_row_specs(columns: list[dict]) -> list[dict]:
    if not columns:
        raise ValueError("columns must not be empty")
    return [
        {"key": "irradiance", "title": "Normalized Irradiance"},
        {"key": "irradiance_error", "title": "Irradiance Error vs Vacuum"},
        {"key": "phase", "title": "Phase [rad]"},
        {"key": "phase_error", "title": "Phase Error vs Vacuum [rad]"},
    ]


def _center_crop(array: np.ndarray, radius: int | None) -> np.ndarray:
    if not radius:
        return array
    radius = min(int(radius), array.shape[0] // 2, array.shape[1] // 2)
    cy = array.shape[0] // 2
    cx = array.shape[1] // 2
    return array[cy - radius:cy + radius, cx - radius:cx + radius]


def _normalized_irradiance(field: np.ndarray, reference_max: float) -> np.ndarray:
    return np.abs(field) ** 2 / max(reference_max, 1e-12)


def _masked_phase(field: np.ndarray, threshold: float) -> np.ma.MaskedArray:
    irradiance = np.abs(field) ** 2
    return np.ma.masked_where(irradiance < threshold, np.angle(field))


def _phase_error(field: np.ndarray, target: np.ndarray, threshold: float) -> np.ma.MaskedArray:
    field_i = np.abs(field) ** 2
    target_i = np.abs(target) ** 2
    phase_err = np.angle(field * np.conj(target))
    return np.ma.masked_where((field_i < threshold) | (target_i < threshold), phase_err)


def _row_image(
    key: str,
    field: np.ndarray,
    *,
    target: np.ndarray,
    reference_max: float,
    phase_threshold: float,
) -> tuple[np.ndarray | np.ma.MaskedArray, dict]:
    if key == "irradiance":
        return _normalized_irradiance(field, reference_max), {"cmap": "inferno", "vmin": 0.0, "vmax": 1.0}
    if key == "irradiance_error":
        image = np.abs(_normalized_irradiance(field, reference_max) - _normalized_irradiance(target, reference_max))
        return image, {"cmap": "viridis", "vmin": 0.0, "vmax": 0.5}
    if key == "phase":
        return _masked_phase(field, phase_threshold), {
            "cmap": "twilight_shifted",
            "vmin": -math.pi,
            "vmax": math.pi,
        }
    if key == "phase_error":
        return _phase_error(field, target, phase_threshold), {
            "cmap": "twilight_shifted",
            "vmin": -math.pi,
            "vmax": math.pi,
        }
    raise ValueError(f"unknown image key: {key}")


def _series_color(source: str) -> str:
    return COLOR_MAP.get(source, "#444444")


def _centerline(field: np.ndarray, *, reference_max: float) -> np.ndarray:
    irradiance = _normalized_irradiance(field, reference_max)
    return irradiance[irradiance.shape[0] // 2]


def _radial_profile(field: np.ndarray, *, reference_max: float) -> np.ndarray:
    irradiance = _normalized_irradiance(field, reference_max)
    cy = irradiance.shape[0] // 2
    cx = irradiance.shape[1] // 2
    yy, xx = np.mgrid[:irradiance.shape[0], :irradiance.shape[1]]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(int)
    r_max = rr.max() + 1
    radial = np.zeros(r_max, dtype=np.float64)
    for radius in range(r_max):
        mask = rr == radius
        radial[radius] = irradiance[mask].mean() if mask.any() else 0.0
    return radial


def _phase_centerline(field: np.ndarray, target: np.ndarray, *, threshold: float) -> np.ndarray:
    phase_err = _phase_error(field, target, threshold).filled(np.nan)
    return phase_err[phase_err.shape[0] // 2]


def _best_run_name(runs: dict[str, dict]) -> str:
    def sort_key(name: str) -> tuple[int, float, float]:
        metrics = runs[name]["test"]
        if "intensity_overlap" not in metrics:
            return (1, float(metrics.get("complex_overlap", 0.0)), 0.0)
        return beam_cleanup_selection_sort_key(metrics)

    return max(runs, key=sort_key)


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_epoch_curves(runs: dict[str, dict], path: str | Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), squeeze=False)
    series = [
        ("train_loss", "Training Loss", axes[0, 0]),
        ("complex_overlap", "Validation Complex Overlap", axes[0, 1]),
        ("phase_rmse_rad", "Validation Phase RMSE [rad]", axes[1, 0]),
        ("intensity_overlap", "Validation Intensity Overlap", axes[1, 1]),
    ]

    for name, run in runs.items():
        history = run["history"]
        color = _series_color(name)
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

    baseline_co = next(iter(runs.values()))["test"]["baseline_complex_overlap"]
    baseline_io = next(iter(runs.values()))["test"]["baseline_intensity_overlap"]
    axes[0, 1].axhline(baseline_co, color="black", linestyle="--", linewidth=1.0, label=f"Baseline {baseline_co:.3f}")
    axes[1, 1].axhline(baseline_io, color="black", linestyle="--", linewidth=1.0, label=f"Baseline {baseline_io:.3f}")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 1].set_ylabel("Complex Overlap")
    axes[1, 0].set_ylabel("RMSE [rad]")
    axes[1, 1].set_ylabel("Intensity Overlap")
    for axis in axes.flat:
        axis.legend(fontsize=8)
    fig.suptitle("FD2NN Sweep Epoch Curves", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_figure(Path(path))
    return Path(path)


def plot_test_summary(runs: dict[str, dict], path: str | Path) -> Path:
    names = list(runs)
    labels = [format_run_label(name) for name in names]
    colors = [_series_color(name) for name in names]
    metrics = {
        "complex_overlap": ("Complex Overlap", next(iter(runs.values()))["test"]["baseline_complex_overlap"]),
        "phase_rmse_rad": ("Phase RMSE [rad]", None),
        "intensity_overlap": ("Intensity Overlap", next(iter(runs.values()))["test"]["baseline_intensity_overlap"]),
        "strehl": ("Strehl Ratio", 1.0),
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for axis, (key, (title, baseline)) in zip(axes.flat, metrics.items()):
        values = [runs[name]["test"][key] for name in names]
        bars = axis.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
        if baseline is not None:
            axis.axhline(baseline, color="black", linestyle="--", linewidth=1.0)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=8)
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.25)
    fig.suptitle("FD2NN Sweep Test Metrics", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_figure(Path(path))
    return Path(path)


def plot_field_comparison(columns: list[dict], row_specs: list[dict], path: str | Path, *, crop_radius: int | None = None) -> Path:
    target = next(column["field"] for column in columns if column["name"] == "target")
    reference_max = max(float((np.abs(column["field"]) ** 2).max()) for column in columns)
    phase_threshold = max(reference_max * 1e-3, 1e-12)
    fig, axes = plt.subplots(len(row_specs), len(columns), figsize=(3.1 * len(columns), 2.8 * len(row_specs)), squeeze=False)

    for col_idx, column in enumerate(columns):
        for row_idx, product in enumerate(row_specs):
            image, kwargs = _row_image(
                product["key"],
                column["field"],
                target=target,
                reference_max=reference_max,
                phase_threshold=phase_threshold,
            )
            image = _center_crop(image, crop_radius)
            im = axes[row_idx, col_idx].imshow(image, origin="lower", **kwargs)
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(column["label"])
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(product["title"])
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
        fig.colorbar(im, ax=axes[row_idx, :].tolist(), shrink=0.72, pad=0.01)

    label = "Center Zoom" if crop_radius else "Full Field"
    fig.suptitle(f"FD2NN Sample Field Comparison ({label})", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_figure(Path(path))
    return Path(path)


def plot_field_profiles(spec: dict, path: str | Path, *, crop_radius: int | None = None) -> Path:
    columns = spec
    target = next(column["field"] for column in columns if column["name"] == "target")
    reference_max = max(float((np.abs(column["field"]) ** 2).max()) for column in columns)
    phase_threshold = max(reference_max * 1e-3, 1e-12)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for column in columns:
        field = column["field"]
        if crop_radius:
            field = _center_crop(field, crop_radius)
        x = np.arange(field.shape[1]) - field.shape[1] // 2
        irradiance_line = _centerline(field, reference_max=reference_max)
        phase_line = _phase_centerline(field, _center_crop(target, crop_radius), threshold=phase_threshold)
        radial = _radial_profile(field, reference_max=reference_max)
        color = _series_color(column["source"])
        label = column["label"]

        axes[0, 0].plot(x, irradiance_line, linewidth=1.6, color=color, label=label)
        axes[0, 1].semilogy(x, np.clip(irradiance_line, 1e-6, None), linewidth=1.6, color=color, label=label)
        axes[1, 0].plot(radial, linewidth=1.6, color=color, label=label)
        axes[1, 1].plot(x, phase_line, linewidth=1.3, color=color, label=label)

    axes[0, 0].set(title="Centerline Irradiance", ylabel="I / Imax")
    axes[0, 1].set(title="Centerline Irradiance (log)", ylabel="I / Imax")
    axes[1, 0].set(title="Radial Irradiance Profile", xlabel="Radius [px]", ylabel="I / Imax")
    axes[1, 1].set(title="Centerline Phase Error vs Vacuum", xlabel="Pixel offset", ylabel="Phase [rad]")
    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    fig.suptitle("FD2NN Sample Field Profiles", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_figure(Path(path))
    return Path(path)


def plot_phase_masks(runs: dict[str, dict], path: str | Path) -> Path:
    best_name = _best_run_name(runs)
    phase_files = runs[best_name]["phase_files"]
    if not phase_files:
        raise ValueError("no phase snapshots available")

    selected = [phase_files[0], phase_files[len(phase_files) // 2], phase_files[-1]]
    first = np.load(selected[0])
    crop_radius = min(128, max(4, first.shape[-1] // 4))
    domains = ["F", "R", "F", "R", "F"]
    fig, axes = plt.subplots(len(selected), first.shape[0], figsize=(15, 3.2 * len(selected)), squeeze=False)
    for row_idx, phase_path in enumerate(selected):
        phases = np.load(phase_path)
        epoch = int(phase_path.stem.split("epoch")[1])
        for col_idx in range(phases.shape[0]):
            image = _center_crop(phases[col_idx], crop_radius)
            im = axes[row_idx, col_idx].imshow(
                image,
                cmap="twilight_shifted",
                origin="lower",
                vmin=-math.pi,
                vmax=math.pi,
            )
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(f"Layer {col_idx} ({domains[col_idx]})")
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(f"Epoch {epoch}")
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
    fig.colorbar(im, ax=axes[:, -1].tolist(), shrink=0.72, pad=0.01)
    fig.suptitle(f"Phase Mask Evolution ({format_run_label(best_name)})", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_figure(Path(path))
    return Path(path)


def generate_figures(sweep_dir: str | Path, fig_dir: str | Path | None = None) -> list[Path]:
    sweep_dir = Path(sweep_dir)
    fig_dir = Path(fig_dir) if fig_dir is not None else sweep_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    runs = get_runs(sweep_dir)
    if not runs:
        raise ValueError(f"no complete sweep runs found in {sweep_dir}")

    columns = build_field_columns(runs)
    row_specs = build_field_row_specs(columns)
    field_size = columns[0]["field"].shape[0]
    zoom_radius = min(256, max(8, field_size // 4))

    paths = [
        plot_epoch_curves(runs, fig_dir / "fig1_epoch_curves.png"),
        plot_test_summary(runs, fig_dir / "fig2_test_metrics.png"),
        plot_field_comparison(columns, row_specs, fig_dir / "fig3_field_full_comparison.png", crop_radius=None),
        plot_field_comparison(columns, row_specs, fig_dir / "fig4_field_zoom_comparison.png", crop_radius=zoom_radius),
        plot_field_profiles(columns, fig_dir / "fig5_field_profiles.png", crop_radius=zoom_radius),
        plot_phase_masks(runs, fig_dir / "fig6_phase_masks.png"),
    ]
    return paths
