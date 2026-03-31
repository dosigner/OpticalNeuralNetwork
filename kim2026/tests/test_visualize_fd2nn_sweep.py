from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "visualize_fd2nn_sweep.py"
VIZ_MODULE_PATH = Path(__file__).resolve().parent.parent / "src" / "kim2026" / "viz" / "fd2nn_sweep.py"


def load_visualizer_module():
    spec = importlib.util.spec_from_file_location("visualize_fd2nn_sweep", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_visualizer_script():
    spec = importlib.util.spec_from_file_location("visualize_fd2nn_sweep_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_visualizer_viz_module():
    spec = importlib.util.spec_from_file_location("kim2026_viz_fd2nn_sweep", VIZ_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_run(
    sweep_dir: Path,
    name: str,
    *,
    spacing_mm: float,
    shape: tuple[int, int] = (16, 16),
    complete: bool = True,
) -> None:
    run_dir = sweep_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)

    if not complete:
        np.save(run_dir / "phases_epoch000.npy", np.zeros((5, *shape), dtype=np.float32))
        return

    epochs = [0, 5, 10]
    history = []
    for idx, epoch in enumerate(epochs):
        history.append(
            {
                "epoch": epoch,
                "train_loss": 1.0 - 0.1 * idx - 0.01 * spacing_mm,
                "time_s": 0.2 + idx,
                "complex_overlap": 0.1 + 0.05 * idx + 0.001 * spacing_mm,
                "phase_rmse_rad": 1.8 - 0.1 * idx + 0.001 * spacing_mm,
                "intensity_overlap": 0.3 + 0.07 * idx,
                "strehl": 0.4 + 0.05 * idx,
                "baseline_complex_overlap": 0.18,
                "baseline_intensity_overlap": 0.92,
            }
        )
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))

    test_metrics = {
        "complex_overlap": 0.19 + 0.01 * spacing_mm,
        "phase_rmse_rad": 1.6 - 0.02 * spacing_mm,
        "amplitude_rmse": 0.15,
        "intensity_overlap": 0.7 - 0.01 * spacing_mm,
        "strehl": 1.1 + 0.02 * spacing_mm,
        "baseline_complex_overlap": 0.18,
        "baseline_intensity_overlap": 0.92,
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))

    y, x = np.mgrid[: shape[0], : shape[1]]
    cy = (shape[0] - 1) / 2
    cx = (shape[1] - 1) / 2
    rr = (x - cx) ** 2 + (y - cy) ** 2
    sigma = 5.0 + 0.2 * spacing_mm
    phase = 0.15 * (x - cx) + 0.05 * spacing_mm

    input_amp = np.exp(-rr / (2 * sigma**2))
    target_amp = np.exp(-rr / (2 * (sigma * 0.8) ** 2))
    pred_amp = np.exp(-rr / (2 * (sigma * 0.9) ** 2))

    input_field = input_amp * np.exp(1j * (phase + 0.2))
    target_field = target_amp * np.exp(1j * (phase * 0.5))
    pred_field = pred_amp * np.exp(1j * (phase * 0.7))

    np.savez(
        run_dir / "sample_fields.npz",
        input_real=input_field.real.astype(np.float32),
        input_imag=input_field.imag.astype(np.float32),
        pred_real=pred_field.real.astype(np.float32),
        pred_imag=pred_field.imag.astype(np.float32),
        target_real=target_field.real.astype(np.float32),
        target_imag=target_field.imag.astype(np.float32),
    )
    np.save(run_dir / "phases_epoch000.npy", np.zeros((5, *shape), dtype=np.float32))
    np.save(run_dir / "phases_epoch010.npy", np.ones((5, *shape), dtype=np.float32) * 0.5)
    np.save(run_dir / "phases_epoch029.npy", np.ones((5, *shape), dtype=np.float32))


def test_get_runs_skips_incomplete_and_sorts_numerically(tmp_path):
    mod = load_visualizer_module()
    sweep_dir = tmp_path / "sweep"
    write_run(sweep_dir, "spacing_10mm", spacing_mm=10.0)
    write_run(sweep_dir, "spacing_1mm", spacing_mm=1.0)
    write_run(sweep_dir, "spacing_5mm", spacing_mm=5.0)
    write_run(sweep_dir, "spacing_0mm", spacing_mm=0.0)
    write_run(sweep_dir, "spacing_0um", spacing_mm=0.0, complete=False)
    write_run(sweep_dir, "spacing_0p1mm", spacing_mm=0.1, complete=False)

    runs = mod.get_runs(sweep_dir)

    assert list(runs) == ["spacing_0mm", "spacing_1mm", "spacing_5mm", "spacing_10mm"]


def test_build_field_columns_include_input_vacuum_and_all_predictions(tmp_path):
    mod = load_visualizer_module()
    sweep_dir = tmp_path / "sweep"
    write_run(sweep_dir, "spacing_0mm", spacing_mm=0.0)
    write_run(sweep_dir, "spacing_1mm", spacing_mm=1.0)
    write_run(sweep_dir, "spacing_5mm", spacing_mm=5.0)
    write_run(sweep_dir, "spacing_10mm", spacing_mm=10.0)

    runs = mod.get_runs(sweep_dir)
    columns = mod.build_field_columns(runs)
    row_specs = mod.build_field_row_specs(columns)

    assert [column["name"] for column in columns] == [
        "input",
        "target",
        "spacing_0mm",
        "spacing_1mm",
        "spacing_5mm",
        "spacing_10mm",
    ]
    assert [column["label"] for column in columns[:2]] == ["Turbulent Input", "Vacuum Target"]
    assert [row["key"] for row in row_specs] == [
        "irradiance",
        "irradiance_error",
        "phase",
        "phase_error",
    ]
    assert columns[1]["field"].shape == (16, 16)


def test_generate_figures_writes_expected_pngs(tmp_path):
    mod = load_visualizer_module()
    sweep_dir = tmp_path / "sweep"
    fig_dir = tmp_path / "figures"
    write_run(sweep_dir, "spacing_0mm", spacing_mm=0.0)
    write_run(sweep_dir, "spacing_1mm", spacing_mm=1.0)
    write_run(sweep_dir, "spacing_5mm", spacing_mm=5.0)
    write_run(sweep_dir, "spacing_10mm", spacing_mm=10.0)

    output_paths = mod.generate_figures(sweep_dir=sweep_dir, fig_dir=fig_dir)

    expected = {
        "fig1_epoch_curves.png",
        "fig2_test_metrics.png",
        "fig3_field_full_comparison.png",
        "fig4_field_zoom_comparison.png",
        "fig5_field_profiles.png",
        "fig6_phase_masks.png",
    }
    assert {path.name for path in output_paths} == expected
    for name in expected:
        assert (fig_dir / name).exists(), name


def test_visualizer_best_run_uses_leakage_gated_overlap_first_selection(tmp_path):
    mod = load_visualizer_viz_module()
    sweep_dir = tmp_path / "sweep"
    write_run(sweep_dir, "spacing_1mm", spacing_mm=1.0)
    write_run(sweep_dir, "spacing_5mm", spacing_mm=5.0)

    leaky_metrics = json.loads((sweep_dir / "spacing_1mm" / "test_metrics.json").read_text())
    leaky_metrics.update(
        {
            "complex_overlap": 0.95,
            "intensity_overlap": 0.95,
            "support_weighted_phase_rmse_rad": 0.40,
            "out_of_support_energy_fraction": 0.18,
        }
    )
    (sweep_dir / "spacing_1mm" / "test_metrics.json").write_text(json.dumps(leaky_metrics, indent=2))

    safe_metrics = json.loads((sweep_dir / "spacing_5mm" / "test_metrics.json").read_text())
    safe_metrics.update(
        {
            "complex_overlap": 0.70,
            "intensity_overlap": 0.88,
            "support_weighted_phase_rmse_rad": 0.55,
            "out_of_support_energy_fraction": 0.10,
        }
    )
    (sweep_dir / "spacing_5mm" / "test_metrics.json").write_text(json.dumps(safe_metrics, indent=2))

    runs = mod.get_runs(sweep_dir)

    assert mod._best_run_name(runs) == "spacing_5mm"


def test_visualizer_defaults_to_dual2f_sweep_dir() -> None:
    mod = load_visualizer_script()

    assert mod.DEFAULT_SWEEP_DIR.name == "fd2nn_metasurface_sweep_dual2f"
