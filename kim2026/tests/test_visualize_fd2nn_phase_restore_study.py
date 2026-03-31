from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "visualize_fd2nn_phase_restore_study.py"


def load_visualizer_module():
    spec = importlib.util.spec_from_file_location("visualize_fd2nn_phase_restore_study", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_run(study_dir: Path, name: str, *, roi_n: int) -> None:
    run_dir = study_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)

    history = [
        {
            "epoch": 0,
            "train_loss": 1.0,
            "complex_overlap": 0.2,
            "phase_rmse_rad": 1.4,
            "full_field_phase_rmse_rad": 2.7,
            "support_weighted_phase_rmse_rad": 0.8,
            "out_of_support_energy_fraction": 0.35,
        },
        {
            "epoch": 5,
            "train_loss": 0.8,
            "complex_overlap": 0.3,
            "phase_rmse_rad": 1.1,
            "full_field_phase_rmse_rad": 2.3,
            "support_weighted_phase_rmse_rad": 0.5,
            "out_of_support_energy_fraction": 0.2,
        },
    ]
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))

    test_metrics = {
        "complex_overlap": 0.31,
        "phase_rmse_rad": 1.05,
        "full_field_phase_rmse_rad": 2.2,
        "support_weighted_phase_rmse_rad": 0.48,
        "out_of_support_energy_fraction": 0.19,
        "amplitude_rmse": 0.13,
        "intensity_overlap": 0.66,
        "strehl": 2.4,
        "baseline_complex_overlap": 0.18,
        "baseline_intensity_overlap": 0.92,
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))

    y, x = np.mgrid[:roi_n, :roi_n]
    cy = (roi_n - 1) / 2
    cx = (roi_n - 1) / 2
    rr = (x - cx) ** 2 + (y - cy) ** 2
    sigma = roi_n / 8.0
    phase = 0.002 * (x - cx)

    input_amp = np.exp(-rr / (2 * (sigma * 1.2) ** 2))
    target_amp = np.exp(-rr / (2 * sigma**2))
    pred_amp = np.exp(-rr / (2 * (sigma * 0.9) ** 2))

    input_field = input_amp * np.exp(1j * (phase + 0.4))
    target_field = target_amp * np.exp(1j * phase)
    pred_field = pred_amp * np.exp(1j * (phase * 0.8))

    np.savez(
        run_dir / "sample_fields.npz",
        input_real=input_field.real.astype(np.float32),
        input_imag=input_field.imag.astype(np.float32),
        pred_real=pred_field.real.astype(np.float32),
        pred_imag=pred_field.imag.astype(np.float32),
        target_real=target_field.real.astype(np.float32),
        target_imag=target_field.imag.astype(np.float32),
    )
    np.save(run_dir / "phases_raw_epoch000.npy", np.ones((5, roi_n, roi_n), dtype=np.float32) * 7.0)
    np.save(run_dir / "phases_wrapped_epoch000.npy", np.ones((5, roi_n, roi_n), dtype=np.float32) * 0.7)


def test_generate_phase_restore_figures(tmp_path):
    mod = load_visualizer_module()
    study_dir = tmp_path / "study"
    fig_dir = tmp_path / "figures"
    write_run(study_dir, "roi512_spacing_0p1mm", roi_n=32)
    write_run(study_dir, "roi1024_spacing_0p1mm", roi_n=48)

    output_paths = mod.generate_figures(study_dir=study_dir, fig_dir=fig_dir)

    expected = {
        "fig1_epoch_curves.png",
        "fig2_phase_metrics.png",
        "fig3_field_comparison.png",
        "fig4_support_and_leakage.png",
        "fig5_phase_masks_raw_vs_wrapped.png",
    }
    assert {path.name for path in output_paths} == expected
    for name in expected:
        assert (fig_dir / name).exists()


def test_phase_restore_visualizer_defaults_to_new_study_dir() -> None:
    mod = load_visualizer_module()

    assert mod.DEFAULT_STUDY_DIR.name == "fd2nn_phase_restore_dual2f_codex"
