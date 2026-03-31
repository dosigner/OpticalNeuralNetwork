from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VIZ_MODULE_PATH = ROOT / "src" / "kim2026" / "viz" / "d2nn_beamreducer_sweep.py"
SCRIPT_PATH = ROOT / "scripts" / "visualize_d2nn_beamreducer_sweep.py"


def load_viz_module():
    spec = importlib.util.spec_from_file_location("d2nn_beamreducer_sweep", VIZ_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_script_module():
    spec = importlib.util.spec_from_file_location("visualize_d2nn_beamreducer_sweep", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_run(
    run_dir: Path,
    *,
    overlap: float,
    strehl: float,
    shape: tuple[int, int] = (24, 24),
    complete: bool = True,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    if not complete:
        (run_dir / "evaluation.json").write_text("{}", encoding="utf-8")
        return

    evaluation = {
        "baseline": {"overlap": overlap - 0.1, "strehl": max(strehl - 0.5, 0.0)},
        "model": {"overlap": overlap, "strehl": strehl},
    }
    (run_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")

    y, x = np.mgrid[: shape[0], : shape[1]]
    cy = (shape[0] - 1) / 2
    cx = (shape[1] - 1) / 2
    rr = (x - cx) ** 2 + (y - cy) ** 2
    sigma = 6.0
    phase = 0.12 * (x - cx)

    input_amp = np.exp(-rr / (2 * (sigma * 1.2) ** 2))
    vacuum_amp = np.exp(-rr / (2 * (sigma * 1.0) ** 2))
    baseline_amp = np.exp(-rr / (2 * (sigma * 1.1) ** 2))
    pred_amp = np.exp(-rr / (2 * (sigma * 0.9) ** 2))
    target_amp = np.exp(-rr / (2 * (sigma * 0.8) ** 2))

    input_field = input_amp * np.exp(1j * (phase + 0.3))
    vacuum_field = vacuum_amp * np.exp(1j * (phase * 0.2))
    baseline_field = baseline_amp * np.exp(1j * (phase * 0.6))
    pred_field = pred_amp * np.exp(1j * (phase * 0.7))
    target_field = target_amp * np.exp(1j * (phase * 0.5))

    np.savez(
        run_dir / "sample_fields.npz",
        input_real=input_field.real.astype(np.float32),
        input_imag=input_field.imag.astype(np.float32),
        vacuum_real=vacuum_field.real.astype(np.float32),
        vacuum_imag=vacuum_field.imag.astype(np.float32),
        baseline_real=baseline_field.real.astype(np.float32),
        baseline_imag=baseline_field.imag.astype(np.float32),
        pred_real=pred_field.real.astype(np.float32),
        pred_imag=pred_field.imag.astype(np.float32),
        target_real=target_field.real.astype(np.float32),
        target_imag=target_field.imag.astype(np.float32),
    )


def test_get_stage_runs_sorts_numerically_and_skips_incomplete(tmp_path: Path) -> None:
    mod = load_viz_module()
    stage1_dir = tmp_path / "stage1"
    stage2_dir = tmp_path / "stage2"

    write_run(stage1_dir / "ls100mm_dd10mm", overlap=0.63, strehl=2.0)
    write_run(stage1_dir / "ls10mm_dd50mm", overlap=0.64, strehl=2.1)
    write_run(stage1_dir / "ls50mm_dd50mm", overlap=0.68, strehl=2.3)
    write_run(stage1_dir / "ls10mm_dd10mm", overlap=0.62, strehl=2.2, complete=False)

    write_run(stage2_dir / "pitch5um", overlap=0.66, strehl=2.4)
    write_run(stage2_dir / "pitch2um", overlap=0.63, strehl=2.1)
    write_run(stage2_dir / "pitch3um", overlap=0.64, strehl=2.2)

    stage1_runs = mod.get_stage1_runs(stage1_dir)
    stage2_runs = mod.get_stage2_runs(stage2_dir)

    assert list(stage1_runs) == ["ls10mm_dd50mm", "ls50mm_dd50mm", "ls100mm_dd10mm"]
    assert list(stage2_runs) == ["pitch2um", "pitch3um", "pitch5um"]


def test_generate_figures_writes_expected_pngs(tmp_path: Path) -> None:
    mod = load_viz_module()
    stage1_dir = tmp_path / "stage1"
    stage2_dir = tmp_path / "stage2"
    fig_dir = tmp_path / "figures"

    write_run(stage1_dir / "ls10mm_dd10mm", overlap=0.62, strehl=2.0)
    write_run(stage1_dir / "ls50mm_dd50mm", overlap=0.68, strehl=2.4)
    write_run(stage1_dir / "ls100mm_dd100mm", overlap=0.58, strehl=1.8)

    write_run(stage2_dir / "pitch2um", overlap=0.63, strehl=2.1)
    write_run(stage2_dir / "pitch3um", overlap=0.64, strehl=2.2)
    write_run(stage2_dir / "pitch5um", overlap=0.66, strehl=2.5)

    output_paths = mod.generate_figures(stage1_dir=stage1_dir, stage2_dir=stage2_dir, fig_dir=fig_dir)

    expected = {
        "fig1_stage1_test_metrics.png",
        "fig2_stage1_best_fields.png",
        "fig3_stage2_test_metrics.png",
        "fig4_stage2_best_fields.png",
    }
    assert {path.name for path in output_paths} == expected
    for name in expected:
        assert (fig_dir / name).exists(), name


def test_visualizer_defaults_to_coarse_sweep_dirs() -> None:
    mod = load_script_module()

    assert mod.DEFAULT_STAGE1_DIR.name == "06_d2nn_beamreducer_distance-sweep_pitch-3um_codex"
    assert mod.DEFAULT_STAGE2_DIR.name == "07_d2nn_beamreducer_pitch-sweep_codex"
    assert mod.DEFAULT_FIG_DIR.name == "d2nn_beamreducer_figures"
