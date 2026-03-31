from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from kim2026.config.schema import load_and_validate_config


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH = ROOT / "configs" / "fso_1024_train_d2nn_beamreducer_base.yaml"
RUNNER_PATH = ROOT / "scripts" / "run_d2nn_beamreducer_coarse_sweep.py"


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("run_d2nn_beamreducer_coarse_sweep", RUNNER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_evaluation_json(run_dir: Path, *, overlap: float, strehl: float = 0.0) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "baseline": {"overlap": 0.1, "strehl": 0.05},
                "model": {"overlap": overlap, "strehl": strehl},
            }
        ),
        encoding="utf-8",
    )


def test_base_config_uses_stage1_beamreducer_regime() -> None:
    cfg = load_and_validate_config(BASE_CONFIG_PATH)

    assert cfg["model"]["type"] == "d2nn"
    assert cfg["grid"]["n"] == 1024
    assert cfg["model"]["layer_spacing_m"] <= 0.10
    assert cfg["model"]["detector_distance_m"] <= 0.10
    assert cfg["grid"]["receiver_window_m"] == pytest.approx(0.003072)


def test_matrix_stage1_runs_cover_distance_grid_at_3um() -> None:
    module = _load_runner_module()

    stage1_runs = module.build_stage1_runs()

    assert len(stage1_runs) == 9
    assert {run["name"] for run in stage1_runs} == {
        "ls10mm_dd10mm",
        "ls10mm_dd50mm",
        "ls10mm_dd100mm",
        "ls50mm_dd10mm",
        "ls50mm_dd50mm",
        "ls50mm_dd100mm",
        "ls100mm_dd10mm",
        "ls100mm_dd50mm",
        "ls100mm_dd100mm",
    }
    assert {run["pitch_um"] for run in stage1_runs} == {3.0}
    for run in stage1_runs:
        assert run["receiver_window_m"] == pytest.approx(0.003072)


def test_matrix_stage2_runs_cover_pitch_grid_with_derived_windows() -> None:
    module = _load_runner_module()

    stage2_runs = module.build_stage2_runs(best_layer_spacing_m=0.01, best_detector_distance_m=0.05)

    assert len(stage2_runs) == 3
    assert [run["name"] for run in stage2_runs] == ["pitch2um", "pitch3um", "pitch5um"]
    assert [run["pitch_um"] for run in stage2_runs] == [2.0, 3.0, 5.0]
    assert [run["receiver_window_m"] for run in stage2_runs] == [
        pytest.approx(0.002048),
        pytest.approx(0.003072),
        pytest.approx(0.00512),
    ]
    assert {run["layer_spacing_m"] for run in stage2_runs} == {0.01}
    assert {run["detector_distance_m"] for run in stage2_runs} == {0.05}


def test_materialize_stage1_config_writes_run_yaml(tmp_path: Path) -> None:
    module = _load_runner_module()
    run = module.build_stage1_runs()[4]
    stage_dir = tmp_path / "stage1"

    config_path = module.materialize_stage1_config(
        run,
        base_config_path=BASE_CONFIG_PATH,
        stage_dir=stage_dir,
    )

    cfg = load_and_validate_config(config_path)

    assert config_path == stage_dir / run["name"] / "config.yaml"
    assert cfg["grid"]["receiver_window_m"] == pytest.approx(run["receiver_window_m"])
    assert cfg["model"]["layer_spacing_m"] == pytest.approx(run["layer_spacing_m"])
    assert cfg["model"]["detector_distance_m"] == pytest.approx(run["detector_distance_m"])
    assert Path(cfg["experiment"]["save_dir"]) == stage_dir / run["name"]


def test_materialize_stage2_config_writes_run_yaml(tmp_path: Path) -> None:
    module = _load_runner_module()
    run = module.build_stage2_runs(best_layer_spacing_m=0.05, best_detector_distance_m=0.10)[2]
    stage_dir = tmp_path / "stage2"

    config_path = module.materialize_stage2_config(
        run,
        base_config_path=BASE_CONFIG_PATH,
        stage_dir=stage_dir,
    )

    cfg = load_and_validate_config(config_path)

    assert config_path == stage_dir / run["name"] / "config.yaml"
    assert cfg["grid"]["receiver_window_m"] == pytest.approx(0.00512)
    assert cfg["model"]["layer_spacing_m"] == pytest.approx(0.05)
    assert cfg["model"]["detector_distance_m"] == pytest.approx(0.10)
    assert Path(cfg["experiment"]["save_dir"]) == stage_dir / run["name"]


def test_cli_stage1_dry_run_prints_9_planned_runs(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_runner_module()

    module.main(["--stage", "stage1", "--dry-run"])

    out = capsys.readouterr().out
    assert "Planned 9 runs for stage1" in out
    assert out.count("config:") == 9
    assert "receiver_window_m=0.003072" in out


def test_cli_stage2_dry_run_prints_3_planned_runs(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_runner_module()

    module.main(
        [
            "--stage",
            "stage2",
            "--best-layer-spacing-mm",
            "10",
            "--best-detector-distance-mm",
            "50",
            "--dry-run",
        ]
    )

    out = capsys.readouterr().out
    assert "Planned 3 runs for stage2" in out
    assert out.count("config:") == 3
    assert "receiver_window_m=0.005120" in out


def test_aggregate_stage1_summary_ranks_by_overlap_and_records_best_distances(tmp_path: Path) -> None:
    module = _load_runner_module()
    runs = module.build_stage1_runs()
    stage_dir = tmp_path / "stage1"

    _write_evaluation_json(stage_dir / "ls10mm_dd10mm", overlap=0.62)
    _write_evaluation_json(stage_dir / "ls10mm_dd50mm", overlap=0.65)
    _write_evaluation_json(stage_dir / "ls10mm_dd100mm", overlap=0.61)
    _write_evaluation_json(stage_dir / "ls50mm_dd10mm", overlap=0.63)
    _write_evaluation_json(stage_dir / "ls50mm_dd50mm", overlap=0.68)
    _write_evaluation_json(stage_dir / "ls50mm_dd100mm", overlap=0.64)
    _write_evaluation_json(stage_dir / "ls100mm_dd10mm", overlap=0.60)
    _write_evaluation_json(stage_dir / "ls100mm_dd50mm", overlap=0.59)
    _write_evaluation_json(stage_dir / "ls100mm_dd100mm", overlap=0.58)

    summary = module.aggregate_stage_summary(stage="stage1", runs=runs, stage_dir=stage_dir)

    assert summary["best_run_name"] == "ls50mm_dd50mm"
    assert summary["best_layer_spacing_m"] == pytest.approx(0.05)
    assert summary["best_detector_distance_m"] == pytest.approx(0.05)
    assert summary["runs"][0]["primary_metric_value"] == pytest.approx(0.68)
    assert (stage_dir / "stage_summary.json").exists()


def test_aggregate_stage2_summary_ranks_by_overlap_and_records_best_pitch(tmp_path: Path) -> None:
    module = _load_runner_module()
    runs = module.build_stage2_runs(best_layer_spacing_m=0.05, best_detector_distance_m=0.05)
    stage_dir = tmp_path / "stage2"

    _write_evaluation_json(stage_dir / "pitch2um", overlap=0.63)
    _write_evaluation_json(stage_dir / "pitch3um", overlap=0.64)
    _write_evaluation_json(stage_dir / "pitch5um", overlap=0.66)

    summary = module.aggregate_stage_summary(stage="stage2", runs=runs, stage_dir=stage_dir)

    assert summary["best_run_name"] == "pitch5um"
    assert summary["best_pitch_um"] == pytest.approx(5.0)
    assert summary["runs"][0]["primary_metric_value"] == pytest.approx(0.66)
    assert (stage_dir / "stage_summary.json").exists()
