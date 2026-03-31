from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "fd2nn-codex-run-interpreter"
    / "scripts"
    / "build_run_registry.py"
)


def load_module():
    assert SCRIPT_PATH.exists()
    spec = importlib.util.spec_from_file_location("build_run_registry", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_run_02(root: Path) -> None:
    run_dir = root / "runs" / "02_fd2nn_spacing-sweep_loss-old_roi-1024_codex"
    write_json(
        run_dir / "spacing_0mm" / "test_metrics.json",
        {
            "complex_overlap": 0.08,
            "intensity_overlap": 0.91,
            "strehl": 2.1,
        },
    )
    write_json(
        run_dir / "spacing_0p1mm" / "test_metrics.json",
        {
            "complex_overlap": 0.10,
            "intensity_overlap": 0.93,
            "strehl": 2.4,
        },
    )


def write_run_02_error_only(root: Path) -> None:
    run_dir = root / "runs" / "02_fd2nn_spacing-sweep_loss-old_roi-1024_codex"
    write_json(
        run_dir / "spacing_0mm" / "test_metrics.json",
        {
            "phase_rmse_rad": 1.5,
        },
    )
    write_json(
        run_dir / "spacing_0p1mm" / "test_metrics.json",
        {
            "phase_rmse_rad": 0.8,
        },
    )


def write_run_03(root: Path) -> None:
    run_dir = root / "runs" / "03_fd2nn_spacing-sweep_loss-shape_roi-1024_codex"
    write_json(
        run_dir / "sweep_summary.json",
        {
            "spacing_0mm": {
                "complex_overlap": 0.07,
                "intensity_overlap": 0.90,
            },
            "spacing_2mm": {
                "complex_overlap": 0.11,
                "intensity_overlap": 0.89,
            },
        },
    )


def write_run_04(root: Path) -> None:
    run_dir = root / "runs" / "04_fd2nn_spacing-sweep_loss-shape_roi-512_codex"
    write_json(
        run_dir / "sweep_summary.json",
        {
            "spacing_0p1mm": {
                "complex_overlap": 0.14,
                "intensity_overlap": 0.89,
            },
            "spacing_1mm": {
                "complex_overlap": 0.36,
                "intensity_overlap": 0.85,
            },
        },
    )


def write_run_05(root: Path) -> None:
    run_dir = root / "runs" / "05_fd2nn_roi-sweep_loss-phase-first_spacing-0p1mm_codex"
    write_json(
        run_dir / "study_summary.json",
        {
            "roi512_spacing_0p1mm": {
                "complex_overlap": 0.068,
                "support_weighted_phase_rmse_rad": 1.77,
                "out_of_support_energy_fraction": 0.225,
            },
            "roi1024_spacing_0p1mm": {
                "complex_overlap": 0.097,
                "support_weighted_phase_rmse_rad": 1.77,
                "out_of_support_energy_fraction": 0.121,
            },
        },
    )


def test_build_run_registry_reads_official_codex_runs(tmp_path: Path) -> None:
    write_run_02(tmp_path)
    write_run_03(tmp_path)
    write_run_04(tmp_path)
    write_run_05(tmp_path)

    mod = load_module()
    registry = mod.build_run_registry(tmp_path)

    assert set(registry) == {"02", "03", "04", "05"}

    run02 = registry["02"]
    assert run02["sweep_axis"] == "spacing-sweep"
    assert run02["fixed_conditions"]["roi"] == 1024
    assert run02["compared_conditions"] == ["spacing_0mm", "spacing_0p1mm", "spacing_1mm", "spacing_2mm"]
    assert run02["available_conditions"] == ["spacing_0mm", "spacing_0p1mm"]
    assert run02["missing_conditions"] == ["spacing_1mm", "spacing_2mm"]
    assert run02["summary_kind"] == "per-condition-test-metrics"

    run05 = registry["05"]
    assert run05["sweep_axis"] == "roi-sweep"
    assert run05["fixed_conditions"]["spacing"] == "0p1mm"
    assert run05["compared_conditions"] == ["roi512_spacing_0p1mm", "roi1024_spacing_0p1mm"]
    assert run05["available_conditions"] == ["roi512_spacing_0p1mm", "roi1024_spacing_0p1mm"]
    assert run05["missing_conditions"] == []
    assert run05["loss_family"] == "phase-first"
    assert run05["summary_kind"] == "study-summary"


def test_build_run_registry_surfaces_missing_official_runs(tmp_path: Path) -> None:
    write_run_05(tmp_path)

    mod = load_module()
    registry = mod.build_run_registry(tmp_path)

    run02 = registry["02"]
    assert run02["status"] == "missing-run"
    assert run02["available_conditions"] == []
    assert run02["missing_conditions"] == ["spacing_0mm", "spacing_0p1mm", "spacing_1mm", "spacing_2mm"]

    run05 = registry["05"]
    assert run05["status"] == "ok"


def test_build_run_registry_uses_lowest_error_metric_as_best(tmp_path: Path) -> None:
    write_run_02_error_only(tmp_path)

    mod = load_module()
    registry = mod.build_run_registry(tmp_path)

    run02 = registry["02"]
    assert run02["representative_metric"] == "phase_rmse_rad"
    assert run02["best_condition"] == "spacing_0p1mm"
    assert run02["best_value"] == 0.8
    assert "lowest phase_rmse_rad" in run02["conclusion"]


def test_skill_instructions_use_repo_local_script_path() -> None:
    skill_md = (
        Path(__file__).resolve().parents[2]
        / "skills"
        / "fd2nn-codex-run-interpreter"
        / "SKILL.md"
    )
    text = skill_md.read_text(encoding="utf-8")

    assert "/root/dj/D2NN/skills/fd2nn-codex-run-interpreter/scripts/build_run_registry.py" in text
