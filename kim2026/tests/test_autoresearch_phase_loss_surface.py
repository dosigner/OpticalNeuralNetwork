from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

EXPERIMENT_PATH = Path(__file__).resolve().parent.parent / "autoresearch" / "experiment.py"
LOSS_SWEEP_PATH = Path(__file__).resolve().parent.parent / "autoresearch" / "loss_sweep.py"
PHASE_STUDY_PATH = Path(__file__).resolve().parent.parent / "scripts" / "run_fd2nn_phase_restore_study.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_autoresearch_experiment_exposes_full_field_phase_knobs() -> None:
    module = _load_module(EXPERIMENT_PATH, "autoresearch_experiment")

    assert module.LOSS_WEIGHTS["full_field_phase"] == 0.0
    assert module.LOSS_WEIGHTS["full_field_phase_gamma"] == 1.0
    assert module.LOSS_WEIGHTS["full_field_phase_threshold"] == 0.05


def test_phase_restore_study_defaults_enable_full_field_phase_auxiliary_loss() -> None:
    module = _load_module(PHASE_STUDY_PATH, "run_fd2nn_phase_restore_study")

    assert module.COMMON["complex_weights"]["full_field_phase"] == 0.15
    assert module.COMMON["complex_weights"]["full_field_phase_gamma"] == 1.0
    assert module.COMMON["complex_weights"]["full_field_phase_threshold"] == 0.05


def test_loss_sweep_exposes_full_field_phase_axis() -> None:
    module = _load_module(LOSS_SWEEP_PATH, "autoresearch_loss_sweep")

    for name, weight in (
        ("co_ffp", 1.0),
        ("co_ffp_strong", 3.0),
    ):
        cfg = module.LOSS_CONFIGS[name]
        assert cfg["weights"]["complex_overlap"] == 1.0
        assert cfg["weights"]["full_field_phase"] == weight


def test_loss_sweep_evaluate_reports_full_field_phase_rmse() -> None:
    module = _load_module(LOSS_SWEEP_PATH, "autoresearch_loss_sweep_eval")

    class IdentityModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    original_prepare_field = module.prepare_field
    module.prepare_field = lambda field: field
    try:
        batch = {
            "u_turb": torch.ones(1, 4, 4, dtype=torch.complex64),
            "u_vacuum": torch.ones(1, 4, 4, dtype=torch.complex64),
        }
        metrics = module.evaluate(IdentityModel(), [batch], torch.device("cpu"))
    finally:
        module.prepare_field = original_prepare_field

    assert "full_field_phase_rmse_rad" in metrics
    assert metrics["full_field_phase_rmse_rad"] == pytest.approx(0.0, abs=1e-6)
