from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch

from kim2026.eval import sanity_check


def _reload_sweep_module():
    module = importlib.import_module("autoresearch.d2nn_focal_pib_sweep")
    return importlib.reload(module)


class _IdentityModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return field


class _FakeLayer:
    def __init__(self) -> None:
        self.phase = torch.zeros((2, 2), dtype=torch.float32)


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.layers = [_FakeLayer()]

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        return field * self.scale.to(field.dtype)


def test_check_unitary_co_uses_same_model_for_target(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _IdentityModel()
    batch = {
        "u_turb": torch.ones((1, 4, 4), dtype=torch.complex64),
        "u_vacuum": torch.ones((1, 4, 4), dtype=torch.complex64),
    }

    monkeypatch.setattr(sanity_check, "prepare_field", lambda field: field)

    def _forbidden_zero_phase_model(*args, **kwargs):
        raise AssertionError("check_unitary_co must not use the zero-phase reference model")

    monkeypatch.setattr(sanity_check, "_make_zero_phase_model", _forbidden_zero_phase_model)

    result = sanity_check.check_unitary_co(model, [batch], tolerance=1e-6)

    assert result.passed is True
    assert model.calls == 2


def test_apply_config_overrides_updates_runtime_globals(tmp_path: Path) -> None:
    sweep = _reload_sweep_module()
    dataset_root = tmp_path / "custom_dataset"
    cfg = {
        "physics": {
            "wavelength_m": 9.9e-7,
            "receiver_window_m": 3.1e-3,
            "aperture_diameter_m": 1.7e-3,
            "focus_f_m": 7.0e-3,
            "pib_bucket_radius_um": 12.5,
        },
        "architecture": {
            "num_layers": 3,
            "layer_spacing_m": 2.5e-2,
            "detector_distance_m": 3.5e-2,
            "propagation_pad_factor": 3,
        },
        "training": {
            "epochs": 7,
            "batch_size": 2,
            "lr": 2e-3,
            "warmup_epochs": 2,
            "tv_weight": 0.0,
        },
        "data": {
            "path": str(dataset_root),
        },
    }

    sweep.apply_config_overrides(cfg)

    assert sweep.WAVELENGTH_M == pytest.approx(9.9e-7)
    assert sweep.RECEIVER_WINDOW_M == pytest.approx(3.1e-3)
    assert sweep.APERTURE_DIAMETER_M == pytest.approx(1.7e-3)
    assert sweep.FOCUS_F_M == pytest.approx(7.0e-3)
    assert sweep.PIB_BUCKET_RADIUS_UM == pytest.approx(12.5)
    assert sweep.ARCH == {
        "num_layers": 3,
        "layer_spacing_m": pytest.approx(2.5e-2),
        "detector_distance_m": pytest.approx(3.5e-2),
        "propagation_pad_factor": 3,
    }
    assert sweep.TRAIN["epochs"] == 7
    assert sweep.TRAIN["batch_size"] == 2
    assert sweep.TRAIN["lr"] == pytest.approx(2e-3)
    assert sweep.TRAIN["warmup_epochs"] == 2
    assert sweep.TRAIN["tv_weight"] == pytest.approx(0.0)
    assert sweep.DATA_DIR == dataset_root / "cache"
    assert sweep.MANIFEST == dataset_root / "split_manifest.json"


def test_train_one_runs_post_training_checks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sweep = _reload_sweep_module()
    batch = {
        "u_turb": torch.ones((1, 2, 2), dtype=torch.complex64),
        "u_vacuum": torch.ones((1, 2, 2), dtype=torch.complex64),
    }
    calls: list[dict] = []

    monkeypatch.setattr(sweep, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(sweep, "prepare_field", lambda field: field)
    monkeypatch.setattr(sweep, "make_model", lambda: _FakeModel())
    monkeypatch.setattr(sweep, "to_focal_plane", lambda field: (field, 1.0e-6))
    monkeypatch.setattr(sweep, "evaluate", lambda *args, **kwargs: {
        "co_output": 1.0,
        "io_output": 1.0,
        "co_baseline": 1.0,
        "focal_pib_10um": 1.0,
        "focal_pib_50um": 1.0,
        "focal_pib_10um_baseline": 1.0,
        "focal_pib_50um_baseline": 1.0,
        "focal_pib_10um_vacuum": 1.0,
        "focal_pib_50um_vacuum": 1.0,
        "focal_strehl": 1.0,
        "dx_focal_um": 1.0,
    })
    monkeypatch.setattr(sweep, "throughput_check", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(sweep, "wf_rms_eval", lambda *args, **kwargs: (0.0, 0.0))
    monkeypatch.setitem(sweep.TRAIN, "epochs", 1)
    monkeypatch.setitem(sweep.TRAIN, "warmup_epochs", 1)
    monkeypatch.setitem(sweep.TRAIN, "batch_size", 1)
    monkeypatch.setitem(sweep.TRAIN, "tv_weight", 0.0)

    def _fake_post_checks(model, loader, device="cpu", config=None):
        calls.append({"model": model, "loader": loader, "device": device, "config": config})

        class _Report:
            all_passed = True

        return _Report()

    monkeypatch.setattr("kim2026.eval.sanity_check.run_post_training_checks", _fake_post_checks)

    result = sweep.train_one(
        "smoke",
        {"fn": lambda dp, dt, fp, ft, dx: dp.abs().mean(), "desc": "smoke"},
        [batch],
        [batch],
        [batch],
        torch.device("cpu"),
        {"enabled": True, "unitary_co_tolerance": 0.02},
    )

    assert result["name"] == "smoke"
    assert len(calls) == 1
    assert calls[0]["config"] == {"enabled": True, "unitary_co_tolerance": 0.02}
    assert calls[0]["device"] == "cpu"


def test_focal_strehl_loss_uses_correct_strehl(monkeypatch: pytest.MonkeyPatch) -> None:
    sweep = _reload_sweep_module()
    pred = torch.ones((1, 4, 4), dtype=torch.complex64)
    target = torch.ones((1, 4, 4), dtype=torch.complex64)
    monkeypatch.setattr(
        sweep,
        "strehl_ratio_correct",
        lambda pred_field, ref_amplitude, pad_factor=4: torch.tensor([0.75]),
        raising=False,
    )

    loss = sweep.focal_strehl_loss(pred, target)

    assert loss.item() == pytest.approx(0.25)


def test_check_strehl_bound_uses_correct_strehl(monkeypatch: pytest.MonkeyPatch) -> None:
    batch = {
        "u_turb": torch.ones((1, 4, 4), dtype=torch.complex64),
        "u_vacuum": torch.ones((1, 4, 4), dtype=torch.complex64),
    }
    model = _IdentityModel()

    monkeypatch.setattr(sanity_check, "prepare_field", lambda field: field)
    monkeypatch.setattr(
        sanity_check,
        "strehl_ratio_correct",
        lambda pred_field, ref_amplitude, pad_factor=4: torch.tensor([0.8]),
        raising=False,
    )

    result = sanity_check.check_strehl_bound(model, [batch], max_strehl=1.05)

    assert result.passed is True
    assert result.value == pytest.approx(0.8)
