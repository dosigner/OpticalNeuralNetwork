from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from kim2026.cli.evaluate_beam_cleanup import _build_eval_model, _save_sample_fields
from kim2026.models.fd2nn import BeamCleanupFD2NN


def test_build_eval_model_fd2nn_uses_dual_2f_parameters() -> None:
    cfg = {
        "optics": {
            "lambda_m": 1.55e-6,
            "dual_2f": {
                "enabled": True,
                "f1_m": 1.0e-3,
                "f2_m": 2.0e-3,
                "na1": 0.16,
                "na2": 0.12,
                "apply_scaling": True,
            },
        },
        "grid": {"receiver_window_m": 0.2},
        "model": {
            "type": "fd2nn",
            "num_layers": 5,
            "layer_spacing_m": 1.0e-4,
            "phase_max": 3.14159265,
            "phase_constraint": "unconstrained",
            "phase_init": "uniform",
            "phase_init_scale": 0.1,
        },
    }

    model = _build_eval_model(cfg, sample_n=32, device=torch.device("cpu"))

    assert isinstance(model, BeamCleanupFD2NN)
    assert model.dual_2f_f1_m == 1.0e-3
    assert model.dual_2f_f2_m == 2.0e-3
    assert model.dual_2f_na1 == 0.16
    assert model.dual_2f_na2 == 0.12
    assert model.dual_2f_apply_scaling is True
    assert model.layer_spacing_m == 1.0e-4
    assert model.layers[0].constraint == "unconstrained"


def test_save_sample_fields_writes_complex_components(tmp_path: Path) -> None:
    output_path = tmp_path / "sample_fields.npz"
    input_field = torch.tensor([[1.0 + 2.0j, 3.0 + 4.0j]], dtype=torch.complex64)
    vacuum_field = torch.tensor([[2.0 + 1.0j, 4.0 + 3.0j]], dtype=torch.complex64)
    baseline_field = torch.tensor([[5.0 + 6.0j, 7.0 + 8.0j]], dtype=torch.complex64)
    pred_field = torch.tensor([[9.0 + 1.0j, 2.0 + 3.0j]], dtype=torch.complex64)
    target_field = torch.tensor([[4.0 + 5.0j, 6.0 + 7.0j]], dtype=torch.complex64)

    _save_sample_fields(
        output_path,
        input_field=input_field,
        vacuum_field=vacuum_field,
        baseline_field=baseline_field,
        pred_field=pred_field,
        target_field=target_field,
    )

    saved = np.load(output_path)
    assert set(saved.files) == {
        "input_real",
        "input_imag",
        "vacuum_real",
        "vacuum_imag",
        "baseline_real",
        "baseline_imag",
        "pred_real",
        "pred_imag",
        "target_real",
        "target_imag",
    }
    np.testing.assert_allclose(saved["input_real"], np.array([[1.0, 3.0]], dtype=np.float32))
    np.testing.assert_allclose(saved["vacuum_real"], np.array([[2.0, 4.0]], dtype=np.float32))
    np.testing.assert_allclose(saved["baseline_imag"], np.array([[6.0, 8.0]], dtype=np.float32))
    np.testing.assert_allclose(saved["pred_real"], np.array([[9.0, 2.0]], dtype=np.float32))
    np.testing.assert_allclose(saved["target_imag"], np.array([[5.0, 7.0]], dtype=np.float32))
