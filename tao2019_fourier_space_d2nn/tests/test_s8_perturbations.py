from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from tao2019_fd2nn.cli.common import build_model
from tao2019_fd2nn.config.schema import load_and_validate_config
from tao2019_fd2nn.models.fd2nn import Fd2nnConfig, Fd2nnModel


def _base_real_space_runtime_cfg(**overrides) -> Fd2nnConfig:
    kwargs = {
        "N": 16,
        "dx_m": 1.0e-6,
        "wavelength_m": 5.32e-7,
        "z_layer_m": 0.0,
        "z_out_m": 0.0,
        "num_layers": 1,
        "phase_max": float(torch.pi),
        "model_type": "real_d2nn",
    }
    kwargs.update(overrides)
    return Fd2nnConfig(**kwargs)


def _set_impulse_phase(model: Fd2nnModel) -> None:
    with torch.no_grad():
        model.layers[0].raw.fill_(-10.0)
        model.layers[0].raw[8, 8] = 10.0


def _ones_field() -> torch.Tensor:
    return torch.ones(1, 16, 16, dtype=torch.complex64)


def test_build_model_plumbs_s8_perturbation_fields() -> None:
    cfg_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "tao2019_fd2nn"
        / "config"
        / "cls_mnist_linear_real_5l.yaml"
    )
    cfg = load_and_validate_config(cfg_path)
    cfg = deepcopy(cfg)
    cfg["model"]["fabrication_blur_sigma_px"] = 0.5
    cfg["model"]["fabrication_blur_kernel_size"] = 3
    cfg["optics"]["alignment_shift_um"] = 2.0

    model = build_model(cfg)

    assert model.cfg.fabrication_blur_sigma_px == pytest.approx(0.5)
    assert model.cfg.fabrication_blur_kernel_size == 3
    assert model.cfg.alignment_shift_um == pytest.approx(2.0)


def test_alignment_shift_zero_matches_baseline_output() -> None:
    base = Fd2nnModel(_base_real_space_runtime_cfg())
    shifted = Fd2nnModel(_base_real_space_runtime_cfg(alignment_shift_um=0.0))
    base.eval()
    shifted.eval()
    _set_impulse_phase(base)
    _set_impulse_phase(shifted)

    y_base = base(_ones_field())
    y_shifted = shifted(_ones_field())

    assert torch.allclose(y_base, y_shifted, atol=1e-6, rtol=1e-6)


def test_alignment_shift_changes_output_for_nonzero_shift() -> None:
    base = Fd2nnModel(_base_real_space_runtime_cfg())
    shifted = Fd2nnModel(_base_real_space_runtime_cfg(alignment_shift_um=2.0))
    base.eval()
    shifted.eval()
    _set_impulse_phase(base)
    _set_impulse_phase(shifted)

    y_base = base(_ones_field())
    y_shifted = shifted(_ones_field())

    assert not torch.allclose(y_base, y_shifted, atol=1e-6, rtol=1e-6)


def test_fabrication_blur_zero_matches_baseline_output() -> None:
    base = Fd2nnModel(_base_real_space_runtime_cfg())
    blurred = Fd2nnModel(
        _base_real_space_runtime_cfg(
            fabrication_blur_sigma_px=0.0,
            fabrication_blur_kernel_size=3,
        )
    )
    base.eval()
    blurred.eval()
    _set_impulse_phase(base)
    _set_impulse_phase(blurred)

    y_base = base(_ones_field())
    y_blurred = blurred(_ones_field())

    assert torch.allclose(y_base, y_blurred, atol=1e-6, rtol=1e-6)


def test_fabrication_blur_changes_output_for_nonzero_sigma() -> None:
    base = Fd2nnModel(_base_real_space_runtime_cfg())
    blurred = Fd2nnModel(
        _base_real_space_runtime_cfg(
            fabrication_blur_sigma_px=0.75,
            fabrication_blur_kernel_size=3,
        )
    )
    base.eval()
    blurred.eval()
    _set_impulse_phase(base)
    _set_impulse_phase(blurred)

    y_base = base(_ones_field())
    y_blurred = blurred(_ones_field())

    assert not torch.allclose(y_base, y_blurred, atol=1e-6, rtol=1e-6)
