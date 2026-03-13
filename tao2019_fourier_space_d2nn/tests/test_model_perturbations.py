from __future__ import annotations

from pathlib import Path

import torch

from tao2019_fd2nn.cli.common import build_model
from tao2019_fd2nn.config.schema import load_and_validate_config
from tao2019_fd2nn.models.fd2nn import Fd2nnConfig, Fd2nnModel


def test_build_model_passes_sensitivity_perturbation_settings() -> None:
    cfg_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "tao2019_fd2nn"
        / "config"
        / "cls_mnist_linear_real_5l.yaml"
    )
    cfg = load_and_validate_config(cfg_path)
    cfg["model"]["fabrication_blur_sigma_px"] = 0.5
    cfg["model"]["fabrication_blur_kernel_size"] = 3
    cfg["optics"]["alignment_shift_um"] = 2.0

    model = build_model(cfg)

    assert model.cfg.fabrication_blur_sigma_px == 0.5
    assert model.cfg.fabrication_blur_kernel_size == 3
    assert model.cfg.alignment_shift_um == 2.0


def _make_single_layer_model(**cfg_overrides: object) -> Fd2nnModel:
    cfg = Fd2nnConfig(
        N=8,
        dx_m=1.0e-6,
        wavelength_m=532.0e-9,
        z_layer_m=0.0,
        z_out_m=0.0,
        num_layers=1,
        phase_max=float(torch.pi),
        model_type="real_d2nn",
        **cfg_overrides,
    )
    model = Fd2nnModel(cfg)
    with torch.no_grad():
        raw = torch.full((8, 8), -12.0)
        raw[2:6, 2:6] = 12.0
        model.layers[0].raw.copy_(raw)
    return model


def test_fabrication_blur_changes_forward_without_mutating_raw_phase() -> None:
    field = torch.ones((1, 8, 8), dtype=torch.complex64)
    baseline = _make_single_layer_model()
    blurred = _make_single_layer_model(fabrication_blur_sigma_px=0.75, fabrication_blur_kernel_size=3)
    baseline.eval()
    blurred.eval()
    raw_before = blurred.layers[0].raw.detach().clone()

    y_baseline = baseline(field)
    y_blurred = blurred(field)

    assert not torch.allclose(y_baseline, y_blurred)
    assert torch.allclose(blurred.layers[0].raw.detach(), raw_before)


def test_alignment_shift_changes_forward_without_mutating_raw_phase() -> None:
    field = torch.ones((1, 8, 8), dtype=torch.complex64)
    baseline = _make_single_layer_model()
    shifted = _make_single_layer_model(alignment_shift_um=1.0)
    baseline.eval()
    shifted.eval()
    raw_before = shifted.layers[0].raw.detach().clone()

    y_baseline = baseline(field)
    y_shifted = shifted(field)

    assert not torch.allclose(y_baseline, y_shifted)
    assert torch.allclose(shifted.layers[0].raw.detach(), raw_before)


def test_perturbations_are_eval_only() -> None:
    field = torch.ones((1, 8, 8), dtype=torch.complex64)
    baseline = _make_single_layer_model()
    perturbed = _make_single_layer_model(fabrication_blur_sigma_px=0.75, alignment_shift_um=1.0)
    baseline.train()
    perturbed.train()

    y_baseline = baseline(field)
    y_perturbed = perturbed(field)

    assert torch.allclose(y_baseline, y_perturbed)
