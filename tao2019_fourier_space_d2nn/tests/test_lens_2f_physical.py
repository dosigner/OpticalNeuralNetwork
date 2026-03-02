from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tao2019_fd2nn.cli.common import build_model
from tao2019_fd2nn.config.schema import load_and_validate_config
from tao2019_fd2nn.optics.lens_2f import lens_2f_forward, lens_2f_inverse


def _rand_field(N: int) -> torch.Tensor:
    return torch.randn(1, N, N) + 1j * torch.randn(1, N, N)


def test_lens_2f_pitch_depends_on_focal_length() -> None:
    field = _rand_field(64)
    _, dx_f1 = lens_2f_forward(
        field,
        dx_in_m=1.0e-6,
        wavelength_m=5.32e-7,
        f_m=1.0e-3,
        na=None,
        apply_scaling=False,
    )
    _, dx_f4 = lens_2f_forward(
        field,
        dx_in_m=1.0e-6,
        wavelength_m=5.32e-7,
        f_m=4.0e-3,
        na=None,
        apply_scaling=False,
    )
    assert dx_f4 / dx_f1 == pytest.approx(4.0, rel=1e-6, abs=1e-6)


def test_lens_2f_inverse_recovers_input_pitch_when_f1_equals_f2() -> None:
    dx_in = 1.0e-6
    field = _rand_field(64)
    fourier, dx_fourier = lens_2f_forward(
        field,
        dx_in_m=dx_in,
        wavelength_m=5.32e-7,
        f_m=1.0e-3,
        na=None,
        apply_scaling=False,
    )
    _, dx_out = lens_2f_inverse(
        fourier,
        dx_fourier_m=dx_fourier,
        wavelength_m=5.32e-7,
        f_m=1.0e-3,
        na=None,
        apply_scaling=False,
    )
    assert dx_out == pytest.approx(dx_in, rel=1e-6, abs=1e-12)


def test_lens_2f_na_filter_reduces_energy() -> None:
    field = _rand_field(64)
    out_lo, _ = lens_2f_forward(
        field,
        dx_in_m=1.0e-6,
        wavelength_m=5.32e-7,
        f_m=1.0e-3,
        na=0.05,
        apply_scaling=False,
    )
    out_hi, _ = lens_2f_forward(
        field,
        dx_in_m=1.0e-6,
        wavelength_m=5.32e-7,
        f_m=1.0e-3,
        na=0.16,
        apply_scaling=False,
    )
    e_lo = float((out_lo.abs() ** 2).sum().item())
    e_hi = float((out_hi.abs() ** 2).sum().item())
    assert e_lo <= e_hi + 1e-7


def test_dual_2f_config_fields_are_plumbed_to_model() -> None:
    cfg_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "tao2019_fd2nn"
        / "config"
        / "cls_mnist_linear_fourier_5l_f1mm.yaml"
    )
    cfg = load_and_validate_config(cfg_path)
    model = build_model(cfg)
    assert model.cfg.use_dual_2f is True
    assert model.cfg.dual_2f_f1_m == pytest.approx(1.0e-3)
    assert model.cfg.dual_2f_f2_m == pytest.approx(1.0e-3)
    assert model.cfg.dual_2f_na1 == pytest.approx(0.16)
    assert model.cfg.dual_2f_na2 == pytest.approx(0.16)
    assert model.cfg.dual_2f_apply_scaling is True
