from __future__ import annotations

import math

import numpy as np
import torch

from tao2019_fd2nn.cli.common import build_model
from tao2019_fd2nn.models.fd2nn import Fd2nnConfig, Fd2nnModel
from tao2019_fd2nn.viz.figure_factory import FigureFactory


def _minimal_model_config(*, alignment_shift_um: float = 0.0) -> dict:
    return {
        "experiment": {"dtype": "float32"},
        "optics": {
            "wavelength_m": 5.32e-7,
            "grid": {"nx": 16, "ny": 16, "dx_m": 1.0e-6, "dy_m": 1.0e-6},
            "propagation": {"layer_spacing_m": 1.0e-4},
            "alignment_shift_um": alignment_shift_um,
        },
        "model": {
            "type": "fd2nn",
            "num_layers": 1,
            "modulation": {
                "kind": "phase_only",
                "phase_constraint": "sigmoid",
                "phase_max_rad": math.pi,
                "init": "zeros",
            },
        },
    }


def _seed_pattern(model: Fd2nnModel) -> None:
    with torch.no_grad():
        model.layers[0].raw.copy_(torch.linspace(-2.0, 2.0, steps=model.cfg.N * model.cfg.N).reshape(model.cfg.N, model.cfg.N))


def test_build_model_converts_alignment_shift_um_to_pixels() -> None:
    model = build_model(_minimal_model_config(alignment_shift_um=2.0))
    assert model.cfg.alignment_shift_um == 2.0
    assert model._alignment_shift_px(1.0e-6) == 2


def test_fabrication_blur_changes_output() -> None:
    base_cfg = Fd2nnConfig(
        N=16,
        dx_m=1.0e-6,
        wavelength_m=5.32e-7,
        z_layer_m=1.0e-4,
        z_out_m=1.0e-4,
        num_layers=1,
        phase_max=float(torch.pi),
        phase_init="zeros",
        fabrication_blur_sigma_px=0.0,
        fabrication_blur_kernel_size=3,
    )
    blurred_cfg = Fd2nnConfig(
        N=16,
        dx_m=1.0e-6,
        wavelength_m=5.32e-7,
        z_layer_m=1.0e-4,
        z_out_m=1.0e-4,
        num_layers=1,
        phase_max=float(torch.pi),
        phase_init="zeros",
        fabrication_blur_sigma_px=0.6,
        fabrication_blur_kernel_size=3,
    )
    base = Fd2nnModel(base_cfg)
    blurred = Fd2nnModel(blurred_cfg)
    _seed_pattern(base)
    blurred.load_state_dict(base.state_dict())
    base.eval()
    blurred.eval()

    field = torch.ones(1, 16, 16, dtype=torch.complex64)
    out_base = base(field)
    out_blurred = blurred(field)

    assert not torch.allclose(out_base, out_blurred)


def test_alignment_shift_changes_output() -> None:
    base_cfg = Fd2nnConfig(
        N=16,
        dx_m=1.0e-6,
        wavelength_m=5.32e-7,
        z_layer_m=1.0e-4,
        z_out_m=1.0e-4,
        num_layers=1,
        phase_max=float(torch.pi),
        phase_init="zeros",
        alignment_shift_um=0.0,
    )
    shifted_cfg = Fd2nnConfig(
        N=16,
        dx_m=1.0e-6,
        wavelength_m=5.32e-7,
        z_layer_m=1.0e-4,
        z_out_m=1.0e-4,
        num_layers=1,
        phase_max=float(torch.pi),
        phase_init="zeros",
        alignment_shift_um=2.0,
    )
    base = Fd2nnModel(base_cfg)
    shifted = Fd2nnModel(shifted_cfg)
    _seed_pattern(base)
    shifted.load_state_dict(base.state_dict())
    base.eval()
    shifted.eval()

    field = torch.ones(1, 16, 16, dtype=torch.complex64)
    out_base = base(field)
    out_shifted = shifted(field)

    assert not torch.allclose(out_base, out_shifted)


def test_alignment_shift_zero_fills_instead_of_wrapping() -> None:
    model = Fd2nnModel(
        Fd2nnConfig(
            N=4,
            dx_m=1.0e-6,
            wavelength_m=5.32e-7,
            z_layer_m=1.0e-4,
            z_out_m=1.0e-4,
            num_layers=1,
            phase_max=float(torch.pi),
            alignment_shift_um=2.0,
        )
    )
    phi = torch.zeros(4, 4)
    phi[:, -1] = 1.0

    shifted = model._apply_phase_perturbations(phi, dx_m=1.0e-6)

    assert torch.count_nonzero(shifted[:, :2]) == 0


def test_figure_factory_writes_s8_sensitivity_plot(tmp_path) -> None:
    factory = FigureFactory(tmp_path)
    fabrication_curves = {
        "Nonlinear Fourier 1L": [0.93, 0.91, 0.88],
        "Nonlinear Fourier 5L": [0.98, 0.97, 0.95],
        "Linear Real 5L": [0.94, 0.92, 0.90],
    }
    alignment_curves = {
        "Nonlinear Fourier 1L": [0.93, 0.89, 0.84],
        "Nonlinear Fourier 5L": [0.98, 0.96, 0.94],
        "Linear Real 5L": [0.94, 0.90, 0.85],
    }

    path = factory.plot_s8_classification_sensitivity(
        fabrication_curves,
        alignment_curves,
        fabrication_x=[0.0, 0.5, 1.0],
        alignment_x=[0.0, 2.0, 4.0],
        name="supp_s8_cls.png",
    )

    assert path.exists()
