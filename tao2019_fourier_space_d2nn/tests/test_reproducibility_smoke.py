from __future__ import annotations

import torch

from tao2019_fd2nn.models.fd2nn import Fd2nnConfig, Fd2nnModel
from tao2019_fd2nn.utils.seed import set_global_seed


def _run_once(seed: int) -> torch.Tensor:
    set_global_seed(seed, deterministic=True)
    cfg = Fd2nnConfig(
        N=32,
        dx_m=8e-6,
        wavelength_m=532e-9,
        z_layer_m=1e-4,
        z_out_m=1e-4,
        num_layers=2,
        phase_max=float(torch.pi),
    )
    model = Fd2nnModel(cfg)
    x = torch.randn(1, 32, 32) + 1j * torch.randn(1, 32, 32)
    return model(x)


def test_seed_reproducibility_smoke() -> None:
    y1 = _run_once(1234)
    y2 = _run_once(1234)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-6)
