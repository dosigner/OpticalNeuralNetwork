from __future__ import annotations

import torch

from tao2019_fd2nn.optics.fft2c import fft2c, ifft2c


def test_fft2c_ifft2c_identity() -> None:
    x = torch.randn(3, 32, 32) + 1j * torch.randn(3, 32, 32)
    y = ifft2c(fft2c(x))
    assert torch.allclose(y, x, atol=1e-5, rtol=1e-5)
