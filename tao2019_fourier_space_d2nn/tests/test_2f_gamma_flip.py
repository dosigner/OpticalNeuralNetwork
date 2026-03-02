from __future__ import annotations

import torch

from tao2019_fd2nn.optics.fft2c import fft2c, gamma_flip2d


def test_double_fft_matches_gamma_up_to_constant() -> None:
    x = torch.randn(1, 16, 16) + 1j * torch.randn(1, 16, 16)
    y = fft2c(fft2c(x))
    target = gamma_flip2d(x)

    # Compensate global phase/scale ambiguity.
    c = (y * torch.conj(target)).sum() / (torch.abs(target) ** 2).sum().clamp_min(1e-8)
    y_aligned = y / c
    assert torch.allclose(y_aligned, target, atol=1e-4, rtol=1e-4)
