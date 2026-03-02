from __future__ import annotations

import torch

from tao2019_fd2nn.optics.asm import asm_propagate, asm_transfer_function


def test_asm_identity_at_z0() -> None:
    field = torch.randn(2, 32, 32) + 1j * torch.randn(2, 32, 32)
    H = asm_transfer_function(N=32, dx_m=8e-6, wavelength_m=532e-9, z_m=0.0, evanescent="keep")
    out = asm_propagate(field, H)
    assert torch.allclose(out, field, atol=1e-5, rtol=1e-5)


def test_asm_energy_sanity() -> None:
    field = torch.ones(1, 64, 64, dtype=torch.complex64)
    H = asm_transfer_function(N=64, dx_m=8e-6, wavelength_m=532e-9, z_m=1e-4, evanescent="keep")
    out = asm_propagate(field, H)
    e0 = (torch.abs(field) ** 2).sum()
    e1 = (torch.abs(out) ** 2).sum()
    rel = torch.abs(e1 - e0) / e0
    assert float(rel.item()) < 1e-3
