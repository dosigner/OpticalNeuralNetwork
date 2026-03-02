from __future__ import annotations

import torch

from d2nn.physics.asm import asm_propagate, asm_transfer_function


def test_asm_identity_z0() -> None:
    N = 32
    dx = 1e-3
    wavelength = 0.75e-3
    z = 0.0

    x = torch.randn(2, N, N, dtype=torch.float32)
    y = torch.randn(2, N, N, dtype=torch.float32)
    field = torch.complex(x, y)

    H = asm_transfer_function(N=N, dx=dx, wavelength=wavelength, z=z, bandlimit=False, dtype="complex64")
    out = asm_propagate(field, H)

    assert torch.allclose(out, field, atol=1e-5, rtol=1e-5)


def test_asm_energy_sanity_phase_only() -> None:
    N = 64
    dx = 1e-3
    wavelength = 0.75e-3
    z = 0.01

    amp = torch.ones(1, N, N, dtype=torch.float32)
    field = torch.complex(amp, torch.zeros_like(amp))

    H = asm_transfer_function(N=N, dx=dx, wavelength=wavelength, z=z, bandlimit=False, dtype="complex64")
    out = asm_propagate(field, H)

    e_in = (torch.abs(field) ** 2).sum()
    e_out = (torch.abs(out) ** 2).sum()
    rel_err = torch.abs(e_in - e_out) / e_in
    assert float(rel_err.item()) < 1e-3
