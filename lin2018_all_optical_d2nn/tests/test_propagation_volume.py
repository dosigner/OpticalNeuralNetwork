from __future__ import annotations

import numpy as np
import torch

from d2nn.viz.propagation import extract_xz_cross_section, generate_phase_masks, simulate_d2nn_volume, simulate_free_space_volume


def _unit_field(N: int) -> torch.Tensor:
    amp = torch.ones((N, N), dtype=torch.float32)
    return torch.complex(amp, torch.zeros_like(amp))


def test_simulate_free_space_volume_xyz_shape() -> None:
    N = 20
    field = _unit_field(N)
    volume_xyz, z_positions = simulate_free_space_volume(
        field,
        dx=0.4e-3,
        wavelength=0.75e-3,
        total_distance=0.02,
        num_segments=4,
        bandlimit=False,
    )

    assert volume_xyz.shape == (N, N, 5)
    assert z_positions.shape == (5,)
    np.testing.assert_allclose(volume_xyz[:, :, 0], field.numpy().T, atol=1e-6, rtol=1e-6)


def test_simulate_d2nn_volume_xyz_shape_and_xz_extract() -> None:
    N = 24
    field = _unit_field(N)
    masks = generate_phase_masks(num_layers=3, N=N, mode="random", seed=7)
    volume_xyz, z_positions = simulate_d2nn_volume(
        field,
        dx=0.4e-3,
        wavelength=0.75e-3,
        num_layers=3,
        layer_spacing=0.004,
        phase_masks=masks,
        bandlimit=False,
    )

    assert volume_xyz.shape == (N, N, 4)
    assert z_positions.shape == (4,)
    assert np.isfinite(np.abs(volume_xyz)).all()

    xz = extract_xz_cross_section(volume_xyz, quantity="amplitude")
    assert xz.shape == (4, N)
