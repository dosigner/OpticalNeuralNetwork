from __future__ import annotations

import torch

from tao2019_fd2nn.models.detectors import integrate_detector_energies, make_detector_masks


def test_detector_masks_and_integration() -> None:
    N = 64
    masks = make_detector_masks(N=N, dx_m=8e-6, num_classes=10, width_um=12.0, gap_um=4.0, layout="default10")
    assert masks.shape == (10, N, N)
    assert int(masks.sum().item()) > 0

    intensity = torch.zeros(1, N, N)
    pos = torch.nonzero(masks[0], as_tuple=False)[0]
    intensity[0, int(pos[0]), int(pos[1])] = 1.0
    energies = integrate_detector_energies(intensity, masks)
    assert energies.shape == (1, 10)
    assert int(torch.argmax(energies, dim=1).item()) == 0


def test_detector_default10_is_3_4_3_layout() -> None:
    masks = make_detector_masks(N=64, dx_m=8e-6, num_classes=10, width_um=12.0, gap_um=4.0, layout="default10")
    y_centers: list[int] = []
    for k in range(10):
        pos = torch.nonzero(masks[k], as_tuple=False)
        assert int(pos.shape[0]) > 0
        y_centers.append(int(pos[:, 0].to(torch.float32).mean().round().item()))

    rows = sorted(set(y_centers))
    assert len(rows) == 3
    counts = [sum(1 for y in y_centers if y == row) for row in rows]
    assert counts == [3, 4, 3]
