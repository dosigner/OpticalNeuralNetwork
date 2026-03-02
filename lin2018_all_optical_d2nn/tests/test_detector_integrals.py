from __future__ import annotations

import numpy as np

from d2nn.detectors.integrate import integrate_regions
from d2nn.detectors.layout import DetectorLayout, DetectorRegion, build_region_masks


def test_detector_integral_indexing() -> None:
    N = 32
    dx = 1e-3
    regions = [
        DetectorRegion(name="a", center_xy=(-0.004, 0.0), size_xy=(0.004, 0.004)),
        DetectorRegion(name="b", center_xy=(0.004, 0.0), size_xy=(0.004, 0.004)),
    ]
    layout = DetectorLayout(regions=regions, plane_size_xy=(N * dx, N * dx))
    masks = build_region_masks(layout, N=N, dx=dx)

    intensity = np.zeros((N, N), dtype=np.float32)
    # Put one bright pixel near left detector and one near right detector
    intensity[N // 2, N // 2 - 4] = 2.0
    intensity[N // 2, N // 2 + 4] = 3.0

    energies = integrate_regions(intensity, masks, reduction="sum")
    assert energies.shape == (2,)
    assert energies[0] > 0
    assert energies[1] > 0
    assert not np.isclose(energies[0], energies[1])
