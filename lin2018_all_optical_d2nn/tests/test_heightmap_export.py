from __future__ import annotations

import numpy as np

from d2nn.physics.materials import phase_to_height


def test_phase_to_height_formula() -> None:
    phase = np.array([[0.0, np.pi]], dtype=np.float64)
    wavelength = 1.0
    delta_n = 2.0
    h = phase_to_height(phase, wavelength, delta_n)

    assert np.isclose(h[0, 0], 0.0)
    assert np.isclose(h[0, 1], 0.25)
