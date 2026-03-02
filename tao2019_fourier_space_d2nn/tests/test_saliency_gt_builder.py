from __future__ import annotations

import numpy as np

from tao2019_fd2nn.data.saliency_gt import SaliencyGtBuilder


def _sample_rgb() -> np.ndarray:
    rng = np.random.default_rng(123)
    return (rng.random((32, 32, 3)) * 255.0).astype(np.float32)


def test_ft_builder_range() -> None:
    b = SaliencyGtBuilder(source="ft", params={"smooth_sigma": 1.0})
    sal = b.build(image=_sample_rgb(), label=3)
    assert sal.shape == (32, 32)
    assert float(sal.min()) >= 0.0
    assert float(sal.max()) <= 1.0 + 1e-6


def test_spectral_residual_builder_range() -> None:
    b = SaliencyGtBuilder(source="spectral_residual", params={"log_smooth_sigma": 2.0, "map_smooth_sigma": 1.5})
    sal = b.build(image=_sample_rgb(), label=1)
    assert sal.shape == (32, 32)
    assert float(sal.min()) >= 0.0
    assert float(sal.max()) <= 1.0 + 1e-6


def test_class_gate_zeroes_non_target() -> None:
    b = SaliencyGtBuilder(source="ft", params={"class_gate": True}, foreground_class=3)
    sal = b.build(image=_sample_rgb(), label=1)
    assert np.allclose(sal, 0.0)
