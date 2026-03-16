"""Tests for grating period estimation."""

import numpy as np
import torch
import pytest

from luo2022_d2nn.data.resolution_targets import SUPPORTED_PERIODS_MM, generate_grating_target
from luo2022_d2nn.eval.grating_period import estimate_grating_period


def _make_synthetic_grating(N: int, dx_mm: float, period_mm: float) -> torch.Tensor:
    """Create a synthetic horizontal 3-bar grating with Gaussian-shaped bars.

    Bars are centered at  center - period_mm, center, center + period_mm
    so that (max_peak - min_peak) / 2 = period_mm.
    """
    x = np.arange(N) * dx_mm
    center = x[N // 2]
    sigma = period_mm / 5.0  # narrow enough to be distinct

    positions = [center - period_mm, center, center + period_mm]
    profile = np.zeros(N)
    for mu in positions:
        profile += np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    # Make it 2D as horizontal bars: replicate the 1D y-profile across x.
    img = np.tile(profile[:, np.newaxis], (1, N)).astype(np.float32)
    return torch.from_numpy(img)


class TestGratingPeriod:
    def test_grating_period_known_period(self):
        """3-bar grating with period_mm=10.8 -> estimate within +-1.0mm"""
        period_mm = 10.8
        intensity = _make_synthetic_grating(240, dx_mm=0.3, period_mm=period_mm)
        estimated = estimate_grating_period(intensity, dx_mm=0.3)
        assert estimated == pytest.approx(period_mm, abs=1.0)

    def test_grating_period_different_periods(self):
        """Test with 12.0mm period"""
        period_mm = 12.0
        intensity = _make_synthetic_grating(240, dx_mm=0.3, period_mm=period_mm)
        estimated = estimate_grating_period(intensity, dx_mm=0.3)
        assert estimated == pytest.approx(period_mm, abs=1.0)

    def test_grating_period_shape_handling(self):
        """Works with both (N,N) and (1,N,N) input."""
        period_mm = 10.8
        intensity_2d = _make_synthetic_grating(240, dx_mm=0.3, period_mm=period_mm)
        intensity_3d = intensity_2d.unsqueeze(0)

        est_2d = estimate_grating_period(intensity_2d, dx_mm=0.3)
        est_3d = estimate_grating_period(intensity_3d, dx_mm=0.3)

        assert est_2d == pytest.approx(period_mm, abs=1.0)
        assert est_3d == pytest.approx(period_mm, abs=1.0)
        assert est_2d == pytest.approx(est_3d, abs=0.01)

    @pytest.mark.parametrize("period_mm", SUPPORTED_PERIODS_MM)
    def test_generated_horizontal_bar_target_matches_requested_period(self, period_mm):
        """Estimator should recover the requested period from generated paper-style targets."""
        target = generate_grating_target(
            period_mm=period_mm,
            dx_mm=0.3,
            active_size=160,
            final_size=240,
        )

        estimated = estimate_grating_period(target, dx_mm=0.3)

        assert estimated == pytest.approx(period_mm, abs=1.0)
