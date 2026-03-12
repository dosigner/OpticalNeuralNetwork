"""Tests for Fresnel thin-lens transmission function."""

from __future__ import annotations

import math

import pytest
import torch

from luo2022_d2nn.optics.lens import fresnel_lens_transmission
from luo2022_d2nn.optics.grids import make_spatial_grid


# Shared test parameters
N = 64
DX_MM = 0.3
WAVELENGTH_MM = 0.75
FOCAL_LENGTH_MM = 109.2
PUPIL_RADIUS_MM = 10.0  # small enough to have inside & outside pixels at N=64


class TestLensShape:
    def test_lens_shape(self):
        """Output shape is (N, N)."""
        t = fresnel_lens_transmission(
            N, DX_MM, WAVELENGTH_MM, FOCAL_LENGTH_MM, PUPIL_RADIUS_MM,
        )
        assert t.shape == (N, N)
        assert t.is_complex()


class TestLensMagnitude:
    def test_lens_unit_magnitude_inside_pupil(self):
        """|t_L| = 1 inside the pupil."""
        t = fresnel_lens_transmission(
            N, DX_MM, WAVELENGTH_MM, FOCAL_LENGTH_MM, PUPIL_RADIUS_MM,
        )
        x, y = make_spatial_grid(N, DX_MM, dtype=torch.float64)
        r_sq = x ** 2 + y ** 2
        inside = r_sq <= PUPIL_RADIUS_MM ** 2

        magnitudes = t[inside].abs()
        torch.testing.assert_close(
            magnitudes,
            torch.ones_like(magnitudes),
            atol=1e-6,
            rtol=0,
        )

    def test_lens_zero_outside_pupil(self):
        """t_L = 0 outside the pupil."""
        t = fresnel_lens_transmission(
            N, DX_MM, WAVELENGTH_MM, FOCAL_LENGTH_MM, PUPIL_RADIUS_MM,
        )
        x, y = make_spatial_grid(N, DX_MM, dtype=torch.float64)
        r_sq = x ** 2 + y ** 2
        outside = r_sq > PUPIL_RADIUS_MM ** 2

        # Must have at least some outside pixels for this test to be meaningful
        assert outside.any(), "No pixels outside pupil; increase grid or decrease pupil"

        magnitudes = t[outside].abs()
        torch.testing.assert_close(
            magnitudes,
            torch.zeros_like(magnitudes),
            atol=1e-7,
            rtol=0,
        )


class TestLensPhase:
    def test_lens_phase_quadratic(self):
        """Phase inside pupil follows -pi * (x^2 + y^2) / (lambda * f)."""
        t = fresnel_lens_transmission(
            N, DX_MM, WAVELENGTH_MM, FOCAL_LENGTH_MM, PUPIL_RADIUS_MM,
        )
        x, y = make_spatial_grid(N, DX_MM, dtype=torch.float64)
        r_sq = x ** 2 + y ** 2
        inside = r_sq <= PUPIL_RADIUS_MM ** 2

        expected_phase = -math.pi * r_sq / (WAVELENGTH_MM * FOCAL_LENGTH_MM)
        expected_phase = expected_phase.float()

        actual_phase = t.angle()

        # Compare phases modulo 2*pi by checking exp(j*phase) agreement
        actual_inside = torch.exp(1j * actual_phase[inside].to(torch.float64))
        expected_inside = torch.exp(1j * expected_phase[inside].to(torch.float64))

        torch.testing.assert_close(
            actual_inside.real,
            expected_inside.real,
            atol=1e-5,
            rtol=0,
        )
        torch.testing.assert_close(
            actual_inside.imag,
            expected_inside.imag,
            atol=1e-5,
            rtol=0,
        )
