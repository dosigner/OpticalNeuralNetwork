"""Tests for FD2NN model and complex field losses."""

from __future__ import annotations

import math

import pytest
import torch

from kim2026.models.fd2nn import BeamCleanupFD2NN, FourierPhaseMask
from kim2026.optics.fft2c import fft2c, ifft2c
from kim2026.training.losses import (
    align_global_phase,
    amplitude_mse_loss,
    beam_radius_loss,
    complex_field_loss,
    complex_overlap_loss,
    encircled_energy_loss,
    full_field_phase_residual_loss,
    out_of_support_leakage_loss,
    roi_complex_loss,
    soft_target_support_weights,
    soft_weighted_phasor_loss,
)
from kim2026.training.metrics import (
    amplitude_rmse,
    beam_cleanup_selection_sort_key,
    beam_cleanup_selection_summary,
    complex_overlap,
    full_field_phase_rmse,
    out_of_support_energy_fraction,
    phase_rmse,
    support_weighted_phase_rmse,
)


# ---------------------------------------------------------------------------
# FFT2C tests
# ---------------------------------------------------------------------------


class TestFft2c:
    def test_round_trip(self):
        x = torch.randn(1, 32, 32, dtype=torch.complex64)
        y = ifft2c(fft2c(x))
        torch.testing.assert_close(y, x, atol=1e-5, rtol=1e-5)

    def test_parseval(self):
        x = torch.randn(1, 32, 32, dtype=torch.complex64)
        energy_spatial = (x.abs().square()).sum()
        energy_fourier = (fft2c(x).abs().square()).sum()
        torch.testing.assert_close(energy_spatial, energy_fourier, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# FourierPhaseMask tests
# ---------------------------------------------------------------------------


class TestFourierPhaseMask:
    def test_output_shape(self):
        mask = FourierPhaseMask(32, phase_max=math.pi)
        field = torch.randn(2, 32, 32, dtype=torch.complex64)
        out = mask(field)
        assert out.shape == field.shape
        assert out.dtype == torch.complex64

    def test_zero_init_identity(self):
        mask = FourierPhaseMask(32, phase_max=math.pi, init_mode="zeros")
        field = torch.randn(1, 32, 32, dtype=torch.complex64)
        out = mask(field)
        # zero raw -> tanh(0) = 0 -> exp(0) = 1 -> identity
        torch.testing.assert_close(out, field, atol=1e-6, rtol=1e-6)

    def test_phase_range_tanh(self):
        mask = FourierPhaseMask(32, phase_max=math.pi, constraint="symmetric_tanh")
        # Set raw to large values
        mask.raw.data.fill_(100.0)
        phase = mask.phase()
        assert phase.max().item() < math.pi + 1e-6
        assert phase.min().item() > -math.pi - 1e-6

    def test_unconstrained_phase_can_exceed_wrapped_range(self):
        mask = FourierPhaseMask(32, phase_max=math.pi, constraint="unconstrained")
        mask.raw.data.fill_(5.0 * math.pi)
        phase = mask.phase()
        assert phase.max().item() > 2.0 * math.pi

    def test_wrapped_phase_is_exported_in_0_to_2pi(self):
        mask = FourierPhaseMask(4, phase_max=math.pi, constraint="unconstrained")
        mask.raw.data = torch.tensor(
            [
                [-3.0 * math.pi, -0.5 * math.pi, 0.0, 0.25 * math.pi],
                [0.5 * math.pi, 2.0 * math.pi, 2.5 * math.pi, 4.0 * math.pi],
                [-2.25 * math.pi, -1.25 * math.pi, 1.25 * math.pi, 3.25 * math.pi],
                [6.0 * math.pi, -6.0 * math.pi, 0.75 * math.pi, -0.75 * math.pi],
            ]
        )
        wrapped = mask.wrapped_phase()
        assert wrapped.min().item() >= 0.0
        assert wrapped.max().item() < 2.0 * math.pi + 1e-6

    def test_gradient_flow(self):
        mask = FourierPhaseMask(16, phase_max=math.pi)
        field = torch.randn(1, 16, 16, dtype=torch.complex64)
        out = mask(field)
        loss = out.abs().square().sum()
        loss.backward()
        assert mask.raw.grad is not None
        assert mask.raw.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# BeamCleanupFD2NN tests
# ---------------------------------------------------------------------------


class TestBeamCleanupFD2NN:
    def _make_model(self, n=32, num_layers=3, layer_spacing_m=0.0):
        return BeamCleanupFD2NN(
            n=n,
            wavelength_m=1.55e-6,
            window_m=0.2,
            num_layers=num_layers,
            layer_spacing_m=layer_spacing_m,
            phase_max=math.pi,
            phase_constraint="symmetric_tanh",
            phase_init="zeros",
            dual_2f_f1_m=1.0e-3,
            dual_2f_f2_m=1.0e-3,
            dual_2f_na1=None,
            dual_2f_na2=None,
            dual_2f_apply_scaling=False,
        )

    def test_forward_shape(self):
        model = self._make_model()
        field = torch.randn(2, 32, 32, dtype=torch.complex64)
        out = model(field)
        assert out.shape == (2, 32, 32)
        assert out.dtype == torch.complex64

    def test_identity_init(self):
        model = self._make_model()
        field = torch.randn(1, 32, 32, dtype=torch.complex64)
        out = model(field)
        torch.testing.assert_close(out, field.to(torch.complex64), atol=1e-4, rtol=1e-4)

    def test_pure_fourier_stack_supports_five_layers(self):
        model = self._make_model(num_layers=5, layer_spacing_m=1e-4)
        field = torch.randn(1, 32, 32, dtype=torch.complex64)
        out = model(field)
        assert out.shape == field.shape

    def test_gradient_to_all_layers(self):
        model = self._make_model()
        field = torch.randn(1, 32, 32, dtype=torch.complex64)
        out = model(field)
        loss = out.abs().square().sum()
        loss.backward()
        for i, layer in enumerate(model.layers):
            assert layer.raw.grad is not None, f"layer {i} has no gradient"
            assert layer.raw.grad.abs().sum() > 0, f"layer {i} gradient is zero"

    def test_dual_2f_parameters_must_be_positive(self):
        with pytest.raises(ValueError, match="dual_2f_f1_m"):
            BeamCleanupFD2NN(
                n=16,
                wavelength_m=1e-6,
                window_m=0.1,
                num_layers=2,
                layer_spacing_m=1e-4,
                dual_2f_f1_m=0.0,
                dual_2f_f2_m=1e-3,
                dual_2f_na1=0.16,
                dual_2f_na2=0.16,
            )

    def test_layer_spacing_allows_zero_baseline(self):
        model = self._make_model(layer_spacing_m=0.0)
        field = torch.randn(1, 32, 32, dtype=torch.complex64)
        out = model(field)
        assert out.shape == field.shape


# ---------------------------------------------------------------------------
# Complex loss tests
# ---------------------------------------------------------------------------


class TestComplexLosses:
    def test_overlap_identical(self):
        u = torch.randn(2, 16, 16, dtype=torch.complex64)
        loss = complex_overlap_loss(u, u)
        assert loss.item() < 1e-6

    def test_overlap_phase_invariant(self):
        u = torch.randn(2, 16, 16, dtype=torch.complex64)
        v = torch.randn(2, 16, 16, dtype=torch.complex64)
        alpha = torch.tensor([0.7, -1.2])
        u_rotated = u * torch.exp(1j * alpha).reshape(-1, 1, 1)
        loss1 = complex_overlap_loss(u, v)
        loss2 = complex_overlap_loss(u_rotated, v)
        torch.testing.assert_close(loss1, loss2, atol=1e-5, rtol=1e-5)

    def test_amplitude_mse_zero(self):
        u = torch.randn(2, 16, 16, dtype=torch.complex64)
        loss = amplitude_mse_loss(u, u)
        assert loss.item() < 1e-6

    def test_align_global_phase_identity(self):
        u = torch.randn(2, 16, 16, dtype=torch.complex64)
        aligned = align_global_phase(u, u)
        torch.testing.assert_close(aligned, u, atol=1e-5, rtol=1e-5)

    def test_align_global_phase_rotation(self):
        u = torch.randn(2, 16, 16, dtype=torch.complex64)
        alpha = torch.tensor([1.5, -0.8])
        u_rotated = u * torch.exp(1j * alpha).reshape(-1, 1, 1)
        aligned = align_global_phase(u_rotated, u)
        torch.testing.assert_close(aligned, u, atol=1e-4, rtol=1e-4)

    def test_complex_field_loss_composite(self):
        u = torch.randn(2, 16, 16, dtype=torch.complex64, requires_grad=True)
        v = torch.randn(2, 16, 16, dtype=torch.complex64)
        loss = complex_field_loss(u, v, weights={"complex_overlap": 1.0, "amplitude_mse": 0.5})
        assert loss.item() > 0
        assert loss.requires_grad

    def test_complex_field_loss_explicit_weights_do_not_inherit_hidden_amplitude_term(self):
        pred = torch.ones(1, 8, 8, dtype=torch.complex64)
        target = 2.0 * torch.ones(1, 8, 8, dtype=torch.complex64)

        overlap_only = complex_field_loss(pred, target, weights={"complex_overlap": 1.0})
        overlap_plus_amp = complex_field_loss(pred, target, weights={"complex_overlap": 1.0, "amplitude_mse": 0.5})

        assert overlap_only.item() < 1e-6
        assert overlap_plus_amp.item() > overlap_only.item()

    def test_soft_target_support_weights_focus_high_amplitude_region(self):
        target = torch.zeros(1, 5, 5, dtype=torch.complex64)
        target[:, 2, 2] = 1.0 + 0j
        target[:, 2, 1:4] += 0.5 + 0j

        weights = soft_target_support_weights(target, gamma=2.0)

        assert weights[0, 2, 2].item() > weights[0, 0, 0].item()
        assert weights[0, 2, 2].item() > weights[0, 2, 1].item()

    def test_soft_weighted_phasor_loss_is_global_phase_invariant(self):
        target = torch.randn(1, 16, 16, dtype=torch.complex64)
        pred = target * torch.exp(1j * torch.tensor(0.9))

        loss = soft_weighted_phasor_loss(pred, target, gamma=2.0)

        assert loss.item() < 1e-6

    def test_soft_weighted_phasor_loss_penalizes_spatially_varying_phase_error(self):
        target = torch.ones(1, 8, 8, dtype=torch.complex64)
        target[:, :4, :] *= 3.0
        pred = target.clone()
        pred[:, :4, :] *= torch.exp(1j * torch.tensor(math.pi / 3))
        pred[:, 4:, :] *= torch.exp(1j * torch.tensor(-math.pi / 6))

        loss = soft_weighted_phasor_loss(pred, target, gamma=2.0)

        assert loss.item() > 0.0

    def test_full_field_phase_residual_loss_is_zero_for_global_phase_offset(self):
        target = torch.randn(1, 16, 16, dtype=torch.complex64)
        pred = target * torch.exp(1j * torch.tensor(0.7))

        loss = full_field_phase_residual_loss(pred, target, support_threshold=0.05, gamma=1.0)

        assert loss.item() < 1e-6

    def test_full_field_phase_residual_loss_penalizes_spatially_varying_phase_error(self):
        target = torch.ones(1, 16, 16, dtype=torch.complex64)
        pred = target.clone()
        pred[:, :, :8] *= torch.exp(1j * torch.tensor(math.pi / 4))
        pred[:, :, 8:] *= torch.exp(1j * torch.tensor(-math.pi / 6))

        loss = full_field_phase_residual_loss(pred, target, support_threshold=0.05, gamma=1.0)

        assert loss.item() > 0.0

    def test_out_of_support_leakage_loss_penalizes_energy_outside_support(self):
        target_amp = torch.zeros(1, 16, 16)
        target_amp[:, 6:10, 6:10] = 1.0
        target = target_amp.to(torch.complex64)

        inside = torch.zeros(1, 16, 16, dtype=torch.complex64)
        inside[:, 6:10, 6:10] = 1.0 + 0j
        outside = torch.zeros(1, 16, 16, dtype=torch.complex64)
        outside[:, 0:4, 0:4] = 1.0 + 0j

        inside_loss = out_of_support_leakage_loss(inside, target, gamma=2.0)
        outside_loss = out_of_support_leakage_loss(outside, target, gamma=2.0)

        assert outside_loss.item() > inside_loss.item()

    def test_phase_first_complex_field_loss_penalizes_ring_like_leakage(self):
        n = 32
        coords = torch.linspace(-1.0, 1.0, n)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        rr = torch.sqrt(xx.square() + yy.square())
        target_amp = torch.exp(-(rr.square()) / 0.08)
        target = target_amp.to(torch.complex64).unsqueeze(0)

        matched = target.clone().requires_grad_(True)
        ring_amp = torch.exp(-((rr - 0.7).square()) / 0.01)
        ring = ring_amp.to(torch.complex64).unsqueeze(0).requires_grad_(True)

        matched_loss = complex_field_loss(
            matched,
            target,
            weights={"soft_phasor": 1.0, "amplitude_mse": 0.05, "leakage": 0.1, "support_gamma": 2.0},
        )
        ring_loss = complex_field_loss(
            ring,
            target,
            weights={"soft_phasor": 1.0, "amplitude_mse": 0.05, "leakage": 0.1, "support_gamma": 2.0},
        )

        assert ring_loss.item() > matched_loss.item()

    def test_complex_field_loss_can_penalize_collapsed_spot(self):
        n = 32
        coords = torch.linspace(-1.0, 1.0, n)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        target_amp = torch.exp(-(xx.square() + yy.square()) / 0.35)
        pred_amp = torch.exp(-(xx.square() + yy.square()) / 0.02)
        target = target_amp.to(torch.complex64).unsqueeze(0)
        pred = pred_amp.to(torch.complex64).unsqueeze(0).requires_grad_(True)

        base = complex_field_loss(
            pred,
            target,
            weights={"complex_overlap": 1.0, "amplitude_mse": 0.5},
            window_m=2.0,
        )
        shaped = complex_field_loss(
            pred,
            target,
            weights={
                "complex_overlap": 1.0,
                "amplitude_mse": 0.5,
                "intensity_overlap": 1.0,
                "beam_radius": 1.0,
                "encircled_energy": 1.0,
            },
            window_m=2.0,
        )

        assert beam_radius_loss(pred.abs().square(), target.abs().square(), window_m=2.0).item() > 0.0
        assert encircled_energy_loss(pred.abs().square(), target.abs().square(), window_m=2.0).item() > 0.0
        assert shaped.item() > base.item()

    def test_complex_field_loss_full_field_phase_zero_preserves_existing_behavior(self):
        pred = torch.randn(1, 16, 16, dtype=torch.complex64, requires_grad=True)
        target = torch.randn(1, 16, 16, dtype=torch.complex64)

        base = complex_field_loss(
            pred,
            target,
            weights={"complex_overlap": 1.0, "amplitude_mse": 0.5},
            window_m=2.0,
        )
        augmented = complex_field_loss(
            pred,
            target,
            weights={
                "complex_overlap": 1.0,
                "amplitude_mse": 0.5,
                "full_field_phase": 0.0,
                "full_field_phase_gamma": 1.0,
                "full_field_phase_threshold": 0.05,
            },
            window_m=2.0,
        )

        torch.testing.assert_close(augmented, base)

    def test_complex_field_loss_full_field_phase_penalizes_collapsed_spot_when_paired_with_intensity_shape_terms(self):
        n = 32
        coords = torch.linspace(-1.0, 1.0, n)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        target_amp = torch.exp(-(xx.square() + yy.square()) / 0.35)
        target_phase = 0.3 * xx
        target = (target_amp * torch.exp(1j * target_phase)).unsqueeze(0).to(torch.complex64)

        matched = target.clone()
        collapsed_amp = torch.exp(-(xx.square() + yy.square()) / 0.02)
        collapsed = (collapsed_amp * torch.exp(1j * target_phase)).unsqueeze(0).to(torch.complex64)

        matched_loss = complex_field_loss(
            matched,
            target,
            weights={
                "amplitude_mse": 0.05,
                "leakage": 0.1,
                "intensity_overlap": 1.0,
                "full_field_phase": 0.15,
                "full_field_phase_gamma": 1.0,
                "full_field_phase_threshold": 0.05,
            },
            window_m=2.0,
        )
        collapsed_loss = complex_field_loss(
            collapsed,
            target,
            weights={
                "amplitude_mse": 0.05,
                "leakage": 0.1,
                "intensity_overlap": 1.0,
                "full_field_phase": 0.15,
                "full_field_phase_gamma": 1.0,
                "full_field_phase_threshold": 0.05,
            },
            window_m=2.0,
        )

        assert collapsed_loss.item() > matched_loss.item()

    def test_roi_complex_loss_prefers_roi50_irradiance_overlap_over_phase_when_phase_is_secondary(self):
        n = 32
        coords = torch.linspace(-1.0, 1.0, n)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        target_amp = torch.exp(-(xx.square() + yy.square()) / 0.12)
        target = target_amp.to(torch.complex64).unsqueeze(0)

        phase_mismatch = target.clone()
        phase_mismatch[:, :, : n // 2] *= torch.exp(1j * torch.tensor(math.pi / 2))

        shifted_amp = torch.exp(-(((xx - 0.35).square()) + yy.square()) / 0.12)
        shifted = shifted_amp.to(torch.complex64).unsqueeze(0)

        phase_first = roi_complex_loss(
            phase_mismatch,
            target,
            roi_threshold=0.5,
            intensity_weight=1.0,
            phase_weight=0.2,
            leakage_weight=0.3,
            phase_gamma=2.0,
            window_m=2.0,
        )
        overlap_worse = roi_complex_loss(
            shifted,
            target,
            roi_threshold=0.5,
            intensity_weight=1.0,
            phase_weight=0.2,
            leakage_weight=0.3,
            phase_gamma=2.0,
            window_m=2.0,
        )

        assert phase_first.item() < overlap_worse.item()

    def test_roi_complex_loss_penalizes_leakage_outside_roi50(self):
        n = 32
        coords = torch.linspace(-1.0, 1.0, n)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        rr = torch.sqrt(xx.square() + yy.square())
        target_amp = torch.exp(-(rr.square()) / 0.08)
        target = target_amp.to(torch.complex64).unsqueeze(0)

        clean = target.clone()
        leaky = target.clone()
        leaky += (0.4 * torch.exp(-((rr - 0.75).square()) / 0.01)).to(torch.complex64).unsqueeze(0)

        clean_loss = roi_complex_loss(
            clean,
            target,
            roi_threshold=0.5,
            intensity_weight=1.0,
            phase_weight=0.2,
            leakage_weight=0.3,
            phase_gamma=2.0,
            window_m=2.0,
        )
        leaky_loss = roi_complex_loss(
            leaky,
            target,
            roi_threshold=0.5,
            intensity_weight=1.0,
            phase_weight=0.2,
            leakage_weight=0.3,
            phase_gamma=2.0,
            window_m=2.0,
        )

        assert leaky_loss.item() > clean_loss.item()

    def test_roi_complex_loss_adds_full_field_phase_auxiliary_term(self):
        n = 32
        coords = torch.linspace(-1.0, 1.0, n)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        target_amp = torch.exp(-(xx.square() + yy.square()) / 0.15)
        target_phase = 0.4 * xx
        target = (target_amp * torch.exp(1j * target_phase)).unsqueeze(0).to(torch.complex64)

        pred = target.clone()
        pred[:, :, : n // 2] *= torch.exp(1j * torch.tensor(math.pi / 2))

        base = roi_complex_loss(
            pred,
            target,
            roi_threshold=0.5,
            intensity_weight=1.0,
            phase_weight=0.2,
            leakage_weight=0.3,
            phase_gamma=2.0,
            full_field_phase_weight=0.0,
            full_field_phase_gamma=1.0,
            full_field_phase_threshold=0.05,
            window_m=2.0,
        )
        augmented = roi_complex_loss(
            pred,
            target,
            roi_threshold=0.5,
            intensity_weight=1.0,
            phase_weight=0.2,
            leakage_weight=0.3,
            phase_gamma=2.0,
            full_field_phase_weight=0.15,
            full_field_phase_gamma=1.0,
            full_field_phase_threshold=0.05,
            window_m=2.0,
        )

        assert augmented.item() > base.item()


# ---------------------------------------------------------------------------
# Complex metrics tests
# ---------------------------------------------------------------------------


class TestComplexLossesExtra:
    """Design Section 4.1 누락 테스트."""

    def test_complex_overlap_orthogonal(self):
        # 직교 필드: 내적 = 0 -> loss = 1
        u = torch.zeros(1, 16, 16, dtype=torch.complex64)
        v = torch.zeros(1, 16, 16, dtype=torch.complex64)
        u[0, 0, 0] = 1.0
        v[0, 0, 1] = 1.0
        loss = complex_overlap_loss(u, v)
        assert abs(loss.item() - 1.0) < 1e-5

    def test_phase_rmse_known_offset(self):
        # Spatially varying phase error (global phase alignment 제거 불가)
        amp = torch.ones(1, 16, 16)
        target = torch.complex(amp, torch.zeros_like(amp))
        # 반쪽에만 pi/4 위상 추가 -> alignment 후에도 잔여 phase error
        phase_err = torch.zeros(1, 16, 16)
        phase_err[:, :8, :] = math.pi / 4
        pred = target * torch.exp(1j * phase_err)
        rmse = phase_rmse(pred, target)
        assert rmse.item() > 0.1  # 유의미한 phase error 존재

    def test_full_field_phase_rmse_penalizes_low_support_predictions(self):
        target = torch.ones(1, 16, 16, dtype=torch.complex64)
        pred = torch.zeros(1, 16, 16, dtype=torch.complex64)
        pred[:, 7:9, 7:9] = 1.0 + 0j

        masked = phase_rmse(pred, target)
        full = full_field_phase_rmse(pred, target)

        assert masked.item() < 1e-6
        assert full.item() > 0.5

    def test_make_target_complex_mode(self):
        from kim2026.training.targets import make_detector_plane_target

        field = torch.randn(1, 32, 32, dtype=torch.complex64)
        target = make_detector_plane_target(
            field,
            wavelength_m=1.55e-6,
            receiver_window_m=0.2,
            aperture_diameter_m=0.1,
            total_distance_m=0.0,
            complex_mode=True,
        )
        assert target.is_complex()

    def test_make_target_intensity_mode(self):
        from kim2026.training.targets import make_detector_plane_target

        field = torch.randn(1, 32, 32, dtype=torch.complex64)
        target = make_detector_plane_target(
            field,
            wavelength_m=1.55e-6,
            receiver_window_m=0.2,
            aperture_diameter_m=0.1,
            total_distance_m=0.0,
            complex_mode=False,
        )
        assert not target.is_complex()


class TestConfigValidation:
    """Design Section 4.1 config 테스트."""

    def test_config_complex_mode_valid(self):
        from kim2026.config.schema import validate_config

        cfg = _make_minimal_config()
        cfg["training"]["loss"]["mode"] = "complex"
        cfg["training"]["loss"]["complex_weights"] = {"complex_overlap": 1.0, "amplitude_mse": 0.5}
        result = validate_config(cfg)
        assert result["training"]["loss"]["mode"] == "complex"

    def test_config_intensity_mode_default(self):
        from kim2026.config.schema import validate_config

        cfg = _make_minimal_config()
        # mode 미지정 -> default "intensity"
        result = validate_config(cfg)
        assert result["training"]["loss"]["mode"] == "intensity"

    def test_config_fd2nn_valid(self):
        from kim2026.config.schema import validate_config

        cfg = _make_minimal_config()
        cfg["optics"]["dual_2f"] = {
            "enabled": True,
            "f1_m": 1.0e-3,
            "f2_m": 1.0e-3,
            "na1": 0.16,
            "na2": 0.16,
            "apply_scaling": False,
        }
        cfg["model"]["type"] = "fd2nn"
        cfg["model"]["num_layers"] = 3
        cfg["model"]["layer_spacing_m"] = 0.0
        cfg["model"].pop("detector_distance_m", None)
        result = validate_config(cfg)
        assert result["model"]["type"] == "fd2nn"

    def test_config_d2nn_backward_compat(self):
        from kim2026.config.schema import validate_config

        cfg = _make_minimal_config()
        # type 미지정 -> default "d2nn"
        result = validate_config(cfg)
        assert result["model"]["type"] == "d2nn"

    def test_config_roi_complex_defaults_to_roi50_overlap_first_objective(self):
        from kim2026.config.schema import validate_config

        cfg = _make_minimal_config()
        cfg["training"]["loss"] = {"mode": "roi_complex"}

        result = validate_config(cfg)

        assert result["training"]["loss"]["roi_threshold"] == pytest.approx(0.5)
        assert result["training"]["loss"]["intensity_weight"] == pytest.approx(1.0)
        assert result["training"]["loss"]["phase_weight"] == pytest.approx(0.2)
        assert result["training"]["loss"]["leakage_weight"] == pytest.approx(0.3)
        assert result["training"]["loss"]["leakage_threshold"] == pytest.approx(0.15)
        assert result["training"]["loss"]["phase_gamma"] == pytest.approx(2.0)


def _make_minimal_config() -> dict:
    """최소 유효 config for testing."""
    return {
        "experiment": {"id": "test"},
        "optics": {"lambda_m": 1.55e-6, "half_angle_rad": 3e-4, "m2": 1.0},
        "grid": {"n": 32, "source_window_m": 0.01, "receiver_window_m": 0.2},
        "channel": {
            "path_length_m": 100.0, "cn2": 1e-15, "outer_scale_m": 30.0,
            "inner_scale_m": 5e-3, "num_screens": 2, "mode": "static",
            "num_realizations": 10,
        },
        "receiver": {"aperture_diameter_m": 0.15},
        "model": {
            "type": "d2nn", "num_layers": 2,
            "layer_spacing_m": 50.0, "detector_distance_m": 50.0,
        },
        "training": {
            "epochs": 1, "batch_size": 2,
            "loss": {"weights": {"overlap": 1.0, "radius": 0.25, "encircled": 0.25}},
        },
        "data": {"cache_dir": "/tmp/test_cache", "split_manifest_path": "/tmp/manifest.json"},
        "evaluation": {},
        "visualization": {},
        "runtime": {"seed": 42},
    }


class TestComplexMetrics:
    def test_complex_overlap_identical(self):
        u = torch.randn(2, 16, 16, dtype=torch.complex64)
        overlap = complex_overlap(u, u)
        torch.testing.assert_close(overlap, torch.ones(2), atol=1e-5, rtol=1e-5)

    def test_phase_rmse_zero(self):
        u = torch.randn(2, 16, 16, dtype=torch.complex64)
        rmse = phase_rmse(u, u)
        assert rmse.max().item() < 1e-5

    def test_amplitude_rmse_zero(self):
        u = torch.randn(2, 16, 16, dtype=torch.complex64)
        rmse = amplitude_rmse(u, u)
        assert rmse.max().item() < 1e-5

    def test_support_weighted_phase_rmse_is_small_for_global_phase_offset(self):
        target = torch.randn(1, 16, 16, dtype=torch.complex64)
        pred = target * torch.exp(1j * torch.tensor(1.1))

        metric = support_weighted_phase_rmse(pred, target, gamma=2.0)

        assert metric.item() < 1e-6

    def test_out_of_support_energy_fraction_detects_leakage(self):
        target = torch.zeros(1, 16, 16, dtype=torch.complex64)
        target[:, 6:10, 6:10] = 1.0 + 0j
        pred = torch.zeros(1, 16, 16, dtype=torch.complex64)
        pred[:, 0:4, 0:4] = 1.0 + 0j

        metric = out_of_support_energy_fraction(pred, target, gamma=2.0)

        assert metric.item() > 0.5

    def test_beam_cleanup_selection_summary_rejects_runs_above_leakage_threshold(self):
        leaky = beam_cleanup_selection_summary(
            {
                "intensity_overlap": 0.94,
                "support_weighted_phase_rmse_rad": 0.4,
                "out_of_support_energy_fraction": 0.16,
            },
            leakage_threshold=0.15,
        )
        safe = beam_cleanup_selection_summary(
            {
                "intensity_overlap": 0.90,
                "support_weighted_phase_rmse_rad": 0.5,
                "out_of_support_energy_fraction": 0.10,
            },
            leakage_threshold=0.15,
        )

        assert leaky["passes_leakage_gate"] is False
        assert safe["passes_leakage_gate"] is True
        assert beam_cleanup_selection_sort_key(leaky) < beam_cleanup_selection_sort_key(safe)
