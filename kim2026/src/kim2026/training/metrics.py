"""Beam-cleanup metrics."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from kim2026.training.losses import (
    beam_radius,
    encircled_energy_fraction,
    roi_mask_from_target,
    soft_target_support_weights,
)


def gaussian_overlap(pred_intensity: torch.Tensor, target_intensity: torch.Tensor) -> torch.Tensor:
    """Return normalized overlap, higher is better."""
    pred_flat = pred_intensity.reshape(pred_intensity.shape[0], -1)
    target_flat = target_intensity.reshape(target_intensity.shape[0], -1)
    numerator = (pred_flat * target_flat).sum(dim=1)
    denominator = torch.linalg.vector_norm(pred_flat, dim=1) * torch.linalg.vector_norm(target_flat, dim=1)
    return numerator / denominator.clamp_min(1e-12)


def strehl_ratio(pred_intensity: torch.Tensor, target_intensity: torch.Tensor) -> torch.Tensor:
    """Return the peak ratio after unit-energy normalization.

    .. deprecated:: Use :func:`strehl_ratio_correct` for physically valid results.
        This function undersamples the PSF and does not use a flat-phase reference.
    """
    pred_norm = pred_intensity / pred_intensity.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    target_norm = target_intensity / target_intensity.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return pred_norm.amax(dim=(-2, -1)) / target_norm.amax(dim=(-2, -1)).clamp_min(1e-12)


def strehl_ratio_correct(
    pred_field: torch.Tensor,
    pad_factor: int = 4,
) -> torch.Tensor:
    """Correct Strehl ratio: compare a field to its own flat-phase version.

    S = max|FT[A·exp(jφ)]|² / max|FT[A]|²

    where A = |pred_field| (amplitude) and φ = angle(pred_field) (phase).
    The reference is the same amplitude with zero phase (flat wavefront).
    This guarantees S <= 1 by Cauchy--Schwarz, regardless of the
    amplitude distribution.

    Both fields are zero-padded by ``pad_factor`` before FFT so that
    the PSF peak is properly sampled (Nyquist-compliant).

    Parameters
    ----------
    pred_field : Tensor [B, H, W], complex
        Pupil-plane field with aberration/phase structure.
    pad_factor : int
        Zero-padding multiplier (default 4 → Airy ≈ 5 px).

    Returns
    -------
    Tensor [B] — Strehl ratio per sample, in [0, 1].
    """
    from kim2026.optics.fft2c import fft2c

    n = pred_field.shape[-1]
    n_pad = n * pad_factor
    pad = (n_pad - n) // 2

    # Reference: same amplitude, flat phase
    ref_field = pred_field.abs().to(torch.complex64)

    # Zero-pad
    pred_pad = torch.nn.functional.pad(pred_field.to(torch.complex64), [pad, pad, pad, pad])
    ref_pad = torch.nn.functional.pad(ref_field, [pad, pad, pad, pad])

    # FFT to focal plane
    focal_pred = fft2c(pred_pad)
    focal_ref = fft2c(ref_pad)

    # Energy-normalized intensities
    pred_i = focal_pred.abs().square()
    ref_i = focal_ref.abs().square()
    pred_n = pred_i / pred_i.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    ref_n = ref_i / ref_i.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)

    # Peak comparison — guaranteed S <= 1 by Cauchy-Schwarz
    return pred_n.amax(dim=(-2, -1)) / ref_n.amax(dim=(-2, -1)).clamp_min(1e-12)


# ---------------------------------------------------------------------------
# Complex field metrics
# ---------------------------------------------------------------------------


def complex_overlap(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Phase-sensitive overlap |<pred, target>| / (||pred|| ||target||). Higher is better."""
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    inner = (pred_flat * target_flat.conj()).sum(dim=1)
    norm_pred = torch.linalg.vector_norm(pred_flat, dim=1)
    norm_target = torch.linalg.vector_norm(target_flat, dim=1)
    return inner.abs() / (norm_pred * norm_target).clamp_min(1e-12)


def phase_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Phase RMSE in radians after global phase alignment. Lower is better."""
    from kim2026.training.losses import align_global_phase

    aligned = align_global_phase(pred, target)
    amp_threshold = 0.1 * target.abs().amax(dim=(-2, -1), keepdim=True)
    mask = (target.abs() > amp_threshold) & (aligned.abs() > amp_threshold)
    phase_diff = torch.angle(aligned) - torch.angle(target)
    phase_diff = torch.remainder(phase_diff + torch.pi, 2 * torch.pi) - torch.pi
    masked_sq = (phase_diff.square() * mask).sum(dim=(-2, -1)) / mask.sum(dim=(-2, -1)).clamp_min(1)
    return masked_sq.sqrt()


def full_field_phase_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Phase RMSE over the full target support after global phase alignment.

    Pixels outside the predicted support count as maximal phase error.
    """
    from kim2026.training.losses import align_global_phase

    aligned = align_global_phase(pred, target)
    amp_threshold = 0.1 * target.abs().amax(dim=(-2, -1), keepdim=True)
    support = target.abs() > amp_threshold
    phase_diff = torch.angle(aligned) - torch.angle(target)
    phase_diff = torch.remainder(phase_diff + torch.pi, 2 * torch.pi) - torch.pi
    pred_support = aligned.abs() > amp_threshold
    penalized_sq = torch.where(pred_support, phase_diff.square(), torch.full_like(phase_diff, torch.pi**2))
    masked_sq = (penalized_sq * support).sum(dim=(-2, -1)) / support.sum(dim=(-2, -1)).clamp_min(1)
    return masked_sq.sqrt()


def support_weighted_phase_rmse(pred: torch.Tensor, target: torch.Tensor, *, gamma: float = 2.0) -> torch.Tensor:
    """Phase RMSE weighted by soft target support after global phase alignment."""
    from kim2026.training.losses import align_global_phase

    aligned = align_global_phase(pred, target)
    phase_diff = torch.angle(aligned) - torch.angle(target)
    phase_diff = torch.remainder(phase_diff + torch.pi, 2 * torch.pi) - torch.pi
    weights = soft_target_support_weights(target, gamma=gamma)
    weighted_sq = (phase_diff.square() * weights).sum(dim=(-2, -1)) / weights.sum(dim=(-2, -1)).clamp_min(1e-12)
    return weighted_sq.sqrt()


def out_of_support_energy_fraction(pred: torch.Tensor, target: torch.Tensor, *, gamma: float = 2.0) -> torch.Tensor:
    """Fraction of predicted energy outside the soft target support."""
    support = soft_target_support_weights(target, gamma=gamma)
    outside = 1.0 - support
    pred_i = pred.abs().square()
    total = pred_i.sum(dim=(-2, -1)).clamp_min(1e-12)
    return (pred_i * outside).sum(dim=(-2, -1)) / total


def roi_intensity_overlap(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    roi_threshold: float = 0.5,
    window_m: float,
) -> torch.Tensor:
    """Intensity overlap inside the target-derived hard ROI."""
    roi_mask = roi_mask_from_target(target, roi_threshold=roi_threshold, window_m=window_m)
    pred_i = pred.abs().square() * roi_mask
    target_i = target.abs().square() * roi_mask
    return gaussian_overlap(pred_i, target_i)


def roi_support_weighted_phase_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    roi_threshold: float = 0.5,
    gamma: float = 2.0,
    window_m: float,
) -> torch.Tensor:
    """Support-weighted phase RMSE inside the target-derived hard ROI."""
    from kim2026.training.losses import align_global_phase

    roi_mask = roi_mask_from_target(target, roi_threshold=roi_threshold, window_m=window_m)
    pred_roi = pred * roi_mask
    target_roi = target * roi_mask
    aligned = align_global_phase(pred_roi, target_roi)
    phase_diff = torch.angle(aligned) - torch.angle(target_roi)
    phase_diff = torch.remainder(phase_diff + torch.pi, 2 * torch.pi) - torch.pi
    weights = soft_target_support_weights(target_roi, gamma=gamma) * roi_mask
    weighted_sq = (phase_diff.square() * weights).sum(dim=(-2, -1)) / weights.sum(dim=(-2, -1)).clamp_min(1e-12)
    return weighted_sq.sqrt()


def roi_out_of_support_energy_fraction(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    roi_threshold: float = 0.5,
    window_m: float,
) -> torch.Tensor:
    """Fraction of predicted energy outside the target-derived hard ROI."""
    roi_mask = roi_mask_from_target(target, roi_threshold=roi_threshold, window_m=window_m)
    pred_i = pred.abs().square()
    total = pred_i.sum(dim=(-2, -1)).clamp_min(1e-12)
    outside = (pred_i * (1.0 - roi_mask)).sum(dim=(-2, -1))
    return outside / total


def amplitude_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Normalized RMSE between amplitudes. Lower is better."""
    diff = pred.abs() - target.abs()
    peak = target.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return (diff.square().mean(dim=(-2, -1)).sqrt() / peak.squeeze(-1).squeeze(-1))


def beam_cleanup_selection_summary(
    metrics: Mapping[str, float] | dict[str, float],
    *,
    leakage_threshold: float = 0.15,
) -> dict[str, float | bool | str]:
    """Summarize whether metrics pass the leakage gate and how they rank."""
    intensity_key = "roi50_intensity_overlap" if "roi50_intensity_overlap" in metrics else "intensity_overlap"
    if "roi50_support_weighted_phase_rmse_rad" in metrics:
        phase_key = "roi50_support_weighted_phase_rmse_rad"
    elif "support_weighted_phase_rmse_rad" in metrics:
        phase_key = "support_weighted_phase_rmse_rad"
    else:
        phase_key = "phase_rmse_rad"
    leakage_key = (
        "roi50_out_of_support_energy_fraction"
        if "roi50_out_of_support_energy_fraction" in metrics
        else "out_of_support_energy_fraction"
    )

    intensity = float(metrics.get(intensity_key, float("-inf")))
    phase = float(metrics.get(phase_key, float("inf")))
    leakage_available = leakage_key in metrics
    leakage = float(metrics.get(leakage_key, 0.0 if leakage_available else 0.0))
    passes = (not leakage_available) or leakage <= float(leakage_threshold)
    return {
        "intensity_metric_name": intensity_key,
        "intensity_metric_value": intensity,
        "phase_metric_name": phase_key,
        "phase_metric_value": phase,
        "leakage_metric_name": leakage_key,
        "leakage_metric_value": leakage,
        "leakage_metric_available": leakage_available,
        "leakage_threshold": float(leakage_threshold),
        "passes_leakage_gate": passes,
    }


def beam_cleanup_selection_sort_key(
    metrics_or_summary: Mapping[str, float | bool | str] | dict[str, float | bool | str],
    *,
    leakage_threshold: float = 0.15,
) -> tuple[int, float, float]:
    """Leakage-gated best-run sort key: pass gate, maximize overlap, minimize phase RMSE."""
    if "passes_leakage_gate" in metrics_or_summary:
        summary = metrics_or_summary
    else:
        summary = beam_cleanup_selection_summary(metrics_or_summary, leakage_threshold=leakage_threshold)
    if not bool(summary["passes_leakage_gate"]):
        return (0, float("-inf"), float("-inf"))
    return (
        1,
        float(summary["intensity_metric_value"]),
        -float(summary["phase_metric_value"]),
    )


def summarize_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    window_m: float | None = None,
    complex_mode: bool = False,
) -> dict[str, float]:
    """Summarize core cleanup metrics."""
    if complex_mode:
        pred_i = pred.abs().square()
        target_i = target.abs().square()
        return {
            "complex_overlap": float(complex_overlap(pred, target).mean().item()),
            "phase_rmse_rad": float(phase_rmse(pred, target).mean().item()),
            "full_field_phase_rmse_rad": float(full_field_phase_rmse(pred, target).mean().item()),
            "support_weighted_phase_rmse_rad": float(support_weighted_phase_rmse(pred, target).mean().item()),
            "out_of_support_energy_fraction": float(out_of_support_energy_fraction(pred, target).mean().item()),
            "amplitude_rmse": float(amplitude_rmse(pred, target).mean().item()),
            "intensity_overlap": float(gaussian_overlap(pred_i, target_i).mean().item()),
            "strehl": float(strehl_ratio(pred_i, target_i).mean().item()),
            "roi50_intensity_overlap": float(roi_intensity_overlap(pred, target, roi_threshold=0.5, window_m=window_m).mean().item()),
            "roi50_support_weighted_phase_rmse_rad": float(
                roi_support_weighted_phase_rmse(pred, target, roi_threshold=0.5, gamma=2.0, window_m=window_m).mean().item()
            ),
            "roi50_out_of_support_energy_fraction": float(
                roi_out_of_support_energy_fraction(pred, target, roi_threshold=0.5, window_m=window_m).mean().item()
            ),
        }
    pred_intensity = pred
    target_intensity = target
    reference_radius = beam_radius(target_intensity, window_m=window_m)
    pred_encircled = encircled_energy_fraction(pred_intensity, reference_radius=reference_radius, window_m=window_m)
    return {
        "overlap": float(gaussian_overlap(pred_intensity, target_intensity).mean().item()),
        "strehl": float(strehl_ratio(pred_intensity, target_intensity).mean().item()),
        "beam_radius": float(beam_radius(pred_intensity, window_m=window_m).mean().item()),
        "encircled_energy": float(pred_encircled.mean().item()),
    }
