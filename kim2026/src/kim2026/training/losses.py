"""Beam-cleanup losses."""

from __future__ import annotations

import torch


def _coordinate_grid(size: int, *, window_m: float | None, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    span = float(window_m) if window_m is not None else 2.0
    step = span / size
    coords = (torch.arange(size, device=device, dtype=dtype) - (size // 2)) * step
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    return xx, yy


def normalized_overlap_loss(pred_intensity: torch.Tensor, target_intensity: torch.Tensor) -> torch.Tensor:
    """Return 1 - normalized overlap between two intensity maps."""
    pred_flat = pred_intensity.reshape(pred_intensity.shape[0], -1)
    target_flat = target_intensity.reshape(target_intensity.shape[0], -1)
    numerator = (pred_flat * target_flat).sum(dim=1)
    denominator = torch.linalg.vector_norm(pred_flat, dim=1) * torch.linalg.vector_norm(target_flat, dim=1)
    overlap = numerator / denominator.clamp_min(1e-12)
    return (1.0 - overlap).mean()


def beam_radius(intensity: torch.Tensor, *, window_m: float | None = None) -> torch.Tensor:
    """Compute the 1/e^2-equivalent beam radius from second moments."""
    xx, yy = _coordinate_grid(
        intensity.shape[-1],
        window_m=window_m,
        device=intensity.device,
        dtype=intensity.dtype,
    )
    total = intensity.sum(dim=(-2, -1)).clamp_min(1e-12)
    r2 = (intensity * (xx.square() + yy.square())).sum(dim=(-2, -1)) / total
    return torch.sqrt(2.0 * r2.clamp_min(0.0))


def beam_radius_loss(pred_intensity: torch.Tensor, target_intensity: torch.Tensor, *, window_m: float | None = None) -> torch.Tensor:
    """MSE on beam radius."""
    pred_radius = beam_radius(pred_intensity, window_m=window_m)
    target_radius = beam_radius(target_intensity, window_m=window_m)
    return torch.mean((pred_radius - target_radius).square())


def encircled_energy_fraction(
    intensity: torch.Tensor,
    *,
    reference_radius: torch.Tensor,
    window_m: float | None = None,
) -> torch.Tensor:
    """Compute fraction of energy within the provided reference radius."""
    xx, yy = _coordinate_grid(
        intensity.shape[-1],
        window_m=window_m,
        device=intensity.device,
        dtype=intensity.dtype,
    )
    r = torch.sqrt(xx.square() + yy.square())
    mask = r.unsqueeze(0) <= reference_radius.view(-1, 1, 1)
    total = intensity.sum(dim=(-2, -1)).clamp_min(1e-12)
    inside = (intensity * mask.to(intensity.dtype)).sum(dim=(-2, -1))
    return inside / total


def encircled_energy_loss(pred_intensity: torch.Tensor, target_intensity: torch.Tensor, *, window_m: float | None = None) -> torch.Tensor:
    """MSE on encircled energy using target beam radius as the reference radius."""
    reference_radius = beam_radius(target_intensity, window_m=window_m)
    pred_fraction = encircled_energy_fraction(pred_intensity, reference_radius=reference_radius, window_m=window_m)
    target_fraction = encircled_energy_fraction(target_intensity, reference_radius=reference_radius, window_m=window_m)
    return torch.mean((pred_fraction - target_fraction).square())


# ---------------------------------------------------------------------------
# Complex field losses
# ---------------------------------------------------------------------------


def align_global_phase(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Remove global phase offset between pred and target.

    Returns pred multiplied by exp(-j*alpha) where alpha minimizes
    ||pred*exp(-j*alpha) - target||^2.
    """
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    inner = (pred_flat * target_flat.conj()).sum(dim=1)
    alpha = torch.angle(inner)
    correction = torch.exp(-1j * alpha).to(pred.dtype)
    return pred * correction.reshape(-1, *([1] * (pred.ndim - 1)))


def soft_target_support_weights(target: torch.Tensor, *, gamma: float = 2.0) -> torch.Tensor:
    """Return soft support weights derived from target amplitude."""
    gamma = float(gamma)
    target_amp = target.abs()
    peak = target_amp.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return (target_amp / peak).clamp_min(0.0).pow(gamma)


def complex_overlap_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - |<pred, target>| / (||pred|| * ||target||).

    Phase-sensitive normalized overlap, invariant to global phase offset.
    """
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    inner = (pred_flat * target_flat.conj()).sum(dim=1)
    norm_pred = torch.linalg.vector_norm(pred_flat, dim=1)
    norm_target = torch.linalg.vector_norm(target_flat, dim=1)
    overlap = inner.abs() / (norm_pred * norm_target).clamp_min(1e-12)
    return (1.0 - overlap).mean()


def amplitude_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE between amplitudes: mean(||pred| - |target||^2)."""
    return torch.mean((pred.abs() - target.abs()).square())


def phasor_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Unit phasor MSE: pure wavefront loss ignoring amplitude.

    Computes |pred/|pred| - target/|target||^2 with amplitude threshold mask.
    Mathematically equivalent to 2(1 - cos(Δφ)) over masked pixels.
    """
    amp_threshold = 0.1 * target.abs().amax(dim=(-2, -1), keepdim=True)
    mask = (target.abs() > amp_threshold) & (pred.abs() > amp_threshold)
    phasor_pred = pred / pred.abs().clamp_min(1e-8)
    phasor_target = target / target.abs().clamp_min(1e-8)
    diff_sq = (phasor_pred - phasor_target).abs().square()
    masked_loss = (diff_sq * mask).sum(dim=(-2, -1)) / mask.sum(dim=(-2, -1)).clamp_min(1)
    return masked_loss.mean()


def soft_weighted_phasor_loss(pred: torch.Tensor, target: torch.Tensor, *, gamma: float = 2.0) -> torch.Tensor:
    """Weighted unit-phasor MSE using soft target support weights."""
    aligned = align_global_phase(pred, target)
    phasor_pred = aligned / aligned.abs().clamp_min(1e-8)
    phasor_target = target / target.abs().clamp_min(1e-8)
    weights = soft_target_support_weights(target, gamma=gamma)
    diff_sq = (phasor_pred - phasor_target).abs().square()
    weighted = (diff_sq * weights).sum(dim=(-2, -1)) / weights.sum(dim=(-2, -1)).clamp_min(1e-12)
    return weighted.mean()


def full_field_phase_residual_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    support_threshold: float = 0.05,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Bounded residual-phase surrogate over the full target support.

    This aligns away the global phase offset, derives support from the target
    amplitude, and penalizes detector-plane residual phase using the unit-phasor
    surrogate 0.25 * |u_hat_pred - u_hat_target|^2 in [0, 1].
    """
    aligned = align_global_phase(pred, target)
    target_amp = target.abs()
    peak = target_amp.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    support = (target_amp > float(support_threshold) * peak).to(target_amp.dtype)
    weights = (target_amp / peak).clamp_min(0.0).pow(float(gamma)) * support
    phasor_pred = aligned / aligned.abs().clamp_min(1e-8)
    phasor_target = target / target.abs().clamp_min(1e-8)
    residual = 0.25 * (phasor_pred - phasor_target).abs().square()
    weighted = (residual * weights).sum(dim=(-2, -1)) / weights.sum(dim=(-2, -1)).clamp_min(1e-12)
    return weighted.mean()


def out_of_support_leakage_loss(pred: torch.Tensor, target: torch.Tensor, *, gamma: float = 2.0) -> torch.Tensor:
    """Penalize predicted energy leaking outside the soft target support."""
    weights = soft_target_support_weights(target, gamma=gamma)
    outside = 1.0 - weights
    intensity = pred.abs().square()
    total = intensity.sum(dim=(-2, -1)).clamp_min(1e-12)
    leakage = (intensity * outside).sum(dim=(-2, -1)) / total
    return leakage.mean()


def complex_field_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    weights: dict[str, float] | None = None,
    window_m: float | None = None,
) -> torch.Tensor:
    """Composite complex field loss."""
    if weights is None:
        weights = {"complex_overlap": 1.0, "amplitude_mse": 0.5}
        w_overlap_default = 1.0
        w_amp_default = 0.5
    else:
        # Explicit weight dictionaries should not inherit hidden loss terms.
        w_overlap_default = 0.0
        w_amp_default = 0.0
    w_overlap = float(weights.get("complex_overlap", w_overlap_default))
    w_amp = float(weights.get("amplitude_mse", w_amp_default))
    w_i_overlap = float(weights.get("intensity_overlap", 0.0))
    w_radius = float(weights.get("beam_radius", 0.0))
    w_encircled = float(weights.get("encircled_energy", 0.0))
    w_phasor = float(weights.get("phasor_mse", 0.0))
    w_soft_phasor = float(weights.get("soft_phasor", 0.0))
    w_leakage = float(weights.get("leakage", 0.0))
    w_full_field_phase = float(weights.get("full_field_phase", 0.0))
    support_gamma = float(weights.get("support_gamma", 2.0))
    full_field_phase_gamma = float(weights.get("full_field_phase_gamma", 1.0))
    full_field_phase_threshold = float(weights.get("full_field_phase_threshold", 0.05))
    terms = []
    if w_soft_phasor > 0:
        terms.append(w_soft_phasor * soft_weighted_phasor_loss(pred, target, gamma=support_gamma))
    if w_phasor > 0:
        terms.append(w_phasor * phasor_mse_loss(pred, target))
    if w_overlap > 0:
        terms.append(w_overlap * complex_overlap_loss(pred, target))
    if w_amp > 0:
        terms.append(w_amp * amplitude_mse_loss(pred, target))
    if w_leakage > 0:
        terms.append(w_leakage * out_of_support_leakage_loss(pred, target, gamma=support_gamma))
    if w_full_field_phase > 0:
        terms.append(
            w_full_field_phase
            * full_field_phase_residual_loss(
                pred,
                target,
                support_threshold=full_field_phase_threshold,
                gamma=full_field_phase_gamma,
            )
        )
    if w_i_overlap > 0:
        terms.append(w_i_overlap * normalized_overlap_loss(pred.abs().square(), target.abs().square()))
    if w_radius > 0:
        terms.append(w_radius * beam_radius_loss(pred.abs().square(), target.abs().square(), window_m=window_m))
    if w_encircled > 0:
        terms.append(w_encircled * encircled_energy_loss(pred.abs().square(), target.abs().square(), window_m=window_m))
    if not terms:
        return torch.tensor(0.0, device=pred.device, dtype=torch.float32)
    return sum(terms)


# ---------------------------------------------------------------------------
# ROI-based complex loss
# ---------------------------------------------------------------------------


def _encircled_energy_roi_radius(
    intensity: torch.Tensor,
    *,
    threshold: float,
    window_m: float,
) -> torch.Tensor:
    """Find the radius where encircled energy of *target* reaches *threshold*.

    Args:
        intensity: [B, H, W] real-valued intensity map.
        threshold: Fraction of total energy (e.g. 0.9 for 90%).
        window_m: Physical window size in metres.

    Returns:
        Tensor of shape [B] with the ROI radius in metres per sample.
    """
    n = intensity.shape[-1]
    xx, yy = _coordinate_grid(n, window_m=window_m, device=intensity.device, dtype=intensity.dtype)
    r = torch.sqrt(xx.square() + yy.square())  # [H, W]
    r_flat = r.reshape(-1)
    sorted_r, sort_idx = r_flat.sort()

    radii = []
    for b in range(intensity.shape[0]):
        i_flat = intensity[b].reshape(-1)
        i_sorted = i_flat[sort_idx]
        cumsum = i_sorted.cumsum(0)
        total = cumsum[-1].clamp_min(1e-12)
        idx = (cumsum >= threshold * total).nonzero(as_tuple=True)[0]
        radii.append(sorted_r[idx[0]] if idx.numel() > 0 else sorted_r[-1])
    return torch.stack(radii)


def _circular_roi_mask(
    n: int,
    *,
    radius: torch.Tensor,
    window_m: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Hard circular mask: 1 inside *radius*, 0 outside.  [B, H, W]."""
    xx, yy = _coordinate_grid(n, window_m=window_m, device=device, dtype=dtype)
    r = torch.sqrt(xx.square() + yy.square())  # [H, W]
    return (r.unsqueeze(0) <= radius.view(-1, 1, 1)).to(dtype)


def roi_mask_from_target(
    target: torch.Tensor,
    *,
    roi_threshold: float = 0.5,
    window_m: float,
) -> torch.Tensor:
    """Return a hard circular ROI mask derived from target encircled energy."""
    target_intensity = target.abs().square()
    roi_radius = _encircled_energy_roi_radius(
        target_intensity,
        threshold=roi_threshold,
        window_m=window_m,
    )
    return _circular_roi_mask(
        target.shape[-1],
        radius=roi_radius,
        window_m=window_m,
        device=target.device,
        dtype=target_intensity.dtype,
    )


def _roi_support_weighted_phase_rmse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    roi_mask: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Support-weighted phase RMSE inside the hard ROI mask."""
    pred_roi = pred * roi_mask
    target_roi = target * roi_mask
    aligned = align_global_phase(pred_roi, target_roi)
    phase_diff = torch.angle(aligned) - torch.angle(target_roi)
    phase_diff = torch.remainder(phase_diff + torch.pi, 2 * torch.pi) - torch.pi
    weights = soft_target_support_weights(target_roi, gamma=gamma) * roi_mask
    weighted_sq = (phase_diff.square() * weights).sum(dim=(-2, -1)) / weights.sum(dim=(-2, -1)).clamp_min(1e-12)
    return weighted_sq.sqrt().mean()


def roi_complex_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    roi_threshold: float = 0.5,
    intensity_weight: float = 1.0,
    phase_weight: float = 0.2,
    leakage_weight: float = 0.3,
    phase_gamma: float = 2.0,
    full_field_phase_weight: float = 0.0,
    full_field_phase_gamma: float = 1.0,
    full_field_phase_threshold: float = 0.05,
    window_m: float,
) -> torch.Tensor:
    """ROI50 cleanup loss: overlap first, phase second, leakage third.

    Args:
        pred: [B, H, W] complex predicted field at detector.
        target: [B, H, W] complex vacuum-beam target at detector.
        roi_threshold: Encircled energy fraction defining the ROI.
        intensity_weight: Weight for ROI irradiance overlap loss.
        phase_weight: Weight for ROI support-weighted phase RMSE.
        leakage_weight: Weight for the energy-leakage penalty term.
        phase_gamma: Soft-support exponent used inside the ROI phase term.
        full_field_phase_weight: Weight for the full-support residual-phase auxiliary term.
        full_field_phase_gamma: Soft-support exponent used in the auxiliary term.
        full_field_phase_threshold: Support threshold used in the auxiliary term.
        window_m: Physical window size in metres.
    """
    roi_mask = roi_mask_from_target(target, roi_threshold=roi_threshold, window_m=window_m)

    # Term 1: ROI irradiance overlap
    pred_intensity = pred.abs().square()
    target_intensity = target.abs().square()
    intensity_term = normalized_overlap_loss(pred_intensity * roi_mask, target_intensity * roi_mask)

    # Term 2: ROI support-weighted phase fidelity
    phase_term = _roi_support_weighted_phase_rmse_loss(
        pred,
        target,
        roi_mask=roi_mask,
        gamma=phase_gamma,
    )

    # Term 3: energy leakage outside ROI
    total_energy = pred_intensity.sum(dim=(-2, -1)).clamp_min(1e-12)
    outside_energy = (pred_intensity * (1.0 - roi_mask)).sum(dim=(-2, -1))
    leakage_term = (outside_energy / total_energy).mean()

    loss = (
        float(intensity_weight) * intensity_term
        + float(phase_weight) * phase_term
        + float(leakage_weight) * leakage_term
    )
    if float(full_field_phase_weight) > 0:
        loss = loss + float(full_field_phase_weight) * full_field_phase_residual_loss(
            pred,
            target,
            support_threshold=full_field_phase_threshold,
            gamma=full_field_phase_gamma,
        )
    return loss


def beam_cleanup_loss(
    pred_intensity: torch.Tensor,
    target_intensity: torch.Tensor,
    *,
    window_m: float | None = None,
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Composite beam-cleanup loss."""
    if weights is None:
        weights = {"overlap": 1.0, "radius": 0.25, "encircled": 0.25}
    return (
        float(weights.get("overlap", 1.0)) * normalized_overlap_loss(pred_intensity, target_intensity)
        + float(weights.get("radius", 0.25)) * beam_radius_loss(pred_intensity, target_intensity, window_m=window_m)
        + float(weights.get("encircled", 0.25)) * encircled_energy_loss(pred_intensity, target_intensity, window_m=window_m)
    )
