"""Detector ROI masks and integration helpers."""

from __future__ import annotations

import torch

from tao2019_fd2nn.optics.scaling import um_to_m


def make_detector_masks(
    N: int,
    dx_m: float,
    *,
    num_classes: int = 10,
    width_um: float = 12.0,
    gap_um: float = 4.0,
    layout: str = "default10",
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create deterministic detector ROI masks.

    Supported layouts:
    - default10: 3-4-3 tiled square ROIs (MNIST/Fashion-MNIST style)
    - row10: single horizontal row of 10 square ROIs near image bottom
    """

    dev = torch.device(device) if device is not None else torch.device("cpu")
    masks = torch.zeros((num_classes, N, N), dtype=torch.bool, device=dev)

    width_px = max(1, int(round(um_to_m(width_um) / dx_m)))
    gap_px = max(0, int(round(um_to_m(gap_um) / dx_m)))

    layout_key = str(layout).lower()
    if layout_key in {"default10", "mnist_3_4_3", "3_4_3"}:
        if num_classes != 10:
            raise ValueError("layout 'default10' requires num_classes=10")

        # Match the 3-4-3 detector arrangement used in the lin2018 detector layouts.
        row_counts = (3, 4, 3)
        row_gap_mul = (3, 2, 3)
        row_pitch_px = width_px + 2 * gap_px
        yc = N // 2
        row_centers = (yc - row_pitch_px, yc, yc + row_pitch_px)

        det_idx = 0
        for n_det, gap_mul, y_center in zip(row_counts, row_gap_mul, row_centers):
            gap_row = gap_mul * gap_px
            span = n_det * width_px + (n_det - 1) * gap_row
            x0 = (N - span) // 2
            y0 = y_center - width_px // 2
            y1 = y0 + width_px
            yy0 = max(0, y0)
            yy1 = min(N, y1)
            for k in range(n_det):
                left = x0 + k * (width_px + gap_row)
                right = left + width_px
                xx0 = max(0, left)
                xx1 = min(N, right)
                if yy0 < yy1 and xx0 < xx1:
                    masks[det_idx, yy0:yy1, xx0:xx1] = True
                det_idx += 1
        return masks

    if layout_key in {"row10", "strip10", "horizontal"}:
        span = num_classes * width_px + (num_classes - 1) * gap_px
        x0 = (N - span) // 2
        y_center = int(round(0.85 * N))
        y0 = max(0, y_center - width_px // 2)
        y1 = min(N, y0 + width_px)
        for k in range(num_classes):
            left = x0 + k * (width_px + gap_px)
            right = min(N, left + width_px)
            if left < N and right > 0 and y0 < y1:
                masks[k, y0:y1, max(left, 0) : right] = True
        return masks

    raise ValueError(f"unknown detector layout: {layout}")


def integrate_detector_energies(intensity_map: torch.Tensor, detector_masks: torch.Tensor) -> torch.Tensor:
    """Integrate detector energies.

    Args:
        intensity_map: (B, N, N)
        detector_masks: (K, N, N)
    Returns:
        energies: (B, K)
    """

    bsz = intensity_map.shape[0]
    masks = detector_masks.to(device=intensity_map.device, dtype=intensity_map.dtype)
    flat_i = intensity_map.reshape(bsz, -1)
    flat_m = masks.reshape(masks.shape[0], -1)
    return flat_i @ flat_m.T
