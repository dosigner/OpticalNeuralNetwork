from __future__ import annotations

import numpy as np

from tao2019_fd2nn.data.gt_variants import apply_gt_variant


def test_raw_variant_is_passthrough() -> None:
    mask = np.array([[0.1, 0.3], [0.7, 0.9]], dtype=np.float32)

    out = apply_gt_variant(mask, "raw")

    assert np.allclose(out, mask)


def test_binary_variant_outputs_only_zero_or_one() -> None:
    mask = np.array(
        [
            [0.05, 0.10, 0.15, 0.20],
            [0.25, 0.30, 0.60, 0.70],
            [0.10, 0.20, 0.80, 0.90],
            [0.05, 0.10, 0.15, 0.95],
        ],
        dtype=np.float32,
    )

    out = apply_gt_variant(mask, "binary")

    assert set(np.unique(out).tolist()).issubset({0.0, 1.0})
    assert out.sum() > 0.0
    assert out.sum() < out.size


def test_sharpened_variant_increases_peak_to_background_contrast() -> None:
    mask = np.array(
        [
            [0.05, 0.05, 0.05, 0.05, 0.05],
            [0.05, 0.20, 0.25, 0.20, 0.05],
            [0.05, 0.25, 0.40, 0.25, 0.05],
            [0.05, 0.20, 0.25, 0.20, 0.05],
            [0.05, 0.05, 0.05, 0.05, 0.05],
        ],
        dtype=np.float32,
    )

    out = apply_gt_variant(mask, "sharpened")

    baseline_gap = float(mask[2, 2] - mask[0, 0])
    new_gap = float(out[2, 2] - out[0, 0])
    assert new_gap > baseline_gap
