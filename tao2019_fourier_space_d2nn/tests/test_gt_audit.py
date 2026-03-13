from __future__ import annotations

import numpy as np

from tao2019_fd2nn.analysis.gt_audit import compute_mask_metrics


def test_compute_mask_metrics_reports_center_bias_for_centered_blob() -> None:
    mask = np.zeros((7, 7), dtype=np.float32)
    mask[2:5, 2:5] = 1.0

    metrics = compute_mask_metrics(mask)

    assert metrics["foreground_ratio"] > 0.15
    assert metrics["center_offset_norm"] < 0.1
    assert metrics["component_count"] == 1
    assert metrics["largest_component_ratio"] == 1.0


def test_compute_mask_metrics_detects_fragmentation_and_off_center_mass() -> None:
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[0:2, 0:2] = 1.0
    mask[4:6, 1:3] = 1.0

    metrics = compute_mask_metrics(mask)

    assert metrics["component_count"] == 2
    assert metrics["largest_component_ratio"] == 0.5
    assert metrics["center_offset_norm"] > 0.15


def test_compute_mask_metrics_reports_higher_edge_density_for_sharp_mask() -> None:
    sharp = np.zeros((9, 9), dtype=np.float32)
    sharp[2:7, 2:7] = 1.0

    soft = np.zeros((9, 9), dtype=np.float32)
    soft[1:8, 1:8] = 0.2
    soft[2:7, 2:7] = 0.4
    soft[3:6, 3:6] = 0.6

    sharp_metrics = compute_mask_metrics(sharp)
    soft_metrics = compute_mask_metrics(soft)

    assert sharp_metrics["edge_density"] > soft_metrics["edge_density"]
