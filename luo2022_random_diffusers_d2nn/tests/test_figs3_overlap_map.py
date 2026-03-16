"""Tests for Supplementary Figure S3 overlap-map generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from luo2022_d2nn.figures.figs3_overlap_map import (
    _compute_island_mask,
    _compute_overlap_counts,
    make_figs3,
)


def _circular_roi(shape: tuple[int, int], radius_px: float) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    cy = (shape[0] - 1) / 2.0
    cx = (shape[1] - 1) / 2.0
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return rr <= radius_px


def test_compute_island_mask_is_deterministic_and_keeps_smooth_islands():
    rng = np.random.default_rng(123)
    wrapped_phase = rng.uniform(0.0, 2.0 * np.pi, size=(64, 64))

    yy, xx = np.ogrid[:64, :64]
    island_a = (xx - 24) ** 2 + (yy - 24) ** 2 <= 4 ** 2
    island_b = (xx - 40) ** 2 + (yy - 34) ** 2 <= 5 ** 2
    wrapped_phase[island_a] = 0.35
    wrapped_phase[island_b] = 1.85

    roi = _circular_roi(wrapped_phase.shape, radius_px=24.0)
    mask_a = _compute_island_mask(wrapped_phase, roi)
    mask_b = _compute_island_mask(wrapped_phase, roi)

    assert np.array_equal(mask_a, mask_b)
    assert mask_a[24, 24]
    assert mask_a[34, 40]
    assert not mask_a[2, 2]


def test_compute_overlap_counts_marks_pairwise_and_four_way_intersections():
    masks = np.zeros((4, 10, 10), dtype=bool)

    masks[0, 2, 2] = True
    masks[1, 2, 2] = True

    masks[:, 5, 5] = True

    masks[3, 8, 1] = True

    pairwise_counts, all_layer_count_map = _compute_overlap_counts(masks)

    assert pairwise_counts.shape == (4, 10, 10)
    assert pairwise_counts[0, 2, 2] == 2
    assert pairwise_counts[1, 2, 2] == 1
    assert pairwise_counts[3, 8, 1] == 1
    assert all_layer_count_map[5, 5] == 4
    assert all_layer_count_map[8, 1] == 1


def test_make_figs3_generates_png_and_expected_arrays(tmp_path: Path, baseline_config):
    grid_size = 64
    baseline_config["grid"]["nx"] = grid_size
    baseline_config["grid"]["ny"] = grid_size
    baseline_config["geometry"]["num_layers"] = 4
    baseline_config["optics"]["wavelength_mm"] = 0.75

    checkpoint_path = tmp_path / "model.pt"
    config_path = tmp_path / "baseline.yaml"
    save_path = tmp_path / "figS3_overlap_map.png"

    state_dict = {}
    for idx in range(4):
        phase = torch.full((grid_size, grid_size), fill_value=0.3 * (idx + 1))
        state_dict[f"layers.{idx}.phase"] = phase
    torch.save({"model_state_dict": state_dict}, checkpoint_path)
    config_path.write_text(yaml.safe_dump(baseline_config))

    result = make_figs3(
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path),
        save_path=str(save_path),
    )

    assert save_path.exists()
    assert result["wrapped_phases"].shape == (4, grid_size, grid_size)
    assert result["masks"].shape == (4, grid_size, grid_size)
    assert result["pairwise_counts"].shape == (4, grid_size, grid_size)
    assert result["all_layer_count_map"].shape == (grid_size, grid_size)
