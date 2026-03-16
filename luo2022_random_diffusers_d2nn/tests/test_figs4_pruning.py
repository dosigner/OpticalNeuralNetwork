"""Tests for Supplementary Figure S4 pruning workflow."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import yaml

from luo2022_d2nn.diffuser.random_phase import generate_diffuser
from luo2022_d2nn.figures.figs4_pruning import (
    CONDITION_ORDER,
    _forward_without_diffractive_layers,
    build_condition_masks,
    load_ood_amplitude_from_image,
    make_figs4,
    materialize_condition_phases,
)
from luo2022_d2nn.models.d2nn import D2NN
from luo2022_d2nn.optics.bl_asm import bl_asm_propagate, bl_asm_transfer_function


def _toy_base_masks() -> tuple[np.ndarray, np.ndarray]:
    base_masks = np.zeros((4, 32, 32), dtype=bool)
    roi_mask = np.zeros((32, 32), dtype=bool)
    roi_mask[4:28, 4:28] = True

    for layer_idx in range(4):
        base_masks[layer_idx, 10:18, 10:18] = True
        base_masks[layer_idx, 12:16, 12:16] = False

    return base_masks, roi_mask


def test_build_condition_masks_returns_expected_order_and_area_relationships():
    base_masks, roi_mask = _toy_base_masks()

    condition_masks, kept_ratios = build_condition_masks(base_masks, roi_mask)

    assert list(condition_masks) == list(CONDITION_ORDER)
    assert condition_masks["full_layers"].shape == base_masks.shape
    assert condition_masks["no_layers"].sum() == 0
    assert 0.0 == kept_ratios["no_layers"]
    assert kept_ratios["islands_only"] < kept_ratios["dilated_islands"]
    assert kept_ratios["dilated_islands"] <= kept_ratios["inside_contour"]
    assert kept_ratios["inside_contour"] <= kept_ratios["aperture_80lambda"]
    assert kept_ratios["aperture_80lambda"] < kept_ratios["full_layers"]


def test_materialize_condition_phases_keeps_phase_inside_masks_and_zeros_elsewhere():
    wrapped_phases = np.arange(4 * 8 * 8, dtype=np.float32).reshape(4, 8, 8)
    base_masks = np.zeros((4, 8, 8), dtype=bool)
    roi_mask = np.ones((8, 8), dtype=bool)
    base_masks[:, 2:6, 2:6] = True

    condition_masks, _ = build_condition_masks(base_masks, roi_mask)
    phase_variants = materialize_condition_phases(wrapped_phases, condition_masks)

    assert np.array_equal(phase_variants["full_layers"], wrapped_phases)
    assert np.count_nonzero(phase_variants["no_layers"]) == 0
    assert phase_variants["islands_only"][0, 3, 3] == wrapped_phases[0, 3, 3]
    assert phase_variants["islands_only"][0, 0, 0] == 0.0
    assert np.count_nonzero(phase_variants["dilated_islands"]) >= np.count_nonzero(
        phase_variants["islands_only"]
    )


def test_inside_contour_fills_region_defined_by_scattered_island_boundary():
    base_masks = np.zeros((4, 48, 48), dtype=bool)
    roi_mask = np.ones((48, 48), dtype=bool)

    yy, xx = np.mgrid[:48, :48]
    rr = np.sqrt((xx - 24) ** 2 + (yy - 24) ** 2)
    ring = np.isclose(rr, 12.0, atol=1.5)
    base_masks[:] = ring

    condition_masks, kept_ratios = build_condition_masks(base_masks, roi_mask)

    assert kept_ratios["inside_contour"] > kept_ratios["dilated_islands"] * 1.3
    assert condition_masks["inside_contour"][0, 24, 24]


def test_load_ood_amplitude_from_image_zeros_white_background(tmp_path: Path):
    image = np.full((10, 10), 255, dtype=np.uint8)
    image[2:8, 3:7] = 80
    image_path = tmp_path / "ood.png"
    Image.fromarray(image, mode="L").save(image_path)

    amplitude = load_ood_amplitude_from_image(
        image_path=str(image_path),
        resize_to=10,
        final_size=16,
    )

    assert amplitude.shape == (1, 16, 16)
    assert torch.all(amplitude[:, 0, :] == 0)
    assert torch.max(amplitude) > 0
    assert amplitude[:, 5:11, 6:10].mean() > 0


def test_load_ood_amplitude_from_image_does_not_leave_resized_halo(tmp_path: Path):
    image = np.full((8, 8), 255, dtype=np.uint8)
    image[3:5, 3:5] = 64
    image_path = tmp_path / "ood_halo.png"
    Image.fromarray(image, mode="L").save(image_path)

    amplitude = load_ood_amplitude_from_image(
        image_path=str(image_path),
        resize_to=16,
        final_size=20,
    )

    nonzero = amplitude.squeeze(0) > 0
    rows, cols = torch.nonzero(nonzero, as_tuple=True)
    assert rows.min().item() >= 6
    assert rows.max().item() <= 13
    assert cols.min().item() >= 6
    assert cols.max().item() <= 13


def test_forward_without_diffractive_layers_matches_single_direct_propagation():
    grid_size = 32
    dx_mm = 0.3
    wavelength_mm = 0.75
    pad_factor = 2
    amplitude = torch.zeros((grid_size, grid_size), dtype=torch.float32)
    amplitude[10:22, 12:20] = 1.0
    diffuser = torch.ones((grid_size, grid_size), dtype=torch.complex64)
    H_obj_to_diff = bl_asm_transfer_function(
        grid_size,
        dx_mm,
        wavelength_mm,
        40.0,
        pad_factor=pad_factor,
    )
    total_distance_mm = 2.0 + 3 * 2.0 + 7.0
    H_diff_to_output = bl_asm_transfer_function(
        grid_size,
        dx_mm,
        wavelength_mm,
        total_distance_mm,
        pad_factor=pad_factor,
    )

    direct = _forward_without_diffractive_layers(
        amplitude=amplitude,
        diffuser_t=diffuser,
        H_obj_to_diff=H_obj_to_diff,
        H_diff_to_output=H_diff_to_output,
        pad_factor=pad_factor,
    )

    field = amplitude.unsqueeze(0).to(torch.complex64)
    field_at_diffuser = bl_asm_propagate(field, H_obj_to_diff, pad_factor=pad_factor)
    manual = bl_asm_propagate(field_at_diffuser, H_diff_to_output, pad_factor=pad_factor).abs() ** 2

    assert torch.allclose(direct, manual, atol=1e-6, rtol=1e-5)


def test_make_figs4_generates_png_and_metadata(tmp_path: Path, baseline_config):
    grid_size = 64
    baseline_config["grid"]["nx"] = grid_size
    baseline_config["grid"]["ny"] = grid_size
    baseline_config["grid"]["pitch_mm"] = 0.3
    baseline_config["geometry"]["num_layers"] = 4
    baseline_config["geometry"]["object_to_diffuser_mm"] = 40.0
    baseline_config["geometry"]["diffuser_to_layer1_mm"] = 2.0
    baseline_config["geometry"]["layer_to_layer_mm"] = 2.0
    baseline_config["geometry"]["last_layer_to_output_mm"] = 7.0
    baseline_config["optics"]["wavelength_mm"] = 0.75
    baseline_config["dataset"] = {
        "name": "mnist",
        "resize_to_px": 40,
        "final_resolution_px": grid_size,
    }
    baseline_config["training"] = {"epochs": 4, "batch_size_objects": 1}
    baseline_config["experiment"] = {"id": "figs4_test", "seed": 123}
    baseline_config["diffuser"] = {
        "type": "thin_random_phase",
        "delta_n": 0.74,
        "height_mean_lambda": 25.0,
        "height_std_lambda": 8.0,
        "smoothing_sigma_lambda": 4.0,
    }
    baseline_config["visualization"] = {
        "contrast_enhancement": {"lower_percentile": 1.0, "upper_percentile": 99.0}
    }

    config_path = tmp_path / "baseline.yaml"
    checkpoint_path = tmp_path / "model.pt"
    output_path = tmp_path / "figS4_pruning.png"

    model = D2NN(
        num_layers=4,
        grid_size=grid_size,
        dx_mm=0.3,
        wavelength_mm=0.75,
        diffuser_to_layer1_mm=2.0,
        layer_to_layer_mm=2.0,
        last_layer_to_output_mm=7.0,
        pad_factor=2,
    )
    state_dict = model.state_dict()
    for idx in range(4):
        phase = torch.zeros((grid_size, grid_size), dtype=torch.float32)
        phase[12:52, 12:52] = 0.5 * (idx + 1)
        phase[24:40, 24:40] = 1.5 + 0.25 * idx
        state_dict[f"layers.{idx}.phase"] = phase
    torch.save({"model_state_dict": state_dict}, checkpoint_path)
    config_path.write_text(yaml.safe_dump(baseline_config))

    digit_amplitude = torch.zeros((1, grid_size, grid_size), dtype=torch.float32)
    digit_amplitude[:, 20:44, 28:36] = 1.0
    digit_amplitude[:, 20:28, 20:44] = 0.75

    ood_amplitude = torch.zeros((1, grid_size, grid_size), dtype=torch.float32)
    ood_amplitude[:, 16:48, 16:48] = 0.4
    ood_amplitude[:, 24:40, 24:40] = 0.9

    known_diffuser = generate_diffuser(
        grid_size,
        dx_mm=0.3,
        wavelength_mm=0.75,
        seed=100,
        device=torch.device("cpu"),
    )
    new_diffuser = generate_diffuser(
        grid_size,
        dx_mm=0.3,
        wavelength_mm=0.75,
        seed=200,
        device=torch.device("cpu"),
    )

    result = make_figs4(
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path),
        save_path=str(output_path),
        digit_amplitude=digit_amplitude,
        ood_amplitude=ood_amplitude,
        known_diffuser=known_diffuser,
        new_diffuser=new_diffuser,
        figure_title="Test Figure S4",
    )

    assert output_path.exists()
    assert output_path.with_suffix(".npy").exists()
    assert result["pccs"].shape == (6, 4)
    assert result["layer_display_phases"].shape == (6, 4, grid_size, grid_size)
    assert len(result["row_labels"]) == 6
    assert len(result["output_column_labels"]) == 4
    assert result["row_labels"][0] == "Full layers"


def test_make_figures_cli_includes_figs4_option():
    from luo2022_d2nn.cli.make_figures import _build_parser

    parser = _build_parser()
    figure_action = parser._option_string_actions["--figure"]

    assert "figs4" in figure_action.choices


def test_make_figures_cli_dispatches_figs4(monkeypatch):
    import luo2022_d2nn.cli.make_figures as cli_make_figures
    import luo2022_d2nn.figures.figs4_pruning as figs4_pruning

    captured = {}

    def fake_make_figs4(checkpoint_path, config_path, save_path=None, **kwargs):
        captured["checkpoint_path"] = checkpoint_path
        captured["config_path"] = config_path
        captured["save_path"] = save_path

    monkeypatch.setattr(figs4_pruning, "make_figs4", fake_make_figs4)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "luo2022-make-figures",
            "--figure",
            "figs4",
            "--n20",
            "runs/n20_L4/model.pt",
            "--config",
            "configs/baseline.yaml",
            "--output",
            "figures/figS4_pruning.png",
        ],
    )

    cli_make_figures.main()

    assert captured == {
        "checkpoint_path": "runs/n20_L4/model.pt",
        "config_path": "configs/baseline.yaml",
        "save_path": "figures/figS4_pruning.png",
    }
