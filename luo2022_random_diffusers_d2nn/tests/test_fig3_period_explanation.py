"""Tests for the Fig. 3 explanation figure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from luo2022_d2nn.figures import fig3_period_explanation
from luo2022_d2nn.figures.fig3_period_explanation import make_fig3_explanation
from luo2022_d2nn.models.d2nn import D2NN
from luo2022_d2nn.utils.viz import contrast_enhance


def test_make_fig3_explanation_renders_three_examples_with_diffuser_baseline_and_caption(
    tmp_path: Path,
    baseline_config,
    monkeypatch,
):
    grid_size = 64
    baseline_config["grid"]["nx"] = grid_size
    baseline_config["grid"]["ny"] = grid_size
    baseline_config["geometry"]["num_layers"] = 4
    baseline_config["optics"]["wavelength_mm"] = 0.75
    baseline_config["training"] = {"epochs": 101}
    baseline_config["dataset"]["resize_to_px"] = 32

    config_path = tmp_path / "baseline.yaml"
    checkpoint_path = tmp_path / "n20.pt"
    save_path = tmp_path / "fig3_period_explanation.png"
    config_path.write_text(yaml.safe_dump(baseline_config))

    model = D2NN(
        num_layers=4,
        grid_size=grid_size,
        dx_mm=baseline_config["grid"]["pitch_mm"],
        wavelength_mm=baseline_config["optics"]["wavelength_mm"],
        diffuser_to_layer1_mm=baseline_config["geometry"]["diffuser_to_layer1_mm"],
        layer_to_layer_mm=baseline_config["geometry"]["layer_to_layer_mm"],
        last_layer_to_output_mm=baseline_config["geometry"]["last_layer_to_output_mm"],
        pad_factor=2,
    )
    with torch.no_grad():
        for idx, layer in enumerate(model.layers):
            layer.phase.fill_(0.1 * (idx + 1))
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    captured = {}

    def _capture_figure(fig, path, dpi=300, tight=True):
        captured["fig"] = fig
        Path(path).touch()

    monkeypatch.setattr(fig3_period_explanation, "save_figure", _capture_figure)

    result = make_fig3_explanation(
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path),
        save_path=str(save_path),
    )

    fig = captured["fig"]
    axes = fig.axes

    assert len(axes) == 12
    assert axes[0].get_title() == "Input Resolution Target"
    assert axes[1].get_title() == "Propagation Through Diffuser"
    assert axes[2].get_title() == "D2NN Reconstruction"
    assert axes[3].get_title() == "Averaged Profile"

    profile_text = "\n".join(text.get_text() for text in axes[3].texts)
    assert "True period" in profile_text
    assert "Measured period" in profile_text

    figure_text = "\n".join(text.get_text() for text in fig.texts)
    assert "Resolution Test Target Period" in figure_text
    assert "Measured Grating Period" in figure_text

    assert result["periods_mm"] == [7.2, 10.8, 12.0]
    assert len(result["free_space_images"]) == 3
    assert len(result["reconstruction_images"]) == 3
    assert len(result["measured_periods_mm"]) == 3

    free_space_display = np.asarray(axes[1].images[0].get_array())
    reconstruction_display = np.asarray(axes[2].images[0].get_array())
    np.testing.assert_allclose(
        free_space_display,
        contrast_enhance(result["free_space_images"][0]),
    )
    np.testing.assert_allclose(
        reconstruction_display,
        contrast_enhance(result["reconstruction_images"][0]),
    )
    assert not np.allclose(result["free_space_images"][0], result["reconstruction_images"][0])

    profile_line = axes[3].lines[0].get_ydata()
    expected_profile = result["reconstruction_images"][0].mean(axis=1)
    np.testing.assert_allclose(profile_line, expected_profile)
    assert not np.allclose(profile_line, result["free_space_images"][0].mean(axis=1))

    assert save_path.exists()
