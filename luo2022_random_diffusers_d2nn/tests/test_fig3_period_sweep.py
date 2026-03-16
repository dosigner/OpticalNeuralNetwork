"""Tests for Fig. 3 paper-style period sweep rendering."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from luo2022_d2nn.figures import fig3_period_sweep
from luo2022_d2nn.figures.fig3_period_sweep import TEST_PERIODS_MM, make_fig3
from luo2022_d2nn.models.d2nn import D2NN


def _count_bar_patches(ax) -> int:
    return sum(
        1
        for patch in ax.patches
        if getattr(patch, "get_height", None) is not None and patch.get_height() > 0.0
    )


def test_make_fig3_uses_paper_style_grouped_bars_and_labels(
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
    save_path = tmp_path / "fig3_period_sweep.png"
    config_path.write_text(yaml.safe_dump(baseline_config))

    checkpoint_paths = {}
    for n in (1, 10, 15, 20):
        checkpoint_path = tmp_path / f"n{n}.pt"
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
                layer.phase.fill_(0.1 * (idx + 1 + n))
        torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
        checkpoint_paths[n] = str(checkpoint_path)

    captured = {}

    def _capture_figure(fig, path, dpi=300, tight=True):
        captured["fig"] = fig
        Path(path).touch()

    monkeypatch.setattr(fig3_period_sweep, "save_figure", _capture_figure)

    make_fig3(
        checkpoint_paths=checkpoint_paths,
        config_path=str(config_path),
        save_path=str(save_path),
    )

    fig = captured["fig"]
    ax_a, ax_b = fig.axes

    assert ax_a.get_xlabel() == "Resolution Test Target Period, mm"
    assert ax_b.get_xlabel() == "Resolution Test Target Period, mm"
    assert ax_a.get_ylabel() == "Measured Grating Period, mm"
    assert ax_a.get_title() == "All-Optical Imaging Through\nLast n Diffusers in Training"
    assert ax_b.get_title() == "All-Optical Imaging Through\n20 New Diffusers"

    expected_bars_per_panel = len(TEST_PERIODS_MM) * 4
    assert _count_bar_patches(ax_a) == expected_bars_per_panel
    assert _count_bar_patches(ax_b) == expected_bars_per_panel

    legend = ax_b.get_legend()
    labels = [text.get_text() for text in legend.get_texts()]
    assert labels == ["n = 1", "n = 10", "n = 15", "n = 20", "True Period"]

    assert ax_a.get_ylim() == (4.0, 15.0)
    assert ax_b.get_ylim() == (4.0, 15.0)
    assert save_path.exists()
