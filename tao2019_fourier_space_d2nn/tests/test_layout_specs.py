from __future__ import annotations

from tao2019_fd2nn.viz.layout_specs import get_layout


def test_layout_fig2_axes_and_canvas() -> None:
    layout = get_layout("fig2")
    assert layout.width_px == 1800
    assert layout.height_px == 1400
    assert len(layout.axes) == 24  # 20 images + 4 colorbars


def test_layout_fig3_axes_and_canvas() -> None:
    layout = get_layout("fig3")
    assert layout.width_px == 1800
    assert layout.height_px == 1400
    assert len(layout.axes) == 24


def test_layout_fig4_axes_and_canvas() -> None:
    layout = get_layout("fig4")
    assert layout.width_px == 2600
    assert layout.height_px == 1600
    assert "a_plot" in layout.axes
    assert "b_plot" in layout.axes
