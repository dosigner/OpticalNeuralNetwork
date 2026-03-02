"""Pixel-precise figure layout specifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

BBoxPx = Tuple[int, int, int, int]


@dataclass(frozen=True)
class LayoutSpec:
    layout_id: str
    width_px: int
    height_px: int
    dpi: int
    axes: Dict[str, BBoxPx]
    texts: Dict[str, dict]


def _px_to_axes_rect(bbox: BBoxPx, W: int, H: int) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    left = x / W
    width = w / W
    bottom = 1.0 - (y + h) / H
    height = h / H
    return (left, bottom, width, height)


def make_axes_from_layout(fig, layout: LayoutSpec) -> Dict[str, object]:
    axd = {}
    for name, bbox in layout.axes.items():
        rect = _px_to_axes_rect(bbox, layout.width_px, layout.height_px)
        axd[name] = fig.add_axes(rect)
    return axd


def get_layout(layout_id: str) -> LayoutSpec:
    if layout_id == "fig2":
        W, H, dpi = 1800, 1400, 300
        axes = {}
        col_x = [150, 460, 770, 1080, 1390]
        row_y = [80, 390, 700, 1010]
        P = 300
        for r in range(4):
            for c in range(5):
                axes[f"img_r{r}_c{c}"] = (col_x[c], row_y[r], P, P)
        cbar_x, cbar_w = 1700, 40
        axes["cbar_r0"] = (cbar_x, row_y[0], cbar_w, P)
        axes["cbar_r1"] = (cbar_x, row_y[1], cbar_w, P)
        axes["cbar_r2"] = (cbar_x, row_y[2], cbar_w, P)
        axes["cbar_phase"] = (cbar_x, row_y[3], cbar_w, P)
        texts = {
            "row0": {"x": 70, "y": 230, "s": "Target\nSpecimen", "rotation": 90, "ha": "center", "va": "center"},
            "row1": {"x": 70, "y": 540, "s": "Co-saliency\nDetection [2]", "rotation": 90, "ha": "center", "va": "center"},
            "row2": {"x": 70, "y": 850, "s": "F-D2NN\nOutput", "rotation": 90, "ha": "center", "va": "center"},
            "row3": {"x": 70, "y": 1160, "s": "F-D2NN\nModulat. Layer", "rotation": 90, "ha": "center", "va": "center"},
            "title_ct1": {"x": 300, "y": 40, "s": "Cell Type 1", "ha": "center", "va": "center"},
            "title_ct2": {"x": 765, "y": 40, "s": "Cell Type 2", "ha": "center", "va": "center"},
            "title_ct3": {"x": 1385, "y": 40, "s": "Cell Type 3", "ha": "center", "va": "center"},
            "cbar_int_label": {"x": 1755, "y": 540, "s": "Intensity", "rotation": 90, "ha": "center", "va": "center"},
            "cbar_phase_label": {"x": 1755, "y": 1160, "s": "Phase", "rotation": 90, "ha": "center", "va": "center"},
        }
        return LayoutSpec("fig2", W, H, dpi, axes, texts)

    if layout_id == "fig3":
        base = get_layout("fig2")
        texts = dict(base.texts)
        texts["row0"] = {"x": 70, "y": 230, "s": "Dynamic\nScene", "rotation": 90, "ha": "center", "va": "center"}
        for k in ["title_ct1", "title_ct2", "title_ct3"]:
            texts.pop(k, None)
        return LayoutSpec("fig3", base.width_px, base.height_px, base.dpi, base.axes, texts)

    if layout_id == "fig4":
        W, H, dpi = 2600, 1600, 300
        axes = {}
        row_y = [110, 300, 490, 680]
        h = 180
        a_x0 = 80
        axes["a_cfg0"] = (a_x0, row_y[0], 420, h)
        axes["a_cfg1"] = (a_x0, row_y[1], 420, h)
        axes["a_cfg2"] = (a_x0, row_y[2], 420, h)
        axes["a_cfg3"] = (a_x0, row_y[3], 420, h)
        img_x = [610, 800, 990]
        for r in range(4):
            for c in range(3):
                axes[f"a_img_r{r}_c{c}"] = (img_x[c], row_y[r], 180, 180)
        axes["a_plot"] = (a_x0, 960, 1180, 560)
        b_x0 = 1340
        axes["b_cfg0"] = (b_x0, row_y[0], 1180, h)
        axes["b_cfg1"] = (b_x0, row_y[1], 1180, h)
        axes["b_cfg2"] = (b_x0, row_y[2], 1180, h)
        axes["b_cfg3"] = (b_x0, row_y[3], 1180, h)
        axes["b_plot"] = (b_x0, 960, 1180, 560)
        texts = {
            "label_a": {"x": 90, "y": 70, "s": "(a)", "ha": "left", "va": "center"},
            "label_b": {"x": 1350, "y": 70, "s": "(b)", "ha": "left", "va": "center"},
            "a_title_cfg": {"x": 250, "y": 90, "s": "D$^2$NN Configurations", "ha": "center", "va": "center"},
            "a_title_sod": {"x": 800, "y": 90, "s": "Salient Object Detection", "ha": "center", "va": "center"},
            "a_title_cifar": {"x": 1080, "y": 90, "s": "CIFAR-10", "ha": "center", "va": "center"},
            "b_title_cfg": {"x": 1750, "y": 90, "s": "D$^2$NN Configurations", "ha": "center", "va": "center"},
        }
        return LayoutSpec("fig4", W, H, dpi, axes, texts)

    raise ValueError(f"Unknown layout_id: {layout_id}")
