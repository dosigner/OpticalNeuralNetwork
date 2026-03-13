from __future__ import annotations

import numpy as np
from pathlib import Path

from tao2019_fd2nn.viz.figure_factory import FigureFactory


def test_figure_factory_writes_files(tmp_path) -> None:
    f = FigureFactory(tmp_path)
    p1 = f.plot_saliency_grid(np.zeros((8, 8)), np.zeros((8, 8)), np.zeros((8, 8)))
    p1b = f.plot_saliency_diagnostics(
        np.zeros((8, 8)),
        np.zeros((8, 8)),
        np.zeros((8, 8)),
        crop_size=6,
        threshold=0.5,
    )
    p2 = f.plot_pr_curve(np.linspace(1, 0, 8), np.linspace(0, 1, 8), max_f=0.5)
    p3 = f.plot_convergence({"val_loss": [1, 0.5], "val_acc": [0.2, 0.7]}, left_key="val_loss", right_key="val_acc")
    rows = [[np.zeros((16, 16), dtype=np.float32) for _ in range(5)] for _ in range(3)]
    phases = [np.zeros((16, 16), dtype=np.float32) for _ in range(5)]
    p4 = f.render_fig2(rows, phases)
    curves = {
        "Linear Real": np.linspace(0.90, 0.927, 30).tolist(),
        "Nonlinear Real": np.linspace(0.91, 0.954, 30).tolist(),
        "Linear Fourier": np.linspace(0.905, 0.935, 30).tolist(),
        "Nonlinear Fourier": np.linspace(0.92, 0.970, 30).tolist(),
    }
    max_acc = {k: max(v) for k, v in curves.items()}
    p5 = f.plot_mnist_fig4a_comparison(curves, max_acc)
    s7c_curves = {
        "Linear Fourier": np.linspace(0.52, 0.55, 30).tolist(),
        "Nonlinear Fourier": np.linspace(0.61, 0.66, 30).tolist(),
    }
    s7c_max = {k: max(v) for k, v in s7c_curves.items()}
    s7d_curves = {
        "Linear Fourier": np.linspace(0.89, 0.90, 30).tolist(),
        "Nonlinear Fourier, Single SBN": np.linspace(0.92, 0.94, 30).tolist(),
        "Nonlinear Fourier, Multi-SBN": np.linspace(0.93, 0.95, 30).tolist(),
    }
    s7d_max = {k: max(v) for k, v in s7d_curves.items()}
    inset_c = [
        {
            "bbox": [0.44, 0.32, 0.48, 0.22],
            "config_key": "linear_fourier",
            "num_layers": 1,
            "border_color": "#1f77b4",
            "label": "Linear Fourier",
        },
        {
            "bbox": [0.44, 0.14, 0.48, 0.22],
            "config_key": "nonlinear_fourier",
            "num_layers": 1,
            "border_color": "#d95319",
            "label": "Nonlinear Fourier",
        },
    ]
    inset_d = [
        {
            "bbox": [0.37, 0.57, 0.51, 0.20],
            "config_key": "linear_fourier",
            "num_layers": 5,
            "border_color": "#1f77b4",
            "label": "Linear Fourier",
        },
        {
            "bbox": [0.37, 0.40, 0.51, 0.20],
            "config_key": "nonlinear_fourier_single_sbn",
            "num_layers": 5,
            "border_color": "#d95319",
            "label": "Nonlinear Fourier, Single SBN",
        },
        {
            "bbox": [0.37, 0.23, 0.51, 0.20],
            "config_key": "nonlinear_fourier_multi_sbn",
            "num_layers": 5,
            "border_color": "#edb120",
            "label": "Nonlinear Fourier, Multi-SBN",
        },
    ]
    p6 = f.plot_s7cd_panel(
        s7c_curves,
        s7c_max,
        ordered_labels=["Linear Fourier", "Nonlinear Fourier"],
        colors={
            "Linear Fourier": "#1f77b4",
            "Nonlinear Fourier": "#d95319",
            "Nonlinear Fourier, Single SBN": "#d95319",
            "Nonlinear Fourier, Multi-SBN": "#edb120",
        },
        title="Single Layer",
        panel_label="(c)",
        ylim=(0.4, 0.8),
        insets=inset_c,
        name="supp_s7c.png",
    )
    p7 = f.plot_s7cd_composite(
        s7c_curves,
        s7c_max,
        left_ordered_labels=["Linear Fourier", "Nonlinear Fourier"],
        left_colors={
            "Linear Fourier": "#1f77b4",
            "Nonlinear Fourier": "#d95319",
            "Nonlinear Fourier, Single SBN": "#d95319",
            "Nonlinear Fourier, Multi-SBN": "#edb120",
        },
        left_title="Single Layer",
        left_panel_label="(c)",
        left_ylim=(0.4, 0.8),
        left_insets=inset_c,
        right_curves=s7d_curves,
        right_max_acc=s7d_max,
        right_ordered_labels=["Linear Fourier", "Nonlinear Fourier, Single SBN", "Nonlinear Fourier, Multi-SBN"],
        right_colors={
            "Linear Fourier": "#1f77b4",
            "Nonlinear Fourier": "#d95319",
            "Nonlinear Fourier, Single SBN": "#d95319",
            "Nonlinear Fourier, Multi-SBN": "#edb120",
        },
        right_title="Five Layer",
        right_panel_label="(d)",
        right_ylim=(0.6, 1.0),
        right_insets=inset_d,
        name="supp_s7cd.png",
    )
    s8_fabrication = {
        "Nonlinear Fourier (1L)": [0.94, 0.91, 0.86],
        "Nonlinear Fourier (2L)": [0.95, 0.93, 0.90],
        "Nonlinear Fourier (5L)": [0.97, 0.96, 0.94],
        "Nonlinear Real (1L)": [0.92, 0.90, 0.88],
        "Nonlinear Real (2L)": [0.94, 0.92, 0.90],
        "Nonlinear Real (5L)": [0.96, 0.95, 0.93],
        "Linear Real (1L)": [0.90, 0.88, 0.85],
        "Linear Real (2L)": [0.91, 0.89, 0.86],
        "Linear Real (5L)": [0.93, 0.91, 0.88],
    }
    s8_alignment = {
        "Nonlinear Fourier (1L)": [0.94, 0.87, 0.78],
        "Nonlinear Fourier (2L)": [0.95, 0.91, 0.86],
        "Nonlinear Fourier (5L)": [0.97, 0.95, 0.92],
        "Nonlinear Real (1L)": [0.92, 0.84, 0.72],
        "Nonlinear Real (2L)": [0.94, 0.88, 0.80],
        "Nonlinear Real (5L)": [0.96, 0.90, 0.83],
        "Linear Real (1L)": [0.90, 0.82, 0.70],
        "Linear Real (2L)": [0.91, 0.85, 0.75],
        "Linear Real (5L)": [0.93, 0.88, 0.81],
    }
    p8 = f.plot_s8_sensitivity(
        fabrication_curves=s8_fabrication,
        fabrication_x=[0.0, 0.5, 1.0],
        alignment_curves=s8_alignment,
        alignment_x=[0.0, 2.0, 4.0],
        name="supp_s8.png",
    )
    assert p1.exists()
    assert p1b.exists()
    assert p2.exists()
    assert p3.exists()
    assert p4.exists()
    assert p5.exists()
    assert p6.exists()
    assert p7.exists()
    assert p8.exists()


def test_s7b_default_ylim_is_paper_style(tmp_path, monkeypatch) -> None:
    factory = FigureFactory(tmp_path)
    captured: dict[str, tuple[float, float]] = {}

    def fake_save(fig, name: str) -> Path:
        captured["ylim"] = tuple(float(v) for v in fig.axes[-1].get_ylim())
        path = tmp_path / name
        fig.savefig(path, bbox_inches="tight")
        return path

    monkeypatch.setattr(factory, "_save", fake_save)

    curves = {
        "Nonlinear Fourier, SBN Front": np.linspace(0.88, 0.91, 30).tolist(),
        "Nonlinear Fourier, SBN Rear": np.linspace(0.90, 0.95, 30).tolist(),
        "Nonlinear Fourier & Real, SBN Front": np.linspace(0.89, 0.90, 30).tolist(),
        "Nonlinear Fourier & Real, SBN Rear": np.linspace(0.91, 0.97, 30).tolist(),
    }
    max_acc = {k: max(v) for k, v in curves.items()}

    path = factory.plot_s7b_convergence_with_schematics(
        curves,
        max_acc,
        ordered_labels=[
            "Nonlinear Fourier, SBN Front",
            "Nonlinear Fourier, SBN Rear",
            "Nonlinear Fourier & Real, SBN Front",
            "Nonlinear Fourier & Real, SBN Rear",
        ],
        colors={
            "Nonlinear Fourier, SBN Front": "#1f77b4",
            "Nonlinear Fourier, SBN Rear": "#d95319",
            "Nonlinear Fourier & Real, SBN Front": "#edb120",
            "Nonlinear Fourier & Real, SBN Rear": "#7e2f8e",
        },
        schematic_keys=[
            "nonlinear_fourier_sbn_front",
            "nonlinear_fourier_sbn_rear",
            "hybrid_sbn_front",
            "hybrid_sbn_rear",
        ],
    )

    assert path.exists()
    assert captured["ylim"] == (0.8, 1.0)


def test_s7a_legend_includes_best_accuracy(tmp_path, monkeypatch) -> None:
    factory = FigureFactory(tmp_path)
    captured: dict[str, list[str]] = {}

    def fake_save(fig, name: str) -> Path:
        legend = fig.axes[-1].get_legend()
        assert legend is not None
        captured["labels"] = [text.get_text() for text in legend.get_texts()]
        path = tmp_path / name
        fig.savefig(path, bbox_inches="tight")
        return path

    monkeypatch.setattr(factory, "_save", fake_save)

    path = factory.plot_accuracy_vs_layers(
        {
            "Linear Fourier": {1: 0.60, 2: 0.88, 3: 0.90},
            "Nonlinear Fourier, Single SBN": {1: 0.66, 2: 0.89, 3: 0.93},
            "Nonlinear Fourier, Muti-SBN": {1: 0.66, 2: 0.89, 3: 0.94},
        },
        ordered_labels=[
            "Linear Fourier",
            "Nonlinear Fourier, Single SBN",
            "Nonlinear Fourier, Muti-SBN",
        ],
        colors={
            "Linear Fourier": "#1f77b4",
            "Nonlinear Fourier, Single SBN": "#d95319",
            "Nonlinear Fourier, Muti-SBN": "#edb120",
        },
        markers={
            "Linear Fourier": "x",
            "Nonlinear Fourier, Single SBN": "x",
            "Nonlinear Fourier, Muti-SBN": "x",
        },
    )

    assert path.exists()
    assert captured["labels"] == [
        "Linear Fourier (90.0%)",
        "Nonlinear Fourier, Single SBN (93.0%)",
        "Nonlinear Fourier, Muti-SBN (94.0%)",
    ]
