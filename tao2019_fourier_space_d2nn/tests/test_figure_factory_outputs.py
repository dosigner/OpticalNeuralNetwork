from __future__ import annotations

import numpy as np

from tao2019_fd2nn.viz.figure_factory import FigureFactory


def test_figure_factory_writes_files(tmp_path) -> None:
    f = FigureFactory(tmp_path)
    p1 = f.plot_saliency_grid(np.zeros((8, 8)), np.zeros((8, 8)), np.zeros((8, 8)))
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
    assert p1.exists()
    assert p2.exists()
    assert p3.exists()
    assert p4.exists()
    assert p5.exists()
