from __future__ import annotations

from pathlib import Path

from tao2019_fd2nn.config.schema import load_and_validate_config


def test_all_spec_configs_validate() -> None:
    cfg_dir = Path(__file__).resolve().parents[1] / "src" / "tao2019_fd2nn" / "config"
    names = [
        "saliency_cell.yaml",
        "saliency_cell_mag2x.yaml",
        "saliency_cifar_video.yaml",
        "cls_mnist_linear_real_5l.yaml",
        "cls_mnist_nonlinear_real_5l.yaml",
        "cls_mnist_linear_fourier_5l_f1mm.yaml",
        "cls_mnist_nonlinear_fourier_5l_f4mm.yaml",
        "cls_mnist_linear_real_10l.yaml",
        "cls_mnist_nonlinear_real_10l.yaml",
        "cls_mnist_hybrid_5l.yaml",
        "cls_mnist_hybrid_10l.yaml",
    ]
    for name in names:
        cfg = load_and_validate_config(cfg_dir / name)
        assert "experiment" in cfg
        assert "optics" in cfg
        assert "model" in cfg
        assert "task" in cfg
