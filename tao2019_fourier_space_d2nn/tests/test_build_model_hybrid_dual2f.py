from __future__ import annotations

from pathlib import Path

from tao2019_fd2nn.cli.common import build_model
from tao2019_fd2nn.config.schema import load_and_validate_config


def test_hybrid_uses_hybrid_2f_as_dual_2f_runtime() -> None:
    cfg_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "tao2019_fd2nn"
        / "config"
        / "cls_mnist_hybrid_10l.yaml"
    )
    cfg = load_and_validate_config(cfg_path)
    model = build_model(cfg)

    assert model.cfg.model_type == "hybrid_d2nn"
    assert model.cfg.use_dual_2f is True
    assert model.cfg.dual_2f_f1_m == cfg["optics"]["hybrid_2f"]["f_m"]
    assert model.cfg.dual_2f_f2_m == cfg["optics"]["hybrid_2f"]["f_m"]
    assert model.cfg.dual_2f_na1 == cfg["optics"]["hybrid_2f"]["na"]
    assert model.cfg.dual_2f_na2 == cfg["optics"]["hybrid_2f"]["na"]
