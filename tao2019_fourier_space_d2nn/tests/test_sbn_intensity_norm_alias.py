from __future__ import annotations

from pathlib import Path

from tao2019_fd2nn.cli.common import build_model
from tao2019_fd2nn.config.schema import load_and_validate_config


def test_sbn_intensity_norm_per_minmax_alias() -> None:
    cfg_dir = Path(__file__).resolve().parents[1] / "src" / "tao2019_fd2nn" / "config"
    cfg = load_and_validate_config(cfg_dir / "saliency_cell.yaml")
    cfg["model"]["nonlinearity"]["intensity_norm"] = "per_minmax"

    model = build_model(cfg)
    assert model.sbn is not None
    assert model.sbn.intensity_norm == "per_sample_minmax"


def test_sbn_intensity_norm_per_sample_minmax_passthrough() -> None:
    cfg_dir = Path(__file__).resolve().parents[1] / "src" / "tao2019_fd2nn" / "config"
    cfg = load_and_validate_config(cfg_dir / "saliency_cell.yaml")
    cfg["model"]["nonlinearity"]["intensity_norm"] = "per_sample_minmax"

    model = build_model(cfg)
    assert model.sbn is not None
    assert model.sbn.intensity_norm == "per_sample_minmax"
