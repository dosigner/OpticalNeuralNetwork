from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "reproduce_supp_s8_sensitivity_cls.py"
    spec = importlib.util.spec_from_file_location("reproduce_supp_s8_sensitivity_cls", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_s8_script_defines_full_experiment_matrix() -> None:
    mod = _load_script_module()

    assert mod.FABRICATION_SIGMAS == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert mod.ALIGNMENT_SHIFTS_UM == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert len(mod.EXPERIMENT_SPECS) == 9

    families = {spec["family"] for spec in mod.EXPERIMENT_SPECS}
    layers = {int(spec["num_layers"]) for spec in mod.EXPERIMENT_SPECS}

    assert families == {"nonlinear_fourier", "nonlinear_real", "linear_real"}
    assert layers == {1, 2, 5}
