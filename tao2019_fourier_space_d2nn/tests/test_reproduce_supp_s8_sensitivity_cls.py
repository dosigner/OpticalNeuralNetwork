from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "reproduce_supp_s8_sensitivity_cls.py"
    spec = importlib.util.spec_from_file_location("reproduce_supp_s8_sensitivity_cls", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_s8_experiment_specs_cover_all_families_and_layers() -> None:
    module = _load_script_module()
    config_dir = Path(__file__).resolve().parents[1] / "src" / "tao2019_fd2nn" / "config"

    specs = module.build_experiment_specs(config_dir)

    assert len(specs) == 9
    assert module.DEFAULT_BLUR_SIGMAS == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert module.DEFAULT_ALIGNMENT_SHIFTS_UM == [0.0, 2.0, 4.0, 6.0]
    got = {(spec["family"], spec["num_layers"]) for spec in specs}
    expected = {
        ("Nonlinear Fourier", 1),
        ("Nonlinear Fourier", 2),
        ("Nonlinear Fourier", 5),
        ("Nonlinear Real", 1),
        ("Nonlinear Real", 2),
        ("Nonlinear Real", 5),
        ("Linear Real", 1),
        ("Linear Real", 2),
        ("Linear Real", 5),
    }
    assert got == expected
    assert all((config_dir / spec["config_name"]).exists() for spec in specs)
