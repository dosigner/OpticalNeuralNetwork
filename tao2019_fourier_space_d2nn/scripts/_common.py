"""Common helpers for reproduction scripts."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dst)
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _run_cli(module_name: str, config_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root / "src")
    cmd = [sys.executable, "-m", module_name, "--config", str(config_path)]
    subprocess.run(cmd, cwd=project_root, env=env, check=True)


def run_with_overrides(base_config: Path, overrides: dict[str, Any], *, task: str) -> None:
    """Load base YAML, apply overrides, execute matching CLI."""

    with base_config.open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}
    merged = _deep_update(base, overrides)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
        yaml.safe_dump(merged, tmp, sort_keys=False)
        tmp_path = Path(tmp.name)
    module = "tao2019_fd2nn.cli.train_classifier" if task == "classification" else "tao2019_fd2nn.cli.train_saliency"
    _run_cli(module, tmp_path)
