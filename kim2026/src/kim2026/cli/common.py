"""Common CLI helpers for kim2026."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from kim2026.config.schema import load_and_validate_config


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a kim2026 YAML config."""
    return load_and_validate_config(path)


def apply_runtime_environment(runtime_cfg: dict[str, Any]) -> None:
    """Apply strict reproducibility environment variables."""
    if bool(runtime_cfg.get("strict_reproducibility", True)):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(runtime_cfg.get("cublas_workspace_config", ":4096:8"))
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_kim2026")


def choose_device(runtime_cfg: dict[str, Any]) -> torch.device:
    """Resolve runtime device with a CUDA preference."""
    requested = str(runtime_cfg.get("device", "cuda")).lower()
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON with indentation."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_run_subdir(cfg: dict[str, Any], name: str) -> Path:
    """Create and return a named subdirectory under the experiment save dir."""
    path = Path(cfg["experiment"]["save_dir"]) / name
    path.mkdir(parents=True, exist_ok=True)
    return path
