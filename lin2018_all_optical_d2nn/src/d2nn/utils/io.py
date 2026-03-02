"""Input/output helpers for YAML/JSON/NPY and run artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML into a dictionary."""

    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    """Save dictionary as YAML."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=True)


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    """Save dictionary as canonical JSON."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2, sort_keys=True)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load JSON dictionary."""

    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_npy(path: str | Path, array: np.ndarray) -> None:
    """Save numpy array as .npy."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, array)


def _resolve_timestamp_run_dir(base_dir: str | Path, exp_name: str) -> Path:
    """Create run directory with YYMMDD_HHMMSS naming (and collision suffix)."""

    base = Path(base_dir) / exp_name
    stem = datetime.now().strftime("%y%m%d_%H%M%S")
    run_dir = base / stem
    suffix = 1
    while run_dir.exists():
        run_dir = base / f"{stem}_{suffix:02d}"
        suffix += 1
    return run_dir


def resolve_run_dir(
    base_dir: str | Path,
    exp_name: str,
    resolved_config: dict[str, Any],
    seed: int,
    run_id_mode: str = "timestamp",
) -> Path:
    """Create run directory.

    Args:
        run_id_mode:
            - "timestamp" (default): YYMMDD_HHMMSS
            - "hash": deterministic hash from config and seed
    """

    mode = str(run_id_mode).lower()
    if mode == "timestamp":
        run_dir = _resolve_timestamp_run_dir(base_dir, exp_name)
    elif mode == "hash":
        payload = json.dumps({"cfg": resolved_config, "seed": seed}, sort_keys=True).encode("utf-8")
        run_id = hashlib.sha1(payload).hexdigest()[:12]
        run_dir = Path(base_dir) / exp_name / run_id
    else:
        raise ValueError("run_id_mode must be one of: timestamp, hash")

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    (run_dir / "exports").mkdir(exist_ok=True)
    return run_dir


def hash_file(path: str | Path) -> str:
    """Return SHA1 hash for file bytes."""

    h = hashlib.sha1()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
