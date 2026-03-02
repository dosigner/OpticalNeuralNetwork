"""I/O helpers for config and reproducible run artifacts."""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into dictionary."""

    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError(f"YAML root must be mapping: {p}")
    return loaded


def dump_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    """Save dictionary to YAML path."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Save dictionary to JSON path."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def resolve_run_dir(runs_dir: str | Path, experiment_name: str) -> Path:
    """Create deterministic run path using UTC timestamp."""

    stamp = dt.datetime.utcnow().strftime("%y%m%d_%H%M%S")
    run_dir = Path(runs_dir) / experiment_name / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_requirements(path: str | Path) -> None:
    """Persist current Python package set (best-effort)."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        cmd = [sys.executable, "-m", "pip", "freeze"]
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        p.write_text(output, encoding="utf-8")
    except Exception as exc:  # pragma: no cover - best effort
        p.write_text(f"# failed to collect requirements: {exc}\n", encoding="utf-8")


def resolve_git_hash(cwd: str | Path | None = None) -> str:
    """Return git commit hash if available, else 'unknown'."""

    try:
        cmd = ["git", "rev-parse", "HEAD"]
        out = subprocess.check_output(cmd, cwd=str(cwd) if cwd else None, stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except Exception:
        return "unknown"


def save_repro_metadata(run_dir: str | Path, *, save_requirements_file: bool = True, cwd: str | Path | None = None) -> None:
    """Save reproducibility metadata files into run directory."""

    rd = Path(run_dir)
    metadata = {
        "python_version": sys.version,
        "platform": os.uname().sysname if hasattr(os, "uname") else "unknown",
        "git_hash": resolve_git_hash(cwd=cwd),
    }
    dump_json(rd / "repro.json", metadata)
    if save_requirements_file:
        write_requirements(rd / "requirements.txt")
