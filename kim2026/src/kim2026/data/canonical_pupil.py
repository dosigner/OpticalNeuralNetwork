"""Canonical telescope-pupil dataset helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

CANONICAL_DATASET_NAME = "1km_cn2_5e-14_tel15cm_pupil1024_v1"
CANONICAL_GENERATOR_VERSION = "pupil1024_v1"
CANONICAL_SPLIT_COUNTS = {"train": 4000, "val": 500, "test": 500}
REDUCER_VALIDATION_CACHE_DIRNAME = "reducer_val_cache"
REDUCER_VALIDATION_SUMMARY_FILENAME = "summary.json"
REDUCER_RECOMMENDATION_IDEAL_OK = "ideal_ok"
REDUCER_RECOMMENDATION_PROMOTE_FULL = "promote_full"
SUPPORTED_PLANE_SELECTORS = {"stored", "pupil", "reduced_ideal"}

_CANONICAL_KEYS = {
    "u_vacuum_pupil_real",
    "u_vacuum_pupil_imag",
    "u_turb_pupil_real",
    "u_turb_pupil_imag",
    "x_pupil_m",
    "y_pupil_m",
    "metadata_json",
}
_REQUIRED_METADATA_KEYS = {
    "plane",
    "generator_version",
    "realization",
    "seed",
    "Dz",
    "Cn2",
    "wvl",
    "theta_div",
    "receiver_window_m",
    "telescope_diameter_m",
    "crop_n",
    "delta_n_pupil_m",
    "beam_reducer_ratio",
    "reducer_output_window_m",
}


def _split_complex(field: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    array = field.detach().cpu().numpy()
    return array.real.astype(np.float32), array.imag.astype(np.float32)


def _join_complex(real: np.ndarray, imag: np.ndarray) -> torch.Tensor:
    return torch.complex(torch.from_numpy(real.astype(np.float32)), torch.from_numpy(imag.astype(np.float32)))


def _validate_metadata(metadata: dict[str, Any], *, expected_n: int) -> dict[str, Any]:
    missing = sorted(_REQUIRED_METADATA_KEYS.difference(metadata))
    if missing:
        raise KeyError(f"canonical pupil metadata missing keys: {missing}")
    if str(metadata["plane"]) != "telescope_pupil":
        raise ValueError("canonical pupil metadata must declare plane='telescope_pupil'")
    if str(metadata["generator_version"]) != CANONICAL_GENERATOR_VERSION:
        raise ValueError(f"generator_version must be '{CANONICAL_GENERATOR_VERSION}'")
    if int(metadata["crop_n"]) != int(expected_n):
        raise ValueError(f"metadata crop_n={metadata['crop_n']} does not match field size {expected_n}")
    return metadata


def write_canonical_pupil_npz(
    path: str | Path,
    *,
    u_vacuum_pupil: torch.Tensor,
    u_turb_pupil: torch.Tensor,
    x_pupil_m: np.ndarray,
    y_pupil_m: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Write a canonical telescope-pupil sample."""
    if u_vacuum_pupil.shape != u_turb_pupil.shape:
        raise ValueError("u_vacuum_pupil and u_turb_pupil must share the same shape")
    if u_vacuum_pupil.ndim != 2 or u_vacuum_pupil.shape[-1] != u_vacuum_pupil.shape[-2]:
        raise ValueError("canonical pupil fields must be 2D square arrays")
    n = int(u_vacuum_pupil.shape[-1])
    metadata = _validate_metadata(dict(metadata), expected_n=n)
    u_v_real, u_v_imag = _split_complex(u_vacuum_pupil.to(torch.complex64))
    u_t_real, u_t_imag = _split_complex(u_turb_pupil.to(torch.complex64))
    np.savez_compressed(
        path,
        u_vacuum_pupil_real=u_v_real,
        u_vacuum_pupil_imag=u_v_imag,
        u_turb_pupil_real=u_t_real,
        u_turb_pupil_imag=u_t_imag,
        x_pupil_m=np.asarray(x_pupil_m, dtype=np.float32),
        y_pupil_m=np.asarray(y_pupil_m, dtype=np.float32),
        metadata_json=np.array(json.dumps(metadata)),
    )


def read_canonical_pupil_npz(path: str | Path) -> dict[str, Any]:
    """Read a canonical telescope-pupil sample."""
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        missing = sorted(_CANONICAL_KEYS.difference(keys))
        if missing:
            raise KeyError(f"canonical pupil NPZ missing keys: {missing}")
        metadata = _validate_metadata(json.loads(str(data["metadata_json"])), expected_n=int(data["u_vacuum_pupil_real"].shape[-1]))
        return {
            "u_vacuum_pupil": _join_complex(data["u_vacuum_pupil_real"], data["u_vacuum_pupil_imag"]),
            "u_turb_pupil": _join_complex(data["u_turb_pupil_real"], data["u_turb_pupil_imag"]),
            "x_pupil_m": data["x_pupil_m"].astype(np.float32),
            "y_pupil_m": data["y_pupil_m"].astype(np.float32),
            "metadata": metadata,
        }


def is_canonical_pupil_npz(path: str | Path) -> bool:
    """Return True when the NPZ contains the canonical pupil schema."""
    with np.load(path, allow_pickle=False) as data:
        return _CANONICAL_KEYS.issubset(set(data.files))


def build_canonical_split_manifest(
    *,
    total_realizations: int,
    split_counts: dict[str, int] | None = None,
) -> dict[str, list[str]]:
    """Build the fixed train/val/test filename manifest for canonical data."""
    counts = dict(CANONICAL_SPLIT_COUNTS if split_counts is None else split_counts)
    total_expected = int(counts["train"]) + int(counts["val"]) + int(counts["test"])
    if int(total_realizations) != total_expected:
        raise ValueError(f"total_realizations={total_realizations} does not match split counts total={total_expected}")
    filenames = [f"realization_{idx:05d}.npz" for idx in range(int(total_realizations))]
    return {
        "train": filenames[: counts["train"]],
        "val": filenames[counts["train"] : counts["train"] + counts["val"]],
        "test": filenames[counts["train"] + counts["val"] :],
    }


def default_reducer_summary_path(cache_dir: str | Path) -> Path:
    """Return the canonical reducer validation summary path for a dataset cache."""
    cache_path = Path(cache_dir)
    return cache_path.parent / REDUCER_VALIDATION_CACHE_DIRNAME / REDUCER_VALIDATION_SUMMARY_FILENAME


def load_reducer_validation_summary(path: str | Path) -> dict[str, Any]:
    """Load the reducer validation summary JSON."""
    summary_path = Path(path)
    if not summary_path.exists():
        raise FileNotFoundError(f"missing reducer validation summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if "passed" not in summary or "recommendation" not in summary:
        raise KeyError("reducer validation summary must contain 'passed' and 'recommendation'")
    return summary


def enforce_reducer_validation_gate(data_cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Block reduced-ideal training until reducer validation approves the path."""
    plane_selector = str(data_cfg.get("plane_selector", "stored"))
    reducer_validation = dict(data_cfg.get("reducer_validation", {}))
    required = bool(reducer_validation.get("required", False))
    if not required:
        return None

    summary_path = reducer_validation.get("summary_path")
    if summary_path is None:
        summary_path = default_reducer_summary_path(data_cfg["cache_dir"])
    summary = load_reducer_validation_summary(summary_path)
    recommendation = str(summary["recommendation"])
    if plane_selector == "reduced_ideal":
        if recommendation != REDUCER_RECOMMENDATION_IDEAL_OK or not bool(summary["passed"]):
            raise RuntimeError(
                "reducer validation does not approve plane_selector='reduced_ideal' "
                f"(recommendation={recommendation})"
            )
    return summary
