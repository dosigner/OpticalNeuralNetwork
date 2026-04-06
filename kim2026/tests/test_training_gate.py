from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from kim2026.training.trainer import train_model
from kim2026.data.canonical_pupil import write_canonical_pupil_npz


def _metadata(*, n: int) -> dict[str, object]:
    receiver_window_m = float(n) * 150e-6
    telescope_diameter_m = receiver_window_m * (1000.0 / 1024.0)
    return {
        "plane": "telescope_pupil",
        "generator_version": "pupil1024_v1",
        "realization": 0,
        "seed": 20260401,
        "Dz": 1000.0,
        "Cn2": 5.0e-14,
        "wvl": 1.55e-6,
        "theta_div": 3.0e-4,
        "receiver_window_m": receiver_window_m,
        "telescope_diameter_m": telescope_diameter_m,
        "crop_n": n,
        "delta_n_pupil_m": 150e-6,
        "beam_reducer_ratio": 75,
        "reducer_output_window_m": float(n) * 2e-6,
        "vacuum_shared_across_realizations": True,
    }


def _write_canonical_cache(root: Path, *, n: int = 32) -> tuple[Path, Path]:
    cache_dir = root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    coords = (np.arange(n, dtype=np.float32) - n // 2) * np.float32(150e-6)
    filenames = []
    for idx in range(2):
        phase = torch.full((n, n), 0.05 * idx, dtype=torch.float32)
        vacuum = torch.ones((n, n), dtype=torch.complex64)
        turb = vacuum * torch.exp(1j * phase)
        filename = f"realization_{idx:05d}.npz"
        write_canonical_pupil_npz(
            cache_dir / filename,
            u_vacuum_pupil=vacuum,
            u_turb_pupil=turb,
            x_pupil_m=coords,
            y_pupil_m=coords.copy(),
            metadata=_metadata(n=n) | {"realization": idx},
        )
        filenames.append(filename)
    manifest_path = root / "split_manifest.json"
    manifest_path.write_text(
        json.dumps({"train": filenames[:1], "val": filenames[1:], "test": []}, indent=2),
        encoding="utf-8",
    )
    return cache_dir, manifest_path


def _base_cfg(cache_dir: Path, manifest_path: Path, summary_path: Path, *, n: int = 32) -> dict:
    output_window_m = float(n) * 2e-6
    return {
        "experiment": {"id": "gate-smoke", "save_dir": str(cache_dir.parent / "run")},
        "optics": {"lambda_m": 1.55e-6, "half_angle_rad": 3.0e-4, "m2": 1.0},
        "grid": {"n": n, "source_window_m": 0.03, "receiver_window_m": output_window_m},
        "channel": {
            "path_length_m": 1000.0,
            "cn2": 5.0e-14,
            "outer_scale_m": 30.0,
            "inner_scale_m": 5.0e-3,
            "num_screens": 1,
            "frozen_flow": {
                "wind_speed_mps": 10.0,
                "wind_direction_mode": "per_episode_random",
                "dt_s": 5.0e-4,
                "frames_per_episode": 1,
                "screen_canvas_scale": 2.0,
            },
        },
        "receiver": {"aperture_diameter_m": output_window_m},
        "model": {"num_layers": 2, "layer_spacing_m": 0.01, "detector_distance_m": 0.01},
        "training": {
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 1e-3,
            "loss": {"weights": {"overlap": 1.0, "radius": 0.25, "encircled": 0.25}},
        },
        "data": {
            "cache_dir": str(cache_dir),
            "split_manifest_path": str(manifest_path),
            "plane_selector": "reduced_ideal",
            "reducer_validation": {"required": True, "summary_path": str(summary_path)},
        },
        "evaluation": {"metrics": ["overlap"], "split": "val", "save_json": False},
        "visualization": {"save_raw": False, "save_plots": False},
        "runtime": {
            "seed": 20260401,
            "strict_reproducibility": True,
            "allow_tf32": False,
            "deterministic_algorithms": True,
            "device": "cpu",
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        },
    }


def test_train_model_blocks_when_reducer_validation_summary_is_missing(tmp_path: Path) -> None:
    cache_dir, manifest_path = _write_canonical_cache(tmp_path)
    summary_path = tmp_path / "reducer_summary.json"
    cfg = _base_cfg(cache_dir, manifest_path, summary_path)

    with pytest.raises(FileNotFoundError, match="reducer validation"):
        train_model(cfg, run_dir=tmp_path / "run")


def test_train_model_blocks_reduced_ideal_when_validation_recommends_full(tmp_path: Path) -> None:
    cache_dir, manifest_path = _write_canonical_cache(tmp_path)
    summary_path = tmp_path / "reducer_summary.json"
    summary_path.write_text(json.dumps({"passed": False, "recommendation": "promote_full"}), encoding="utf-8")
    cfg = _base_cfg(cache_dir, manifest_path, summary_path)

    with pytest.raises(RuntimeError, match="promote_full"):
        train_model(cfg, run_dir=tmp_path / "run")


def test_train_model_allows_reduced_ideal_when_validation_passes(tmp_path: Path) -> None:
    cache_dir, manifest_path = _write_canonical_cache(tmp_path)
    summary_path = tmp_path / "reducer_summary.json"
    summary_path.write_text(json.dumps({"passed": True, "recommendation": "ideal_ok"}), encoding="utf-8")
    cfg = _base_cfg(cache_dir, manifest_path, summary_path)

    result = train_model(cfg, run_dir=tmp_path / "run")

    assert len(result["history"]) == 1
