from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from kim2026.cli.debug_vacuum_split_step import main as debug_vacuum_main
from kim2026.turbulence.channel import generate_pair_cache


def test_generate_pair_cache_writes_expected_files(tmp_path) -> None:
    cfg = {
        "experiment": {"id": "smoke", "save_dir": str(tmp_path / "runs")},
        "optics": {"lambda_m": 1.55e-6, "half_angle_rad": 3.0e-4, "m2": 1.0},
        "grid": {"n": 32, "source_window_m": 0.03, "receiver_window_m": 0.2},
        "channel": {
            "path_length_m": 100.0,
            "cn2": 2.0e-14,
            "outer_scale_m": 30.0,
            "inner_scale_m": 5.0e-3,
            "num_screens": 2,
            "frozen_flow": {
                "wind_speed_mps": 1.0,
                "wind_direction_mode": "per_episode_random",
                "dt_s": 1.0e-3,
                "frames_per_episode": 2,
                "screen_canvas_scale": 2.0,
            },
        },
        "receiver": {"aperture_diameter_m": 0.15},
        "model": {"num_layers": 2, "layer_spacing_m": 0.02, "detector_distance_m": 0.03},
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "loss": {"weights": {"overlap": 1.0, "radius": 0.25, "encircled": 0.25}},
        },
        "data": {
            "cache_dir": str(tmp_path / "cache"),
            "split_manifest_path": str(tmp_path / "cache" / "split_manifest.json"),
            "episode_manifest_path": str(tmp_path / "cache" / "episodes.json"),
            "split_episode_counts": {"train": 1, "val": 0, "test": 0},
        },
        "evaluation": {"metrics": ["overlap"], "split": "train", "save_json": False},
        "visualization": {"save_raw": True, "save_plots": False, "output_dir": str(tmp_path / "figures")},
        "runtime": {
            "seed": 5,
            "strict_reproducibility": True,
            "allow_tf32": False,
            "deterministic_algorithms": True,
        },
    }

    generate_pair_cache(cfg)

    split_manifest = json.loads((tmp_path / "cache" / "split_manifest.json").read_text(encoding="utf-8"))
    assert split_manifest["train"] == [
        "episode_00000_frame_000.npz",
        "episode_00000_frame_001.npz",
    ]
    sample = np.load(tmp_path / "cache" / "episode_00000_frame_000.npz", allow_pickle=False)
    assert np.isfinite(sample["u_vacuum_real"]).all()
    assert np.isfinite(sample["u_vacuum_imag"]).all()
    assert np.isfinite(sample["u_turb_real"]).all()
    assert np.isfinite(sample["u_turb_imag"]).all()
    metadata = json.loads(str(sample["metadata_json"].item()))
    assert metadata["path_cell_count"] == 2
    assert metadata["screen_count"] == 1
    assert len(metadata["screen_seeds"]) == 1


def test_debug_vacuum_cli_writes_preview(tmp_path: Path) -> None:
    cfg = {
        "experiment": {"id": "debug_vacuum_split_step", "save_dir": str(tmp_path / "runs" / "debug_vacuum_split_step")},
        "optics": {"lambda_m": 1.55e-6, "half_angle_rad": 3.0e-4, "m2": 1.0},
        "grid": {"n": 32, "source_window_m": 0.03, "receiver_window_m": 0.2},
        "channel": {
            "path_length_m": 100.0,
            "cn2": 2.0e-14,
            "outer_scale_m": 30.0,
            "inner_scale_m": 5.0e-3,
            "num_screens": 2,
            "frozen_flow": {
                "wind_speed_mps": 1.0,
                "wind_direction_mode": "per_episode_random",
                "dt_s": 1.0e-3,
                "frames_per_episode": 2,
                "screen_canvas_scale": 2.0,
            },
        },
        "receiver": {"aperture_diameter_m": 0.15},
        "model": {"num_layers": 2, "layer_spacing_m": 0.02, "detector_distance_m": 0.03},
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "loss": {"weights": {"overlap": 1.0, "radius": 0.25, "encircled": 0.25}},
        },
        "data": {
            "cache_dir": str(tmp_path / "cache"),
            "split_manifest_path": str(tmp_path / "cache" / "split_manifest.json"),
            "episode_manifest_path": str(tmp_path / "cache" / "episodes.json"),
            "split_episode_counts": {"train": 1, "val": 0, "test": 0},
        },
        "evaluation": {"metrics": ["overlap"], "split": "train", "save_json": False},
        "visualization": {"save_raw": True, "save_plots": False, "output_dir": str(tmp_path / "figures")},
        "runtime": {
            "seed": 5,
            "strict_reproducibility": True,
            "allow_tf32": False,
            "deterministic_algorithms": True,
            "device": "cpu",
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "fft_warmup_iters": 0,
        },
    }
    config_path = tmp_path / "debug_vacuum.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    debug_vacuum_main(["--config", str(config_path)])

    preview_dir = tmp_path / "runs" / "debug_vacuum_split_step" / "vacuum_split_step"
    manifest_path = preview_dir / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["events"][0]["event_type"] == "source"
    assert manifest["events"][-1]["event_type"] == "receiver"
    assert manifest["events"][-1]["window_m"] == 0.2
    assert any(event["event_type"] == "screen_plane" for event in manifest["events"])
    assert (preview_dir / manifest["events"][0]["path"]).exists()
    assert (preview_dir / manifest["events"][-1]["path"]).exists()
