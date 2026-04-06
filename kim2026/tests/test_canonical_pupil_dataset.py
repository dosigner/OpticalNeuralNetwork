from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from kim2026.data.canonical_pupil import (
    build_canonical_split_manifest,
    read_canonical_pupil_npz,
    write_canonical_pupil_npz,
)
from kim2026.data.dataset import CachedFieldDataset


def _energy(field: torch.Tensor, *, window_m: float) -> float:
    dx = float(window_m) / field.shape[-1]
    return float((field.abs().square().sum() * (dx * dx)).item())


def _aperture_mask(n: int, window_m: float, aperture_diameter_m: float) -> torch.Tensor:
    axis = (torch.arange(n, dtype=torch.float32) - n / 2 + 0.5) * (window_m / n)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return (torch.sqrt(xx.square() + yy.square()) <= (aperture_diameter_m / 2.0)).to(torch.float32)


def _metadata(*, n: int = 1024, realization: int = 0) -> dict[str, object]:
    receiver_window_m = float(n) * 150e-6
    telescope_diameter_m = receiver_window_m * (1000.0 / 1024.0)
    return {
        "plane": "telescope_pupil",
        "generator_version": "pupil1024_v1",
        "realization": realization,
        "seed": 20260401 + realization,
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


def _write_manifest(root: Path, filenames: list[str]) -> Path:
    manifest = {
        "train": filenames[:1],
        "val": filenames[1:2],
        "test": filenames[2:],
    }
    path = root / "split_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def test_build_canonical_split_manifest_uses_fixed_counts() -> None:
    manifest = build_canonical_split_manifest(total_realizations=5000)

    assert len(manifest["train"]) == 4000
    assert len(manifest["val"]) == 500
    assert len(manifest["test"]) == 500
    assert manifest["train"][0] == "realization_00000.npz"
    assert manifest["test"][-1] == "realization_04999.npz"


def test_canonical_npz_round_trip_preserves_required_schema(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    n = 1024
    coords = (np.arange(n, dtype=np.float32) - n // 2) * np.float32(150e-6)
    vacuum = torch.ones((n, n), dtype=torch.complex64)
    turb = vacuum * torch.exp(1j * torch.zeros((n, n), dtype=torch.float32))
    path = cache_dir / "realization_00000.npz"

    write_canonical_pupil_npz(
        path,
        u_vacuum_pupil=vacuum,
        u_turb_pupil=turb,
        x_pupil_m=coords,
        y_pupil_m=coords.copy(),
        metadata=_metadata(n=n),
    )

    record = read_canonical_pupil_npz(path)

    assert record["u_vacuum_pupil"].shape == (1024, 1024)
    assert record["u_vacuum_pupil"].dtype == torch.complex64
    assert record["u_turb_pupil"].dtype == torch.complex64
    assert record["metadata"]["plane"] == "telescope_pupil"
    assert "reducer_output_window_m" in record["metadata"]


def test_cached_field_dataset_returns_pupil_plane_from_canonical_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    n = 32
    coords = (np.arange(n, dtype=np.float32) - n // 2) * np.float32(150e-6)
    vacuum = torch.ones((n, n), dtype=torch.complex64)
    turb = vacuum * torch.exp(1j * torch.full((n, n), 0.1, dtype=torch.float32))
    filename = "realization_00000.npz"

    write_canonical_pupil_npz(
        cache_dir / filename,
        u_vacuum_pupil=vacuum,
        u_turb_pupil=turb,
        x_pupil_m=coords,
        y_pupil_m=coords.copy(),
        metadata=_metadata(n=n),
    )
    manifest_path = _write_manifest(tmp_path, [filename])

    dataset = CachedFieldDataset(
        cache_dir=cache_dir,
        manifest_path=manifest_path,
        split="train",
        plane_selector="pupil",
    )

    sample = dataset[0]

    assert torch.allclose(sample["u_vacuum"], vacuum)
    assert torch.allclose(sample["u_turb"], turb)
    assert sample["metadata"]["plane"] == "telescope_pupil"


def test_cached_field_dataset_can_reduce_canonical_pupil_on_the_fly(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    n = 128
    coords = (np.arange(n, dtype=np.float32) - n // 2) * np.float32(150e-6)
    vacuum = torch.ones((n, n), dtype=torch.complex64)
    turb = vacuum * torch.exp(1j * torch.full((n, n), 0.1, dtype=torch.float32))
    filename = "realization_00000.npz"

    write_canonical_pupil_npz(
        cache_dir / filename,
        u_vacuum_pupil=vacuum,
        u_turb_pupil=turb,
        x_pupil_m=coords,
        y_pupil_m=coords.copy(),
        metadata=_metadata(n=n),
    )
    manifest_path = _write_manifest(tmp_path, [filename])

    dataset = CachedFieldDataset(
        cache_dir=cache_dir,
        manifest_path=manifest_path,
        split="train",
        plane_selector="reduced_ideal",
    )

    sample = dataset[0]

    magnification = sample["metadata"]["telescope_diameter_m"] / sample["metadata"]["reducer_output_window_m"]
    apertured_vacuum = vacuum * _aperture_mask(
        n,
        sample["metadata"]["receiver_window_m"],
        sample["metadata"]["telescope_diameter_m"],
    ).to(vacuum.dtype)

    assert sample["u_vacuum"].dtype == torch.complex64
    assert sample["u_turb"].dtype == torch.complex64
    assert _energy(sample["u_vacuum"], window_m=sample["metadata"]["reducer_output_window_m"]) == pytest.approx(
        _energy(apertured_vacuum, window_m=sample["metadata"]["receiver_window_m"]),
        rel=1.0e-2,
        abs=1.0e-6,
    )
    assert torch.abs(sample["u_vacuum"][n // 2, n // 2]).item() == pytest.approx(magnification, rel=1.0e-3)
    assert torch.angle(sample["u_turb"][n // 2, n // 2]).item() == pytest.approx(0.1, abs=1.0e-3)
    assert sample["metadata"]["plane"] == "reduced_ideal"
