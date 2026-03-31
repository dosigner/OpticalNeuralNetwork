from __future__ import annotations

from pathlib import Path

import torch

from kim2026.data.dataset import CachedFieldDataset
from kim2026.data.npz_pairs import write_pair_npz
from kim2026.models.d2nn import BeamCleanupD2NN
from kim2026.optics.gaussian_beam import make_collimated_gaussian_field
from kim2026.training.losses import (
    beam_cleanup_loss,
    encircled_energy_loss,
    normalized_overlap_loss,
)
from kim2026.training.targets import center_crop_field, make_detector_plane_target
from kim2026.training.trainer import _build_model, train_model
from kim2026.utils.seed import set_global_seed


def _write_tiny_cache(cache_dir: Path, *, count: int = 4, n: int = 32) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "split_manifest.json"
    train_files = []
    val_files = []
    for idx in range(count):
        vacuum, x_m, y_m = make_collimated_gaussian_field(
            n=n,
            window_m=0.2,
            wavelength_m=1.55e-6,
            half_angle_rad=3.0e-4,
        )
        phase = torch.linspace(0.0, 0.2 * (idx + 1), n * n, dtype=torch.float32).reshape(n, n)
        turb = vacuum * torch.exp(1j * phase)
        path = cache_dir / f"episode_{idx:05d}_frame_000.npz"
        write_pair_npz(
            path,
            u_vacuum=vacuum,
            u_turb=turb,
            x_m=x_m.numpy(),
            y_m=y_m.numpy(),
            metadata={
                "episode_id": idx,
                "frame_index": 0,
                "global_seed": 1,
                "episode_seed": idx,
                "screen_seeds": [idx + 10],
                "wind_dir_rad": 0.1,
                "dt_s": 5e-4,
                "lambda_m": 1.55e-6,
                "path_length_m": 1000.0,
                "cn2": 2.0e-14,
                "half_angle_rad": 3.0e-4,
                "aperture_diameter_m": 0.15,
                "receiver_window_m": 0.2,
                "L0_m": 30.0,
                "l0_m": 5.0e-3,
                "screen_count": 1,
            },
        )
        if idx < count - 1:
            train_files.append(path.name)
        else:
            val_files.append(path.name)

    manifest_path.write_text(
        '{"train": ' + str(train_files).replace("'", '"') + ', "val": ' + str(val_files).replace("'", '"') + ', "test": []}',
        encoding="utf-8",
    )
    return manifest_path


def test_beam_cleanup_d2nn_preserves_shape() -> None:
    model = BeamCleanupD2NN(
        n=32,
        wavelength_m=1.55e-6,
        window_m=0.2,
        num_layers=4,
        layer_spacing_m=0.02,
        detector_distance_m=0.03,
        propagation_pad_factor=2,
    )
    field = torch.ones(2, 32, 32, dtype=torch.complex64)

    output = model(field)

    assert output.shape == field.shape
    assert output.dtype == torch.complex64


def test_make_detector_plane_target_matches_direct_free_space() -> None:
    vacuum, _, _ = make_collimated_gaussian_field(
        n=32,
        window_m=0.2,
        wavelength_m=1.55e-6,
        half_angle_rad=3.0e-4,
    )

    direct = make_detector_plane_target(
        vacuum.unsqueeze(0),
        wavelength_m=1.55e-6,
        receiver_window_m=0.2,
        aperture_diameter_m=0.15,
        total_distance_m=0.08,
    )
    via_batch = make_detector_plane_target(
        vacuum.unsqueeze(0),
        wavelength_m=1.55e-6,
        receiver_window_m=0.2,
        aperture_diameter_m=0.15,
        total_distance_m=0.08,
    )

    assert torch.allclose(direct, via_batch)


def test_losses_are_zero_when_prediction_matches_target() -> None:
    intensity = torch.ones(2, 16, 16)

    assert torch.isclose(normalized_overlap_loss(intensity, intensity), torch.tensor(0.0))
    assert torch.isclose(encircled_energy_loss(intensity, intensity), torch.tensor(0.0))
    assert torch.isclose(beam_cleanup_loss(intensity, intensity), torch.tensor(0.0))


def test_resume_training_matches_uninterrupted(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    manifest_path = _write_tiny_cache(cache_dir)
    common_cfg = {
        "optics": {"lambda_m": 1.55e-6},
        "receiver": {"aperture_diameter_m": 0.15},
        "grid": {"receiver_window_m": 0.2},
        "model": {"num_layers": 2, "layer_spacing_m": 0.02, "detector_distance_m": 0.03},
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "learning_rate": 1e-2,
            "loss": {"weights": {"overlap": 1.0, "radius": 0.25, "encircled": 0.25}},
        },
        "runtime": {"seed": 123, "strict_reproducibility": True},
        "data": {"cache_dir": str(cache_dir), "split_manifest_path": str(manifest_path)},
    }

    set_global_seed(123, strict_reproducibility=True)
    uninterrupted_dir = tmp_path / "uninterrupted"
    uninterrupted = train_model(common_cfg, run_dir=uninterrupted_dir)

    set_global_seed(123, strict_reproducibility=True)
    staged_dir = tmp_path / "staged"
    first_stage = train_model(common_cfg | {"training": common_cfg["training"] | {"epochs": 1}}, run_dir=staged_dir)
    resumed = train_model(common_cfg, run_dir=staged_dir, resume_path=first_stage["checkpoint_path"])

    for key, tensor in uninterrupted["model"].state_dict().items():
        assert torch.equal(tensor, resumed["model"].state_dict()[key])

    assert uninterrupted["history"] == resumed["history"]


def test_cached_field_dataset_reads_manifest(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    manifest_path = _write_tiny_cache(cache_dir, count=2)

    ds = CachedFieldDataset(cache_dir=cache_dir, manifest_path=manifest_path, split="train")

    sample = ds[0]
    assert sample["u_turb"].dtype == torch.complex64
    assert sample["u_vacuum"].dtype == torch.complex64


def test_build_model_fd2nn_uses_dual_2f_parameters() -> None:
    cfg = {
        "optics": {
            "lambda_m": 1.55e-6,
            "dual_2f": {
                "enabled": True,
                "f1_m": 1.0e-3,
                "f2_m": 1.0e-3,
                "na1": 0.16,
                "na2": 0.16,
                "apply_scaling": False,
            },
        },
        "grid": {"receiver_window_m": 0.2},
        "model": {
            "type": "fd2nn",
            "num_layers": 5,
            "layer_spacing_m": 1.0e-4,
            "phase_max": 3.14159265,
            "phase_constraint": "unconstrained",
            "phase_init": "uniform",
            "phase_init_scale": 0.1,
        },
    }

    model = _build_model(cfg, 32)

    assert model.dual_2f_f1_m == 1.0e-3
    assert model.dual_2f_f2_m == 1.0e-3
    assert model.layer_spacing_m == 1.0e-4
    assert model.layers[0].constraint == "unconstrained"


def test_build_model_fd2nn_defaults_to_unconstrained_phase() -> None:
    cfg = {
        "optics": {
            "lambda_m": 1.55e-6,
            "dual_2f": {
                "enabled": True,
                "f1_m": 1.0e-3,
                "f2_m": 1.0e-3,
                "na1": 0.16,
                "na2": 0.16,
                "apply_scaling": False,
            },
        },
        "grid": {"receiver_window_m": 0.2},
        "model": {
            "type": "fd2nn",
            "num_layers": 5,
            "layer_spacing_m": 1.0e-4,
            "phase_init": "uniform",
            "phase_init_scale": 0.1,
        },
    }

    model = _build_model(cfg, 32)

    assert model.layers[0].constraint == "unconstrained"


def test_center_crop_field_reduces_window_around_image_center() -> None:
    field = torch.arange(36, dtype=torch.float32).reshape(1, 6, 6).to(torch.complex64)

    cropped = center_crop_field(field, crop_n=4)

    expected = field[:, 1:5, 1:5]
    torch.testing.assert_close(cropped, expected)


def test_center_crop_field_returns_input_when_crop_matches_size() -> None:
    field = torch.randn(1, 8, 8, dtype=torch.complex64)

    cropped = center_crop_field(field, crop_n=8)

    torch.testing.assert_close(cropped, field)
