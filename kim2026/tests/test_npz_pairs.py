from __future__ import annotations

import numpy as np
import torch

from kim2026.data.manifest import build_split_manifest
from kim2026.data.npz_pairs import read_pair_npz, write_pair_npz


def test_write_and_read_pair_npz_round_trip(tmp_path) -> None:
    u_vacuum = torch.complex(torch.ones(4, 4), torch.zeros(4, 4))
    u_turb = torch.complex(torch.full((4, 4), 2.0), torch.full((4, 4), -1.0))
    x_m = np.linspace(-0.1, 0.1, 4)
    y_m = np.linspace(-0.1, 0.1, 4)
    metadata = {
        "episode_id": 3,
        "frame_index": 5,
        "global_seed": 9,
        "episode_seed": 10,
        "screen_seeds": [11, 12],
        "wind_dir_rad": 0.5,
        "dt_s": 5e-4,
        "lambda_m": 1.55e-6,
        "path_length_m": 1000.0,
        "cn2": 2.0e-14,
        "half_angle_rad": 3.0e-4,
        "aperture_diameter_m": 0.6,
        "receiver_window_m": 0.72,
        "L0_m": 30.0,
        "l0_m": 5.0e-3,
        "screen_count": 8,
    }
    path = tmp_path / "episode_00003_frame_005.npz"

    write_pair_npz(path, u_vacuum=u_vacuum, u_turb=u_turb, x_m=x_m, y_m=y_m, metadata=metadata)
    loaded = read_pair_npz(path)

    assert torch.equal(loaded["u_vacuum"], u_vacuum)
    assert torch.equal(loaded["u_turb"], u_turb)
    assert np.array_equal(loaded["x_m"], x_m)
    assert loaded["metadata"]["episode_seed"] == 10
    assert loaded["metadata"]["screen_seeds"] == [11, 12]


def test_build_split_manifest_is_deterministic() -> None:
    manifest = build_split_manifest(
        episode_ids=list(range(10)),
        split_counts={"train": 6, "val": 2, "test": 2},
    )

    assert manifest["train"] == [0, 1, 2, 3, 4, 5]
    assert manifest["val"] == [6, 7]
    assert manifest["test"] == [8, 9]
