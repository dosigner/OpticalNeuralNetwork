from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tao2019_fd2nn.training.trainer import run_saliency_epoch


def _loader_from_intensity(pred_intensity: torch.Tensor, target: torch.Tensor) -> DataLoader:
    pred_amp = pred_intensity.sqrt()
    fields = torch.complex(pred_amp, torch.zeros_like(pred_amp))
    return DataLoader(TensorDataset(fields, target), batch_size=1, shuffle=False)


def test_saliency_loss_can_compare_pred_and_target_after_crop_local_symmetric_normalization() -> None:
    pred = torch.tensor(
        [
            [
                [0.00, 0.00, 0.00, 0.00],
                [0.00, 0.40, 0.55, 0.00],
                [0.00, 0.50, 0.70, 0.00],
                [0.00, 0.00, 0.00, 0.00],
            ]
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [
            [
                [1.00, 1.00, 1.00, 1.00],
                [1.00, 0.20, 0.35, 1.00],
                [1.00, 0.30, 0.50, 1.00],
                [1.00, 1.00, 1.00, 1.00],
            ]
        ],
        dtype=torch.float32,
    )
    loader = _loader_from_intensity(pred, target)

    res = run_saliency_epoch(
        nn.Identity(),
        loader,
        device=torch.device("cpu"),
        optimizer=None,
        gamma_flip=False,
        pr_thresholds=32,
        compute_fmax=False,
        eval_crop_box=(1, 3, 1, 3),
        loss_mode="mse",
        loss_normalization="pred_and_target",
        loss_scope="crop",
        phase="val",
    )

    assert res.loss < 1e-6


def test_saliency_loss_crop_scope_ignores_difference_outside_center_crop() -> None:
    pred = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.2, 0.4, 0.0],
                [0.0, 0.6, 0.8, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    target = pred.clone()
    target[:, 0, 0] = 1.0
    loader = _loader_from_intensity(pred, target)

    res = run_saliency_epoch(
        nn.Identity(),
        loader,
        device=torch.device("cpu"),
        optimizer=None,
        gamma_flip=False,
        pr_thresholds=32,
        compute_fmax=False,
        eval_crop_box=(1, 3, 1, 3),
        loss_mode="mse",
        loss_normalization="pred_and_target",
        loss_scope="crop",
        phase="val",
    )

    assert res.loss < 1e-6
