from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tao2019_fd2nn.optics.fft2c import gamma_flip2d
from tao2019_fd2nn.training.trainer import run_saliency_epoch


def test_saliency_fmax_uses_flipped_prediction_when_gamma_flip_enabled() -> None:
    gt = torch.zeros((1, 8, 8), dtype=torch.float32)
    gt[:, 1:3, 5:7] = 1.0

    pred_intensity = gamma_flip2d(gt)
    pred_amp = pred_intensity.sqrt()
    fields = torch.complex(pred_amp, torch.zeros_like(pred_amp))

    loader = DataLoader(TensorDataset(fields, gt), batch_size=1, shuffle=False)
    model = nn.Identity()

    res_flip = run_saliency_epoch(
        model,
        loader,
        device=torch.device("cpu"),
        optimizer=None,
        gamma_flip=True,
        pr_thresholds=64,
        compute_fmax=True,
        phase="val",
    )
    res_no_flip = run_saliency_epoch(
        model,
        loader,
        device=torch.device("cpu"),
        optimizer=None,
        gamma_flip=False,
        pr_thresholds=64,
        compute_fmax=True,
        phase="val",
    )

    assert res_flip.fmax > 0.999
    assert res_no_flip.fmax < 0.95
