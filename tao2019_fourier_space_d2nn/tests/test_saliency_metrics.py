from __future__ import annotations

import torch

from tao2019_fd2nn.training.metrics_saliency import max_f_measure, pr_curve


def test_saliency_pr_shapes_and_range() -> None:
    pred = torch.rand(4, 32, 32)
    gt = (torch.rand(4, 32, 32) > 0.5).to(torch.float32)
    p, r = pr_curve(pred, gt, thresholds=64)
    assert p.shape == (64,)
    assert r.shape == (64,)
    assert (p >= 0.0).all() and (p <= 1.0).all()
    assert (r >= 0.0).all() and (r <= 1.0).all()


def test_saliency_fmax_range() -> None:
    pred = torch.rand(3, 24, 24)
    gt = (torch.rand(3, 24, 24) > 0.5).to(torch.float32)
    f = max_f_measure(pred, gt, thresholds=32)
    assert 0.0 <= f <= 1.0
