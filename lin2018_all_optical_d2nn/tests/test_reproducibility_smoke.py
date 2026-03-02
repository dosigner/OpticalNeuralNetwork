from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from d2nn.models.d2nn import build_d2nn_model
from d2nn.training.callbacks import save_metrics
from d2nn.training.loops import run_classifier_epoch
from d2nn.utils.io import hash_file
from d2nn.utils.seed import make_torch_generator, set_global_seed


def _run_once(tmp_dir: Path):
    set_global_seed(1234, deterministic=True)

    N = 32
    model = build_d2nn_model(
        N=N,
        dx=1e-3,
        wavelength=0.75e-3,
        num_layers=2,
        z_layer=0.01,
        z_out=0.01,
        phase_max=float(torch.pi),
        dtype="complex64",
    )

    x_real = torch.randn(8, N, N)
    fields = torch.complex(x_real, torch.zeros_like(x_real))
    labels = torch.randint(low=0, high=2, size=(8,))

    ds = TensorDataset(fields, labels)
    loader = DataLoader(ds, batch_size=4, shuffle=True, generator=make_torch_generator(999))

    masks = torch.zeros((2, N, N), dtype=torch.bool)
    masks[0, :, : N // 2] = True
    masks[1, :, N // 2 :] = True

    out = run_classifier_epoch(
        model,
        loader,
        optimizer=None,
        detector_masks=masks,
        device=torch.device("cpu"),
        leakage_weight=0.1,
        temperature=1.0,
        max_steps=2,
    )

    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))[0]
        first_out = model(batch)

    metrics_path = tmp_dir / "metrics.json"
    save_metrics(metrics_path, {"loss": out.loss, "acc": out.acc})
    return first_out, hash_file(metrics_path)


def test_reproducibility_smoke(tmp_path: Path) -> None:
    out1, h1 = _run_once(tmp_path / "a")
    out2, h2 = _run_once(tmp_path / "b")

    assert h1 == h2
    assert torch.allclose(out1, out2, atol=1e-6, rtol=1e-6)
