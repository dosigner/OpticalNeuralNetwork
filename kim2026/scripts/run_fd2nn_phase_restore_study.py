#!/usr/bin/env python
"""Controlled phase-first dual-2f FD2NN study at 0.1 mm spacing for ROI512 and ROI1024."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from kim2026.data.dataset import CachedFieldDataset
from kim2026.models.fd2nn import BeamCleanupFD2NN
from kim2026.training.losses import complex_field_loss
from kim2026.training.metrics import (
    amplitude_rmse,
    complex_overlap,
    full_field_phase_rmse,
    gaussian_overlap,
    out_of_support_energy_fraction,
    phase_rmse,
    strehl_ratio,
    support_weighted_phase_rmse,
)
from kim2026.training.targets import apply_receiver_aperture, center_crop_field


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kim2026" / "1km_cn2e-14_w2m_n1024_dx2mm" / "cache"
MANIFEST = DATA_DIR.parent / "split_manifest.json"
OUT_ROOT = Path(__file__).resolve().parent.parent / "runs" / "fd2nn_phase_restore_dual2f_codex"

STUDIES = {
    "roi512_spacing_0p1mm": {"roi_n": 512, "spacing_m": 0.1e-3},
    "roi1024_spacing_0p1mm": {"roi_n": 1024, "spacing_m": 0.1e-3},
}

COMMON = dict(
    wavelength_m=1.55e-6,
    receiver_window_m=0.002048,
    aperture_diameter_m=0.002,
    n=1024,
    num_layers=5,
    dual_2f_f1_m=1.0e-3,
    dual_2f_f2_m=1.0e-3,
    dual_2f_na1=0.16,
    dual_2f_na2=0.16,
    dual_2f_apply_scaling=False,
    phase_max=3.14159265,
    phase_constraint="unconstrained",
    phase_init="uniform",
    phase_init_scale=0.1,
    lr=5e-4,
    epochs=30,
    batch_size=2,
    complex_weights={
        "soft_phasor": 1.0,
        "amplitude_mse": 0.05,
        "leakage": 0.1,
        "support_gamma": 2.0,
        "full_field_phase": 0.15,
        "full_field_phase_gamma": 1.0,
        "full_field_phase_threshold": 0.05,
    },
    seed=20260324,
)


def roi_window_m(roi_n: int) -> float:
    return COMMON["receiver_window_m"] * (roi_n / COMMON["n"])


def prepare_field(field: torch.Tensor, *, aperture_diameter_m: float, roi_n: int) -> torch.Tensor:
    apertured = apply_receiver_aperture(
        field,
        receiver_window_m=COMMON["receiver_window_m"],
        aperture_diameter_m=aperture_diameter_m,
    )
    return center_crop_field(apertured, crop_n=roi_n)


def make_model(*, roi_n: int, spacing_m: float) -> BeamCleanupFD2NN:
    return BeamCleanupFD2NN(
        n=roi_n,
        wavelength_m=COMMON["wavelength_m"],
        window_m=roi_window_m(roi_n),
        num_layers=COMMON["num_layers"],
        layer_spacing_m=spacing_m,
        phase_max=COMMON["phase_max"],
        phase_constraint=COMMON["phase_constraint"],
        phase_init=COMMON["phase_init"],
        phase_init_scale=COMMON["phase_init_scale"],
        dual_2f_f1_m=COMMON["dual_2f_f1_m"],
        dual_2f_f2_m=COMMON["dual_2f_f2_m"],
        dual_2f_na1=COMMON["dual_2f_na1"],
        dual_2f_na2=COMMON["dual_2f_na2"],
        dual_2f_apply_scaling=COMMON["dual_2f_apply_scaling"],
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, *, device: torch.device, roi_n: int) -> dict[str, float]:
    model.eval()
    ap = COMMON["aperture_diameter_m"]
    all_co, all_pr, all_pr_full, all_pr_support = [], [], [], []
    all_ar, all_io, all_sr, all_leak = [], [], [], []
    all_co_bl, all_io_bl = [], []

    for batch in loader:
        u_turb = batch["u_turb"].to(device)
        u_vac = batch["u_vacuum"].to(device)
        target = prepare_field(u_vac, aperture_diameter_m=ap, roi_n=roi_n)
        inp = prepare_field(u_turb, aperture_diameter_m=ap, roi_n=roi_n)
        pred = model(inp)
        pred_i = pred.abs().square()
        tgt_i = target.abs().square()

        all_co.append(complex_overlap(pred, target).cpu())
        all_pr.append(phase_rmse(pred, target).cpu())
        all_pr_full.append(full_field_phase_rmse(pred, target).cpu())
        all_pr_support.append(support_weighted_phase_rmse(pred, target).cpu())
        all_ar.append(amplitude_rmse(pred, target).cpu())
        all_io.append(gaussian_overlap(pred_i, tgt_i).cpu())
        all_sr.append(strehl_ratio(pred_i, tgt_i).cpu())
        all_leak.append(out_of_support_energy_fraction(pred, target).cpu())
        all_co_bl.append(complex_overlap(inp, target).cpu())
        all_io_bl.append(gaussian_overlap(inp.abs().square(), tgt_i).cpu())

    return {
        "complex_overlap": float(torch.cat(all_co).mean()),
        "phase_rmse_rad": float(torch.cat(all_pr).mean()),
        "full_field_phase_rmse_rad": float(torch.cat(all_pr_full).mean()),
        "support_weighted_phase_rmse_rad": float(torch.cat(all_pr_support).mean()),
        "out_of_support_energy_fraction": float(torch.cat(all_leak).mean()),
        "amplitude_rmse": float(torch.cat(all_ar).mean()),
        "intensity_overlap": float(torch.cat(all_io).mean()),
        "strehl": float(torch.cat(all_sr).mean()),
        "baseline_complex_overlap": float(torch.cat(all_co_bl).mean()),
        "baseline_intensity_overlap": float(torch.cat(all_io_bl).mean()),
    }


def train_one(name: str, *, roi_n: int, spacing_m: float, device: torch.device) -> dict[str, object]:
    run_dir = OUT_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(COMMON["seed"])
    np.random.seed(COMMON["seed"])

    train_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="train")
    val_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="val")
    test_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="test")

    model = make_model(roi_n=roi_n, spacing_m=spacing_m).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=COMMON["lr"])
    train_loader = DataLoader(train_ds, batch_size=COMMON["batch_size"], shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=COMMON["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=COMMON["batch_size"])

    ap = COMMON["aperture_diameter_m"]
    history = []

    print(f"\n{'=' * 60}")
    print(f"{name}: spacing={spacing_m*1e3:.1f} mm, roi_n={roi_n}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'=' * 60}")

    for epoch in range(COMMON["epochs"]):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for batch in train_loader:
            u_turb = batch["u_turb"].to(device)
            u_vac = batch["u_vacuum"].to(device)
            target = prepare_field(u_vac, aperture_diameter_m=ap, roi_n=roi_n)
            inp = prepare_field(u_turb, aperture_diameter_m=ap, roi_n=roi_n)

            optimizer.zero_grad(set_to_none=True)
            pred = model(inp)
            loss = complex_field_loss(pred, target, weights=COMMON["complex_weights"], window_m=roi_window_m(roi_n))
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / max(len(train_loader), 1)
        dt = time.time() - t0
        if epoch % 5 == 0 or epoch == COMMON["epochs"] - 1:
            val_metrics = evaluate(model, val_loader, device=device, roi_n=roi_n)
            entry = {"epoch": epoch, "train_loss": avg_loss, "time_s": dt, **val_metrics}
            raw_phases = [layer.phase().detach().cpu().numpy() for layer in model.layers]
            wrapped_phases = [layer.wrapped_phase().detach().cpu().numpy() for layer in model.layers]
            np.save(run_dir / f"phases_raw_epoch{epoch:03d}.npy", np.stack(raw_phases))
            np.save(run_dir / f"phases_wrapped_epoch{epoch:03d}.npy", np.stack(wrapped_phases))
        else:
            entry = {"epoch": epoch, "train_loss": avg_loss, "time_s": dt}
        history.append(entry)
        co_str = f"{entry['complex_overlap']:.4f}" if "complex_overlap" in entry else "---"
        pr_str = f"{entry['support_weighted_phase_rmse_rad']:.4f}" if "support_weighted_phase_rmse_rad" in entry else "---"
        print(f"  Epoch {epoch:3d}/{COMMON['epochs'] - 1} | loss={avg_loss:.5f} | co={co_str:>6} | sw_pr={pr_str:>6} | {dt:.1f}s")

    test_metrics = evaluate(model, test_loader, device=device, roi_n=roi_n)
    print(
        f"\n  TEST: co={test_metrics['complex_overlap']:.4f} "
        f"sw_pr={test_metrics['support_weighted_phase_rmse_rad']:.4f} "
        f"full_pr={test_metrics['full_field_phase_rmse_rad']:.4f} "
        f"leak={test_metrics['out_of_support_energy_fraction']:.4f}"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history,
            "test_metrics": test_metrics,
            "config": {
                "name": name,
                "roi_n": roi_n,
                "spacing_m": spacing_m,
                **COMMON,
            },
        },
        run_dir / "checkpoint.pt",
    )

    with open(run_dir / "history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    with open(run_dir / "test_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(test_metrics, handle, indent=2)

    model.eval()
    with torch.no_grad():
        sample = test_ds[0]
        u_t = sample["u_turb"].unsqueeze(0).to(device)
        u_v = sample["u_vacuum"].unsqueeze(0).to(device)
        target = prepare_field(u_v, aperture_diameter_m=ap, roi_n=roi_n)
        inp = prepare_field(u_t, aperture_diameter_m=ap, roi_n=roi_n)
        pred = model(inp)
        np.savez(
            run_dir / "sample_fields.npz",
            input_real=inp[0].real.cpu().numpy(),
            input_imag=inp[0].imag.cpu().numpy(),
            pred_real=pred[0].real.cpu().numpy(),
            pred_imag=pred[0].imag.cpu().numpy(),
            target_real=target[0].real.cpu().numpy(),
            target_imag=target[0].imag.cpu().numpy(),
        )

    return {"name": name, "history": history, "test_metrics": test_metrics}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for name, cfg in STUDIES.items():
        result = train_one(name, roi_n=cfg["roi_n"], spacing_m=cfg["spacing_m"], device=device)
        all_results[name] = result["test_metrics"]

    with open(OUT_ROOT / "study_summary.json", "w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2)
    print(f"\nResults saved to {OUT_ROOT}")


if __name__ == "__main__":
    main()
