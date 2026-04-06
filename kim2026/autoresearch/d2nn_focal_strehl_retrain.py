#!/usr/bin/env python
"""Retrain focal_strehl_only with CORRECT Strehl loss.

Fix: 4x zero-padding + flat-phase reference → S ≤ 1 guaranteed.

The original focal_strehl_only was trained with undersampled PSF (Airy=1.25px)
and curved-wavefront vacuum reference, producing invalid S≈28.

Usage:
    cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python -m autoresearch.d2nn_focal_strehl_retrain
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from kim2026.data.canonical_pupil import enforce_reducer_validation_gate
from kim2026.data.dataset import CachedFieldDataset
from kim2026.models.d2nn import BeamCleanupD2NN
from kim2026.optics.lens_2f import lens_2f_forward
from kim2026.training.metrics import complex_overlap, gaussian_overlap, strehl_ratio_correct
from kim2026.training.targets import apply_receiver_aperture, center_crop_field

# ══════════════════════════════════════════════════════════════════════
WAVELENGTH_M = 1.55e-6
RECEIVER_WINDOW_M = 0.002048
APERTURE_DIAMETER_M = 0.002
N_FULL = 1024
ROI_N = 1024
SEED = 20260331

FOCUS_F_M = 4.5e-3
PIB_BUCKET_RADIUS_UM = 10.0
STREHL_PAD_FACTOR = 4  # 4x zero-padding → Airy ≈ 5 px

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kim2026" / "1km_cn2_5e-14_tel15cm_pupil1024_v1" / "cache"
MANIFEST = DATA_DIR.parent / "split_manifest.json"
DATA_PLANE_SELECTOR = "reduced_ideal"
REDUCER_VALIDATION_SUMMARY_PATH = DATA_DIR.parent / "reducer_val_cache" / "summary.json"
OUT_ROOT = Path(__file__).resolve().parent / "runs" / "d2nn_focal_pib_sweep"

ARCH = dict(num_layers=5, layer_spacing_m=10e-3, detector_distance_m=10e-3)

TRAIN = dict(
    lr=2e-3,            # conservative for Strehl (smoother landscape)
    epochs=100,
    batch_size=8,       # 4x padding → larger tensors
    tv_weight=0.05,
    warmup_epochs=10,
)


def roi_window_m():
    return RECEIVER_WINDOW_M * (ROI_N / N_FULL)


def prepare_field(field: torch.Tensor) -> torch.Tensor:
    a = apply_receiver_aperture(field, receiver_window_m=RECEIVER_WINDOW_M,
                                 aperture_diameter_m=APERTURE_DIAMETER_M)
    return center_crop_field(a, crop_n=ROI_N)


def make_model() -> BeamCleanupD2NN:
    return BeamCleanupD2NN(n=ROI_N, wavelength_m=WAVELENGTH_M, window_m=roi_window_m(), **ARCH)


def to_focal_plane(field: torch.Tensor) -> tuple[torch.Tensor, float]:
    dx_in_m = roi_window_m() / ROI_N
    return lens_2f_forward(field, dx_in_m=dx_in_m, wavelength_m=WAVELENGTH_M,
                            f_m=FOCUS_F_M, na=None, apply_scaling=False)


def total_variation(model: BeamCleanupD2NN) -> torch.Tensor:
    tv = torch.tensor(0.0, device=next(model.parameters()).device)
    for layer in model.layers:
        phase = layer.phase
        tv = tv + (phase[:, :-1] - phase[:, 1:]).abs().mean()
        tv = tv + (phase[:-1, :] - phase[1:, :]).abs().mean()
    return tv


def focal_strehl_loss_correct(d2nn_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - correct Strehl using the field's own flat-phase reference."""
    sr = strehl_ratio_correct(d2nn_pred, pad_factor=STREHL_PAD_FACTOR)
    return 1.0 - sr.mean()


def compute_focal_pib(focal_field, dx_focal, radius_um):
    intensity = focal_field.abs().square()
    n = focal_field.shape[-1]; c = n // 2
    ref_radius = radius_um * 1e-6
    yy, xx = torch.meshgrid(torch.arange(n, device=focal_field.device) - c,
                             torch.arange(n, device=focal_field.device) - c, indexing="ij")
    r = torch.sqrt((xx * dx_focal) ** 2 + (yy * dx_focal) ** 2)
    mask = (r <= ref_radius).float()
    pib = (intensity * mask).sum(dim=(-2, -1)) / intensity.sum(dim=(-2, -1)).clamp(min=1e-12)
    return pib


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_co, all_co_bl = [], []
    all_pib_10, all_pib_10_bl, all_pib_10_vac = [], [], []
    all_pib_50, all_pib_50_bl, all_pib_50_vac = [], [], []
    all_strehl = []

    for batch in loader:
        u_turb = batch["u_turb"].to(device)
        u_vac = batch["u_vacuum"].to(device)
        target = prepare_field(u_vac)
        inp = prepare_field(u_turb)
        d2nn_pred = model(inp)

        # CO at output plane
        all_co.append(complex_overlap(d2nn_pred, target).cpu())
        all_co_bl.append(complex_overlap(inp, target).cpu())

        # Correct Strehl (4x padded, flat-phase ref)
        sr = strehl_ratio_correct(d2nn_pred, pad_factor=STREHL_PAD_FACTOR)
        all_strehl.append(sr.cpu())

        # Focal PIB
        focal_pred, dx_f = to_focal_plane(d2nn_pred)
        focal_target, _ = to_focal_plane(target)
        focal_inp, _ = to_focal_plane(inp)

        all_pib_10.append(compute_focal_pib(focal_pred, dx_f, 10.0).cpu())
        all_pib_50.append(compute_focal_pib(focal_pred, dx_f, 50.0).cpu())
        all_pib_10_bl.append(compute_focal_pib(focal_inp, dx_f, 10.0).cpu())
        all_pib_50_bl.append(compute_focal_pib(focal_inp, dx_f, 50.0).cpu())
        all_pib_10_vac.append(compute_focal_pib(focal_target, dx_f, 10.0).cpu())
        all_pib_50_vac.append(compute_focal_pib(focal_target, dx_f, 50.0).cpu())

        del focal_pred, focal_target, focal_inp
        torch.cuda.empty_cache()

    return {
        "co_output": float(torch.cat(all_co).mean()),
        "co_baseline": float(torch.cat(all_co_bl).mean()),
        "focal_strehl_correct": float(torch.cat(all_strehl).mean()),
        "focal_pib_10um": float(torch.cat(all_pib_10).mean()),
        "focal_pib_50um": float(torch.cat(all_pib_50).mean()),
        "focal_pib_10um_baseline": float(torch.cat(all_pib_10_bl).mean()),
        "focal_pib_50um_baseline": float(torch.cat(all_pib_50_bl).mean()),
        "focal_pib_10um_vacuum": float(torch.cat(all_pib_10_vac).mean()),
        "focal_pib_50um_vacuum": float(torch.cat(all_pib_50_vac).mean()),
        "dx_focal_um": dx_f * 1e6,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dx_in = roi_window_m() / ROI_N
    dx_focal = WAVELENGTH_M * FOCUS_F_M / (ROI_N * dx_in)
    airy = 1.22 * WAVELENGTH_M * FOCUS_F_M / APERTURE_DIAMETER_M
    dx_focal_pad = WAVELENGTH_M * FOCUS_F_M / (ROI_N * STREHL_PAD_FACTOR * dx_in)
    print(f"\nFocal Strehl Retrain (CORRECT: 4x padding + flat-phase ref)")
    print(f"  Airy first zero: {airy*1e6:.2f}μm")
    print(f"  dx_focal (no pad): {dx_focal*1e6:.3f}μm → Airy={airy/dx_focal:.2f}px (undersampled!)")
    print(f"  dx_focal (4x pad): {dx_focal_pad*1e6:.3f}μm → Airy={airy/dx_focal_pad:.2f}px (OK)")
    print(f"  batch_size={TRAIN['batch_size']}, lr={TRAIN['lr']}, epochs={TRAIN['epochs']}")
    enforce_reducer_validation_gate(
        {
            "cache_dir": str(DATA_DIR),
            "plane_selector": DATA_PLANE_SELECTOR,
            "reducer_validation": {
                "required": True,
                "summary_path": str(REDUCER_VALIDATION_SUMMARY_PATH),
            },
        }
    )

    train_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="train", plane_selector=DATA_PLANE_SELECTOR)
    val_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="val", plane_selector=DATA_PLANE_SELECTOR)
    test_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="test", plane_selector=DATA_PLANE_SELECTOR)
    print(f"  Data: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=TRAIN["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=8, num_workers=0)

    name = "focal_strehl_only"
    run_dir = OUT_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED); np.random.seed(SEED)
    model = make_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=TRAIN["epochs"] - TRAIN["warmup_epochs"], eta_min=1e-5)

    print(f"\n{'='*70}")
    print(f"  [focal_strehl_only] CORRECT Strehl loss (4x pad, flat-phase ref)")
    print(f"  TV={TRAIN['tv_weight']}, LR={TRAIN['lr']}→cosine, epochs={TRAIN['epochs']}")
    print(f"{'='*70}")

    t_start = time.time()
    history = {"epoch": [], "loss": [], "val_co": [], "val_strehl": [], "val_focal_pib_10": [], "lr": []}
    log_file = run_dir / "epoch_log.txt"

    for epoch in range(TRAIN["epochs"]):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        if epoch < TRAIN["warmup_epochs"]:
            lr_scale = (epoch + 1) / TRAIN["warmup_epochs"]
            for pg in optimizer.param_groups:
                pg["lr"] = TRAIN["lr"] * lr_scale

        for batch in train_loader:
            u_turb = batch["u_turb"].to(device)
            u_vac = batch["u_vacuum"].to(device)
            target = prepare_field(u_vac)
            inp = prepare_field(u_turb)

            optimizer.zero_grad(set_to_none=True)
            d2nn_pred = model(inp)

            # Correct Strehl loss: 4x padded FFT, flat-phase reference
            loss = focal_strehl_loss_correct(d2nn_pred, target)

            if TRAIN["tv_weight"] > 0:
                loss = loss + TRAIN["tv_weight"] * total_variation(model)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch >= TRAIN["warmup_epochs"]:
            scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        dt = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]

        with open(log_file, "a") as lf:
            lf.write(f"ep{epoch:3d} loss={avg_loss:.5f} lr={cur_lr:.2e} {dt:.1f}s\n")

        if epoch % 50 == 0 or epoch == TRAIN["epochs"] - 1:
            with open(log_file, "a") as lf:
                lf.write(f"  -> eval at epoch {epoch}...\n")
            val_m = evaluate(model, val_loader, device)
            history["epoch"].append(epoch)
            history["loss"].append(avg_loss)
            history["val_co"].append(val_m["co_output"])
            history["val_strehl"].append(val_m["focal_strehl_correct"])
            history["val_focal_pib_10"].append(val_m["focal_pib_10um"])
            history["lr"].append(cur_lr)
            msg = (f"  Epoch {epoch:3d}/{TRAIN['epochs']-1} | loss={avg_loss:.5f} | "
                   f"co={val_m['co_output']:.4f} | "
                   f"strehl={val_m['focal_strehl_correct']:.4f} | "
                   f"fpib10={val_m['focal_pib_10um']:.4f} | "
                   f"lr={cur_lr:.2e} | {dt:.1f}s")
            print(msg, flush=True)
            with open(log_file, "a") as lf:
                lf.write(msg + "\n")

    train_time = time.time() - t_start

    # Test
    test_m = evaluate(model, test_loader, device)
    phases = [layer.phase.detach().cpu().numpy() for layer in model.layers]
    np.save(run_dir / "phases_wrapped.npy", np.stack(phases))

    result = {
        "name": name,
        "description": "Correct focal Strehl loss (4x pad + flat-phase ref)",
        "arch": ARCH,
        "strehl_pad_factor": STREHL_PAD_FACTOR,
        **test_m,
        "training_seconds": train_time,
        "history": history,
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    torch.save({"model_state_dict": model.state_dict()}, run_dir / "checkpoint.pt")

    print(f"\n  TEST: strehl={test_m['focal_strehl_correct']:.4f} | "
          f"fpib10={test_m['focal_pib_10um']:.4f} | co={test_m['co_output']:.4f}")
    print(f"  Time: {train_time:.0f}s")
    print(f"  Saved to {run_dir}")


if __name__ == "__main__":
    main()
