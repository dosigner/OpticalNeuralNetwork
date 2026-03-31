#!/usr/bin/env python
"""FD2NN-64 Sweep: Low-resolution Fourier masks on strong turbulence.

Sweeps n_mask = {32, 48, 64} with 3 loss configs each.
Uses CroppedFourierD2NN: Input(1024) → FFT → crop(n_mask) → masks → pad(1024) → IFFT

Usage:
    cd /root/dj/D2NN/kim2026 && python -m autoresearch.fd2nn_64_strong_turb
"""
from __future__ import annotations

import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from kim2026.data.dataset import CachedFieldDataset
from kim2026.models.fd2nn import CroppedFourierD2NN
from kim2026.training.losses import complex_field_loss
from kim2026.training.metrics import (
    amplitude_rmse,
    complex_overlap,
    full_field_phase_rmse,
    gaussian_overlap,
    phase_rmse,
    strehl_ratio,
)
from kim2026.training.targets import apply_receiver_aperture, center_crop_field

# ══════════════════════════════════════════════════════════════════════
# DATA — Strong turbulence (Cn²=5e-14)
# ══════════════════════════════════════════════════════════════════════
WAVELENGTH_M = 1.55e-6
RECEIVER_WINDOW_M = 0.002048
APERTURE_DIAMETER_M = 0.002
N_FULL = 1024
ROI_N = 1024
SEED = 20260327

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kim2026" / "1km_cn2_5e-14_tel15cm_n1024_br75" / "cache"
MANIFEST = DATA_DIR.parent / "split_manifest.json"
OUT_ROOT = Path(__file__).resolve().parent / "runs" / "fd2nn_64_strong_turb"

# ══════════════════════════════════════════════════════════════════════
# FIXED OPTICS
# ══════════════════════════════════════════════════════════════════════
OPTICS = dict(
    dual_2f_f1_m=25e-3,
    dual_2f_f2_m=25e-3,
    dual_2f_na1=0.508,
    dual_2f_na2=0.508,
    dual_2f_apply_scaling=False,
)

TRAIN = dict(
    lr=5e-4,
    epochs=30,
    batch_size=16,
)

# ══════════════════════════════════════════════════════════════════════
# SWEEP: n_mask × loss_config
# ══════════════════════════════════════════════════════════════════════
N_MASKS = [32, 48, 64]

LOSS_CONFIGS: dict[str, dict] = OrderedDict({
    "baseline_co": {"weights": {"complex_overlap": 1.0}},
    "co_phasor": {"weights": {"complex_overlap": 1.0, "phasor_mse": 1.0}},
    "co_amp": {"weights": {"complex_overlap": 1.0, "amplitude_mse": 0.5}},
})


def roi_window_m():
    return RECEIVER_WINDOW_M * (ROI_N / N_FULL)


def prepare_field(field: torch.Tensor) -> torch.Tensor:
    a = apply_receiver_aperture(field, receiver_window_m=RECEIVER_WINDOW_M,
                                 aperture_diameter_m=APERTURE_DIAMETER_M)
    return center_crop_field(a, crop_n=ROI_N)


def make_model(n_mask: int) -> CroppedFourierD2NN:
    return CroppedFourierD2NN(
        n_input=ROI_N,
        n_mask=n_mask,
        wavelength_m=WAVELENGTH_M,
        window_m=roi_window_m(),
        num_layers=5,
        layer_spacing_m=5e-3,
        **OPTICS,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_co, all_pr_full, all_ar, all_io, all_sr = [], [], [], [], []
    all_co_bl = []
    for batch in loader:
        u_turb = batch["u_turb"].to(device)
        u_vac = batch["u_vacuum"].to(device)
        target = prepare_field(u_vac)
        inp = prepare_field(u_turb)
        pred = model(inp)
        all_co.append(complex_overlap(pred, target).cpu())
        all_pr_full.append(full_field_phase_rmse(pred, target).cpu())
        all_ar.append(amplitude_rmse(pred, target).cpu())
        pred_i, tgt_i = pred.abs().square(), target.abs().square()
        all_io.append(gaussian_overlap(pred_i, tgt_i).cpu())
        all_sr.append(strehl_ratio(pred_i, tgt_i).cpu())
        all_co_bl.append(complex_overlap(inp, target).cpu())
    return {
        "complex_overlap": float(torch.cat(all_co).mean()),
        "full_field_phase_rmse_rad": float(torch.cat(all_pr_full).mean()),
        "amplitude_rmse": float(torch.cat(all_ar).mean()),
        "intensity_overlap": float(torch.cat(all_io).mean()),
        "strehl": float(torch.cat(all_sr).mean()),
        "baseline_co": float(torch.cat(all_co_bl).mean()),
    }


@torch.no_grad()
def throughput_check(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_in, total_out = 0.0, 0.0
    for batch in loader:
        inp = prepare_field(batch["u_turb"].to(device))
        pred = model(inp)
        total_in += inp.abs().square().sum().item()
        total_out += pred.abs().square().sum().item()
    return total_out / max(total_in, 1e-12)


def train_one(
    name: str,
    n_mask: int,
    loss_config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> dict:
    run_dir = OUT_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.json"
    if results_path.exists():
        print(f"\n  [{name}] already completed, skipping")
        with open(results_path) as f:
            return json.load(f)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = make_model(n_mask).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN["lr"])
    window = roi_window_m()
    weights = loss_config.get("weights", {})
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*70}")
    print(f"  [{name}] n_mask={n_mask}, params={n_params:,}")
    print(f"  {' + '.join(f'{k}:{v}' for k, v in weights.items() if v > 0)}")
    print(f"{'='*70}")

    t_start = time.time()
    history = {"epoch": [], "loss": [], "val_co": []}

    for epoch in range(TRAIN["epochs"]):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for batch in train_loader:
            u_turb = batch["u_turb"].to(device)
            u_vac = batch["u_vacuum"].to(device)
            target = prepare_field(u_vac)
            inp = prepare_field(u_turb)
            optimizer.zero_grad(set_to_none=True)
            pred = model(inp)
            loss = complex_field_loss(pred, target, weights=weights, window_m=window)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        dt = time.time() - t0

        if epoch % 5 == 0 or epoch == TRAIN["epochs"] - 1:
            val_m = evaluate(model, val_loader, device)
            history["epoch"].append(epoch)
            history["loss"].append(avg_loss)
            history["val_co"].append(val_m["complex_overlap"])
            print(f"  Epoch {epoch:3d}/{TRAIN['epochs']-1} | loss={avg_loss:.5f} | "
                  f"co={val_m['complex_overlap']:.4f} | bl={val_m['baseline_co']:.4f} | {dt:.1f}s")

    train_time = time.time() - t_start
    test_m = evaluate(model, test_loader, device)
    tp = throughput_check(model, test_loader, device)

    result = {
        "name": name,
        "model_type": "fd2nn_cropped",
        "n_mask": n_mask,
        "n_params": n_params,
        "loss_config": loss_config,
        **test_m,
        "throughput": tp,
        "training_seconds": train_time,
        "history": history,
    }
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    torch.save({"model_state_dict": model.state_dict()}, run_dir / "checkpoint.pt")
    phases = [layer.phase().detach().cpu().numpy() for layer in model.layers]
    np.save(run_dir / "phases_wrapped.npy", np.stack(phases))

    delta = test_m["complex_overlap"] - test_m["baseline_co"]
    status = "IMPROVED" if delta > 0 else "WORSE"
    print(f"  TEST: co={test_m['complex_overlap']:.4f} | baseline={test_m['baseline_co']:.4f} | "
          f"delta={delta:+.4f} | tp={tp:.4f} | {status}")
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"FD2NN Cropped Fourier Mask Sweep")
    print(f"n_mask sizes: {N_MASKS}")
    print(f"Loss configs: {list(LOSS_CONFIGS.keys())}")
    print(f"Total runs: {len(N_MASKS) * len(LOSS_CONFIGS)}")
    print(f"batch_size={TRAIN['batch_size']}, epochs={TRAIN['epochs']}")

    train_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="train")
    val_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="val")
    test_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="test")
    print(f"Data: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=TRAIN["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=TRAIN["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=TRAIN["batch_size"])

    # Zero-phase throughput for each n_mask
    print("\n--- Throughput diagnostics ---")
    for nm in N_MASKS:
        m = make_model(nm).to(device)
        m.eval()
        tp = throughput_check(m, test_loader, device)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"  n_mask={nm}: throughput={tp:.4f}, params={n_params:,}")
        del m
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Run all combos
    all_results = []
    for nm in N_MASKS:
        for loss_name, loss_config in LOSS_CONFIGS.items():
            name = f"nm{nm}_{loss_name}"
            result = train_one(name, nm, loss_config, train_loader, val_loader, test_loader, device)
            all_results.append(result)

    # Summary
    print(f"\n{'='*90}")
    print("FD2NN CROPPED FOURIER SWEEP SUMMARY")
    print(f"{'='*90}")
    ranked = sorted(all_results, key=lambda r: r["complex_overlap"], reverse=True)
    bl = ranked[0]["baseline_co"]
    print(f"Baseline (no correction): CO={bl:.4f}\n")

    header = f"{'Rank':>4} | {'Name':>25} | {'n_mask':>6} | {'Params':>8} | {'CO':>7} | {'Delta':>7} | {'TP':>6}"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(ranked):
        delta = r["complex_overlap"] - r["baseline_co"]
        print(f"{i+1:4d} | {r['name']:>25} | {r['n_mask']:6d} | {r['n_params']:>8,} | "
              f"{r['complex_overlap']:7.4f} | {delta:+7.4f} | {r['throughput']:6.4f}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUT_ROOT / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUT_ROOT}")


if __name__ == "__main__":
    main()
