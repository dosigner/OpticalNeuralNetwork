#!/usr/bin/env python
"""D2NN Pilot: 2-layer, N=512, 30 epochs on strong turbulence data (Cn²=5e-14).

Quick validation that D2NN can improve beam quality with stronger turbulence.
Success = any config achieves CO > baseline CO.

Usage:
    cd /root/dj/D2NN/kim2026 && python -m autoresearch.d2nn_pilot_strong_turb
"""
from __future__ import annotations

import json
import math
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from kim2026.data.dataset import CachedFieldDataset
from kim2026.models.d2nn import BeamCleanupD2NN
from kim2026.training.losses import (
    complex_field_loss,
    phasor_mse_loss,
)
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
# DATA — strong turbulence (Cn²=5e-14)
# ══════════════════════════════════════════════════════════════════════
WAVELENGTH_M = 1.55e-6
RECEIVER_WINDOW_M = 0.002048       # 2.048mm (after 75:1 beam reducer)
APERTURE_DIAMETER_M = 0.002         # 2mm
N_FULL = 1024
SEED = 20260327

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kim2026" / "1km_cn2_5e-14_tel15cm_n1024_br75" / "cache"
MANIFEST = DATA_DIR.parent / "split_manifest.json"
OUT_ROOT = Path(__file__).resolve().parent / "runs" / "d2nn_pilot_strong_turb"

# ══════════════════════════════════════════════════════════════════════
# PILOT ARCHITECTURE — 2 layers, N=512
# ══════════════════════════════════════════════════════════════════════
PILOT_N = 512                        # center-crop from 1024
ARCH = dict(
    num_layers=2,                    # 2 layers (was 5)
    layer_spacing_m=10.0e-3,         # 10mm between layers
    detector_distance_m=10.0e-3,     # 10mm to detector
)

TRAIN = dict(
    lr=5e-4,
    epochs=30,                       # 30 epochs (quick pilot)
    batch_size=4,                    # larger batch OK with smaller model
)

# ══════════════════════════════════════════════════════════════════════
# LOSS CONFIGURATIONS — 3 core configs
# ══════════════════════════════════════════════════════════════════════
LOSS_CONFIGS: dict[str, dict] = OrderedDict({
    "baseline_co": {
        "weights": {"complex_overlap": 1.0},
    },
    "co_phasor": {
        "weights": {"complex_overlap": 1.0, "phasor_mse": 1.0},
    },
    "co_amp": {
        "weights": {"complex_overlap": 1.0, "amplitude_mse": 0.5},
    },
})


# ══════════════════════════════════════════════════════════════════════
# IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════

def pilot_window_m() -> float:
    return RECEIVER_WINDOW_M * (PILOT_N / N_FULL)


def prepare_field(field: torch.Tensor) -> torch.Tensor:
    """Aperture → center-crop to PILOT_N."""
    apertured = apply_receiver_aperture(
        field,
        receiver_window_m=RECEIVER_WINDOW_M,
        aperture_diameter_m=APERTURE_DIAMETER_M,
    )
    return center_crop_field(apertured, crop_n=PILOT_N)


def make_model() -> BeamCleanupD2NN:
    return BeamCleanupD2NN(
        n=PILOT_N,
        wavelength_m=WAVELENGTH_M,
        window_m=pilot_window_m(),
        **ARCH,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_co, all_pr, all_pr_full, all_ar, all_io, all_sr = [], [], [], [], [], []
    all_co_bl = []

    for batch in loader:
        u_turb = batch["u_turb"].to(device)
        u_vac = batch["u_vacuum"].to(device)
        target = prepare_field(u_vac)
        inp = prepare_field(u_turb)
        pred = model(inp)

        all_co.append(complex_overlap(pred, target).cpu())
        all_pr.append(phase_rmse(pred, target).cpu())
        all_pr_full.append(full_field_phase_rmse(pred, target).cpu())
        all_ar.append(amplitude_rmse(pred, target).cpu())
        pred_i, tgt_i = pred.abs().square(), target.abs().square()
        all_io.append(gaussian_overlap(pred_i, tgt_i).cpu())
        all_sr.append(strehl_ratio(pred_i, tgt_i).cpu())
        all_co_bl.append(complex_overlap(inp, target).cpu())

    return {
        "complex_overlap": float(torch.cat(all_co).mean()),
        "phase_rmse_rad": float(torch.cat(all_pr).mean()),
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
    loss_config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> dict:
    run_dir = OUT_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Resume: skip if already completed
    results_path = run_dir / "results.json"
    if results_path.exists():
        print(f"\n  [{name}] already completed, skipping")
        with open(results_path) as f:
            return json.load(f)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = make_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN["lr"])
    window = pilot_window_m()

    weights = loss_config.get("weights", {})
    active = {k: v for k, v in weights.items() if v > 0}
    desc = " + ".join(f"{k}:{v}" for k, v in active.items())

    print(f"\n{'='*70}")
    print(f"  [{name}] {desc}")
    print(f"  PILOT: {ARCH['num_layers']} layers, N={PILOT_N}, {TRAIN['epochs']} epochs")
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

        # Evaluate every 5 epochs
        if epoch % 5 == 0 or epoch == TRAIN["epochs"] - 1:
            val_m = evaluate(model, val_loader, device)
            history["epoch"].append(epoch)
            history["loss"].append(avg_loss)
            history["val_co"].append(val_m["complex_overlap"])
            print(f"  Epoch {epoch:3d}/{TRAIN['epochs']-1} | loss={avg_loss:.5f} | "
                  f"co={val_m['complex_overlap']:.4f} | io={val_m['intensity_overlap']:.4f} | "
                  f"sr={val_m['strehl']:.4f} | bl_co={val_m['baseline_co']:.4f} | {dt:.1f}s")

    train_time = time.time() - t_start

    # Test evaluation
    test_m = evaluate(model, test_loader, device)
    tp = throughput_check(model, test_loader, device)

    # Save
    result = {
        "name": name,
        "description": desc,
        "model_type": "d2nn_pilot",
        "pilot_n": PILOT_N,
        "loss_config": loss_config,
        "arch": ARCH,
        **test_m,
        "throughput": tp,
        "training_seconds": train_time,
        "history": history,
    }
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    torch.save({"model_state_dict": model.state_dict()}, run_dir / "checkpoint.pt")

    # Save phases
    phases = [layer.phase.detach().cpu().numpy() for layer in model.layers]
    np.save(run_dir / "phases_wrapped.npy", np.stack(phases))

    improved = test_m["complex_overlap"] > test_m["baseline_co"]
    status = "IMPROVED" if improved else "WORSE"
    delta = test_m["complex_overlap"] - test_m["baseline_co"]

    print(f"  TEST: co={test_m['complex_overlap']:.4f} | baseline={test_m['baseline_co']:.4f} | "
          f"delta={delta:+.4f} | tp={tp:.4f} | {status}")

    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PILOT: D2NN {ARCH['num_layers']}-layer, N={PILOT_N}, {TRAIN['epochs']} epochs")
    print(f"Data: strong turbulence (Cn²=5e-14)")
    print(f"Sweeping {len(LOSS_CONFIGS)} loss configurations")

    # Load data
    train_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="train")
    val_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="val")
    test_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="test")
    print(f"Data: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=TRAIN["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=TRAIN["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=TRAIN["batch_size"])

    # Zero-phase throughput diagnostic
    print("\n--- Throughput diagnostic (zero-phase D2NN) ---")
    diag_model = make_model().to(device)
    diag_model.eval()
    tp_zero = throughput_check(diag_model, test_loader, device)
    print(f"  Zero-phase throughput: {tp_zero:.4f}")
    del diag_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Baseline (no D2NN) evaluation
    print("\n--- Baseline (no D2NN) ---")
    identity_model = make_model().to(device)
    identity_model.eval()
    bl_m = evaluate(identity_model, test_loader, device)
    print(f"  Baseline CO (no correction): {bl_m['baseline_co']:.4f}")
    print(f"  Zero-phase D2NN CO: {bl_m['complex_overlap']:.4f}")
    del identity_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Run all configs
    all_results = []
    for name, config in LOSS_CONFIGS.items():
        result = train_one(name, config, train_loader, val_loader, test_loader, device)
        all_results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print("PILOT SUMMARY")
    print(f"{'='*80}")

    baseline_co = all_results[0].get("baseline_co", 0)
    any_improved = False

    for r in all_results:
        delta = r["complex_overlap"] - r["baseline_co"]
        status = "IMPROVED" if delta > 0 else "WORSE"
        if delta > 0:
            any_improved = True
        print(f"  {r['name']:>15} | CO={r['complex_overlap']:.4f} | "
              f"baseline={r['baseline_co']:.4f} | delta={delta:+.4f} | {status}")

    print(f"\n{'='*80}")
    if any_improved:
        print("SUCCESS: D2NN improves beam quality with stronger turbulence!")
        print("→ Proceed to Step 3: Scale up to N=1024, 5 layers")
    else:
        print("FAIL: D2NN still cannot improve beam quality.")
        print("→ Proceed to Step 4: Add TV regularization")
    print(f"{'='*80}")

    # Save summary
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUT_ROOT / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUT_ROOT}")


if __name__ == "__main__":
    main()
