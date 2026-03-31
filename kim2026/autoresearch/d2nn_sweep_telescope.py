#!/usr/bin/env python
"""Non-Fourier D2NN loss sweep on telescope data.

BeamCleanupD2NN uses angular spectrum free-space propagation between layers.
No Fourier lens → no under-resolution problem. Phase masks operate in spatial domain.

Usage:
    cd /root/dj/D2NN/kim2026 && python -m autoresearch.d2nn_sweep_telescope
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
    roi_complex_loss,
    phasor_mse_loss,
    soft_weighted_phasor_loss,
    beam_radius,
    encircled_energy_fraction,
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
# IMMUTABLE — must match data
# ══════════════════════════════════════════════════════════════════════
WAVELENGTH_M = 1.55e-6
RECEIVER_WINDOW_M = 0.002048       # 2.048mm (after 75:1 beam reducer)
APERTURE_DIAMETER_M = 0.002         # 2mm
N_FULL = 1024
ROI_N = 1024
SEED = 20260327

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kim2026" / "1km_cn2e-14_tel15cm_n1024_br75" / "cache"
MANIFEST = DATA_DIR.parent / "split_manifest.json"
OUT_ROOT = Path(__file__).resolve().parent / "runs" / "d2nn_sweep_telescope"

# ══════════════════════════════════════════════════════════════════════
# FIXED ARCHITECTURE — Non-Fourier D2NN
# ══════════════════════════════════════════════════════════════════════
ARCH = dict(
    num_layers=5,
    layer_spacing_m=10.0e-3,         # 10mm between layers (free-space propagation)
    detector_distance_m=10.0e-3,     # 10mm to detector
    # Total length: 5 masks × 10mm + 10mm = 50mm
)

TRAIN = dict(
    lr=5e-4,
    epochs=100,
    batch_size=2,
)

# ══════════════════════════════════════════════════════════════════════
# LOSS CONFIGURATIONS — Phase 1: 핵심 5개
# ══════════════════════════════════════════════════════════════════════
LOSS_CONFIGS: dict[str, dict] = OrderedDict({

    # ── Baseline ──
    "baseline_co": {
        "weights": {"complex_overlap": 1.0},
    },

    # ── CO + amplitude ──
    "co_amp": {
        "weights": {"complex_overlap": 1.0, "amplitude_mse": 0.5},
    },

    # ── Phase correction ──
    "co_phasor": {
        "weights": {"complex_overlap": 1.0, "phasor_mse": 1.0},
    },

    # ── Full-field phase ──
    "co_ffp": {
        "weights": {"complex_overlap": 1.0, "full_field_phase": 1.0},
    },

    # ── ROI (should work better than FD2NN since no Fourier limitation) ──
    "roi80": {
        "roi": {"roi_threshold": 0.80, "leakage_weight": 1.0},
    },
})


# ══════════════════════════════════════════════════════════════════════
# IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════

def roi_window_m() -> float:
    return RECEIVER_WINDOW_M * (ROI_N / N_FULL)


def prepare_field(field: torch.Tensor) -> torch.Tensor:
    apertured = apply_receiver_aperture(
        field,
        receiver_window_m=RECEIVER_WINDOW_M,
        aperture_diameter_m=APERTURE_DIAMETER_M,
    )
    return center_crop_field(apertured, crop_n=ROI_N)


def make_model() -> BeamCleanupD2NN:
    return BeamCleanupD2NN(
        n=ROI_N,
        wavelength_m=WAVELENGTH_M,
        window_m=roi_window_m(),
        **ARCH,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    window = roi_window_m()
    all_co, all_pr, all_pr_full, all_ar, all_io, all_sr, all_ee = [], [], [], [], [], [], []
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
        ref_r = beam_radius(tgt_i, window_m=window)
        all_ee.append(encircled_energy_fraction(pred_i, reference_radius=ref_r, window_m=window).cpu())
        all_co_bl.append(complex_overlap(inp, target).cpu())

    return {
        "complex_overlap": float(torch.cat(all_co).mean()),
        "phase_rmse_rad": float(torch.cat(all_pr).mean()),
        "full_field_phase_rmse_rad": float(torch.cat(all_pr_full).mean()),
        "amplitude_rmse": float(torch.cat(all_ar).mean()),
        "intensity_overlap": float(torch.cat(all_io).mean()),
        "strehl": float(torch.cat(all_sr).mean()),
        "encircled_energy": float(torch.cat(all_ee).mean()),
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
    """Train one D2NN model with the given loss config."""
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
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model = make_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN["lr"])
    window = roi_window_m()

    is_roi = "roi" in loss_config
    weights = loss_config.get("weights", {})
    roi_params = loss_config.get("roi", {})
    phasor_w = loss_config.get("phasor_weight", 0.0)
    soft_phasor_w = loss_config.get("soft_phasor_weight", 0.0)
    support_gamma = loss_config.get("support_gamma", 2.0)

    # Describe
    if is_roi:
        parts = [f"ROI(th={roi_params['roi_threshold']}, lk={roi_params['leakage_weight']})"]
        if phasor_w > 0:
            parts.append(f"ph:{phasor_w}")
        desc = " + ".join(parts)
    else:
        active = {k: v for k, v in weights.items() if v > 0}
        desc = " + ".join(f"{k}:{v}" for k, v in active.items())

    print(f"\n{'='*70}")
    print(f"  [{name}] {desc}")
    print(f"  D2NN: {ARCH['num_layers']} layers, spacing={ARCH['layer_spacing_m']*1e3:.0f}mm, "
          f"det={ARCH['detector_distance_m']*1e3:.0f}mm")
    print(f"{'='*70}")

    t_start = time.time()
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

            if is_roi:
                loss = roi_complex_loss(pred, target, window_m=window, **roi_params)
                if phasor_w > 0:
                    loss = loss + phasor_w * phasor_mse_loss(pred, target)
                if soft_phasor_w > 0:
                    loss = loss + soft_phasor_w * soft_weighted_phasor_loss(
                        pred, target, gamma=support_gamma)
            else:
                loss = complex_field_loss(pred, target, weights=weights, window_m=window)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        dt = time.time() - t0

        if epoch % 10 == 0 or epoch == TRAIN["epochs"] - 1:
            val_m = evaluate(model, val_loader, device)
            print(f"  Epoch {epoch:3d}/{TRAIN['epochs']-1} | loss={avg_loss:.5f} | "
                  f"co={val_m['complex_overlap']:.4f} | io={val_m['intensity_overlap']:.4f} | "
                  f"sr={val_m['strehl']:.4f} | {dt:.1f}s")

    train_time = time.time() - t_start

    # Test evaluation
    test_m = evaluate(model, test_loader, device)
    tp = throughput_check(model, test_loader, device)

    peak_vram = 0.0
    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(device) / 1024**2

    # Save phases
    phases = [layer.phase.detach().cpu().numpy() for layer in model.layers]
    np.save(run_dir / "phases_wrapped.npy", np.stack(phases))

    # Save results
    result = {
        "name": name,
        "description": desc,
        "model_type": "d2nn",
        "loss_config": loss_config,
        "arch": ARCH,
        **test_m,
        "throughput": tp,
        "training_seconds": train_time,
        "peak_vram_mb": peak_vram,
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    torch.save({"model_state_dict": model.state_dict()}, run_dir / "checkpoint.pt")

    # Save sample fields
    model.eval()
    with torch.no_grad():
        sample = test_loader.dataset[0]
        u_t = sample["u_turb"].unsqueeze(0).to(device)
        u_v = sample["u_vacuum"].unsqueeze(0).to(device)
        target = prepare_field(u_v)
        inp = prepare_field(u_t)
        pred = model(inp)

    np.savez(
        run_dir / "sample_fields.npz",
        input_intensity=inp[0].abs().square().cpu().numpy(),
        input_phase=torch.angle(inp[0]).cpu().numpy(),
        pred_intensity=pred[0].abs().square().cpu().numpy(),
        pred_phase=torch.angle(pred[0]).cpu().numpy(),
        target_intensity=target[0].abs().square().cpu().numpy(),
        target_phase=torch.angle(target[0]).cpu().numpy(),
        window_m=roi_window_m(),
        n=ROI_N,
    )

    # Warnings
    if tp < 0.90 or tp > 1.10:
        print(f"  ⚠ WARN: throughput={tp:.4f} outside [0.90, 1.10]")

    print(f"  TEST: co={test_m['complex_overlap']:.4f} | io={test_m['intensity_overlap']:.4f} | "
          f"sr={test_m['strehl']:.4f} | tp={tp:.4f} | {train_time:.0f}s")

    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: D2NN (non-Fourier, angular spectrum propagation)")
    print(f"Data: telescope (15cm + 75:1)")
    print(f"Sweeping {len(LOSS_CONFIGS)} loss configurations")
    print(f"Fixed: layers={ARCH['num_layers']}, spacing={ARCH['layer_spacing_m']*1e3:.0f}mm, "
          f"detector={ARCH['detector_distance_m']*1e3:.0f}mm, "
          f"lr={TRAIN['lr']}, epochs={TRAIN['epochs']}")

    # Load data
    train_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="train")
    val_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="val")
    test_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="test")
    train_loader = DataLoader(train_ds, batch_size=TRAIN["batch_size"], shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=TRAIN["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=TRAIN["batch_size"])

    # Zero-phase throughput diagnostic
    print("\n--- Throughput diagnostic (zero-phase D2NN) ---")
    diag_model = make_model().to(device)
    diag_model.eval()
    tp_zero = throughput_check(diag_model, test_loader, device)
    print(f"  Zero-phase throughput: {tp_zero:.4f}")
    if tp_zero < 0.90:
        print(f"  ⚠ D2NN path loses {(1-tp_zero)*100:.1f}% energy!")
    else:
        print(f"  ✓ Energy conserved (within 10%)")
    del diag_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Run all configs
    all_results = []
    for name, config in LOSS_CONFIGS.items():
        result = train_one(name, config, train_loader, val_loader, test_loader, device)
        all_results.append(result)

    # Summary table
    print(f"\n{'='*100}")
    print("D2NN TELESCOPE SWEEP SUMMARY")
    print(f"{'='*100}")

    ranked = sorted(all_results, key=lambda r: r["complex_overlap"], reverse=True)

    header = f"{'Rank':>4} | {'Name':>22} | {'CO':>7} | {'IO':>7} | {'Strehl':>7} | {'PhRMSE':>7} | {'TP':>6} | {'Time':>5}"
    print(header)
    print("-" * len(header))

    for i, r in enumerate(ranked):
        flag = " ⚠" if r["throughput"] < 0.90 or r["throughput"] > 1.10 else ""
        print(f"{i+1:4d} | {r['name']:>22} | {r['complex_overlap']:7.4f} | "
              f"{r['intensity_overlap']:7.4f} | {r['strehl']:7.4f} | "
              f"{r['full_field_phase_rmse_rad']:7.4f} | "
              f"{r['throughput']:6.4f} | {r['training_seconds']:5.0f}s{flag}")

    bl = ranked[0]["baseline_co"]
    print(f"\nBaseline (no correction): co={bl:.4f}")
    print(f"Best: [{ranked[0]['name']}] co={ranked[0]['complex_overlap']:.4f}")

    # Save summary
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUT_ROOT / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUT_ROOT}")


if __name__ == "__main__":
    main()
