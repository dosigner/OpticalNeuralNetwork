#!/usr/bin/env python
"""D2NN Full Sweep on Strong Turbulence Data (Cn²=5e-14, D/r₀≈5).

Scale-up from pilot: N=1024, 5 layers, 100 epochs.

Usage:
    cd /root/dj/D2NN/kim2026 && python -m autoresearch.d2nn_strong_turb_sweep
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
OUT_ROOT = Path(__file__).resolve().parent / "runs" / "d2nn_strong_turb_sweep"

# ══════════════════════════════════════════════════════════════════════
# ARCHITECTURE — Full 5-layer D2NN
# ══════════════════════════════════════════════════════════════════════
ARCH = dict(
    num_layers=5,
    layer_spacing_m=10.0e-3,
    detector_distance_m=10.0e-3,
)

TRAIN = dict(
    lr=5e-4,
    epochs=100,
    batch_size=2,
)

# ══════════════════════════════════════════════════════════════════════
# LOSS CONFIGURATIONS — 5 configs (same as previous weak-turb sweep)
# ══════════════════════════════════════════════════════════════════════
LOSS_CONFIGS: dict[str, dict] = OrderedDict({
    "baseline_co": {
        "weights": {"complex_overlap": 1.0},
    },
    "co_amp": {
        "weights": {"complex_overlap": 1.0, "amplitude_mse": 0.5},
    },
    "co_phasor": {
        "weights": {"complex_overlap": 1.0, "phasor_mse": 1.0},
    },
    "co_ffp": {
        "weights": {"complex_overlap": 1.0, "full_field_phase": 1.0},
    },
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
        field, receiver_window_m=RECEIVER_WINDOW_M, aperture_diameter_m=APERTURE_DIAMETER_M,
    )
    return center_crop_field(apertured, crop_n=ROI_N)


def make_model() -> BeamCleanupD2NN:
    return BeamCleanupD2NN(
        n=ROI_N, wavelength_m=WAVELENGTH_M, window_m=roi_window_m(), **ARCH,
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
    run_dir = OUT_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)

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

    if is_roi:
        parts = [f"ROI(th={roi_params['roi_threshold']}, lk={roi_params['leakage_weight']})"]
        desc = " + ".join(parts)
    else:
        active = {k: v for k, v in weights.items() if v > 0}
        desc = " + ".join(f"{k}:{v}" for k, v in active.items())

    print(f"\n{'='*70}")
    print(f"  [{name}] {desc}")
    print(f"  D2NN: {ARCH['num_layers']} layers, N={ROI_N}, {TRAIN['epochs']} epochs")
    print(f"  Data: strong turbulence (Cn²=5e-14, D/r₀≈5)")
    print(f"{'='*70}")

    t_start = time.time()
    history = {"epoch": [], "loss": [], "val_co": [], "val_io": []}

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
            else:
                loss = complex_field_loss(pred, target, weights=weights, window_m=window)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        dt = time.time() - t0

        if epoch % 10 == 0 or epoch == TRAIN["epochs"] - 1:
            val_m = evaluate(model, val_loader, device)
            history["epoch"].append(epoch)
            history["loss"].append(avg_loss)
            history["val_co"].append(val_m["complex_overlap"])
            history["val_io"].append(val_m["intensity_overlap"])
            print(f"  Epoch {epoch:3d}/{TRAIN['epochs']-1} | loss={avg_loss:.5f} | "
                  f"co={val_m['complex_overlap']:.4f} | io={val_m['intensity_overlap']:.4f} | "
                  f"sr={val_m['strehl']:.4f} | bl_co={val_m['baseline_co']:.4f} | {dt:.1f}s")

    train_time = time.time() - t_start

    # Test evaluation
    test_m = evaluate(model, test_loader, device)
    tp = throughput_check(model, test_loader, device)

    peak_vram = 0.0
    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(device) / 1024**2

    # Save
    phases = [layer.phase.detach().cpu().numpy() for layer in model.layers]
    np.save(run_dir / "phases_wrapped.npy", np.stack(phases))

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
        "history": history,
    }
    with open(results_path, "w") as f:
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

    if tp < 0.90 or tp > 1.10:
        print(f"  WARNING: throughput={tp:.4f} outside [0.90, 1.10]")

    improved = test_m["complex_overlap"] > test_m["baseline_co"]
    delta = test_m["complex_overlap"] - test_m["baseline_co"]
    status = "IMPROVED" if improved else "WORSE"

    print(f"  TEST: co={test_m['complex_overlap']:.4f} | baseline={test_m['baseline_co']:.4f} | "
          f"delta={delta:+.4f} | tp={tp:.4f} | {status} | {train_time:.0f}s")

    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: D2NN (non-Fourier, angular spectrum)")
    print(f"Data: strong turbulence (Cn²=5e-14, D/r₀≈5)")
    print(f"Sweeping {len(LOSS_CONFIGS)} loss configurations")
    print(f"Fixed: layers={ARCH['num_layers']}, spacing={ARCH['layer_spacing_m']*1e3:.0f}mm, "
          f"N={ROI_N}, lr={TRAIN['lr']}, epochs={TRAIN['epochs']}")

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
    if tp_zero < 0.90:
        print(f"  WARNING: D2NN path loses {(1-tp_zero)*100:.1f}% energy!")
    else:
        print(f"  OK: Energy conserved (within 10%)")

    # Baseline evaluation
    bl_m = evaluate(diag_model, test_loader, device)
    print(f"  Baseline CO (no correction): {bl_m['baseline_co']:.4f}")
    del diag_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Run all configs
    all_results = []
    for name, config in LOSS_CONFIGS.items():
        result = train_one(name, config, train_loader, val_loader, test_loader, device)
        all_results.append(result)

    # Summary
    print(f"\n{'='*100}")
    print("D2NN STRONG TURBULENCE SWEEP SUMMARY")
    print(f"{'='*100}")

    ranked = sorted(all_results, key=lambda r: r["complex_overlap"], reverse=True)

    header = f"{'Rank':>4} | {'Name':>22} | {'CO':>7} | {'Baseline':>8} | {'Delta':>7} | {'IO':>7} | {'TP':>6} | {'Status':>8}"
    print(header)
    print("-" * len(header))

    for i, r in enumerate(ranked):
        delta = r["complex_overlap"] - r["baseline_co"]
        status = "BETTER" if delta > 0 else "WORSE"
        print(f"{i+1:4d} | {r['name']:>22} | {r['complex_overlap']:7.4f} | "
              f"{r['baseline_co']:8.4f} | {delta:+7.4f} | "
              f"{r['intensity_overlap']:7.4f} | {r['throughput']:6.4f} | {status:>8}")

    bl = ranked[0]["baseline_co"]
    best = ranked[0]
    delta_best = best["complex_overlap"] - bl
    print(f"\nBaseline (no correction): co={bl:.4f}")
    print(f"Best: [{best['name']}] co={best['complex_overlap']:.4f} (delta={delta_best:+.4f})")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUT_ROOT / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUT_ROOT}")


if __name__ == "__main__":
    main()
