#!/usr/bin/env python
"""FD2NN loss function sweep — systematic comparison of loss combinations.

Holds all other hyperparameters fixed. Varies only the loss composition.
Trains one model per config, evaluates, and prints a comparison table.

Usage:
    cd /root/dj/D2NN/kim2026 && python -m autoresearch.loss_sweep
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
from kim2026.models.fd2nn import BeamCleanupFD2NN
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
# IMMUTABLE — must match data, DO NOT CHANGE
# ══════════════════════════════════════════════════════════════════════
WAVELENGTH_M = 1.55e-6
RECEIVER_WINDOW_M = 0.002048
APERTURE_DIAMETER_M = 0.002
N_FULL = 1024
ROI_N = 1024  # no crop — use full grid to avoid aliasing from truncation
SEED = 20260323

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kim2026" / "1km_cn2e-14_tel15cm_n1024_br75" / "cache"
MANIFEST = DATA_DIR.parent / "split_manifest.json"
OUT_ROOT = Path(__file__).resolve().parent / "runs" / "loss_sweep_telescope"

# ══════════════════════════════════════════════════════════════════════
# FIXED ARCHITECTURE — same across all loss configs
# ══════════════════════════════════════════════════════════════════════
ARCH = dict(
    num_layers=5,
    layer_spacing_m=5.0e-3,    # 5mm inter-layer spacing
    phase_constraint="unconstrained",
    phase_max=math.pi,
    phase_init="uniform",
    phase_init_scale=0.1,
    # Thorlabs AC254-025-C: f=25mm, Ø25.4mm, C-coat (1050-1700nm)
    # NA = 25.4/(2×25) = 0.508
    # At n=1024, dx_in=2μm: dx_fourier=18.92μm, fourier_window=19.38mm < Ø25.4mm ✓
    # (AC127 Ø12.7mm would clip: 19.38mm > 12.7mm — only valid for n≤512)
    dual_2f_f1_m=25.0e-3,
    dual_2f_f2_m=25.0e-3,
    dual_2f_na1=0.508,
    dual_2f_na2=0.508,
    dual_2f_apply_scaling=False,
)

# FIXED TRAINING — same across all loss configs
TRAIN = dict(
    lr=5e-4,
    epochs=100,
    batch_size=2,
    phasor_smoothness_weight=0.01,
)

# ══════════════════════════════════════════════════════════════════════
# LOSS CONFIGURATIONS TO SWEEP — Telescope data (15cm + 75:1)
# ══════════════════════════════════════════════════════════════════════
#
# Telescope data: Fourier spot = 7px (under-resolved).
# ROI 기반 loss는 degenerate할 가능성 높음.
# Phase correction과 full-field 접근 위주로 설계.
#
# Phase 1: 핵심 5개 먼저 (~10h)
# Phase 2: 나머지 5개 (결과 보고 결정)
#

LOSS_CONFIGS: dict[str, dict] = OrderedDict({

    # ══════════════════════════════════════════════════════════════
    # Phase 1: 핵심 5개
    # ══════════════════════════════════════════════════════════════

    # ── Baseline ──
    "baseline_co": {
        "weights": {"complex_overlap": 1.0},
    },

    # ── Phase correction: CO + phasor (direct phase restoration) ──
    "co_phasor": {
        "weights": {"complex_overlap": 1.0, "phasor_mse": 1.0},
    },

    # ── Full-field phase (no ROI, works with sparse beam) ──
    "co_ffp": {
        "weights": {"complex_overlap": 1.0, "full_field_phase": 1.0},
    },

    # ── ROI reference (기존 방식, under-resolved에서 얼마나 나쁜지 확인) ──
    "roi80": {
        "roi": {"roi_threshold": 0.80, "leakage_weight": 1.0},
    },

    # ── Soft phasor: amplitude-weighted phase (밝은 곳 위주 보정) ──
    "co_soft_phasor_g2": {
        "weights": {"complex_overlap": 1.0, "soft_phasor": 1.0},
        "support_gamma": 2.0,
    },

    # ══════════════════════════════════════════════════════════════
    # Phase 2: 나머지 5개 (Phase 1 결과 후 실행)
    # ══════════════════════════════════════════════════════════════

    "baseline_co_amp": {
        "weights": {"complex_overlap": 1.0, "amplitude_mse": 0.5},
    },
    "phasor_only": {
        "weights": {"phasor_mse": 1.0},
    },
    "co_phasor_strong": {
        "weights": {"complex_overlap": 1.0, "phasor_mse": 3.0},
    },
    "co_ffp_strong": {
        "weights": {"complex_overlap": 1.0, "full_field_phase": 3.0},
    },
    "roi80_ph1": {
        "roi": {"roi_threshold": 0.80, "leakage_weight": 1.0},
        "phasor_weight": 1.0,
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


def make_model() -> BeamCleanupFD2NN:
    return BeamCleanupFD2NN(
        n=ROI_N,
        wavelength_m=WAVELENGTH_M,
        window_m=roi_window_m(),
        **ARCH,
    )


def phasor_smoothness_loss(model: nn.Module) -> torch.Tensor:
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for layer in model.layers:
        phase = layer.phase()
        phasor = torch.exp(1j * phase.to(torch.float32))
        grad_x = phasor[:, 1:] - phasor[:, :-1]
        grad_y = phasor[1:, :] - phasor[:-1, :]
        total = total + (grad_x.abs().pow(2).mean() + grad_y.abs().pow(2).mean()) / 2
    return total / len(model.layers)


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
    """Train one model with the given loss config. Return results dict."""
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
    psw = TRAIN["phasor_smoothness_weight"]

    is_roi = "roi" in loss_config
    weights = loss_config.get("weights", {})
    roi_params = loss_config.get("roi", {})
    phasor_w = loss_config.get("phasor_weight", 0.0)
    soft_phasor_w = loss_config.get("soft_phasor_weight", 0.0)
    support_gamma = loss_config.get("support_gamma", 2.0)

    # Describe the config
    if is_roi:
        parts = [f"ROI(th={roi_params['roi_threshold']}, lk={roi_params['leakage_weight']})"]
        ffp_w = roi_params.get("full_field_phase_weight", 0.0)
        if ffp_w > 0:
            parts.append(
                "ffp:"
                f"{ffp_w}/g{roi_params.get('full_field_phase_gamma', 1.0)}"
                f"/th{roi_params.get('full_field_phase_threshold', 0.05)}"
            )
        if phasor_w > 0:
            parts.append(f"ph:{phasor_w}")
        if soft_phasor_w > 0:
            parts.append(f"sph:{soft_phasor_w}/g{support_gamma}")
        desc = " + ".join(parts)
    else:
        active = {k: v for k, v in weights.items() if v > 0 and k != "support_gamma"}
        desc = " + ".join(f"{k}:{v}" for k, v in active.items())

    print(f"\n{'='*70}")
    print(f"  [{name}] {desc}")
    print(f"  + phasor_smoothness: {psw}")
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
                # + explicit residual phase penalty: drives φ_pred - φ_vac → 0
                if phasor_w > 0:
                    loss = loss + phasor_w * phasor_mse_loss(pred, target)
                if soft_phasor_w > 0:
                    loss = loss + soft_phasor_w * soft_weighted_phasor_loss(
                        pred, target, gamma=support_gamma,
                    )
            else:
                loss = complex_field_loss(pred, target, weights=weights, window_m=window)

            if psw > 0:
                loss = loss + psw * phasor_smoothness_loss(model)

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

    # Save wrapped phase
    phases = [layer.wrapped_phase().detach().cpu().numpy() for layer in model.layers]
    np.save(run_dir / "phases_wrapped.npy", np.stack(phases))

    # Save results
    result = {
        "name": name,
        "description": desc,
        "loss_config": loss_config,
        **test_m,
        "throughput": tp,
        "training_seconds": train_time,
        "peak_vram_mb": peak_vram,
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    torch.save({"model_state_dict": model.state_dict()}, run_dir / "checkpoint.pt")

    # Save sample fields for irradiance comparison (corrected vs vacuum)
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
            # Turbulent input (before correction)
            input_intensity=inp[0].abs().square().cpu().numpy(),
            input_phase=torch.angle(inp[0]).cpu().numpy(),
            # FD2NN corrected output
            pred_intensity=pred[0].abs().square().cpu().numpy(),
            pred_phase=torch.angle(pred[0]).cpu().numpy(),
            # Vacuum reference (ideal beam)
            target_intensity=target[0].abs().square().cpu().numpy(),
            target_phase=torch.angle(target[0]).cpu().numpy(),
            # Residual phase: pred_phase - vacuum_phase (should be ~0 if correction works)
            residual_phase=(torch.angle(pred[0]) - torch.angle(target[0])).cpu().numpy(),
            # Window info for physical coordinates
            window_m=roi_window_m(),
            n=ROI_N,
        )

    # Warnings
    if tp < 0.95 or tp > 1.05:
        print(f"  ⚠ WARN: throughput={tp:.4f} outside [0.95, 1.05]")
    if test_m["strehl"] > 1.05:
        print(f"  ⚠ WARN: strehl={test_m['strehl']:.4f} > 1.05")

    print(f"  TEST: co={test_m['complex_overlap']:.4f} | io={test_m['intensity_overlap']:.4f} | "
          f"sr={test_m['strehl']:.4f} | tp={tp:.4f} | {train_time:.0f}s")

    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Sweeping {len(LOSS_CONFIGS)} loss configurations")
    print(f"Fixed: layers={ARCH['num_layers']}, spacing={ARCH['layer_spacing_m']*1e3:.0f}mm, "
          f"constraint={ARCH['phase_constraint']}, lr={TRAIN['lr']}, epochs={TRAIN['epochs']}")

    # Load data once
    train_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="train")
    val_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="val")
    test_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="test")
    train_loader = DataLoader(train_ds, batch_size=TRAIN["batch_size"], shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=TRAIN["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=TRAIN["batch_size"])

    # ── Zero-phase throughput diagnostic ─────────────────────────
    print("\n--- Throughput diagnostic (zero-phase model) ---")
    diag_model = make_model().to(device)
    for layer in diag_model.layers:
        layer.raw.data.zero_()
    tp_zero = throughput_check(diag_model, test_loader, device)
    print(f"  Zero-phase throughput: {tp_zero:.4f}")
    if tp_zero < 0.95:
        print(f"  ⚠ Optical path loses {(1-tp_zero)*100:.1f}% energy even without phase masks!")
        print(f"    Possible causes: NA clipping, apply_scaling, inter-layer diffraction")
    else:
        print(f"  ✓ Optical path conserves energy (within 5%)")
    del diag_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Run all configs
    all_results = []
    for name, config in LOSS_CONFIGS.items():
        result = train_one(name, config, train_loader, val_loader, test_loader, device)
        all_results.append(result)

    # ── Summary table ─────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("LOSS SWEEP SUMMARY")
    print(f"{'='*100}")

    # Sort by complex_overlap descending
    ranked = sorted(all_results, key=lambda r: r["complex_overlap"], reverse=True)

    header = f"{'Rank':>4} | {'Name':>22} | {'CO':>7} | {'IO':>7} | {'Strehl':>7} | {'FullPr':>7} | {'EE':>7} | {'TP':>6} | {'Time':>5}"
    print(header)
    print("-" * len(header))

    for i, r in enumerate(ranked):
        flag = " *" if r["throughput"] < 0.95 or r["throughput"] > 1.05 else ""
        print(f"{i+1:4d} | {r['name']:>22} | {r['complex_overlap']:7.4f} | "
              f"{r['intensity_overlap']:7.4f} | {r['strehl']:7.4f} | "
              f"{r['full_field_phase_rmse_rad']:7.4f} | {r['encircled_energy']:7.4f} | "
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
