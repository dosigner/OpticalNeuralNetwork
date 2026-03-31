#!/usr/bin/env python
"""Sweep 06 + 07: Loss weight ablation & curriculum learning.

Sweep 06 — remove amplitude_mse from complex loss, test alternatives.
Sweep 07 — curriculum: irradiance first, then complex overlap.

All configs use tanh_2pi, spacing=1mm, dx=2um (best from prior sweeps).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from kim2026.data.dataset import CachedFieldDataset
from kim2026.models.fd2nn import BeamCleanupFD2NN
from kim2026.training.losses import complex_field_loss, beam_cleanup_loss
from kim2026.training.metrics import (
    amplitude_rmse, complex_overlap, phase_rmse,
    gaussian_overlap, strehl_ratio,
)
from kim2026.training.targets import apply_receiver_aperture

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kim2026" / "1km_cn2e-14_w2m_n1024_dx2mm" / "cache"
MANIFEST = DATA_DIR.parent / "split_manifest.json"
SWEEP06_DIR = Path(__file__).resolve().parent.parent / "runs" / "06_fd2nn_loss_ablation_roi1024_claude"
SWEEP07_DIR = Path(__file__).resolve().parent.parent / "runs" / "07_fd2nn_curriculum_roi1024_claude"

COMMON = dict(
    wavelength_m=1.55e-6,
    receiver_window_m=0.002048,
    aperture_diameter_m=0.002,
    n=1024, num_layers=5,
    layer_spacing_m=1e-3,
    phase_max=2 * math.pi,
    phase_constraint="symmetric_tanh",
    dual_2f_f1_m=1e-3, dual_2f_f2_m=1e-3,
    dual_2f_na1=0.16, dual_2f_na2=0.16,
    dual_2f_apply_scaling=False,
    lr=5e-4, epochs=30, batch_size=2, seed=20260323,
)

# ── Sweep 06: loss weight ablation ──

ABLATION_CONFIGS = {
    "co_only": {
        "label": "co:1 (no amp)",
        "weights": {"complex_overlap": 1.0},
    },
    "co_amp01": {
        "label": "co:1 + amp:0.1",
        "weights": {"complex_overlap": 1.0, "amplitude_mse": 0.1},
    },
    "co_io": {
        "label": "co:1 + io:0.5",
        "weights": {"complex_overlap": 1.0, "intensity_overlap": 0.5},
    },
    "co_io_br": {
        "label": "co:1 + io:0.25 + br:0.25",
        "weights": {"complex_overlap": 1.0, "intensity_overlap": 0.25, "beam_radius": 0.25},
    },
}

# ── Sweep 07: curriculum configs ──

CURRICULUM_CONFIGS = {
    "cur_10_20": {
        "label": "irr 0-10 → co 10-30",
        "switch_epoch": 10,
        "blend": False,
    },
    "cur_15_15": {
        "label": "irr 0-15 → co 15-30",
        "switch_epoch": 15,
        "blend": False,
    },
    "cur_20_10": {
        "label": "irr 0-20 → co 20-30",
        "switch_epoch": 20,
        "blend": False,
    },
    "cur_blend": {
        "label": "linear blend irr→co over 30ep",
        "switch_epoch": None,
        "blend": True,
    },
}


def make_model(device):
    return BeamCleanupFD2NN(
        n=COMMON["n"], wavelength_m=COMMON["wavelength_m"],
        window_m=COMMON["receiver_window_m"],
        num_layers=COMMON["num_layers"], layer_spacing_m=COMMON["layer_spacing_m"],
        phase_max=COMMON["phase_max"], phase_constraint=COMMON["phase_constraint"],
        dual_2f_f1_m=COMMON["dual_2f_f1_m"], dual_2f_f2_m=COMMON["dual_2f_f2_m"],
        dual_2f_na1=COMMON["dual_2f_na1"], dual_2f_na2=COMMON["dual_2f_na2"],
        dual_2f_apply_scaling=COMMON["dual_2f_apply_scaling"],
    ).to(device)


def make_loaders():
    train_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="train")
    val_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="val")
    test_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="test")
    kw = dict(batch_size=COMMON["batch_size"], shuffle=False)
    return (DataLoader(train_ds, **kw), DataLoader(val_ds, **kw),
            DataLoader(test_ds, **kw), test_ds)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    w, ap = COMMON["receiver_window_m"], COMMON["aperture_diameter_m"]
    all_co, all_pr, all_ar, all_io, all_sr = [], [], [], [], []
    all_co_bl, all_io_bl = [], []
    for batch in loader:
        u_t = batch["u_turb"].to(device)
        u_v = batch["u_vacuum"].to(device)
        tgt = apply_receiver_aperture(u_v, receiver_window_m=w, aperture_diameter_m=ap)
        inp = apply_receiver_aperture(u_t, receiver_window_m=w, aperture_diameter_m=ap)
        pred = model(inp)
        all_co.append(complex_overlap(pred, tgt).cpu())
        all_pr.append(phase_rmse(pred, tgt).cpu())
        all_ar.append(amplitude_rmse(pred, tgt).cpu())
        pred_i, tgt_i = pred.abs().square(), tgt.abs().square()
        all_io.append(gaussian_overlap(pred_i, tgt_i).cpu())
        all_sr.append(strehl_ratio(pred_i, tgt_i).cpu())
        all_co_bl.append(complex_overlap(inp, tgt).cpu())
        all_io_bl.append(gaussian_overlap(inp.abs().square(), tgt_i).cpu())
    return {
        "complex_overlap": float(torch.cat(all_co).mean()),
        "phase_rmse_rad": float(torch.cat(all_pr).mean()),
        "amplitude_rmse": float(torch.cat(all_ar).mean()),
        "intensity_overlap": float(torch.cat(all_io).mean()),
        "strehl": float(torch.cat(all_sr).mean()),
        "baseline_co": float(torch.cat(all_co_bl).mean()),
        "baseline_io": float(torch.cat(all_io_bl).mean()),
    }


def save_results(run_dir, model, history, test, test_ds, device):
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "history": history,
                "test_metrics": test}, run_dir / "checkpoint.pt")
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(test, f, indent=2)

    w, ap = COMMON["receiver_window_m"], COMMON["aperture_diameter_m"]
    model.eval()
    with torch.no_grad():
        sample = test_ds[0]
        u_t = sample["u_turb"].unsqueeze(0).to(device)
        u_v = sample["u_vacuum"].unsqueeze(0).to(device)
        tgt = apply_receiver_aperture(u_v, receiver_window_m=w, aperture_diameter_m=ap)
        inp = apply_receiver_aperture(u_t, receiver_window_m=w, aperture_diameter_m=ap)
        pred = model(inp)
        np.savez(run_dir / "sample_fields.npz",
                 input_real=inp[0].real.cpu().numpy(), input_imag=inp[0].imag.cpu().numpy(),
                 pred_real=pred[0].real.cpu().numpy(), pred_imag=pred[0].imag.cpu().numpy(),
                 target_real=tgt[0].real.cpu().numpy(), target_imag=tgt[0].imag.cpu().numpy())


def collate(batch):
    return {
        "u_vacuum": torch.stack([b["u_vacuum"] for b in batch]),
        "u_turb": torch.stack([b["u_turb"] for b in batch]),
        "metadata": [b["metadata"] for b in batch],
    }


# ── Sweep 06: ablation ──

def run_ablation(name, cfg, device, loaders):
    train_loader, val_loader, test_loader, test_ds = loaders
    run_dir = SWEEP06_DIR / name
    if (run_dir / "test_metrics.json").exists():
        print(f"  {name}: already done, skipping")
        with open(run_dir / "test_metrics.json") as f:
            return json.load(f)

    torch.manual_seed(COMMON["seed"])
    np.random.seed(COMMON["seed"])

    model = make_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=COMMON["lr"])
    w, ap = COMMON["receiver_window_m"], COMMON["aperture_diameter_m"]
    weights = cfg["weights"]

    print(f"\n{'='*60}")
    print(f"  [06] {name}: {cfg['label']}")
    print(f"  Weights: {weights}")
    print(f"{'='*60}")

    history = []
    for epoch in range(COMMON["epochs"]):
        model.train()
        eloss = 0.0
        t0 = time.time()
        for batch in train_loader:
            u_t = batch["u_turb"].to(device)
            u_v = batch["u_vacuum"].to(device)
            tgt = apply_receiver_aperture(u_v, receiver_window_m=w, aperture_diameter_m=ap)
            inp = apply_receiver_aperture(u_t, receiver_window_m=w, aperture_diameter_m=ap)
            optimizer.zero_grad(set_to_none=True)
            pred = model(inp)
            loss = complex_field_loss(pred, tgt, weights=weights, window_m=w)
            loss.backward()
            optimizer.step()
            eloss += loss.item()
        avg = eloss / max(len(train_loader), 1)
        dt = time.time() - t0

        if epoch % 5 == 0 or epoch == COMMON["epochs"] - 1:
            vm = evaluate(model, val_loader, device)
            entry = {"epoch": epoch, "train_loss": avg, "time_s": dt, **vm}
            print(f"    ep {epoch:2d} | loss={avg:.5f} | co={vm['complex_overlap']:.4f} "
                  f"pr={vm['phase_rmse_rad']:.4f} io={vm['intensity_overlap']:.4f} | {dt:.1f}s")
        else:
            entry = {"epoch": epoch, "train_loss": avg, "time_s": dt}
        history.append(entry)

    test = evaluate(model, test_loader, device)
    print(f"  TEST: co={test['complex_overlap']:.4f} pr={test['phase_rmse_rad']:.4f} "
          f"io={test['intensity_overlap']:.4f} amp={test['amplitude_rmse']:.4f}")
    save_results(run_dir, model, history, test, test_ds, device)
    return test


# ── Sweep 07: curriculum ──

def run_curriculum(name, cfg, device, loaders):
    train_loader, val_loader, test_loader, test_ds = loaders
    run_dir = SWEEP07_DIR / name
    if (run_dir / "test_metrics.json").exists():
        print(f"  {name}: already done, skipping")
        with open(run_dir / "test_metrics.json") as f:
            return json.load(f)

    torch.manual_seed(COMMON["seed"])
    np.random.seed(COMMON["seed"])

    model = make_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=COMMON["lr"])
    w, ap = COMMON["receiver_window_m"], COMMON["aperture_diameter_m"]

    # Irradiance loss weights (same as sweep 04)
    irr_weights = {"overlap": 1.0, "radius": 0.25, "encircled": 0.25}
    # Complex loss weights (co only, no amp_mse — informed by strategy 1)
    co_weights = {"complex_overlap": 1.0}

    switch_epoch = cfg["switch_epoch"]
    blend = cfg["blend"]
    total_epochs = COMMON["epochs"]

    print(f"\n{'='*60}")
    print(f"  [07] {name}: {cfg['label']}")
    print(f"{'='*60}")

    history = []
    for epoch in range(total_epochs):
        model.train()
        eloss = 0.0
        t0 = time.time()

        # Determine loss mode for this epoch
        if blend:
            # Linear blend: α goes from 1 (pure irradiance) to 0 (pure complex)
            alpha = 1.0 - epoch / max(total_epochs - 1, 1)
        else:
            alpha = 1.0 if epoch < switch_epoch else 0.0

        for batch in train_loader:
            u_t = batch["u_turb"].to(device)
            u_v = batch["u_vacuum"].to(device)
            tgt = apply_receiver_aperture(u_v, receiver_window_m=w, aperture_diameter_m=ap)
            inp = apply_receiver_aperture(u_t, receiver_window_m=w, aperture_diameter_m=ap)
            optimizer.zero_grad(set_to_none=True)
            pred = model(inp)

            if alpha == 1.0:
                # Pure irradiance loss
                pred_i = pred.abs().square()
                loss = beam_cleanup_loss(pred_i, tgt.abs().square(), window_m=w, weights=irr_weights)
            elif alpha == 0.0:
                # Pure complex loss
                loss = complex_field_loss(pred, tgt, weights=co_weights, window_m=w)
            else:
                # Blended loss
                pred_i = pred.abs().square()
                loss_irr = beam_cleanup_loss(pred_i, tgt.abs().square(), window_m=w, weights=irr_weights)
                loss_co = complex_field_loss(pred, tgt, weights=co_weights, window_m=w)
                loss = alpha * loss_irr + (1.0 - alpha) * loss_co

            loss.backward()
            optimizer.step()
            eloss += loss.item()
        avg = eloss / max(len(train_loader), 1)
        dt = time.time() - t0

        if epoch % 5 == 0 or epoch == total_epochs - 1:
            vm = evaluate(model, val_loader, device)
            mode = f"α={alpha:.2f}" if blend else ("IRR" if alpha == 1.0 else "CO")
            entry = {"epoch": epoch, "train_loss": avg, "time_s": dt, "alpha": alpha, **vm}
            print(f"    ep {epoch:2d} [{mode:>8}] | loss={avg:.5f} | co={vm['complex_overlap']:.4f} "
                  f"pr={vm['phase_rmse_rad']:.4f} io={vm['intensity_overlap']:.4f} | {dt:.1f}s")
        else:
            entry = {"epoch": epoch, "train_loss": avg, "time_s": dt, "alpha": alpha}
        history.append(entry)

    test = evaluate(model, test_loader, device)
    print(f"  TEST: co={test['complex_overlap']:.4f} pr={test['phase_rmse_rad']:.4f} "
          f"io={test['intensity_overlap']:.4f} amp={test['amplitude_rmse']:.4f}")
    save_results(run_dir, model, history, test, test_ds, device)
    return test


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Override collate for DataLoader
    train_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="train")
    val_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="val")
    test_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="test")
    kw = dict(batch_size=COMMON["batch_size"], shuffle=False, collate_fn=collate)
    loaders = (DataLoader(train_ds, **kw), DataLoader(val_ds, **kw),
               DataLoader(test_ds, **kw), test_ds)

    # ── Sweep 06 ──
    print("\n" + "█" * 60)
    print("  SWEEP 06: Loss Weight Ablation")
    print("█" * 60)

    results_06 = {}
    for name, cfg in ABLATION_CONFIGS.items():
        results_06[name] = run_ablation(name, cfg, device, loaders)

    SWEEP06_DIR.mkdir(parents=True, exist_ok=True)
    with open(SWEEP06_DIR / "ablation_summary.json", "w") as f:
        json.dump(results_06, f, indent=2)

    # ── Sweep 07 ──
    print("\n" + "█" * 60)
    print("  SWEEP 07: Curriculum Learning")
    print("█" * 60)

    results_07 = {}
    for name, cfg in CURRICULUM_CONFIGS.items():
        results_07[name] = run_curriculum(name, cfg, device, loaders)

    SWEEP07_DIR.mkdir(parents=True, exist_ok=True)
    with open(SWEEP07_DIR / "curriculum_summary.json", "w") as f:
        json.dump(results_07, f, indent=2)

    # ── Summary ──
    print("\n" + "=" * 90)
    print("COMBINED SUMMARY — Sweeps 06 + 07")
    print("=" * 90)

    # Reference baselines from prior sweeps
    print(f"\n{'Strategy':<25} | {'co':>7} | {'pr':>7} | {'io':>7} | {'amp':>7} | Source")
    print("-" * 80)
    print(f"{'Turbulent (no D2NN)':<25} | {'0.191':>7} | {'—':>7} | {'0.973':>7} | {'~0.02':>7} | baseline")
    print(f"{'Complex (co+amp0.5)':<25} | {'0.270':>7} | {'0.359':>7} | {'0.378':>7} | {'0.176':>7} | sweep 02")
    print(f"{'Irradiance (io+br+ee)':<25} | {'0.099':>7} | {'1.679':>7} | {'0.933':>7} | {'0.159':>7} | sweep 04")
    print("-" * 80)

    for name, r in results_06.items():
        label = ABLATION_CONFIGS[name]["label"]
        print(f"{label:<25} | {r['complex_overlap']:>7.4f} | {r['phase_rmse_rad']:>7.4f} | "
              f"{r['intensity_overlap']:>7.4f} | {r['amplitude_rmse']:>7.4f} | sweep 06")

    print("-" * 80)
    for name, r in results_07.items():
        label = CURRICULUM_CONFIGS[name]["label"]
        print(f"{label:<25} | {r['complex_overlap']:>7.4f} | {r['phase_rmse_rad']:>7.4f} | "
              f"{r['intensity_overlap']:>7.4f} | {r['amplitude_rmse']:>7.4f} | sweep 07")

    print("\nDone.")


if __name__ == "__main__":
    main()
