#!/usr/bin/env python
"""Loss function sweep with f=100mm.

Compares 4 loss strategies at fixed spacing=600mm (z/z_R≈0.5).
dx_fourier=75.7µm (49λ) — SLM-scale pixel pitch.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from kim2026.data.dataset import CachedFieldDataset
from kim2026.models.fd2nn import BeamCleanupFD2NN
from kim2026.training.losses import complex_field_loss
from kim2026.training.losses import beam_radius, encircled_energy_fraction
from kim2026.training.metrics import (
    amplitude_rmse, complex_overlap, full_field_phase_rmse,
    gaussian_overlap, phase_rmse, strehl_ratio,
)
from kim2026.training.targets import apply_receiver_aperture

PROJ = Path(__file__).resolve().parent.parent
DATA_DIR = PROJ / "data" / "kim2026" / "1km_cn2e-14_w2m_n1024_dx2mm" / "cache"
MANIFEST = DATA_DIR.parent / "split_manifest.json"
OUT_ROOT = PROJ / "runs" / "06_fd2nn_loss_sweep_f10mm_sp50mm_claude"

# 4 loss strategies
LOSS_CONFIGS = {
    "complex": {
        "label": "Complex (CO+IO+BR+EE)",
        "weights": {
            "complex_overlap": 1.0,
            "intensity_overlap": 1.0, "beam_radius": 1.0, "encircled_energy": 1.0,
        },
    },
    "phasor": {
        "label": "Phasor (phase-only)",
        "weights": {
            "soft_phasor": 1.0, "leakage": 0.5,
        },
    },
    "irradiance": {
        "label": "Irradiance (IO+BR+EE)",
        "weights": {
            "intensity_overlap": 1.0, "beam_radius": 1.0, "encircled_energy": 1.0,
        },
    },
    "hybrid": {
        "label": "Hybrid (CO+IO+BR)",
        "weights": {
            "complex_overlap": 1.0, "intensity_overlap": 0.5, "beam_radius": 0.5,
        },
    },
}

COMMON = dict(
    wavelength_m=1.55e-6,
    receiver_window_m=0.002048,
    aperture_diameter_m=0.002,
    n=1024,
    num_layers=5,
    layer_spacing_m=50e-3,             # 50mm → z/z_R≈4.3 (f=10mm)
    dual_2f_f1_m=10.0e-3,             # 10 mm
    dual_2f_f2_m=10.0e-3,
    dual_2f_na1=0.16,
    dual_2f_na2=0.16,
    dual_2f_apply_scaling=False,
    phase_max=2 * math.pi,
    phase_constraint="symmetric_tanh",
    phase_init="uniform",
    phase_init_scale=0.1,
    lr=5e-4,
    epochs=100,
    batch_size=2,
    seed=20260323,
)

PHASE_SNAPSHOT_EPOCHS = {0, 15, 29}


def prepare_field(field, *, aperture_diameter_m):
    return apply_receiver_aperture(
        field, receiver_window_m=COMMON["receiver_window_m"],
        aperture_diameter_m=aperture_diameter_m,
    )


def make_model():
    return BeamCleanupFD2NN(
        n=COMMON["n"], wavelength_m=COMMON["wavelength_m"],
        window_m=COMMON["receiver_window_m"], num_layers=COMMON["num_layers"],
        layer_spacing_m=COMMON["layer_spacing_m"],
        phase_max=COMMON["phase_max"], phase_constraint=COMMON["phase_constraint"],
        phase_init=COMMON["phase_init"], phase_init_scale=COMMON["phase_init_scale"],
        dual_2f_f1_m=COMMON["dual_2f_f1_m"], dual_2f_f2_m=COMMON["dual_2f_f2_m"],
        dual_2f_na1=COMMON["dual_2f_na1"], dual_2f_na2=COMMON["dual_2f_na2"],
        dual_2f_apply_scaling=COMMON["dual_2f_apply_scaling"],
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    w, ap = COMMON["receiver_window_m"], COMMON["aperture_diameter_m"]
    all_co, all_pr, all_ar, all_io, all_sr = [], [], [], [], []
    all_co_bl, all_io_bl = [], []
    for batch in loader:
        u_t = batch["u_turb"].to(device)
        u_v = batch["u_vacuum"].to(device)
        tgt = prepare_field(u_v, aperture_diameter_m=ap)
        inp = prepare_field(u_t, aperture_diameter_m=ap)
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
        "baseline_complex_overlap": float(torch.cat(all_co_bl).mean()),
        "baseline_intensity_overlap": float(torch.cat(all_io_bl).mean()),
    }


def train_one(name, loss_weights, device):
    run_dir = OUT_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)

    if (run_dir / "test_metrics.json").exists():
        print(f"  {name}: already done, skipping")
        with open(run_dir / "test_metrics.json") as f:
            return json.load(f)

    torch.manual_seed(COMMON["seed"])
    np.random.seed(COMMON["seed"])

    train_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="train")
    val_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="val")
    test_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="test")

    model = make_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=COMMON["lr"])

    train_loader = DataLoader(train_ds, batch_size=COMMON["batch_size"], shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=COMMON["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=COMMON["batch_size"])

    w, ap = COMMON["receiver_window_m"], COMMON["aperture_diameter_m"]

    print(f"\n{'='*60}")
    print(f"  {name}: {LOSS_CONFIGS[name]['label']}")
    print(f"  weights: {loss_weights}")
    print(f"{'='*60}")

    history = []
    for epoch in range(COMMON["epochs"]):
        model.train()
        eloss = 0.0
        t0 = time.time()
        for batch in train_loader:
            u_t = batch["u_turb"].to(device)
            u_v = batch["u_vacuum"].to(device)
            tgt = prepare_field(u_v, aperture_diameter_m=ap)
            inp = prepare_field(u_t, aperture_diameter_m=ap)
            optimizer.zero_grad(set_to_none=True)
            pred = model(inp)
            loss = complex_field_loss(pred, tgt, weights=loss_weights, window_m=w)
            loss.backward()
            optimizer.step()
            eloss += loss.item()
        avg = eloss / len(train_loader)
        dt = time.time() - t0

        if epoch % 5 == 0 or epoch == COMMON["epochs"] - 1:
            vm = evaluate(model, val_loader, device)
            entry = {"epoch": epoch, "train_loss": avg, "time_s": dt, **vm}
        else:
            entry = {"epoch": epoch, "train_loss": avg, "time_s": dt}

        if epoch in PHASE_SNAPSHOT_EPOCHS:
            phases = [layer.phase().detach().cpu().numpy() for layer in model.layers]
            np.save(run_dir / f"phases_epoch{epoch:03d}.npy", np.stack(phases))

        history.append(entry)
        co = entry.get("complex_overlap", "")
        io = entry.get("intensity_overlap", "")
        print(f"  ep {epoch:2d} | loss={avg:.5f} | co={f'{co:.4f}' if co else '---':>6} | "
              f"io={f'{io:.4f}' if io else '---':>6} | {dt:.1f}s")

    test = evaluate(model, test_loader, device)
    print(f"\n  TEST: co={test['complex_overlap']:.4f} pr={test['phase_rmse_rad']:.4f} "
          f"io={test['intensity_overlap']:.4f} sr={test['strehl']:.4f}")

    torch.save({"model_state_dict": model.state_dict(), "history": history,
                "test_metrics": test}, run_dir / "checkpoint.pt")
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(test, f, indent=2)

    # Save sample fields
    model.eval()
    with torch.no_grad():
        sample = test_ds[0]
        u_t = sample["u_turb"].unsqueeze(0).to(device)
        u_v = sample["u_vacuum"].unsqueeze(0).to(device)
        tgt = prepare_field(u_v, aperture_diameter_m=ap)
        inp = prepare_field(u_t, aperture_diameter_m=ap)
        pred = model(inp)
        np.savez(run_dir / "sample_fields.npz",
                 input_real=inp[0].real.cpu().numpy(), input_imag=inp[0].imag.cpu().numpy(),
                 pred_real=pred[0].real.cpu().numpy(), pred_imag=pred[0].imag.cpu().numpy(),
                 target_real=tgt[0].real.cpu().numpy(), target_imag=tgt[0].imag.cpu().numpy())
    return test


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dx = COMMON["receiver_window_m"] / COMMON["n"]
    dx_f = COMMON["wavelength_m"] * COMMON["dual_2f_f1_m"] / (COMMON["n"] * dx)
    z_R = math.pi * (10 * dx_f)**2 / COMMON["wavelength_m"]
    z_ratio = COMMON["layer_spacing_m"] / z_R

    print(f"\nSystem: f={COMMON['dual_2f_f1_m']*1e3:.0f}mm, "
          f"dx_f={dx_f*1e6:.1f}µm ({dx_f/COMMON['wavelength_m']:.0f}λ)")
    print(f"Spacing: {COMMON['layer_spacing_m']*1e3:.0f}mm (z/z_R={z_ratio:.3f})")
    print(f"z_R(10px) = {z_R:.2f} m")

    results = {}
    for name, cfg in LOSS_CONFIGS.items():
        results[name] = train_one(name, cfg["weights"], device)

    print(f"\n{'='*80}")
    print("LOSS SWEEP SUMMARY (f=100mm)")
    print(f"{'='*80}")
    print(f"{'Name':>12} | {'Loss Type':>25} | {'CO':>7} | {'IO':>7} | {'PhRMSE':>7} | {'Strehl':>7}")
    print("-" * 80)
    for name, cfg in LOSS_CONFIGS.items():
        r = results[name]
        print(f"{name:>12} | {cfg['label']:>25} | {r['complex_overlap']:>7.4f} | "
              f"{r['intensity_overlap']:>7.4f} | {r['phase_rmse_rad']:>7.4f} | {r['strehl']:>7.4f}")

    with open(OUT_ROOT / "loss_sweep_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_ROOT}")


if __name__ == "__main__":
    main()
