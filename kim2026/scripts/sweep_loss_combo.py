#!/usr/bin/env python
"""Loss combination sweep: 4 hybrid loss strategies, fixed tanh_2pi phase range.

Uses best config from phase range sweep: tanh_2pi, spacing=1mm, dx=2um.
"""

from __future__ import annotations

import json
import math
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
    amplitude_rmse, complex_overlap, phase_rmse, full_field_phase_rmse,
    gaussian_overlap, strehl_ratio,
)
from kim2026.training.losses import beam_radius, encircled_energy_fraction
from kim2026.training.targets import apply_receiver_aperture

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kim2026" / "1km_cn2e-14_w2m_n1024_dx2mm" / "cache"
MANIFEST = DATA_DIR.parent / "split_manifest.json"
OUT_ROOT = Path(__file__).resolve().parent.parent / "runs" / "fd2nn_loss_combo_sweep"

LOSS_COMBOS = {
    "combo1_io_co": {
        "label": "io:1 + co:0.5",
        "weights": {"intensity_overlap": 1.0, "complex_overlap": 0.5},
    },
    "combo2_io_br_ee": {
        "label": "io:1 + br:0.5 + ee:0.5",
        "weights": {"intensity_overlap": 1.0, "beam_radius": 0.5, "encircled_energy": 0.5},
    },
    "combo3_co_io_br": {
        "label": "co:1 + io:0.5 + br:0.5",
        "weights": {"complex_overlap": 1.0, "intensity_overlap": 0.5, "beam_radius": 0.5},
    },
    "combo4_sp_leak_io": {
        "label": "sp:1 + leak:0.5 + io:0.5",
        "weights": {"soft_phasor": 1.0, "leakage": 0.5, "intensity_overlap": 0.5},
    },
}

COMMON = dict(
    wavelength_m=1.55e-6,
    receiver_window_m=0.002048,
    aperture_diameter_m=0.002,
    n=1024, num_layers=5,
    layer_spacing_m=1e-3,
    phase_max=2 * math.pi,  # tanh_2pi (best)
    phase_constraint="symmetric_tanh",
    dual_2f_f1_m=1e-3, dual_2f_f2_m=1e-3,
    dual_2f_na1=0.16, dual_2f_na2=0.16,
    dual_2f_apply_scaling=False,
    lr=5e-4, epochs=30, batch_size=2, seed=20260323,
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


def train_one(name, loss_cfg, device):
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
    n = train_ds[0]["u_turb"].shape[-1]

    model = BeamCleanupFD2NN(
        n=n, wavelength_m=COMMON["wavelength_m"], window_m=COMMON["receiver_window_m"],
        num_layers=COMMON["num_layers"], layer_spacing_m=COMMON["layer_spacing_m"],
        phase_max=COMMON["phase_max"], phase_constraint=COMMON["phase_constraint"],
        dual_2f_f1_m=COMMON["dual_2f_f1_m"], dual_2f_f2_m=COMMON["dual_2f_f2_m"],
        dual_2f_na1=COMMON["dual_2f_na1"], dual_2f_na2=COMMON["dual_2f_na2"],
        dual_2f_apply_scaling=COMMON["dual_2f_apply_scaling"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=COMMON["lr"])

    train_loader = DataLoader(train_ds, batch_size=COMMON["batch_size"], shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=COMMON["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=COMMON["batch_size"])

    w, ap = COMMON["receiver_window_m"], COMMON["aperture_diameter_m"]
    weights = loss_cfg["weights"]

    print(f"\n{'='*60}")
    print(f"  {name}: {loss_cfg['label']}")
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
        avg = eloss / len(train_loader)
        dt = time.time() - t0

        if epoch % 10 == 0 or epoch == COMMON["epochs"] - 1:
            vm = evaluate(model, val_loader, device)
            entry = {"epoch": epoch, "train_loss": avg, "time_s": dt, **vm}
            print(f"    ep {epoch:2d} | loss={avg:.5f} | co={vm['complex_overlap']:.4f} "
                  f"pr={vm['phase_rmse_rad']:.4f} io={vm['intensity_overlap']:.4f} | {dt:.1f}s")
        else:
            entry = {"epoch": epoch, "train_loss": avg, "time_s": dt}
        history.append(entry)

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
        tgt = apply_receiver_aperture(u_v, receiver_window_m=w, aperture_diameter_m=ap)
        inp = apply_receiver_aperture(u_t, receiver_window_m=w, aperture_diameter_m=ap)
        pred = model(inp)
        np.savez(run_dir / "sample_fields.npz",
                 input_real=inp[0].real.cpu().numpy(), input_imag=inp[0].imag.cpu().numpy(),
                 pred_real=pred[0].real.cpu().numpy(), pred_imag=pred[0].imag.cpu().numpy(),
                 target_real=tgt[0].real.cpu().numpy(), target_imag=tgt[0].imag.cpu().numpy())
    return test


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loss Combo Sweep (tanh_2pi fixed, spacing=1mm)")

    results = {}
    for name, cfg in LOSS_COMBOS.items():
        results[name] = train_one(name, cfg, device)

    print(f"\n{'='*80}")
    print("LOSS COMBO SWEEP SUMMARY")
    print(f"{'='*80}")
    print(f"{'Name':>20} | {'Loss combo':>25} | {'co':>7} | {'pr':>7} | {'io':>7} | {'sr':>7}")
    print("-" * 85)
    for name, cfg in LOSS_COMBOS.items():
        r = results[name]
        print(f"{name:>20} | {cfg['label']:>25} | {r['complex_overlap']:>7.4f} | "
              f"{r['phase_rmse_rad']:>7.4f} | {r['intensity_overlap']:>7.4f} | {r['strehl']:>7.4f}")
    bl = results[list(results.keys())[0]]
    print(f"\n  Baseline: co={bl['baseline_co']:.4f}, io={bl['baseline_io']:.4f}")

    with open(OUT_ROOT / "loss_combo_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_ROOT}")


if __name__ == "__main__":
    main()
