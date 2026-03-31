#!/usr/bin/env python
"""Phase range sweep: which phase_max is best for FD2NN metalens?

Fixed: spacing=1mm, dx=2um, 5 layers, 30 epochs
Sweep: phase_max × constraint combinations
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
from kim2026.training.metrics import complex_overlap, phase_rmse, amplitude_rmse, gaussian_overlap, strehl_ratio
from kim2026.training.targets import apply_receiver_aperture

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kim2026" / "1km_cn2e-14_w2m_n1024_dx2mm" / "cache"
MANIFEST = DATA_DIR.parent / "split_manifest.json"
OUT_ROOT = Path(__file__).resolve().parent.parent / "runs" / "fd2nn_phase_range_sweep"

# Phase range configurations
# symmetric_tanh: range = [-phase_max, +phase_max], total = 2*phase_max
# sigmoid: range = [0, phase_max], total = phase_max
CONFIGS = {
    "tanh_pi2": {"constraint": "symmetric_tanh", "phase_max": math.pi / 2, "label": "[-pi/2,pi/2] (pi total)"},
    "tanh_pi":  {"constraint": "symmetric_tanh", "phase_max": math.pi,     "label": "[-pi,pi] (2pi total)"},
    "tanh_2pi": {"constraint": "symmetric_tanh", "phase_max": 2 * math.pi, "label": "[-2pi,2pi] (4pi total)"},
    "sig_pi":   {"constraint": "sigmoid",        "phase_max": math.pi,     "label": "[0,pi] (pi total)"},
    "sig_2pi":  {"constraint": "sigmoid",        "phase_max": 2 * math.pi, "label": "[0,2pi] (2pi total)"},
    "sig_4pi":  {"constraint": "sigmoid",        "phase_max": 4 * math.pi, "label": "[0,4pi] (4pi total)"},
}

COMMON = dict(
    wavelength_m=1.55e-6,
    receiver_window_m=0.002048,
    aperture_diameter_m=0.002,
    n=1024, num_layers=5,
    layer_spacing_m=1e-3,  # best from previous sweep
    dual_2f_f1_m=1e-3,
    dual_2f_f2_m=1e-3,
    dual_2f_na1=0.16,
    dual_2f_na2=0.16,
    dual_2f_apply_scaling=False,
    lr=5e-4, epochs=30, batch_size=2, seed=20260323,
    complex_weights={
        "intensity_overlap": 1.0,
        "beam_radius": 1.0,
        "encircled_energy": 1.0,
    },
)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    w, ap = COMMON["receiver_window_m"], COMMON["aperture_diameter_m"]
    all_co, all_pr, all_ar, all_io = [], [], [], []
    all_co_bl = []
    for batch in loader:
        u_t = batch["u_turb"].to(device)
        u_v = batch["u_vacuum"].to(device)
        tgt = apply_receiver_aperture(u_v, receiver_window_m=w, aperture_diameter_m=ap)
        inp = apply_receiver_aperture(u_t, receiver_window_m=w, aperture_diameter_m=ap)
        pred = model(inp)
        all_co.append(complex_overlap(pred, tgt).cpu())
        all_pr.append(phase_rmse(pred, tgt).cpu())
        all_ar.append(amplitude_rmse(pred, tgt).cpu())
        all_io.append(gaussian_overlap(pred.abs().square(), tgt.abs().square()).cpu())
        all_co_bl.append(complex_overlap(inp, tgt).cpu())
    return {
        "complex_overlap": float(torch.cat(all_co).mean()),
        "phase_rmse_rad": float(torch.cat(all_pr).mean()),
        "amplitude_rmse": float(torch.cat(all_ar).mean()),
        "intensity_overlap": float(torch.cat(all_io).mean()),
        "baseline_co": float(torch.cat(all_co_bl).mean()),
    }


def train_one(name, cfg, device):
    run_dir = OUT_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already done
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
        num_layers=COMMON["num_layers"],
        layer_spacing_m=COMMON["layer_spacing_m"],
        phase_max=cfg["phase_max"], phase_constraint=cfg["constraint"],
        phase_init="uniform", phase_init_scale=0.1,
        dual_2f_f1_m=COMMON["dual_2f_f1_m"], dual_2f_f2_m=COMMON["dual_2f_f2_m"],
        dual_2f_na1=COMMON["dual_2f_na1"], dual_2f_na2=COMMON["dual_2f_na2"],
        dual_2f_apply_scaling=COMMON["dual_2f_apply_scaling"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=COMMON["lr"])

    train_loader = DataLoader(train_ds, batch_size=COMMON["batch_size"], shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=COMMON["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=COMMON["batch_size"])

    w, ap = COMMON["receiver_window_m"], COMMON["aperture_diameter_m"]
    weights = COMMON["complex_weights"]

    print(f"\n  {name}: {cfg['label']}")
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
            print(f"    ep {epoch:2d} | loss={avg:.5f} co={vm['complex_overlap']:.4f} "
                  f"pr={vm['phase_rmse_rad']:.4f} ar={vm['amplitude_rmse']:.4f} | {dt:.1f}s")
        else:
            entry = {"epoch": epoch, "train_loss": avg, "time_s": dt}
        history.append(entry)

    test = evaluate(model, test_loader, device)
    print(f"    TEST: co={test['complex_overlap']:.4f} pr={test['phase_rmse_rad']:.4f} "
          f"ar={test['amplitude_rmse']:.4f} io={test['intensity_overlap']:.4f}")

    # Check phase utilization
    phases = [layer.phase().detach().cpu() for layer in model.layers]
    phase_stats = {
        "mean_abs": float(torch.stack([p.abs().mean() for p in phases]).mean()),
        "max_abs": float(torch.stack([p.abs().max() for p in phases]).max()),
        "std": float(torch.stack([p.std() for p in phases]).mean()),
        "fraction_saturated": float(torch.stack(
            [(p.abs() > 0.95 * cfg["phase_max"]).float().mean() for p in phases]
        ).mean()),
    }

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump({**test, "phase_stats": phase_stats}, f, indent=2)
    torch.save({"model_state_dict": model.state_dict(), "history": history,
                "test_metrics": test, "phase_stats": phase_stats}, run_dir / "checkpoint.pt")
    return {**test, "phase_stats": phase_stats}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Phase range sweep (spacing=1mm fixed, dx=2um)")

    results = {}
    for name, cfg in CONFIGS.items():
        results[name] = train_one(name, cfg, device)

    print(f"\n{'='*80}")
    print("PHASE RANGE SWEEP SUMMARY")
    print(f"{'='*80}")
    print(f"{'Config':>12} | {'Range':>20} | {'COverlap':>9} | {'PhaseRMSE':>9} | {'AmpRMSE':>8} | {'IntOvlp':>8} | {'Saturated':>9}")
    print("-" * 95)
    for name, cfg in CONFIGS.items():
        r = results[name]
        ps = r.get("phase_stats", {})
        print(f"{name:>12} | {cfg['label']:>20} | {r['complex_overlap']:>9.4f} | "
              f"{r['phase_rmse_rad']:>9.4f} | {r['amplitude_rmse']:>8.4f} | "
              f"{r['intensity_overlap']:>8.4f} | {ps.get('fraction_saturated', 0):>8.1%}")

    bl = results[list(results.keys())[0]]["baseline_co"]
    print(f"\n  Baseline (no D2NN): co={bl:.4f}")

    best = max(results, key=lambda n: results[n]["complex_overlap"])
    print(f"\n  >>> Best: {best} ({CONFIGS[best]['label']})")
    print(f"      co={results[best]['complex_overlap']:.4f}")

    with open(OUT_ROOT / "phase_range_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
