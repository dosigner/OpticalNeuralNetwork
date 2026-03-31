#!/usr/bin/env python
"""D2NN Loss Strategy Sweep — Phase A.

Test different loss functions targeting intensity metrics (not just CO).
Also includes TV regularization and cosine LR scheduler.

4 loss strategies:
  1. pib_only: maximize Power in Bucket (direct PIB loss)
  2. strehl_only: maximize Strehl ratio
  3. intensity_overlap: Gaussian overlap of intensity patterns
  4. co_pib_hybrid: CO + PIB combined

All with TV regularization (weight=0.05) and cosine LR scheduler.

Usage:
    cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python -m autoresearch.d2nn_loss_strategy_sweep
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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from kim2026.data.dataset import CachedFieldDataset
from kim2026.models.d2nn import BeamCleanupD2NN
from kim2026.training.losses import beam_radius, encircled_energy_fraction
from kim2026.training.metrics import (
    complex_overlap,
    gaussian_overlap,
    strehl_ratio,
)
from kim2026.training.targets import apply_receiver_aperture, center_crop_field

# ══════════════════════════════════════════════════════════════════════
WAVELENGTH_M = 1.55e-6
RECEIVER_WINDOW_M = 0.002048
APERTURE_DIAMETER_M = 0.002
N_FULL = 1024
ROI_N = 1024
SEED = 20260327

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kim2026" / "1km_cn2_5e-14_tel15cm_n1024_br75" / "cache"
MANIFEST = DATA_DIR.parent / "split_manifest.json"
OUT_ROOT = Path(__file__).resolve().parent / "runs" / "d2nn_loss_strategy"

ARCH = dict(num_layers=5, layer_spacing_m=10e-3, detector_distance_m=10e-3)

TRAIN = dict(
    lr=1e-3,
    epochs=200,
    batch_size=2,
    tv_weight=0.05,       # Total Variation regularization
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


def total_variation(model: BeamCleanupD2NN) -> torch.Tensor:
    """Total variation of all phase masks — encourages smoothness."""
    tv = torch.tensor(0.0, device=next(model.parameters()).device)
    for layer in model.layers:
        phase = layer.phase
        tv = tv + (phase[:, :-1] - phase[:, 1:]).abs().mean()
        tv = tv + (phase[:-1, :] - phase[1:, :]).abs().mean()
    return tv


def pib_loss(pred: torch.Tensor, target: torch.Tensor, window_m: float, radius_um: float = 50.0) -> torch.Tensor:
    """1 - PIB: power outside the bucket."""
    pred_i = pred.abs().square()
    target_i = target.abs().square()
    ref_radius = radius_um * 1e-6  # fixed 50um bucket
    n = pred.shape[-1]
    dx = window_m / n
    c = n // 2
    yy, xx = torch.meshgrid(torch.arange(n, device=pred.device) - c,
                             torch.arange(n, device=pred.device) - c, indexing="ij")
    r = torch.sqrt((xx * dx) ** 2 + (yy * dx) ** 2)
    mask = (r <= ref_radius).float()
    pib = (pred_i * mask).sum(dim=(-2, -1)) / pred_i.sum(dim=(-2, -1)).clamp(min=1e-12)
    return 1.0 - pib.mean()


def strehl_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - Strehl ratio."""
    pred_i = pred.abs().square()
    target_i = target.abs().square()
    sr = strehl_ratio(pred_i, target_i)
    return 1.0 - sr.mean()


def intensity_overlap_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - Gaussian overlap of intensity patterns."""
    pred_i = pred.abs().square()
    target_i = target.abs().square()
    io = gaussian_overlap(pred_i, target_i)
    return 1.0 - io.mean()


def co_pib_hybrid_loss(pred: torch.Tensor, target: torch.Tensor, window_m: float) -> torch.Tensor:
    """Combined CO + PIB loss."""
    co = complex_overlap(pred, target).mean()
    pib = pib_loss(pred, target, window_m)
    return (1.0 - co) + 0.5 * pib


# Loss config registry
LOSS_CONFIGS = OrderedDict({
    "pib_only": {
        "fn": lambda pred, tgt, w: pib_loss(pred, tgt, w),
        "desc": "PIB loss (50um bucket)",
    },
    "strehl_only": {
        "fn": lambda pred, tgt, w: strehl_loss(pred, tgt),
        "desc": "Strehl ratio loss",
    },
    "intensity_overlap": {
        "fn": lambda pred, tgt, w: intensity_overlap_loss(pred, tgt),
        "desc": "Intensity overlap loss",
    },
    "co_pib_hybrid": {
        "fn": lambda pred, tgt, w: co_pib_hybrid_loss(pred, tgt, w),
        "desc": "CO + 0.5*PIB hybrid",
    },
})


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    window = roi_window_m()
    all_co, all_io, all_sr, all_pib = [], [], [], []
    all_co_bl, all_pib_bl = [], []

    for batch in loader:
        u_turb = batch["u_turb"].to(device)
        u_vac = batch["u_vacuum"].to(device)
        target = prepare_field(u_vac)
        inp = prepare_field(u_turb)
        pred = model(inp)

        pred_i = pred.abs().square()
        target_i = target.abs().square()
        inp_i = inp.abs().square()

        all_co.append(complex_overlap(pred, target).cpu())
        all_io.append(gaussian_overlap(pred_i, target_i).cpu())
        all_sr.append(strehl_ratio(pred_i, target_i).cpu())

        # PIB at 50um
        n = pred.shape[-1]; dx = window / n; c = n // 2
        yy, xx = torch.meshgrid(torch.arange(n, device=device) - c,
                                 torch.arange(n, device=device) - c, indexing="ij")
        r = torch.sqrt((xx * dx) ** 2 + (yy * dx) ** 2)
        mask = (r <= 50e-6).float()
        pib_pred = (pred_i * mask).sum(dim=(-2, -1)) / pred_i.sum(dim=(-2, -1)).clamp(min=1e-12)
        pib_inp = (inp_i * mask).sum(dim=(-2, -1)) / inp_i.sum(dim=(-2, -1)).clamp(min=1e-12)
        all_pib.append(pib_pred.cpu())
        all_pib_bl.append(pib_inp.cpu())

        all_co_bl.append(complex_overlap(inp, target).cpu())

    return {
        "complex_overlap": float(torch.cat(all_co).mean()),
        "intensity_overlap": float(torch.cat(all_io).mean()),
        "strehl": float(torch.cat(all_sr).mean()),
        "pib_50um": float(torch.cat(all_pib).mean()),
        "baseline_co": float(torch.cat(all_co_bl).mean()),
        "baseline_pib_50um": float(torch.cat(all_pib_bl).mean()),
    }


@torch.no_grad()
def throughput_check(model, loader, device):
    model.eval()
    ti, to = 0.0, 0.0
    for batch in loader:
        inp = prepare_field(batch["u_turb"].to(device))
        pred = model(inp)
        ti += inp.abs().square().sum().item()
        to += pred.abs().square().sum().item()
    return to / max(ti, 1e-12)


def wf_rms_eval(model, loader, device):
    """Compute intensity-weighted WF RMS over test set."""
    model.eval()
    # Need zero-phase reference
    d0 = make_model().to(device); d0.eval()
    all_wf = []
    all_wf_bl = []
    for batch in loader:
        u_turb = batch["u_turb"].to(device)
        u_vac = batch["u_vacuum"].to(device)
        target = prepare_field(u_vac)
        inp = prepare_field(u_turb)
        with torch.no_grad():
            pred = model(inp)
            uv_d = d0(target)
            ut_d = d0(inp)
        for b in range(pred.shape[0]):
            # D2NN output vs vacuum
            p_ph = torch.angle(pred[b])
            t_ph = torch.angle(uv_d[b])
            diff = torch.remainder(p_ph - t_ph + math.pi, 2*math.pi) - math.pi
            w = uv_d[b].abs().square(); w = w / w.sum()
            all_wf.append(torch.sqrt((w * diff.square()).sum()).item())
            # Baseline: turbulent vs vacuum (both through zero-phase D2NN)
            p_ph2 = torch.angle(ut_d[b])
            diff2 = torch.remainder(p_ph2 - t_ph + math.pi, 2*math.pi) - math.pi
            all_wf_bl.append(torch.sqrt((w * diff2.square()).sum()).item())
    del d0; torch.cuda.empty_cache()
    return np.mean(all_wf), np.mean(all_wf_bl)


def train_one(name, loss_config, train_loader, val_loader, test_loader, device):
    run_dir = OUT_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.json"
    if results_path.exists():
        print(f"\n  [{name}] already completed, skipping")
        with open(results_path) as f:
            return json.load(f)

    torch.manual_seed(SEED); np.random.seed(SEED)
    model = make_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=TRAIN["epochs"] - TRAIN["warmup_epochs"], eta_min=1e-5)
    window = roi_window_m()
    loss_fn = loss_config["fn"]

    print(f"\n{'='*70}")
    print(f"  [{name}] {loss_config['desc']}")
    print(f"  TV_weight={TRAIN['tv_weight']}, LR={TRAIN['lr']} → cosine, epochs={TRAIN['epochs']}")
    print(f"{'='*70}")

    t_start = time.time()
    history = {"epoch": [], "loss": [], "val_co": [], "val_pib": [], "lr": []}

    for epoch in range(TRAIN["epochs"]):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        # Warmup LR
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
            pred = model(inp)

            loss = loss_fn(pred, target, window)
            # TV regularization
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

        if epoch % 10 == 0 or epoch == TRAIN["epochs"] - 1:
            val_m = evaluate(model, val_loader, device)
            history["epoch"].append(epoch)
            history["loss"].append(avg_loss)
            history["val_co"].append(val_m["complex_overlap"])
            history["val_pib"].append(val_m["pib_50um"])
            history["lr"].append(cur_lr)
            print(f"  Epoch {epoch:3d}/{TRAIN['epochs']-1} | loss={avg_loss:.5f} | "
                  f"co={val_m['complex_overlap']:.4f} | pib={val_m['pib_50um']:.4f} | "
                  f"io={val_m['intensity_overlap']:.4f} | sr={val_m['strehl']:.4f} | "
                  f"lr={cur_lr:.2e} | {dt:.1f}s")

    train_time = time.time() - t_start

    # Test
    test_m = evaluate(model, test_loader, device)
    tp = throughput_check(model, test_loader, device)
    wf_rms_d2nn, wf_rms_bl = wf_rms_eval(model, test_loader, device)

    # Save
    phases = [layer.phase.detach().cpu().numpy() for layer in model.layers]
    np.save(run_dir / "phases_wrapped.npy", np.stack(phases))

    result = {
        "name": name,
        "description": loss_config["desc"],
        "arch": ARCH,
        **test_m,
        "throughput": tp,
        "wf_rms_rad": wf_rms_d2nn,
        "wf_rms_baseline_rad": wf_rms_bl,
        "wf_rms_nm": wf_rms_d2nn * WAVELENGTH_M / (2 * math.pi) * 1e9,
        "wf_rms_baseline_nm": wf_rms_bl * WAVELENGTH_M / (2 * math.pi) * 1e9,
        "training_seconds": train_time,
        "history": history,
    }
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
    torch.save({"model_state_dict": model.state_dict()}, run_dir / "checkpoint.pt")

    # Report
    pib_delta = (test_m["pib_50um"] - test_m["baseline_pib_50um"]) / max(test_m["baseline_pib_50um"], 1e-12) * 100
    wf_delta = (wf_rms_bl - wf_rms_d2nn) / max(wf_rms_bl, 1e-12) * 100
    co_delta = test_m["complex_overlap"] - test_m["baseline_co"]

    print(f"\n  TEST RESULTS:")
    print(f"    CO:     {test_m['baseline_co']:.4f} → {test_m['complex_overlap']:.4f} ({co_delta:+.4f})")
    print(f"    PIB:    {test_m['baseline_pib_50um']:.4f} → {test_m['pib_50um']:.4f} ({pib_delta:+.1f}%)")
    print(f"    WF RMS: {wf_rms_bl*WAVELENGTH_M/(2*math.pi)*1e9:.1f} → {wf_rms_d2nn*WAVELENGTH_M/(2*math.pi)*1e9:.1f} nm ({wf_delta:+.1f}%)")
    print(f"    Strehl: {test_m['strehl']:.4f}")
    print(f"    TP:     {tp:.4f}")

    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"D2NN Loss Strategy Sweep — Phase A")
    print(f"4 loss functions × TV reg + cosine LR")
    print(f"Arch: {ARCH['num_layers']} layers, {ARCH['layer_spacing_m']*1e3:.0f}mm spacing")

    train_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="train")
    val_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="val")
    test_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="test")
    print(f"Data: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=TRAIN["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=TRAIN["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=TRAIN["batch_size"])

    all_results = []
    for name, config in LOSS_CONFIGS.items():
        result = train_one(name, config, train_loader, val_loader, test_loader, device)
        all_results.append(result)

    # Summary
    print(f"\n{'='*100}")
    print("LOSS STRATEGY SWEEP SUMMARY")
    print(f"{'='*100}")
    header = f"{'Name':>20} | {'CO':>7} | {'PIB':>7} | {'PIB%':>7} | {'WF nm':>8} | {'WF%':>7} | {'Strehl':>7} | {'TP':>6}"
    print(header)
    print("-" * len(header))

    for r in all_results:
        pib_pct = (r["pib_50um"] - r["baseline_pib_50um"]) / max(r["baseline_pib_50um"], 1e-12) * 100
        wf_pct = (r["wf_rms_baseline_rad"] - r["wf_rms_rad"]) / max(r["wf_rms_baseline_rad"], 1e-12) * 100
        print(f"{r['name']:>20} | {r['complex_overlap']:>7.4f} | {r['pib_50um']:>7.4f} | "
              f"{pib_pct:>+6.1f}% | {r['wf_rms_nm']:>7.1f} | {wf_pct:>+6.1f}% | "
              f"{r['strehl']:>7.4f} | {r['throughput']:>6.4f}")

    print(f"\nBaseline: CO={all_results[0]['baseline_co']:.4f}, PIB={all_results[0]['baseline_pib_50um']:.4f}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUT_ROOT / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {OUT_ROOT}")


if __name__ == "__main__":
    main()
