#!/usr/bin/env python
"""FD2NN 4f Multi-layer Sweep — Fourier-domain phase masks with ASM propagation.

Architecture: single 4f system with N phase masks between Lens1 and Lens2.
    Input → Lens1(f) → Mask1 → ASM(z) → ... → MaskN → Lens2(f) → Focal Lens(f3) → PIB

Phase 1: f × z sweep (6 runs)
  f ∈ {10, 25} mm, z ∈ {1, 3, 5} mm
Phase 2: distance sweep (5 runs) — uses best (f, z) from Phase 1

Loss: focal_raw_received_power = -log(bucket_energy / input_energy + eps)

Usage:
    cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python -m autoresearch.fd2nn_4f_multilayer_sweep
    cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python -m autoresearch.fd2nn_4f_multilayer_sweep --phase 1
    cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python -m autoresearch.fd2nn_4f_multilayer_sweep --phase 2 --best-f 25 --best-z 3
"""
from __future__ import annotations

import argparse
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
from kim2026.models.fd2nn import MultiLayerFD2NN
from kim2026.optics.lens_2f import lens_2f_forward, fourier_plane_pitch
from kim2026.training.metrics import complex_overlap, gaussian_overlap, strehl_ratio_correct
from kim2026.training.targets import apply_receiver_aperture, center_crop_field

# ═══════════════════════════════════ Constants ═══════════════════════════════
KIM2026_ROOT = Path(__file__).resolve().parent.parent
WAVELENGTH_M = 1.55e-6
RECEIVER_WINDOW_M = 0.002048
APERTURE_DIAMETER_M = 0.002
N_FULL = 1024
ROI_N = 1024
SEED = 20260405

# Focusing lens (detector)
FOCUS_F_M = 6.5e-3
PIB_BUCKET_RADIUS_UM = 10.0

# Data
DATA_ROOT = KIM2026_ROOT / "data" / "kim2026" / "1km_cn2_5e-14_tel15cm_pitch_rescale"
DATA_DIR = DATA_ROOT / "cache"
MANIFEST = DATA_ROOT / "split_manifest.json"
DATA_PLANE_SELECTOR = "reduced_ideal"
DISTANCE_SWEEP_ROOT = KIM2026_ROOT / "data" / "kim2026" / "distance_sweep_cn2_5e-14"

# Output
OUT_ROOT_P1 = Path(__file__).resolve().parent / "runs" / "0405-fd2nn-4f-sweep-pitchrescale"
OUT_ROOT_P2 = Path(__file__).resolve().parent / "runs" / "0405-fd2nn-4f-distance-sweep-pitchrescale"

# ═══════════════════════════════ Hyperparameters ═════════════════════════════
NUM_LAYERS = 5
LR = 5e-4
EPOCHS = 30
BATCH_SIZE = 32

# Sweep axes
SWEEP_F_MM = [10, 25]
SWEEP_Z_MM = [1, 3, 5]
SWEEP_DISTANCES_M = [100, 500, 1000, 2000, 3000]


def roi_window_m():
    return RECEIVER_WINDOW_M * (ROI_N / N_FULL)


def prepare_field(field: torch.Tensor) -> torch.Tensor:
    a = apply_receiver_aperture(field, receiver_window_m=RECEIVER_WINDOW_M,
                                 aperture_diameter_m=APERTURE_DIAMETER_M)
    return center_crop_field(a, crop_n=ROI_N)


def make_model(f_m: float, z_m: float) -> MultiLayerFD2NN:
    return MultiLayerFD2NN(
        n=ROI_N,
        wavelength_m=WAVELENGTH_M,
        window_m=roi_window_m(),
        num_layers=NUM_LAYERS,
        f_m=f_m,
        layer_spacing_m=z_m,
    )


def to_focal_plane(field: torch.Tensor) -> tuple[torch.Tensor, float]:
    dx_in_m = roi_window_m() / ROI_N
    return lens_2f_forward(
        field, dx_in_m=dx_in_m, wavelength_m=WAVELENGTH_M,
        f_m=FOCUS_F_M, na=None, apply_scaling=False,
    )


def _bucket_mask(field: torch.Tensor, dx_focal: float) -> torch.Tensor:
    n = field.shape[-1]
    c = n // 2
    yy, xx = torch.meshgrid(torch.arange(n, device=field.device) - c,
                             torch.arange(n, device=field.device) - c, indexing="ij")
    r = torch.sqrt((xx * dx_focal) ** 2 + (yy * dx_focal) ** 2)
    return (r <= PIB_BUCKET_RADIUS_UM * 1e-6).float()


def focal_raw_received_power_loss(d2nn_pred, d2nn_target, focal_pred, focal_target, dx_focal):
    """loss = -log(bucket_energy / input_energy + eps)"""
    mask = _bucket_mask(focal_pred, dx_focal)
    pred_bucket = (focal_pred.abs().square() * mask).sum(dim=(-2, -1))
    input_energy = d2nn_target.abs().square().sum(dim=(-2, -1)).clamp(min=1e-12)
    focal_tp = pred_bucket / input_energy
    return -torch.log(focal_tp + 1e-8).mean()


def compute_focal_pib(focal_field, dx_focal, radius_um):
    intensity = focal_field.abs().square()
    n = focal_field.shape[-1]
    c = n // 2
    yy, xx = torch.meshgrid(torch.arange(n, device=focal_field.device) - c,
                             torch.arange(n, device=focal_field.device) - c, indexing="ij")
    r = torch.sqrt((xx * dx_focal) ** 2 + (yy * dx_focal) ** 2)
    mask = (r <= radius_um * 1e-6).float()
    pib = (intensity * mask).sum(dim=(-2, -1)) / intensity.sum(dim=(-2, -1)).clamp(min=1e-12)
    return pib


def total_variation(model: MultiLayerFD2NN) -> torch.Tensor:
    tv = torch.tensor(0.0, device=next(model.parameters()).device)
    for layer in model.layers:
        phase = layer.phase
        tv = tv + (phase[:, :-1] - phase[:, 1:]).abs().mean()
        tv = tv + (phase[:-1, :] - phase[1:, :]).abs().mean()
    return tv


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_co, all_co_bl = [], []
    all_pib_10, all_pib_50, all_pib_10_bl, all_pib_50_bl = [], [], [], []
    all_pib_10_vac, all_pib_50_vac = [], []

    for batch in loader:
        u_turb = batch["u_turb"].to(device)
        u_vac = batch["u_vacuum"].to(device)
        target = prepare_field(u_vac)
        inp = prepare_field(u_turb)
        pred = model(inp)

        all_co.append(complex_overlap(pred, target).cpu())
        all_co_bl.append(complex_overlap(inp, target).cpu())

        focal_pred, dx_focal = to_focal_plane(pred)
        all_pib_10.append(compute_focal_pib(focal_pred, dx_focal, 10.0).cpu())
        all_pib_50.append(compute_focal_pib(focal_pred, dx_focal, 50.0).cpu())
        del focal_pred

        focal_target, _ = to_focal_plane(target)
        all_pib_10_vac.append(compute_focal_pib(focal_target, dx_focal, 10.0).cpu())
        all_pib_50_vac.append(compute_focal_pib(focal_target, dx_focal, 50.0).cpu())
        del focal_target

        focal_inp, _ = to_focal_plane(inp)
        all_pib_10_bl.append(compute_focal_pib(focal_inp, dx_focal, 10.0).cpu())
        all_pib_50_bl.append(compute_focal_pib(focal_inp, dx_focal, 50.0).cpu())
        del focal_inp
        torch.cuda.empty_cache()

    return {
        "co_output": float(torch.cat(all_co).mean()),
        "co_baseline": float(torch.cat(all_co_bl).mean()),
        "focal_pib_10um": float(torch.cat(all_pib_10).mean()),
        "focal_pib_50um": float(torch.cat(all_pib_50).mean()),
        "focal_pib_10um_baseline": float(torch.cat(all_pib_10_bl).mean()),
        "focal_pib_50um_baseline": float(torch.cat(all_pib_50_bl).mean()),
        "focal_pib_10um_vacuum": float(torch.cat(all_pib_10_vac).mean()),
        "focal_pib_50um_vacuum": float(torch.cat(all_pib_50_vac).mean()),
        "dx_focal_um": dx_focal * 1e6,
    }


@torch.no_grad()
def throughput_check(model, loader, device):
    model.eval()
    ti, to_ = 0.0, 0.0
    for batch in loader:
        inp = prepare_field(batch["u_turb"].to(device))
        pred = model(inp)
        ti += inp.abs().square().sum().item()
        to_ += pred.abs().square().sum().item()
    return to_ / max(ti, 1e-12)


def train_one(name: str, f_m: float, z_m: float, out_root: Path,
              train_loader, val_loader, test_loader, device):
    run_dir = out_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.json"
    if results_path.exists():
        print(f"\n  [{name}] already completed, skipping")
        with open(results_path) as f:
            return json.load(f)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = make_model(f_m, z_m).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    dx_f = model.dx_fourier_m
    print(f"\n{'='*70}")
    print(f"  [{name}] f={f_m*1e3:.0f}mm, z={z_m*1e3:.0f}mm, layers={NUM_LAYERS}")
    print(f"  dx_fourier={dx_f*1e6:.1f}μm, fourier_window={model.fourier_window_m*1e3:.1f}mm")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  LR={LR}, epochs={EPOCHS}, batch={BATCH_SIZE}")
    print(f"{'='*70}")

    t_start = time.time()
    log_file = run_dir / "epoch_log.txt"

    for epoch in range(EPOCHS):
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
            focal_pred, dx_focal = to_focal_plane(pred)
            focal_target, _ = to_focal_plane(target)

            loss = focal_raw_received_power_loss(pred, target, focal_pred, focal_target, dx_focal)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        dt = time.time() - t0

        with open(log_file, "a") as lf:
            lf.write(f"ep{epoch:3d} loss={avg_loss:.5f} {dt:.1f}s\n")

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            val_m = evaluate(model, val_loader, device)
            msg = (f"  Epoch {epoch:3d}/{EPOCHS-1} | loss={avg_loss:.5f} | "
                   f"co={val_m['co_output']:.4f} | "
                   f"fpib10={val_m['focal_pib_10um']:.4f} | {dt:.1f}s")
            print(msg, flush=True)
            with open(log_file, "a") as lf:
                lf.write(msg + "\n")

    train_time = time.time() - t_start

    # Test evaluation
    test_m = evaluate(model, test_loader, device)
    tp = throughput_check(model, test_loader, device)

    # Save phases
    phases = [layer.phase.detach().cpu().numpy() for layer in model.layers]
    np.save(run_dir / "phases_wrapped.npy", np.stack(phases))

    result = {
        "name": name,
        "f_mm": f_m * 1e3,
        "z_mm": z_m * 1e3,
        "num_layers": NUM_LAYERS,
        "dx_fourier_um": dx_f * 1e6,
        "focus_f_mm": FOCUS_F_M * 1e3,
        **test_m,
        "throughput": tp,
        "training_seconds": train_time,
    }
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
    torch.save({"model_state_dict": model.state_dict()}, run_dir / "checkpoint.pt")

    # Report
    pib10_delta = test_m["focal_pib_10um"] - test_m["focal_pib_10um_baseline"]
    print(f"\n  TEST: PIB@10μm={test_m['focal_pib_10um']:.4f} "
          f"(baseline={test_m['focal_pib_10um_baseline']:.4f}, Δ={pib10_delta:+.4f}) "
          f"| TP={tp:.4f} | {train_time:.0f}s")
    return result


def run_phase1(train_loader, val_loader, test_loader, device):
    """Phase 1: f × z sweep (6 runs)."""
    print(f"\n{'#'*70}")
    print("  PHASE 1: f × z Sweep")
    print(f"  f ∈ {SWEEP_F_MM} mm, z ∈ {SWEEP_Z_MM} mm")
    print(f"{'#'*70}")

    results = []
    for f_mm in SWEEP_F_MM:
        for z_mm in SWEEP_Z_MM:
            name = f"f{f_mm}mm_z{z_mm}mm"
            r = train_one(name, f_mm * 1e-3, z_mm * 1e-3, OUT_ROOT_P1,
                          train_loader, val_loader, test_loader, device)
            results.append(r)

    # Summary
    print(f"\n{'='*90}")
    print("PHASE 1 SUMMARY")
    print(f"{'='*90}")
    header = f"{'Config':>15} | {'dx_f(μm)':>8} | {'fPIB10':>7} | {'fPIB50':>7} | {'CO':>7} | {'TP':>6} | {'Δ PIB10':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        delta = r["focal_pib_10um"] - r["focal_pib_10um_baseline"]
        print(f"{r['name']:>15} | {r['dx_fourier_um']:>8.1f} | {r['focal_pib_10um']:>7.4f} | "
              f"{r['focal_pib_50um']:>7.4f} | {r['co_output']:>7.4f} | "
              f"{r['throughput']:>6.4f} | {delta:>+8.4f}")

    r0 = results[0]
    print(f"\nBaseline: fPIB@10μm={r0['focal_pib_10um_baseline']:.4f}, "
          f"fPIB@50μm={r0['focal_pib_50um_baseline']:.4f}")

    OUT_ROOT_P1.mkdir(parents=True, exist_ok=True)
    with open(OUT_ROOT_P1 / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # Find best config
    best = max(results, key=lambda r: r["focal_pib_10um"])
    print(f"\nBest config: {best['name']} (fPIB@10μm={best['focal_pib_10um']:.4f})")
    return best


def run_phase2(best_f_mm: float, best_z_mm: float, device):
    """Phase 2: distance sweep with best (f, z), loading per-distance data."""
    print(f"\n{'#'*70}")
    print(f"  PHASE 2: Distance Sweep with f={best_f_mm}mm, z={best_z_mm}mm")
    print(f"  Distances: {SWEEP_DISTANCES_M} m")
    print(f"{'#'*70}")

    results = []
    for dist_m in SWEEP_DISTANCES_M:
        dist_data = DISTANCE_SWEEP_ROOT / f"L{dist_m}m"
        if not dist_data.exists():
            print(f"\n  [{dist_m}m] data not found at {dist_data}, skipping")
            continue

        dist_cache = dist_data / "cache"
        dist_manifest = dist_data / "split_manifest.json"
        train_ds = CachedFieldDataset(cache_dir=str(dist_cache), manifest_path=str(dist_manifest),
                                       split="train", plane_selector=DATA_PLANE_SELECTOR)
        val_ds = CachedFieldDataset(cache_dir=str(dist_cache), manifest_path=str(dist_manifest),
                                     split="val", plane_selector=DATA_PLANE_SELECTOR)
        test_ds = CachedFieldDataset(cache_dir=str(dist_cache), manifest_path=str(dist_manifest),
                                      split="test", plane_selector=DATA_PLANE_SELECTOR)
        print(f"\n  [{dist_m}m] train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

        tl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        vl = DataLoader(val_ds, batch_size=8, num_workers=0)
        tel = DataLoader(test_ds, batch_size=8, num_workers=0)

        name = f"dist_{dist_m}m"
        r = train_one(name, best_f_mm * 1e-3, best_z_mm * 1e-3, OUT_ROOT_P2, tl, vl, tel, device)
        r["distance_m"] = dist_m
        results.append(r)

    OUT_ROOT_P2.mkdir(parents=True, exist_ok=True)
    with open(OUT_ROOT_P2 / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def main():
    parser = argparse.ArgumentParser(description="FD2NN 4f Multi-layer Sweep")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2],
                        help="1 = f×z sweep, 2 = distance sweep")
    parser.add_argument("--best-f", type=float, default=None,
                        help="Best f in mm (for phase 2)")
    parser.add_argument("--best-z", type=float, default=None,
                        help="Best z in mm (for phase 2)")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Override data directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Resolve data paths
    data_dir = Path(args.data_path) if args.data_path else DATA_DIR
    manifest = data_dir.parent / "split_manifest.json"

    print(f"\nFD2NN 4f Multi-layer Sweep")
    print(f"  λ={WAVELENGTH_M*1e6:.2f}μm, N={ROI_N}, dx={roi_window_m()/ROI_N*1e6:.1f}μm")
    print(f"  Focus lens f3={FOCUS_F_M*1e3:.1f}mm, PIB bucket={PIB_BUCKET_RADIUS_UM:.0f}μm")
    print(f"  Loss: focal_raw_received_power")
    print(f"  {NUM_LAYERS} layers, LR={LR}, epochs={EPOCHS}, batch={BATCH_SIZE}")
    print(f"  Data: {data_dir}")

    train_ds = CachedFieldDataset(cache_dir=str(data_dir), manifest_path=str(manifest),
                                   split="train", plane_selector=DATA_PLANE_SELECTOR)
    val_ds = CachedFieldDataset(cache_dir=str(data_dir), manifest_path=str(manifest),
                                 split="val", plane_selector=DATA_PLANE_SELECTOR)
    test_ds = CachedFieldDataset(cache_dir=str(data_dir), manifest_path=str(manifest),
                                  split="test", plane_selector=DATA_PLANE_SELECTOR)
    print(f"  Samples: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=8, num_workers=0)

    if args.phase == 1:
        best = run_phase1(train_loader, val_loader, test_loader, device)
        print(f"\nTo run Phase 2:")
        print(f"  PYTHONPATH=src python -m autoresearch.fd2nn_4f_multilayer_sweep "
              f"--phase 2 --best-f {best['f_mm']:.0f} --best-z {best['z_mm']:.0f}")
    else:
        if args.best_f is None or args.best_z is None:
            # Try to load from phase 1 summary
            summary_path = OUT_ROOT_P1 / "summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    p1_results = json.load(f)
                best = max(p1_results, key=lambda r: r["focal_pib_10um"])
                best_f = best["f_mm"]
                best_z = best["z_mm"]
                print(f"\n  Auto-selected best from Phase 1: f={best_f}mm, z={best_z}mm")
            else:
                raise ValueError("Phase 2 requires --best-f and --best-z, "
                                 "or Phase 1 results in summary.json")
        else:
            best_f = args.best_f
            best_z = args.best_z
        run_phase2(best_f, best_z, device)


if __name__ == "__main__":
    main()
