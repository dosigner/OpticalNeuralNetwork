#!/usr/bin/env python
"""FD2NN spacing sweep with corrected focal length f=10mm.

Previous run01 used f=1mm → dx_fourier=0.76µm (λ/2, sub-wavelength, unfabricable).
This script uses f=10mm → dx_fourier=7.57µm (4.9λ, realistic metasurface pitch).

Sweeps 7 layer spacings spanning near-field to far-field (0 to 4.3 z_R).
Saves checkpoint, history, test_metrics, sample_fields, and phase snapshots.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

# Ensure kim2026 package is importable
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
    amplitude_rmse,
    complex_overlap,
    full_field_phase_rmse,
    gaussian_overlap,
    phase_rmse,
    strehl_ratio,
)
from kim2026.training.targets import apply_receiver_aperture

# ─── Config ──────────────────────────────────────────────────────
PROJ = Path(__file__).resolve().parent.parent
DATA_DIR = PROJ / "data" / "kim2026" / "1km_cn2e-14_w2m_n1024_dx2mm" / "cache"
MANIFEST = DATA_DIR.parent / "split_manifest.json"
OUT_ROOT = PROJ / "runs" / "01_fd2nn_spacing_sweep_f10mm_claude"

# Spacing sweep: near-field → far-field
# z_R(10px feature) = π·(10·dx_fourier)²/λ = 11.6 mm
SPACINGS = {
    "spacing_0mm":  0.0,       # stacked, no propagation
    "spacing_1mm":  1e-3,      # 0.09 z_R — near-field
    "spacing_3mm":  3e-3,      # 0.26 z_R — weak diffraction coupling
    "spacing_6mm":  6e-3,      # 0.52 z_R — moderate diffraction
    "spacing_12mm": 12e-3,     # 1.0  z_R — Rayleigh range
    "spacing_25mm": 25e-3,     # 2.2  z_R — strong feature mixing
    "spacing_50mm": 50e-3,     # 4.3  z_R — far-field approach
}

COMMON = dict(
    wavelength_m=1.55e-6,
    receiver_window_m=0.002048,    # 2.048mm → dx=2µm
    aperture_diameter_m=0.002,     # 2mm
    n=1024,
    num_layers=5,
    dual_2f_f1_m=10.0e-3,         # 10 mm (corrected from 1mm)
    dual_2f_f2_m=10.0e-3,         # 10 mm
    dual_2f_na1=0.16,
    dual_2f_na2=0.16,
    dual_2f_apply_scaling=False,
    phase_max=2 * math.pi,        # 2π (best from phase range sweep)
    phase_constraint="symmetric_tanh",
    phase_init="uniform",
    phase_init_scale=0.1,
    lr=5e-4,
    epochs=30,
    batch_size=2,
    complex_weights={
        "complex_overlap": 1.0,
        "amplitude_mse": 0.5,
        "intensity_overlap": 1.0,
        "beam_radius": 1.0,
        "encircled_energy": 1.0,
    },
    seed=20260323,
)

PHASE_SNAPSHOT_EPOCHS = {0, 15, 29}


def prepare_field(field: torch.Tensor, *, aperture_diameter_m: float) -> torch.Tensor:
    """Apply receiver aperture (no center crop — full 1024×1024 field)."""
    return apply_receiver_aperture(
        field,
        receiver_window_m=COMMON["receiver_window_m"],
        aperture_diameter_m=aperture_diameter_m,
    )


def make_model(spacing_m: float) -> BeamCleanupFD2NN:
    return BeamCleanupFD2NN(
        n=COMMON["n"],
        wavelength_m=COMMON["wavelength_m"],
        window_m=COMMON["receiver_window_m"],
        num_layers=COMMON["num_layers"],
        layer_spacing_m=spacing_m,
        phase_max=COMMON["phase_max"],
        phase_constraint=COMMON["phase_constraint"],
        phase_init=COMMON["phase_init"],
        phase_init_scale=COMMON["phase_init_scale"],
        dual_2f_f1_m=COMMON["dual_2f_f1_m"],
        dual_2f_f2_m=COMMON["dual_2f_f2_m"],
        dual_2f_na1=COMMON["dual_2f_na1"],
        dual_2f_na2=COMMON["dual_2f_na2"],
        dual_2f_apply_scaling=COMMON["dual_2f_apply_scaling"],
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    window = COMMON["receiver_window_m"]
    ap = COMMON["aperture_diameter_m"]
    all_co, all_pr, all_pr_full, all_ar, all_io, all_sr = [], [], [], [], [], []
    all_br, all_ee = [], []
    all_co_bl, all_io_bl = [], []

    for batch in loader:
        u_turb = batch["u_turb"].to(device)
        u_vac = batch["u_vacuum"].to(device)
        target = prepare_field(u_vac, aperture_diameter_m=ap)
        inp = prepare_field(u_turb, aperture_diameter_m=ap)
        pred = model(inp)

        all_co.append(complex_overlap(pred, target).cpu())
        all_pr.append(phase_rmse(pred, target).cpu())
        all_pr_full.append(full_field_phase_rmse(pred, target).cpu())
        all_ar.append(amplitude_rmse(pred, target).cpu())
        pred_i = pred.abs().square()
        tgt_i = target.abs().square()
        all_io.append(gaussian_overlap(pred_i, tgt_i).cpu())
        all_sr.append(strehl_ratio(pred_i, tgt_i).cpu())
        ref_radius = beam_radius(tgt_i, window_m=window)
        all_br.append(beam_radius(pred_i, window_m=window).cpu())
        all_ee.append(encircled_energy_fraction(pred_i, reference_radius=ref_radius, window_m=window).cpu())
        all_co_bl.append(complex_overlap(inp, target).cpu())
        all_io_bl.append(gaussian_overlap(inp.abs().square(), tgt_i).cpu())

    return {
        "complex_overlap": float(torch.cat(all_co).mean()),
        "phase_rmse_rad": float(torch.cat(all_pr).mean()),
        "full_field_phase_rmse_rad": float(torch.cat(all_pr_full).mean()),
        "amplitude_rmse": float(torch.cat(all_ar).mean()),
        "intensity_overlap": float(torch.cat(all_io).mean()),
        "strehl": float(torch.cat(all_sr).mean()),
        "beam_radius_m": float(torch.cat(all_br).mean()),
        "encircled_energy": float(torch.cat(all_ee).mean()),
        "baseline_complex_overlap": float(torch.cat(all_co_bl).mean()),
        "baseline_intensity_overlap": float(torch.cat(all_io_bl).mean()),
    }


def train_one(name: str, spacing_m: float, device: torch.device) -> dict:
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

    model = make_model(spacing_m).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=COMMON["lr"])

    train_loader = DataLoader(train_ds, batch_size=COMMON["batch_size"], shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=COMMON["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=COMMON["batch_size"])

    window = COMMON["receiver_window_m"]
    ap = COMMON["aperture_diameter_m"]
    weights = COMMON["complex_weights"]

    # Fourier plane analysis
    dx_in = window / COMMON["n"]
    dx_f = COMMON["wavelength_m"] * COMMON["dual_2f_f1_m"] / (COMMON["n"] * dx_in)
    w10 = 10 * dx_f
    z_R = math.pi * w10**2 / COMMON["wavelength_m"]
    z_ratio = spacing_m / z_R if spacing_m > 0 else 0.0

    print(f"\n{'='*70}")
    print(f"  {name}: spacing={spacing_m*1e3:.0f}mm, z/z_R={z_ratio:.2f}")
    print(f"  dx_fourier={dx_f*1e6:.2f}um ({dx_f/COMMON['wavelength_m']:.1f}λ)")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*70}")

    history = []
    for epoch in range(COMMON["epochs"]):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for batch in train_loader:
            u_turb = batch["u_turb"].to(device)
            u_vac = batch["u_vacuum"].to(device)
            target = prepare_field(u_vac, aperture_diameter_m=ap)
            inp = prepare_field(u_turb, aperture_diameter_m=ap)

            optimizer.zero_grad(set_to_none=True)
            pred = model(inp)
            loss = complex_field_loss(pred, target, weights=weights, window_m=window)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        dt = time.time() - t0

        # Validate every 5 epochs + first + last
        if epoch % 5 == 0 or epoch == COMMON["epochs"] - 1:
            val_metrics = evaluate(model, val_loader, device)
            entry = {"epoch": epoch, "train_loss": avg_loss, "time_s": dt, **val_metrics}
        else:
            entry = {"epoch": epoch, "train_loss": avg_loss, "time_s": dt}

        # Phase snapshots at epoch 0, 15, 29
        if epoch in PHASE_SNAPSHOT_EPOCHS:
            phases = [layer.phase().detach().cpu().numpy() for layer in model.layers]
            np.save(run_dir / f"phases_epoch{epoch:03d}.npy", np.stack(phases))

        history.append(entry)
        co = entry.get("complex_overlap", "")
        pr = entry.get("phase_rmse_rad", "")
        co_str = f"{co:.4f}" if co else "---"
        pr_str = f"{pr:.4f}" if pr else "---"
        print(f"  Epoch {epoch:3d}/{COMMON['epochs']-1} | loss={avg_loss:.5f} | "
              f"co={co_str:>6} | pr={pr_str:>6} | {dt:.1f}s")

    # Final test
    test_metrics = evaluate(model, test_loader, device)
    print(f"\n  TEST: co={test_metrics['complex_overlap']:.4f} "
          f"pr={test_metrics['phase_rmse_rad']:.4f} "
          f"io={test_metrics['intensity_overlap']:.4f} "
          f"sr={test_metrics['strehl']:.4f}")
    print(f"  BASELINE: co={test_metrics['baseline_complex_overlap']:.4f} "
          f"io={test_metrics['baseline_intensity_overlap']:.4f}")

    # Save checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "history": history,
        "test_metrics": test_metrics,
        "config": {
            "name": name, "spacing_m": spacing_m,
            "z_over_zR": z_ratio, "dx_fourier_um": dx_f * 1e6,
            **COMMON,
        },
    }, run_dir / "checkpoint.pt")

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Save sample fields for visualization
    model.eval()
    with torch.no_grad():
        sample = test_ds[0]
        u_t = sample["u_turb"].unsqueeze(0).to(device)
        u_v = sample["u_vacuum"].unsqueeze(0).to(device)
        tgt = prepare_field(u_v, aperture_diameter_m=ap)
        inp = prepare_field(u_t, aperture_diameter_m=ap)
        pred = model(inp)
        np.savez(
            run_dir / "sample_fields.npz",
            input_real=inp[0].real.cpu().numpy(),
            input_imag=inp[0].imag.cpu().numpy(),
            pred_real=pred[0].real.cpu().numpy(),
            pred_imag=pred[0].imag.cpu().numpy(),
            target_real=tgt[0].real.cpu().numpy(),
            target_imag=tgt[0].imag.cpu().numpy(),
        )

    return test_metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # System info
    dx_in = COMMON["receiver_window_m"] / COMMON["n"]
    dx_f = COMMON["wavelength_m"] * COMMON["dual_2f_f1_m"] / (COMMON["n"] * dx_in)
    w10 = 10 * dx_f
    z_R = math.pi * w10**2 / COMMON["wavelength_m"]
    print(f"\nSystem: λ={COMMON['wavelength_m']*1e6:.2f}µm, f={COMMON['dual_2f_f1_m']*1e3:.0f}mm")
    print(f"  dx_in={dx_in*1e6:.1f}µm, dx_fourier={dx_f*1e6:.2f}µm ({dx_f/COMMON['wavelength_m']:.1f}λ)")
    print(f"  z_R(10px)={z_R*1e3:.2f}mm")

    all_results = {}
    for name, spacing in SPACINGS.items():
        result = train_one(name, spacing, device)
        all_results[name] = result

    # Summary table
    print(f"\n{'='*80}")
    print("SPACING SWEEP SUMMARY (f=10mm)")
    print(f"{'='*80}")
    print(f"{'Name':>15} | {'Spacing':>8} | {'z/z_R':>6} | {'CO':>7} | {'PhRMSE':>7} | {'IO':>7} | {'Strehl':>7}")
    print("-" * 80)
    for name, spacing in SPACINGS.items():
        m = all_results[name]
        z_ratio = spacing / z_R if spacing > 0 else 0.0
        sp_str = f"{spacing*1e3:.0f}mm" if spacing > 0 else "0(FFT)"
        print(f"{name:>15} | {sp_str:>8} | {z_ratio:>6.2f} | "
              f"{m['complex_overlap']:>7.4f} | {m['phase_rmse_rad']:>7.4f} | "
              f"{m['intensity_overlap']:>7.4f} | {m['strehl']:>7.4f}")
    bl = all_results["spacing_0mm"]
    print(f"\n  Baseline (no D2NN): co={bl['baseline_complex_overlap']:.4f}, "
          f"io={bl['baseline_intensity_overlap']:.4f}")

    with open(OUT_ROOT / "sweep_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUT_ROOT}")


if __name__ == "__main__":
    main()
