#!/usr/bin/env python
"""FD2NN dual-2f metasurface sweep: train 4 models (0/0.1/1/2mm spacing) on existing data.

Uses existing 1km_cn2e-14_w2m_n1024_dx2mm data (Cn2=1e-14, 1024x1024, dx=2mm).
Logs per-epoch metrics for visualization.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

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
from kim2026.training.targets import apply_receiver_aperture, center_crop_field

# ─── Config ──────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kim2026" / "1km_cn2e-14_w2m_n1024_dx2mm" / "cache"
MANIFEST = DATA_DIR.parent / "split_manifest.json"
OUT_ROOT = Path(__file__).resolve().parent.parent / "runs" / "fd2nn_metasurface_sweep_dual2f"

SPACINGS = {
    "spacing_0mm": 0.0,         # baseline: stacked Fourier masks without inter-layer propagation
    "spacing_0p1mm": 0.1e-3,    # 100um metalens stack
    "spacing_1mm": 1e-3,        # 1mm metalens stack
    "spacing_2mm": 2e-3,        # 2mm metalens stack
}

COMMON = dict(
    wavelength_m=1.55e-6,
    receiver_window_m=0.002048,    # 2.048mm → dx=2μm (metalens pixel pitch)
    aperture_diameter_m=0.002,     # 2mm (beam reducer output)
    n=1024,
    roi_n=512,
    num_layers=5,
    dual_2f_f1_m=1.0e-3,
    dual_2f_f2_m=1.0e-3,
    dual_2f_na1=0.16,
    dual_2f_na2=0.16,
    dual_2f_apply_scaling=False,
    phase_max=3.14159265,
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


def roi_window_m() -> float:
    return COMMON["receiver_window_m"] * (COMMON["roi_n"] / COMMON["n"])


def prepare_field(field: torch.Tensor, *, aperture_diameter_m: float) -> torch.Tensor:
    apertured = apply_receiver_aperture(
        field,
        receiver_window_m=COMMON["receiver_window_m"],
        aperture_diameter_m=aperture_diameter_m,
    )
    return center_crop_field(apertured, crop_n=COMMON["roi_n"])


def make_model(spacing_m: float, n: int) -> BeamCleanupFD2NN:
    return BeamCleanupFD2NN(
        n=n,
        wavelength_m=COMMON["wavelength_m"],
        window_m=roi_window_m(),
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
    window = roi_window_m()
    ap = COMMON["aperture_diameter_m"]
    all_co, all_pr, all_pr_full, all_ar, all_io, all_sr = [], [], [], [], [], []
    all_br, all_ee = [], []
    all_co_bl, all_io_bl, all_pr_full_bl, all_br_bl, all_ee_bl = [], [], [], [], []

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
        # baseline (no correction)
        all_co_bl.append(complex_overlap(inp, target).cpu())
        all_io_bl.append(gaussian_overlap(inp.abs().square(), tgt_i).cpu())
        all_pr_full_bl.append(full_field_phase_rmse(inp, target).cpu())
        all_br_bl.append(beam_radius(inp.abs().square(), window_m=window).cpu())
        all_ee_bl.append(encircled_energy_fraction(inp.abs().square(), reference_radius=ref_radius, window_m=window).cpu())

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
        "baseline_full_field_phase_rmse_rad": float(torch.cat(all_pr_full_bl).mean()),
        "baseline_beam_radius_m": float(torch.cat(all_br_bl).mean()),
        "baseline_encircled_energy": float(torch.cat(all_ee_bl).mean()),
    }


def train_one(name: str, spacing_m: float, device: torch.device) -> dict:
    run_dir = OUT_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(COMMON["seed"])
    np.random.seed(COMMON["seed"])

    train_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="train")
    val_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="val")
    test_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="test")

    n = COMMON["roi_n"]
    model = make_model(spacing_m, n).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=COMMON["lr"])

    train_loader = DataLoader(train_ds, batch_size=COMMON["batch_size"], shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=COMMON["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=COMMON["batch_size"])

    window = roi_window_m()
    ap = COMMON["aperture_diameter_m"]
    weights = COMMON["complex_weights"]

    history = []
    dx = COMMON["receiver_window_m"] / COMMON["n"]  # 2um
    nf = dx ** 2 / (COMMON["wavelength_m"] * spacing_m) if spacing_m > 0 else float("inf")
    print(f"\n{'='*60}")
    print(f"  {name}: spacing={spacing_m*1e3:.0f}mm, dx={dx*1e6:.0f}um, N_F={nf:.4f}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}")

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
            # Save phase snapshots for visualization
            phases = [layer.phase().detach().cpu().numpy() for layer in model.layers]
            np.save(run_dir / f"phases_epoch{epoch:03d}.npy", np.stack(phases))
        else:
            entry = {"epoch": epoch, "train_loss": avg_loss, "time_s": dt}

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
          f"full_pr={test_metrics['full_field_phase_rmse_rad']:.4f} "
          f"sr={test_metrics['strehl']:.4f}")
    print(f"  BASELINE: co={test_metrics['baseline_complex_overlap']:.4f} "
          f"io={test_metrics['baseline_intensity_overlap']:.4f}")

    # Save checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "history": history,
        "test_metrics": test_metrics,
        "config": {"name": name, "spacing_m": spacing_m, "fresnel_number": nf, **COMMON},
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

    return {"history": history, "test_metrics": test_metrics, "name": name}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    all_results = {}
    for name, spacing in SPACINGS.items():
        result = train_one(name, spacing, device)
        all_results[name] = result["test_metrics"]

    # Summary table
    print(f"\n{'='*70}")
    print("SWEEP SUMMARY")
    print(f"{'='*70}")
    dx = COMMON["receiver_window_m"] / COMMON["n"]
    print(f"  Metalens pixel pitch: dx={dx*1e6:.0f}um")
    print(f"\n{'Name':>15} | {'Spacing':>8} | {'N_F':>8} | {'COverlap':>9} | {'PhaseRMSE':>9} | {'Strehl':>7}")
    print("-" * 75)
    for name, spacing in SPACINGS.items():
        m = all_results[name]
        nf = dx**2 / (COMMON["wavelength_m"] * spacing) if spacing > 0 else float("inf")
        sp_str = f"{spacing*1e3:.0f}mm" if spacing > 0 else "0(FFT)"
        print(f"{name:>15} | {sp_str:>8} | {nf:>8.4f} | "
              f"{m['complex_overlap']:>9.4f} | {m['phase_rmse_rad']:>9.4f} | {m['strehl']:>7.4f}")
    bl = all_results["spacing_0mm"]
    print(f"\n  Baseline (no D2NN): co={bl['baseline_complex_overlap']:.4f}, "
          f"io={bl['baseline_intensity_overlap']:.4f}")

    with open(OUT_ROOT / "sweep_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUT_ROOT}")


if __name__ == "__main__":
    main()
