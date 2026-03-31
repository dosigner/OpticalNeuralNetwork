#!/usr/bin/env python
"""FD2NN autoresearch experiment — the single file the agent modifies.

Trains one FD2NN model with the specified hyperparameters, evaluates it,
and prints results in a parseable format. Imports from the immutable
kim2026 physics/optics modules.

Usage:
    cd /root/dj/D2NN/kim2026 && python -m autoresearch.experiment
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
from kim2026.training.losses import (
    complex_field_loss,
    beam_radius,
    encircled_energy_fraction,
)
from kim2026.training.metrics import (
    amplitude_rmse,
    complex_overlap,
    full_field_phase_rmse,
    gaussian_overlap,
    out_of_support_energy_fraction,
    phase_rmse,
    strehl_ratio,
)
from kim2026.training.targets import apply_receiver_aperture, center_crop_field

# ══════════════════════════════════════════════════════════════════════
# IMMUTABLE CONSTANTS — must match data generation, DO NOT CHANGE
# ══════════════════════════════════════════════════════════════════════
WAVELENGTH_M = 1.55e-6
RECEIVER_WINDOW_M = 0.002048       # 2.048mm → dx=2μm at n=1024
APERTURE_DIAMETER_M = 0.002        # 2mm (beam reducer output)
N_FULL = 1024                      # full grid (must match data)
ROI_N = 1024                       # no crop — full grid to avoid aliasing
# Thorlabs AC127-025-C: f=25mm, Ø12.7mm, C-coat (1050-1700nm)
DUAL_2F_F1_M = 25.0e-3            # forward lens focal length
DUAL_2F_F2_M = 25.0e-3            # inverse lens focal length (MUST == F1)
DUAL_2F_NA1 = 0.254                # forward lens NA = 12.7/(2×25)
DUAL_2F_NA2 = 0.254                # inverse lens NA
DUAL_2F_APPLY_SCALING = False      # must be False with ortho FFT
SEED = 20260323

# Data paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kim2026" / "1km_cn2e-14_w2m_n1024_dx2mm" / "cache"
MANIFEST = DATA_DIR.parent / "split_manifest.json"
OUT_DIR = Path(__file__).resolve().parent / "runs"

# ══════════════════════════════════════════════════════════════════════
# TUNABLE HYPERPARAMETERS — the agent modifies these freely
# ══════════════════════════════════════════════════════════════════════

# --- Architecture ---
NUM_LAYERS = 5
LAYER_SPACING_M = 5.0e-3           # inter-layer distance (5mm, max 50e-3)
PHASE_MAX = math.pi                # max phase excursion for constrained modes
PHASE_CONSTRAINT = "unconstrained" # "symmetric_tanh", "sigmoid", "unconstrained"
PHASE_INIT = "uniform"             # "zeros" or "uniform"
PHASE_INIT_SCALE = 0.1             # init range for uniform

# --- Optimizer ---
LR = 5e-4
OPTIMIZER = "adam"                 # "adam" or "adamw"
WEIGHT_DECAY = 0.0                 # only used with adamw
GRAD_CLIP_NORM = 0.0              # 0 = no clipping

# --- Scheduler ---
SCHEDULER = "none"                 # "none", "cosine", "cosine_warm_restarts"
COSINE_T0 = 20                     # for cosine_warm_restarts
COSINE_ETA_MIN = 1e-6

# --- Training ---
EPOCHS = 30
BATCH_SIZE = 2

# --- Loss ---
LOSS_MODE = "composite"            # "composite" or "roi_complex"
LOSS_WEIGHTS = {
    "complex_overlap": 1.0,
    "amplitude_mse": 0.5,
    "intensity_overlap": 0.0,
    "beam_radius": 0.0,
    "encircled_energy": 0.0,
    "phasor_mse": 0.0,
    "soft_phasor": 0.0,
    "leakage": 0.0,
    "support_gamma": 2.0,
    "full_field_phase": 0.0,
    "full_field_phase_gamma": 1.0,
    "full_field_phase_threshold": 0.05,
}
# ROI mode params (only used when LOSS_MODE == "roi_complex")
ROI_THRESHOLD = 0.9
ROI_LEAKAGE_WEIGHT = 1.0

# --- Phasor smoothness regularization ---
PHASOR_SMOOTHNESS_WEIGHT = 0.01   # 0 = disabled


# ══════════════════════════════════════════════════════════════════════
# IMPLEMENTATION — agent may modify training logic below
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
        num_layers=NUM_LAYERS,
        layer_spacing_m=LAYER_SPACING_M,
        phase_max=PHASE_MAX,
        phase_constraint=PHASE_CONSTRAINT,
        phase_init=PHASE_INIT,
        phase_init_scale=PHASE_INIT_SCALE,
        dual_2f_f1_m=DUAL_2F_F1_M,
        dual_2f_f2_m=DUAL_2F_F2_M,
        dual_2f_na1=DUAL_2F_NA1,
        dual_2f_na2=DUAL_2F_NA2,
        dual_2f_apply_scaling=DUAL_2F_APPLY_SCALING,
    )


def phasor_smoothness_loss(model: nn.Module) -> torch.Tensor:
    """Phase smoothness on circular manifold: |exp(jφ_i) - exp(jφ_j)|².

    2π-periodic: correctly ignores full-wrap jumps.
    Penalizes only local phase discontinuities between adjacent pixels.
    """
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for layer in model.layers:
        phase = layer.phase()
        phasor = torch.exp(1j * phase.to(torch.float32))
        grad_x = phasor[:, 1:] - phasor[:, :-1]
        grad_y = phasor[1:, :] - phasor[:-1, :]
        total = total + (grad_x.abs().pow(2).mean() + grad_y.abs().pow(2).mean()) / 2
    return total / len(model.layers)


def compute_loss(pred: torch.Tensor, target: torch.Tensor, model: nn.Module) -> torch.Tensor:
    window = roi_window_m()
    if LOSS_MODE == "roi_complex":
        from kim2026.training.losses import roi_complex_loss
        loss = roi_complex_loss(
            pred, target,
            roi_threshold=ROI_THRESHOLD,
            leakage_weight=ROI_LEAKAGE_WEIGHT,
            window_m=window,
        )
    else:
        loss = complex_field_loss(pred, target, weights=LOSS_WEIGHTS, window_m=window)

    if PHASOR_SMOOTHNESS_WEIGHT > 0:
        loss = loss + PHASOR_SMOOTHNESS_WEIGHT * phasor_smoothness_loss(model)

    return loss


def make_optimizer(model: nn.Module):
    if OPTIMIZER == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    return torch.optim.Adam(model.parameters(), lr=LR)


def make_scheduler(optimizer):
    if SCHEDULER == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=COSINE_ETA_MIN,
        )
    if SCHEDULER == "cosine_warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=COSINE_T0, eta_min=COSINE_ETA_MIN,
        )
    return None


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    window = roi_window_m()
    ap = APERTURE_DIAMETER_M
    all_co, all_pr, all_pr_full, all_ar = [], [], [], []
    all_io, all_sr, all_br, all_ee = [], [], [], []
    all_leak = []
    all_co_bl, all_io_bl = [], []

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
        pred_i = pred.abs().square()
        tgt_i = target.abs().square()
        all_io.append(gaussian_overlap(pred_i, tgt_i).cpu())
        all_sr.append(strehl_ratio(pred_i, tgt_i).cpu())
        ref_radius = beam_radius(tgt_i, window_m=window)
        all_br.append(beam_radius(pred_i, window_m=window).cpu())
        all_ee.append(encircled_energy_fraction(
            pred_i, reference_radius=ref_radius, window_m=window,
        ).cpu())
        all_leak.append(out_of_support_energy_fraction(pred, target).cpu())
        # baseline
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
        "out_of_support_energy_fraction": float(torch.cat(all_leak).mean()),
        "baseline_complex_overlap": float(torch.cat(all_co_bl).mean()),
        "baseline_intensity_overlap": float(torch.cat(all_io_bl).mean()),
    }


@torch.no_grad()
def throughput_check(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Verify energy conservation: sum|out|^2 / sum|in|^2 should be ~1.0."""
    model.eval()
    total_in, total_out = 0.0, 0.0
    for batch in loader:
        u_turb = batch["u_turb"].to(device)
        inp = prepare_field(u_turb)
        pred = model(inp)
        total_in += inp.abs().square().sum().item()
        total_out += pred.abs().square().sum().item()
    return total_out / max(total_in, 1e-12)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # --- Data ---
    train_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="train")
    val_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="val")
    test_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="test")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # --- Model ---
    model = make_model().to(device)
    optimizer = make_optimizer(model)
    scheduler = make_scheduler(optimizer)
    num_params = sum(p.numel() for p in model.parameters())

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Params: {num_params:,}")
    print(f"Config: layers={NUM_LAYERS}, spacing={LAYER_SPACING_M*1e3:.1f}mm, "
          f"constraint={PHASE_CONSTRAINT}, lr={LR}, epochs={EPOCHS}")

    # --- Train ---
    t_start = time.time()
    best_val_co = 0.0

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
            loss = compute_loss(pred, target, model)
            loss.backward()

            if GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            optimizer.step()
            epoch_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        dt = time.time() - t0

        # Validate every 5 epochs + first + last
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            val_m = evaluate(model, val_loader, device)
            co = val_m["complex_overlap"]
            if co > best_val_co:
                best_val_co = co
            print(f"Epoch {epoch:3d}/{EPOCHS-1} | loss={avg_loss:.5f} | "
                  f"co={co:.4f} | pr={val_m['phase_rmse_rad']:.4f} | {dt:.1f}s")
        else:
            print(f"Epoch {epoch:3d}/{EPOCHS-1} | loss={avg_loss:.5f} | {dt:.1f}s")

    training_seconds = time.time() - t_start

    # --- Test ---
    test_metrics = evaluate(model, test_loader, device)
    tp = throughput_check(model, test_loader, device)

    # --- VRAM ---
    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / 1024**2

    # --- Save ---
    run_dir = OUT_DIR / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save wrapped phase for fabrication view
    phases_wrapped = []
    phases_raw = []
    for layer in model.layers:
        phases_wrapped.append(layer.wrapped_phase().detach().cpu().numpy())
        phases_raw.append(layer.phase().detach().cpu().numpy())
    np.save(run_dir / "phases_wrapped.npy", np.stack(phases_wrapped))
    np.save(run_dir / "phases_raw.npy", np.stack(phases_raw))

    torch.save({
        "model_state_dict": model.state_dict(),
        "test_metrics": test_metrics,
        "throughput": tp,
    }, run_dir / "checkpoint.pt")

    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump({**test_metrics, "throughput": tp}, f, indent=2)

    # --- Parseable output ---
    print("\n---")
    print(f"complex_overlap:   {test_metrics['complex_overlap']:.6f}")
    print(f"intensity_overlap: {test_metrics['intensity_overlap']:.6f}")
    print(f"strehl:            {test_metrics['strehl']:.6f}")
    print(f"phase_rmse_rad:    {test_metrics['phase_rmse_rad']:.6f}")
    print(f"full_field_phase_rmse_rad: {test_metrics['full_field_phase_rmse_rad']:.6f}")
    print(f"amplitude_rmse:    {test_metrics['amplitude_rmse']:.6f}")
    print(f"encircled_energy:  {test_metrics['encircled_energy']:.6f}")
    print(f"out_of_support_energy_fraction: {test_metrics['out_of_support_energy_fraction']:.6f}")
    print(f"throughput:        {tp:.6f}")
    print(f"training_seconds:  {training_seconds:.1f}")
    print(f"peak_vram_mb:      {peak_vram_mb:.1f}")
    print(f"num_params:        {num_params}")
    print(f"num_layers:        {NUM_LAYERS}")
    print(f"baseline_co:       {test_metrics['baseline_complex_overlap']:.6f}")

    # --- Sanity checks ---
    warnings = []
    if tp < 0.95 or tp > 1.05:
        warnings.append(f"WARN: throughput={tp:.4f} outside [0.95, 1.05] — energy conservation violated")
    if test_metrics["strehl"] > 1.05:
        warnings.append(f"WARN: strehl={test_metrics['strehl']:.4f} > 1.05 — passive system cannot amplify")
    if warnings:
        print("\n" + "\n".join(warnings))


if __name__ == "__main__":
    main()
