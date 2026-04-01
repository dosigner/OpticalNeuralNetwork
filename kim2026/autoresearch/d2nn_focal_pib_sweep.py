#!/usr/bin/env python
"""D2NN Focal-Plane PIB Sweep — Phase B.

Retrain D2NN with lens_2f_forward in the forward pass so that PIB/Strehl
are optimized at the actual focal plane (after f=4.5mm focusing lens),
NOT the D2NN output plane.

Metric plane separation:
  - CO, WF RMS → D2NN output plane (unitary theorem verification)
  - PIB, Strehl → focal plane (actual detector location)

4 loss strategies:
  1. focal_pib_only:        maximize focal-plane PIB@10μm
  2. focal_strehl_only:     maximize focal-plane Strehl ratio
  3. focal_intensity_overlap: Gaussian overlap of focal intensities
  4. focal_co_pib_hybrid:   D2NN-output CO + focal PIB combined

Usage:
    cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python -m autoresearch.d2nn_focal_pib_sweep
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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from kim2026.data.dataset import CachedFieldDataset
from kim2026.models.d2nn import BeamCleanupD2NN
from kim2026.optics import MAX_ALIAS_SAFE_DISTANCE_M, MIN_PAD_FACTOR
from kim2026.optics.lens_2f import lens_2f_forward
from kim2026.training.losses import beam_radius, encircled_energy_fraction
from kim2026.training.metrics import (
    complex_overlap,
    gaussian_overlap,
    strehl_ratio_correct,
)
from kim2026.training.targets import apply_receiver_aperture, center_crop_field

# ══════════════════════════════════════════════════════════════════════
KIM2026_ROOT = Path(__file__).resolve().parent.parent
WAVELENGTH_M = 1.55e-6
RECEIVER_WINDOW_M = 0.002048
APERTURE_DIAMETER_M = 0.002
N_FULL = 1024
ROI_N = 1024
SEED = 20260329

# Focusing lens parameters
FOCUS_F_M = 4.5e-3  # f=4.5mm focusing lens
PIB_BUCKET_RADIUS_UM = 10.0  # 10μm bucket (SMF coupling proxy)

DATA_DIR = KIM2026_ROOT / "data" / "kim2026" / "1km_cn2_5e-14_tel15cm_n1024_br75" / "cache"
MANIFEST = DATA_DIR.parent / "split_manifest.json"
OUT_ROOT = Path(__file__).resolve().parent / "runs" / "d2nn_focal_pib_sweep"

ARCH = dict(num_layers=5, layer_spacing_m=10e-3, detector_distance_m=10e-3)
ARCH["propagation_pad_factor"] = MIN_PAD_FACTOR

TRAIN = dict(
    lr=8e-3,           # linear scaling: 1e-3 × 32/2 × 0.5
    epochs=100,
    batch_size=32,
    tv_weight=0.05,
    warmup_epochs=10,
)
STREHL_PAD_FACTOR = 4
STREHL_BATCH_CHUNK = 4


def roi_window_m():
    return RECEIVER_WINDOW_M * (ROI_N / N_FULL)


def prepare_field(field: torch.Tensor) -> torch.Tensor:
    a = apply_receiver_aperture(field, receiver_window_m=RECEIVER_WINDOW_M,
                                 aperture_diameter_m=APERTURE_DIAMETER_M)
    return center_crop_field(a, crop_n=ROI_N)


def make_model() -> BeamCleanupD2NN:
    return BeamCleanupD2NN(n=ROI_N, wavelength_m=WAVELENGTH_M, window_m=roi_window_m(), **ARCH)


def corrected_strehl_batch(
    pred_field: torch.Tensor,
    *,
    pad_factor: int = STREHL_PAD_FACTOR,
    chunk_size: int = STREHL_BATCH_CHUNK,
) -> torch.Tensor:
    """Compute corrected Strehl in memory-safe chunks."""
    outputs = []
    for start in range(0, pred_field.shape[0], chunk_size):
        stop = start + chunk_size
        outputs.append(
            strehl_ratio_correct(
                pred_field[start:stop],
                pad_factor=pad_factor,
            )
        )
    return torch.cat(outputs, dim=0)


def _resolve_data_paths(data_path: str) -> tuple[Path, Path]:
    root = Path(data_path)
    if not root.is_absolute():
        root = KIM2026_ROOT / root
    if root.name == "cache":
        return root, root.parent / "split_manifest.json"
    return root / "cache", root / "split_manifest.json"


def apply_config_overrides(cfg: dict | None) -> None:
    """Apply YAML configuration values to module-level runtime parameters."""
    if not cfg:
        return

    global WAVELENGTH_M, RECEIVER_WINDOW_M, APERTURE_DIAMETER_M
    global FOCUS_F_M, PIB_BUCKET_RADIUS_UM, N_FULL, ROI_N
    global DATA_DIR, MANIFEST, SEED, OUT_ROOT, ARCH

    physics = cfg.get("physics", {})
    if "wavelength_m" in physics:
        WAVELENGTH_M = float(physics["wavelength_m"])
    if "receiver_window_m" in physics:
        RECEIVER_WINDOW_M = float(physics["receiver_window_m"])
    if "aperture_diameter_m" in physics:
        APERTURE_DIAMETER_M = float(physics["aperture_diameter_m"])
    if "focus_f_m" in physics:
        FOCUS_F_M = float(physics["focus_f_m"])
    if "pib_bucket_radius_um" in physics:
        PIB_BUCKET_RADIUS_UM = float(physics["pib_bucket_radius_um"])

    architecture = cfg.get("architecture", {})
    if "n_grid" in architecture:
        N_FULL = int(architecture["n_grid"])
    if "roi_n" in architecture:
        ROI_N = int(architecture["roi_n"])
    arch_overrides = {
        key: architecture[key]
        for key in ("num_layers", "layer_spacing_m", "detector_distance_m", "propagation_pad_factor")
        if key in architecture
    }
    if arch_overrides:
        ARCH = {**ARCH, **arch_overrides}

    training = cfg.get("training", {})
    if "seed" in training:
        SEED = int(training["seed"])
    train_overrides = {key: training[key] for key in TRAIN if key in training}
    if train_overrides:
        TRAIN.update(train_overrides)

    data = cfg.get("data", {})
    if "path" in data:
        DATA_DIR, MANIFEST = _resolve_data_paths(data["path"])

    output = cfg.get("output", {})
    if "dir" in output:
        out_root = Path(output["dir"])
        if not out_root.is_absolute():
            out_root = KIM2026_ROOT / out_root
        OUT_ROOT = out_root


def print_resolved_parameters(config_path: str | None) -> None:
    dx_in = roi_window_m() / ROI_N
    dx_focal = WAVELENGTH_M * FOCUS_F_M / (ROI_N * dx_in)
    print(f"\nD2NN Focal-Plane PIB Sweep")
    if config_path:
        print(f"  Config: {config_path}")
    print("  Resolved parameters:")
    print(f"    wavelength_m={WAVELENGTH_M:.6e}")
    print(f"    receiver_window_m={RECEIVER_WINDOW_M:.6e}")
    print(f"    aperture_diameter_m={APERTURE_DIAMETER_M:.6e}")
    print(f"    focus_f_m={FOCUS_F_M:.6e}")
    print(f"    pib_bucket_radius_um={PIB_BUCKET_RADIUS_UM:.3f}")
    print(f"    arch={ARCH}")
    print(f"    train={TRAIN}")
    print(f"    strehl_pad_factor={STREHL_PAD_FACTOR}")
    print(f"    alias_safe_distance_guard_m={MAX_ALIAS_SAFE_DISTANCE_M:.3f}")
    print(f"    data_dir={DATA_DIR}")
    print(f"    manifest={MANIFEST}")
    print(f"    out_root={OUT_ROOT}")
    print(f"  dx_in={dx_in*1e6:.1f}μm → dx_focal={dx_focal*1e6:.3f}μm")
    print(f"  Focal window: {ROI_N * dx_focal * 1e6:.0f}μm")
    print(f"  PIB bucket: {PIB_BUCKET_RADIUS_UM:.0f}μm = {PIB_BUCKET_RADIUS_UM*1e-6/dx_focal:.1f} pixels")
    print(f"  4 loss strategies × TV reg + cosine LR")


def to_focal_plane(field: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Propagate field through focusing lens to focal plane."""
    dx_in_m = roi_window_m() / ROI_N  # 2μm
    return lens_2f_forward(
        field,
        dx_in_m=dx_in_m,
        wavelength_m=WAVELENGTH_M,
        f_m=FOCUS_F_M,
        na=None,
        apply_scaling=False,
    )


def total_variation(model: BeamCleanupD2NN) -> torch.Tensor:
    """Total variation of all phase masks — encourages smoothness."""
    tv = torch.tensor(0.0, device=next(model.parameters()).device)
    for layer in model.layers:
        phase = layer.phase
        tv = tv + (phase[:, :-1] - phase[:, 1:]).abs().mean()
        tv = tv + (phase[:-1, :] - phase[1:, :]).abs().mean()
    return tv


def focal_pib_loss(focal_pred: torch.Tensor, focal_target: torch.Tensor,
                   dx_focal: float, radius_um: float = PIB_BUCKET_RADIUS_UM) -> torch.Tensor:
    """1 - PIB at focal plane with given bucket radius."""
    pred_i = focal_pred.abs().square()
    ref_radius = radius_um * 1e-6
    n = focal_pred.shape[-1]
    c = n // 2
    yy, xx = torch.meshgrid(torch.arange(n, device=focal_pred.device) - c,
                             torch.arange(n, device=focal_pred.device) - c, indexing="ij")
    r = torch.sqrt((xx * dx_focal) ** 2 + (yy * dx_focal) ** 2)
    mask = (r <= ref_radius).float()
    pib = (pred_i * mask).sum(dim=(-2, -1)) / pred_i.sum(dim=(-2, -1)).clamp(min=1e-12)
    return 1.0 - pib.mean()


def focal_strehl_loss(d2nn_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - corrected Strehl with a flat-phase raw-vacuum reference."""
    sr = corrected_strehl_batch(d2nn_pred)
    return 1.0 - sr.mean()


def focal_intensity_overlap_loss(focal_pred: torch.Tensor, focal_target: torch.Tensor) -> torch.Tensor:
    """1 - Gaussian overlap of focal-plane intensity patterns."""
    pred_i = focal_pred.abs().square()
    target_i = focal_target.abs().square()
    io = gaussian_overlap(pred_i, target_i)
    return 1.0 - io.mean()


def focal_co_pib_hybrid_loss(d2nn_pred: torch.Tensor, d2nn_target: torch.Tensor,
                             focal_pred: torch.Tensor, dx_focal: float) -> torch.Tensor:
    """CO at D2NN output + PIB at focal plane.

    CO is computed at D2NN output plane (unitary theorem plane).
    PIB is computed at focal plane (detector location).
    """
    co = complex_overlap(d2nn_pred, d2nn_target).mean()
    pib = focal_pib_loss(focal_pred, None, dx_focal)
    return (1.0 - co) + 0.5 * pib


# Loss config registry — each fn receives (d2nn_pred, d2nn_target, focal_pred, focal_target, dx_focal)
LOSS_CONFIGS = OrderedDict({
    "focal_pib_only": {
        "fn": lambda dp, dt, fp, ft, dx: focal_pib_loss(fp, ft, dx),
        "desc": "Focal-plane PIB@10μm loss",
    },
    "focal_strehl_only": {
        "fn": lambda dp, dt, fp, ft, dx: focal_strehl_loss(dp, dt),
        "desc": "Focal-plane Strehl ratio loss",
    },
    "focal_intensity_overlap": {
        "fn": lambda dp, dt, fp, ft, dx: focal_intensity_overlap_loss(fp, ft),
        "desc": "Focal-plane intensity overlap loss",
    },
    "focal_co_pib_hybrid": {
        "fn": lambda dp, dt, fp, ft, dx: focal_co_pib_hybrid_loss(dp, dt, fp, dx),
        "desc": "D2NN-output CO + focal PIB@10μm hybrid",
    },
})


def compute_focal_pib(focal_field: torch.Tensor, dx_focal: float,
                      radius_um: float) -> torch.Tensor:
    """Compute PIB at focal plane for given bucket radius."""
    intensity = focal_field.abs().square()
    n = focal_field.shape[-1]
    c = n // 2
    ref_radius = radius_um * 1e-6
    yy, xx = torch.meshgrid(torch.arange(n, device=focal_field.device) - c,
                             torch.arange(n, device=focal_field.device) - c, indexing="ij")
    r = torch.sqrt((xx * dx_focal) ** 2 + (yy * dx_focal) ** 2)
    mask = (r <= ref_radius).float()
    pib = (intensity * mask).sum(dim=(-2, -1)) / intensity.sum(dim=(-2, -1)).clamp(min=1e-12)
    return pib


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Evaluate with separate metric planes:
    - CO, IO (output plane) → D2NN output
    - PIB, Strehl → focal plane (after lens)
    """
    model.eval()

    # D2NN output plane metrics
    all_co, all_io_output = [], []
    all_co_bl = []
    # Focal plane metrics
    all_pib_10, all_pib_50 = [], []
    all_pib_10_bl, all_pib_50_bl = [], []
    all_pib_10_vac, all_pib_50_vac = [], []
    all_strehl_focal = []

    for batch in loader:
        u_turb = batch["u_turb"].to(device)
        u_vac = batch["u_vacuum"].to(device)
        target = prepare_field(u_vac)
        inp = prepare_field(u_turb)
        d2nn_pred = model(inp)

        # --- D2NN output plane metrics ---
        pred_i = d2nn_pred.abs().square()
        target_i = target.abs().square()
        all_co.append(complex_overlap(d2nn_pred, target).cpu())
        all_io_output.append(gaussian_overlap(pred_i, target_i).cpu())
        all_co_bl.append(complex_overlap(inp, target).cpu())

        # --- Focal plane metrics (compute and release immediately) ---
        focal_pred, dx_focal = to_focal_plane(d2nn_pred)
        all_pib_10.append(compute_focal_pib(focal_pred, dx_focal, 10.0).cpu())
        all_pib_50.append(compute_focal_pib(focal_pred, dx_focal, 50.0).cpu())
        del focal_pred

        focal_target, _ = to_focal_plane(target)
        all_pib_10_vac.append(compute_focal_pib(focal_target, dx_focal, 10.0).cpu())
        all_pib_50_vac.append(compute_focal_pib(focal_target, dx_focal, 50.0).cpu())
        all_strehl_focal.append(corrected_strehl_batch(d2nn_pred).cpu())
        del focal_target

        focal_inp, _ = to_focal_plane(inp)
        all_pib_10_bl.append(compute_focal_pib(focal_inp, dx_focal, 10.0).cpu())
        all_pib_50_bl.append(compute_focal_pib(focal_inp, dx_focal, 50.0).cpu())
        del focal_inp

        torch.cuda.empty_cache()

    return {
        # D2NN output plane
        "co_output": float(torch.cat(all_co).mean()),
        "io_output": float(torch.cat(all_io_output).mean()),
        "co_baseline": float(torch.cat(all_co_bl).mean()),
        # Focal plane
        "focal_pib_10um": float(torch.cat(all_pib_10).mean()),
        "focal_pib_50um": float(torch.cat(all_pib_50).mean()),
        "focal_pib_10um_baseline": float(torch.cat(all_pib_10_bl).mean()),
        "focal_pib_50um_baseline": float(torch.cat(all_pib_50_bl).mean()),
        "focal_pib_10um_vacuum": float(torch.cat(all_pib_10_vac).mean()),
        "focal_pib_50um_vacuum": float(torch.cat(all_pib_50_vac).mean()),
        "focal_strehl": float(torch.cat(all_strehl_focal).mean()),
        "dx_focal_um": dx_focal * 1e6,
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
    """Compute intensity-weighted WF RMS over test set at D2NN output plane."""
    model.eval()
    d0 = make_model().to(device); d0.eval()
    all_wf, all_wf_bl = [], []
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
            p_ph = torch.angle(pred[b])
            t_ph = torch.angle(uv_d[b])
            diff = torch.remainder(p_ph - t_ph + math.pi, 2*math.pi) - math.pi
            w = uv_d[b].abs().square(); w = w / w.sum()
            all_wf.append(torch.sqrt((w * diff.square()).sum()).item())
            p_ph2 = torch.angle(ut_d[b])
            diff2 = torch.remainder(p_ph2 - t_ph + math.pi, 2*math.pi) - math.pi
            all_wf_bl.append(torch.sqrt((w * diff2.square()).sum()).item())
    del d0; torch.cuda.empty_cache()
    return np.mean(all_wf), np.mean(all_wf_bl)


def train_one(name, loss_config, train_loader, val_loader, test_loader, device, sanity_cfg=None):
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
    loss_fn = loss_config["fn"]

    print(f"\n{'='*70}")
    print(f"  [{name}] {loss_config['desc']}")
    print(f"  TV={TRAIN['tv_weight']}, LR={TRAIN['lr']}→cosine, epochs={TRAIN['epochs']}")
    print(f"  Focus lens f={FOCUS_F_M*1e3:.1f}mm, PIB bucket={PIB_BUCKET_RADIUS_UM:.0f}μm")
    print(f"{'='*70}")

    t_start = time.time()
    history = {"epoch": [], "loss": [], "val_co": [], "val_focal_pib_10": [], "lr": []}

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

            # Forward: D2NN → lens → focal plane
            d2nn_pred = model(inp)
            focal_pred, dx_focal = to_focal_plane(d2nn_pred)
            focal_target, _ = to_focal_plane(target)

            loss = loss_fn(d2nn_pred, target, focal_pred, focal_target, dx_focal)

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

        # Log every epoch to file for monitoring
        log_file = run_dir / "epoch_log.txt"
        with open(log_file, "a") as lf:
            lf.write(f"ep{epoch:3d} loss={avg_loss:.5f} lr={cur_lr:.2e} {dt:.1f}s\n")

        if epoch % 50 == 0 or epoch == TRAIN["epochs"] - 1:
            with open(log_file, "a") as lf:
                lf.write(f"  -> starting eval at epoch {epoch}...\n")
            val_m = evaluate(model, val_loader, device)
            history["epoch"].append(epoch)
            history["loss"].append(avg_loss)
            history["val_co"].append(val_m["co_output"])
            history["val_focal_pib_10"].append(val_m["focal_pib_10um"])
            history["lr"].append(cur_lr)
            msg = (f"  Epoch {epoch:3d}/{TRAIN['epochs']-1} | loss={avg_loss:.5f} | "
                   f"co={val_m['co_output']:.4f} | "
                   f"fpib10={val_m['focal_pib_10um']:.4f} | "
                   f"fpib50={val_m['focal_pib_50um']:.4f} | "
                   f"fstrehl={val_m['focal_strehl']:.4f} | "
                   f"lr={cur_lr:.2e} | {dt:.1f}s")
            print(msg, flush=True)
            with open(log_file, "a") as lf:
                lf.write(msg + "\n")

    train_time = time.time() - t_start

    # Test evaluation
    test_m = evaluate(model, test_loader, device)
    tp = throughput_check(model, test_loader, device)
    wf_rms_d2nn, wf_rms_bl = wf_rms_eval(model, test_loader, device)

    # Save phases
    phases = [layer.phase.detach().cpu().numpy() for layer in model.layers]
    np.save(run_dir / "phases_wrapped.npy", np.stack(phases))

    sanity_results = []
    if sanity_cfg is None or sanity_cfg.get("enabled", True):
        from kim2026.eval.sanity_check import run_post_training_checks

        post_report = run_post_training_checks(
            model,
            test_loader,
            device=str(device),
            config=sanity_cfg or {},
        )
        if hasattr(post_report, "to_dict"):
            sanity_results = post_report.to_dict()
        if not getattr(post_report, "all_passed", True):
            print(f"WARNING: Post-training sanity checks failed for {name}.")

    result = {
        "name": name,
        "description": loss_config["desc"],
        "arch": ARCH,
        "focus_f_mm": FOCUS_F_M * 1e3,
        "pib_bucket_radius_um": PIB_BUCKET_RADIUS_UM,
        "propagation_pad_factor": ARCH["propagation_pad_factor"],
        "strehl_pad_factor": STREHL_PAD_FACTOR,
        **test_m,
        "throughput": tp,
        "wf_rms_rad": wf_rms_d2nn,
        "wf_rms_baseline_rad": wf_rms_bl,
        "wf_rms_nm": wf_rms_d2nn * WAVELENGTH_M / (2 * math.pi) * 1e9,
        "wf_rms_baseline_nm": wf_rms_bl * WAVELENGTH_M / (2 * math.pi) * 1e9,
        "training_seconds": train_time,
        "sanity_checks": sanity_results,
        "history": history,
    }
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
    torch.save({"model_state_dict": model.state_dict()}, run_dir / "checkpoint.pt")

    # Report
    pib10_delta = test_m["focal_pib_10um"] - test_m["focal_pib_10um_baseline"]
    pib10_vs_vac = test_m["focal_pib_10um"] / max(test_m["focal_pib_10um_vacuum"], 1e-12) * 100
    co_delta = test_m["co_output"] - test_m["co_baseline"]
    wf_delta = (wf_rms_bl - wf_rms_d2nn) / max(wf_rms_bl, 1e-12) * 100

    print(f"\n  TEST RESULTS (metric plane separation):")
    print(f"    === D2NN Output Plane ===")
    print(f"    CO:     {test_m['co_baseline']:.4f} → {test_m['co_output']:.4f} ({co_delta:+.4f})")
    print(f"    IO:     {test_m['io_output']:.4f}")
    print(f"    WF RMS: {wf_rms_bl*WAVELENGTH_M/(2*math.pi)*1e9:.1f} → {wf_rms_d2nn*WAVELENGTH_M/(2*math.pi)*1e9:.1f} nm ({wf_delta:+.1f}%)")
    print(f"    === Focal Plane (f={FOCUS_F_M*1e3:.1f}mm) ===")
    print(f"    PIB@10μm: {test_m['focal_pib_10um_baseline']:.4f} → {test_m['focal_pib_10um']:.4f} "
          f"(Δ={pib10_delta:+.4f}, {pib10_vs_vac:.1f}% of vacuum={test_m['focal_pib_10um_vacuum']:.4f})")
    print(f"    PIB@50μm: {test_m['focal_pib_50um_baseline']:.4f} → {test_m['focal_pib_50um']:.4f} "
          f"(vacuum={test_m['focal_pib_50um_vacuum']:.4f})")
    print(f"    Strehl:   {test_m['focal_strehl']:.4f}")
    print(f"    dx_focal: {test_m['dx_focal_um']:.3f} μm")
    print(f"    TP:       {tp:.4f}")

    return result


def load_config(config_path: str | None) -> dict | None:
    """Load YAML config if provided, otherwise return None (use defaults)."""
    if config_path is None:
        return None
    try:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    except ImportError:
        print("WARNING: pyyaml not installed, using hardcoded defaults")
        return None


def main():
    parser = argparse.ArgumentParser(description="D2NN Focal-Plane PIB Sweep")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_config_overrides(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print_resolved_parameters(args.config)

    train_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="train")
    val_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="val")
    test_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="test")
    print(f"  Data: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=TRAIN["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=16, num_workers=0)

    # ── Pre-training sanity checks ──
    sanity_cfg = cfg.get("sanity_checks", {}) if cfg else {}
    if sanity_cfg.get("enabled", True):
        try:
            from kim2026.eval.sanity_check import run_pre_training_checks
            pre_report = run_pre_training_checks(test_loader, device=str(device), config=sanity_cfg)
            if not pre_report.all_passed:
                print("WARNING: Pre-training sanity checks failed. Proceeding anyway.")
        except ImportError:
            print("  (sanity_check module not available, skipping pre-checks)")

    all_results = []
    for name, config in LOSS_CONFIGS.items():
        result = train_one(
            name,
            config,
            train_loader,
            val_loader,
            test_loader,
            device,
            sanity_cfg,
        )
        all_results.append(result)

    # Summary
    dx_in = roi_window_m() / ROI_N
    dx_focal = WAVELENGTH_M * FOCUS_F_M / (ROI_N * dx_in)
    print(f"\n{'='*120}")
    print("FOCAL-PLANE PIB SWEEP SUMMARY")
    print(f"Focus lens f={FOCUS_F_M*1e3:.1f}mm | PIB bucket={PIB_BUCKET_RADIUS_UM:.0f}μm | dx_focal={dx_focal*1e6:.3f}μm")
    print(f"{'='*120}")
    header = (f"{'Name':>25} | {'CO(out)':>7} | {'fPIB10':>7} | {'fPIB50':>7} | "
              f"{'fStrehl':>7} | {'WF nm':>7} | {'TP':>6} | {'vs vac%':>7}")
    print(header)
    print("-" * len(header))

    for r in all_results:
        vs_vac = r["focal_pib_10um"] / max(r["focal_pib_10um_vacuum"], 1e-12) * 100
        print(f"{r['name']:>25} | {r['co_output']:>7.4f} | {r['focal_pib_10um']:>7.4f} | "
              f"{r['focal_pib_50um']:>7.4f} | {r['focal_strehl']:>7.4f} | "
              f"{r['wf_rms_nm']:>7.1f} | {r['throughput']:>6.4f} | {vs_vac:>6.1f}%")

    print(f"\nBaselines:")
    r0 = all_results[0]
    print(f"  Turbulent: CO={r0['co_baseline']:.4f}, fPIB@10μm={r0['focal_pib_10um_baseline']:.4f}, fPIB@50μm={r0['focal_pib_50um_baseline']:.4f}")
    print(f"  Vacuum:    fPIB@10μm={r0['focal_pib_10um_vacuum']:.4f}, fPIB@50μm={r0['focal_pib_50um_vacuum']:.4f}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUT_ROOT / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUT_ROOT}")


if __name__ == "__main__":
    main()
