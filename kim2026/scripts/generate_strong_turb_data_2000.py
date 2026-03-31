#!/usr/bin/env python
"""Generate 10000 D2NN training realizations (extend from existing 500).

Same physics as generate_strong_turb_data.py:
  TX (0.3mrad) → 1km → 15cm telescope → 75:1 beam reducer → D2NN (2mm)
  Cn² = 5e-14, D/r₀ ≈ 5.0

Split: 8000 train / 1000 val / 1000 test
Existing 500 files are reused (skipped), only 9500 new generated.

Usage:
    cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python scripts/generate_strong_turb_data_2000.py
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from kim2026.data.npz_pairs import write_pair_npz
from kim2026.fso.config import SimulationConfig
from kim2026.fso.atmosphere import compute_atmospheric_params
from kim2026.fso.sampling import analyze_sampling
from kim2026.fso.phase_screen import generate_phase_screens
from kim2026.fso.propagation import make_gaussian_source, ang_spec_multi_prop, compute_irradiance

# ─── Physical parameters (unchanged) ─────────────────────────
WAVELENGTH_M = 1.55e-6
THETA_DIV = 3.0e-4
PATH_LENGTH_M = 1000.0
CN2 = 5.0e-14

TELESCOPE_DIAMETER_M = 0.150
BEAM_REDUCER_RATIO = 75
FDNN_WINDOW_M = 0.002048

RECEIVER_WINDOW_M = FDNN_WINDOW_M * BEAM_REDUCER_RATIO
RECEIVER_DX_M = RECEIVER_WINDOW_M / 1024

# ─── Changed parameters ──────────────────────────────────────
N_REALIZATIONS = 10000       # was 500
SEED = 20260327              # same seed base for reproducibility

# Split: 8000/1000/1000
TRAIN_END = 8000
VAL_END = 9000

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "kim2026" / "1km_cn2_5e-14_tel15cm_n1024_br75"
CACHE_DIR = OUT_DIR / "cache"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Check GPU memory
    if device == "cuda":
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {total_mem:.1f} GB")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ─── Setup ────────────────────────────────────────────────
    config = SimulationConfig(
        Dz=PATH_LENGTH_M, Cn2=CN2, theta_div=THETA_DIV,
        D_roi=RECEIVER_WINDOW_M, delta_n=RECEIVER_DX_M,
        D_aperture=TELESCOPE_DIAMETER_M, n_reals=N_REALIZATIONS, wvl=WAVELENGTH_M,
    )

    sampling = analyze_sampling(config)
    delta1 = sampling.delta1
    delta_n = sampling.delta_n
    N = sampling.N
    n_scr = sampling.n_scr
    z_planes = sampling.z_planes
    delta_values = sampling.delta_values

    atm = compute_atmospheric_params(config.k, CN2, PATH_LENGTH_M)
    r0 = atm["r0_pw"]

    print(f"N={N}, n_scr={n_scr}, delta1={delta1*1e6:.1f}μm, delta_n={delta_n*1e6:.1f}μm")
    print(f"r₀={r0*1e2:.1f}cm, D/r₀={TELESCOPE_DIAMETER_M/r0:.2f}")
    print(f"VRAM per realization: ~{N**2 * 16 * 3 / 1e9 + N**2 * 8 * n_scr / 1e9:.1f} GB")
    print(f"Target: {N_REALIZATIONS} realizations ({TRAIN_END}/{VAL_END-TRAIN_END}/{N_REALIZATIONS-VAL_END} split)")

    # ─── r0 optimization ──────────────────────────────────────
    from kim2026.fso.atmosphere import optimize_screen_r0
    r0_vals = optimize_screen_r0(atm["r0_sw"], atm["sigma2_chi_sw"], config.k, PATH_LENGTH_M, n_scr)

    # ─── Vacuum propagation (once) ────────────────────────────
    print("\nVacuum propagation...")
    U_in = make_gaussian_source(N, delta1, config.w0, device=device)
    _, _, U_vac = ang_spec_multi_prop(
        U_in, WAVELENGTH_M, delta1, delta_n, z_planes,
        phase_screens=None, device=device,
    )

    CROP_N = 1024
    c = N // 2
    half = CROP_N // 2
    U_vac_crop = U_vac[c - half:c + half, c - half:c + half]
    crop_window_m = CROP_N * delta_n
    U_vac_save = U_vac_crop.detach().cpu()
    x_m = (np.arange(CROP_N) - CROP_N // 2) * delta_n
    y_m = x_m.copy()

    I_vac = compute_irradiance(U_vac)
    I_vac_np = I_vac.cpu().numpy()
    I_norm = I_vac_np / I_vac_np.max()
    r_px = math.sqrt((I_norm > 1/math.e**2).sum() / math.pi)
    print(f"Vacuum 1/e² radius: {r_px:.0f}px = {r_px*delta_n*1e3:.1f}mm")

    # ─── Generate realizations ────────────────────────────────
    print(f"\nGenerating {N_REALIZATIONS} realizations...")
    split_files = {"train": [], "val": [], "test": []}
    n_skipped = 0
    n_generated = 0
    t0 = time.time()

    for i in range(N_REALIZATIONS):
        filename = f"realization_{i:05d}.npz"
        if i < TRAIN_END:
            split_key = "train"
        elif i < VAL_END:
            split_key = "val"
        else:
            split_key = "test"

        if (CACHE_DIR / filename).exists():
            split_files[split_key].append(filename)
            n_skipped += 1
            continue

        torch.manual_seed(SEED + i * 1009)
        screens = generate_phase_screens(
            r0_vals.tolist(), N, delta_values, device=device,
        )
        _, _, U_turb_full = ang_spec_multi_prop(
            U_in, WAVELENGTH_M, delta1, delta_n, z_planes,
            phase_screens=screens, device=device,
        )
        U_turb = U_turb_full[c - half:c + half, c - half:c + half]

        write_pair_npz(
            CACHE_DIR / filename,
            u_vacuum=torch.from_numpy(U_vac_save.numpy()),
            u_turb=U_turb.detach().cpu(),
            x_m=x_m, y_m=y_m,
            metadata={
                "realization": i,
                "Dz": PATH_LENGTH_M, "Cn2": CN2,
                "theta_div": THETA_DIV, "wvl": WAVELENGTH_M,
                "delta_n": delta_n, "N": int(N),
                "crop_n": CROP_N,
                "receiver_window_m": float(crop_window_m),
                "telescope_diameter_m": TELESCOPE_DIAMETER_M,
                "beam_reducer_ratio": BEAM_REDUCER_RATIO,
            },
        )
        split_files[split_key].append(filename)
        n_generated += 1

        if n_generated % 50 == 0:
            elapsed = time.time() - t0
            rate = n_generated / elapsed
            remaining = (N_REALIZATIONS - n_skipped - n_generated) / max(rate, 0.01)
            mem_used = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0
            print(f"  [{i+1}/{N_REALIZATIONS}] generated={n_generated}, skipped={n_skipped} | "
                  f"{rate:.1f} real/s | ETA {remaining:.0f}s | VRAM {mem_used:.1f}GB")

    # ─── Save manifest ────────────────────────────────────────
    manifest_path = OUT_DIR / "split_manifest.json"
    # Backup old manifest
    if manifest_path.exists():
        import shutil
        shutil.copy(manifest_path, OUT_DIR / "split_manifest_500.json.bak")

    with open(manifest_path, "w") as f:
        json.dump(split_files, f, indent=2)

    total_time = time.time() - t0

    # ─── Summary ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("DATA GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {N_REALIZATIONS} realizations")
    print(f"  Skipped (existing): {n_skipped}")
    print(f"  Generated (new):    {n_generated}")
    print(f"Split: train={len(split_files['train'])}, val={len(split_files['val'])}, test={len(split_files['test'])}")
    print(f"Time: {total_time:.0f}s ({n_generated/(total_time+1e-6):.1f} real/s)")
    print(f"Saved to: {CACHE_DIR}")
    print(f"Manifest: {manifest_path}")
    sz_gb = sum(f.stat().st_size for f in CACHE_DIR.iterdir() if f.suffix == ".npz") / 1e9
    print(f"Total disk: {sz_gb:.1f} GB")


if __name__ == "__main__":
    main()
