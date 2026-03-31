#!/usr/bin/env python
"""Generate D2NN training data with STRONGER turbulence (Cn²=5e-14).

Physical system:
  TX (0.3mrad full div) → 1km → 15cm telescope → 75:1 beam reducer → D2NN (2mm)

Key difference from generate_telescope_data.py:
  Cn² = 5e-14 (5x stronger) → D/r₀ ≈ 4.4 → meaningful wavefront distortion
  500 realizations (400 train / 50 val / 50 test)

Usage:
    cd /root/dj/D2NN/kim2026 && python scripts/generate_strong_turb_data.py
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

# ─── Physical parameters ──────────────────────────────────────
WAVELENGTH_M = 1.55e-6
THETA_DIV = 3.0e-4            # full divergence 0.3 mrad
PATH_LENGTH_M = 1000.0        # 1 km
CN2 = 5.0e-14                 # STRONGER turbulence (was 1e-14)

# Telescope: 15cm aperture → 75:1 → 2.048mm D2NN window
TELESCOPE_DIAMETER_M = 0.150  # 15cm
BEAM_REDUCER_RATIO = 75
FDNN_WINDOW_M = 0.002048      # 2.048mm

# Derived: receiver grid = telescope aperture
RECEIVER_WINDOW_M = FDNN_WINDOW_M * BEAM_REDUCER_RATIO  # 153.6mm
RECEIVER_DX_M = RECEIVER_WINDOW_M / 1024                # 150μm

N_REALIZATIONS = 500
SEED = 20260327

# Output
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "kim2026" / "1km_cn2_5e-14_tel15cm_n1024_br75"
CACHE_DIR = OUT_DIR / "cache"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ─── Setup simulation config ──────────────────────────────
    config = SimulationConfig(
        Dz=PATH_LENGTH_M,
        Cn2=CN2,
        theta_div=THETA_DIV,
        D_roi=RECEIVER_WINDOW_M,
        delta_n=RECEIVER_DX_M,
        D_aperture=TELESCOPE_DIAMETER_M,
        n_reals=N_REALIZATIONS,
        wvl=WAVELENGTH_M,
    )

    print(f"w₀ = {config.w0*1e3:.3f} mm")
    print(f"D1 (source aperture) = {config.D1*1e3:.1f} mm")
    print(f"Receiver window = {RECEIVER_WINDOW_M*1e3:.1f} mm")
    print(f"Receiver dx = {RECEIVER_DX_M*1e6:.1f} μm")
    print(f"Cn² = {CN2:.1e} (5x stronger than previous)")

    # ─── Sampling analysis ────────────────────────────────────
    sampling = analyze_sampling(config)
    delta1 = sampling.delta1
    delta_n = sampling.delta_n
    N = sampling.N
    n_scr = sampling.n_scr
    z_planes = sampling.z_planes
    delta_values = sampling.delta_values

    print(f"\n--- Sampling analysis ---")
    print(f"N = {N}")
    print(f"n_scr = {n_scr}")
    print(f"delta1 (source) = {delta1*1e6:.1f} μm")
    print(f"delta_n (receiver) = {delta_n*1e6:.1f} μm")

    # ─── Atmospheric params ───────────────────────────────────
    atm = compute_atmospheric_params(config.k, CN2, PATH_LENGTH_M)
    r0 = atm["r0_pw"]
    print(f"\nr₀ (plane wave) = {r0*1e2:.1f} cm")
    print(f"D/r₀ = {TELESCOPE_DIAMETER_M/r0:.2f}")

    # ─── Optimize screen r0 ──────────────────────────────────
    from kim2026.fso.atmosphere import optimize_screen_r0
    r0_vals = optimize_screen_r0(atm["r0_sw"], atm["sigma2_chi_sw"], config.k, PATH_LENGTH_M, n_scr)

    # ─── Vacuum propagation ──────────────────────────────────
    print(f"\n--- Vacuum propagation ---")
    U_in = make_gaussian_source(N, delta1, config.w0, device=device)
    xn, yn, U_vac = ang_spec_multi_prop(
        U_in, WAVELENGTH_M, delta1, delta_n, z_planes,
        phase_screens=None, device=device,
    )

    # Check vacuum beam
    I_vac = compute_irradiance(U_vac)
    I_vac_np = I_vac.cpu().numpy()
    I_norm = I_vac_np / I_vac_np.max()
    above = I_norm > (1 / math.e**2)
    r_px = math.sqrt(above.sum() / math.pi)
    print(f"Vacuum beam 1/e² radius: {r_px:.0f} px = {r_px*delta_n*1e3:.1f} mm")

    # ─── Crop to telescope aperture ───────────────────────────
    CROP_N = 1024
    c = N // 2
    half_crop = CROP_N // 2
    U_vac_crop = U_vac[c - half_crop:c + half_crop, c - half_crop:c + half_crop]
    crop_window_m = CROP_N * delta_n
    print(f"\nCropping: {N}→{CROP_N}, window {N*delta_n*1e3:.1f}mm→{crop_window_m*1e3:.1f}mm")

    # ─── Generate realizations ────────────────────────────────
    print(f"\n--- Generating {N_REALIZATIONS} realizations ---")
    U_vac_save = U_vac_crop.detach().cpu()
    x_m = (np.arange(CROP_N) - CROP_N // 2) * delta_n
    y_m = x_m.copy()

    split_files = {"train": [], "val": [], "test": []}
    t0 = time.time()

    for i in range(N_REALIZATIONS):
        filename = f"realization_{i:05d}.npz"
        # Split: 400/50/50
        if i < 400:
            split_key = "train"
        elif i < 450:
            split_key = "val"
        else:
            split_key = "test"

        if (CACHE_DIR / filename).exists():
            split_files[split_key].append(filename)
            continue

        torch.manual_seed(SEED + i * 1009)
        screens = generate_phase_screens(
            r0_vals.tolist(), N, delta_values, device=device,
        )
        _, _, U_turb_full = ang_spec_multi_prop(
            U_in, WAVELENGTH_M, delta1, delta_n, z_planes,
            phase_screens=screens, device=device,
        )
        U_turb = U_turb_full[c - half_crop:c + half_crop, c - half_crop:c + half_crop]

        write_pair_npz(
            CACHE_DIR / filename,
            u_vacuum=torch.from_numpy(U_vac_save.numpy()),
            u_turb=U_turb.detach().cpu(),
            x_m=x_m, y_m=y_m,
            metadata={
                "realization": i,
                "Dz": PATH_LENGTH_M,
                "Cn2": CN2,
                "theta_div": THETA_DIV,
                "wvl": WAVELENGTH_M,
                "delta_n": delta_n,
                "N": int(N),
                "crop_n": CROP_N,
                "receiver_window_m": float(crop_window_m),
                "telescope_diameter_m": TELESCOPE_DIAMETER_M,
                "beam_reducer_ratio": BEAM_REDUCER_RATIO,
            },
        )
        split_files[split_key].append(filename)

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (N_REALIZATIONS - i - 1)
            print(f"  [{i+1}/{N_REALIZATIONS}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    # Save manifest
    manifest_path = OUT_DIR / "split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(split_files, f, indent=2)

    total_time = time.time() - t0
    print(f"\nDone! {N_REALIZATIONS} realizations in {total_time:.0f}s")
    print(f"Saved to: {CACHE_DIR}")

    # ─── Summary ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("DATA GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Physical: TX 0.3mrad → 1km → 15cm telescope → {BEAM_REDUCER_RATIO}:1 → {FDNN_WINDOW_M*1e3:.3f}mm")
    print(f"Cn² = {CN2:.1e}, D/r₀ = {TELESCOPE_DIAMETER_M/r0:.2f}")
    print(f"Grid: N={N} (cropped to {CROP_N})")
    print(f"Splits: train={len(split_files['train'])}, val={len(split_files['val'])}, test={len(split_files['test'])}")


if __name__ == "__main__":
    main()
