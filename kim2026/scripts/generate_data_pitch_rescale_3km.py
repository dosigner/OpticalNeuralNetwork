#!/usr/bin/env python
"""Production: 3km propagation, Cn²=5e-14, pitch rescale.

Same as 1km version but with PATH_LENGTH_M=3000.
3km → r0 smaller → stronger turbulence (D/r0 ≈ 3.8).

Usage:
    cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python scripts/generate_data_pitch_rescale_3km.py
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
from kim2026.fso.atmosphere import compute_atmospheric_params, optimize_screen_r0
from kim2026.fso.sampling import analyze_sampling
from kim2026.fso.phase_screen import generate_phase_screens
from kim2026.fso.propagation import make_gaussian_source, ang_spec_multi_prop
from kim2026.optics.aperture import circular_aperture

# ─── Physical parameters ──────────────────────────────────────
WAVELENGTH = 1.55e-6
THETA_DIV = 3.0e-4
PATH_LENGTH_M = 3000.0
CN2 = 5.0e-14

TELESCOPE_DIAMETER_M = 0.150
BEAM_REDUCER_M = 75           # f1/f2 = 75
FDNN_WINDOW_M = 0.002048      # output: 1024 × 2μm

DELTA_N = 150e-6              # receiver grid spacing
CROP_N = 1024                 # 1024 × 150μm = 153.6mm (covers telescope)
OUTPUT_N = 1024
OUTPUT_DX = FDNN_WINDOW_M / OUTPUT_N  # 2μm

N_REALIZATIONS = 5000
SEED = 20260403

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "kim2026" / "3km_cn2_5e-14_tel15cm_pitch_rescale"
CACHE_DIR = OUT_DIR / "cache"


def compute_defocus_correction(n, dx_m, device):
    """Analytical defocus: removes beam curvature after M=75 reduction."""
    w0 = 2 * WAVELENGTH / (math.pi * THETA_DIV)
    z_R = math.pi * w0**2 / WAVELENGTH
    R = PATH_LENGTH_M * (1 + (z_R / PATH_LENGTH_M)**2)
    k = 2 * math.pi / WAVELENGTH
    M = TELESCOPE_DIAMETER_M / FDNN_WINDOW_M
    c = n // 2
    idx = torch.arange(n, dtype=torch.float64, device=device)
    y, x = torch.meshgrid(idx - c, idx - c, indexing="ij")
    return torch.exp(-1j * k * M**2 / (2 * R) * ((x * dx_m)**2 + (y * dx_m)**2)).to(torch.complex64)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Production: 3km Pitch Rescale (delta_n=150μm, Cn²=5e-14) ===")
    print(f"Device: {device}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ─── Sampling ─────────────────────────────────────────────
    config = SimulationConfig(
        Dz=PATH_LENGTH_M, Cn2=CN2, theta_div=THETA_DIV,
        D_roi=CROP_N * DELTA_N, delta_n=DELTA_N,
        D_aperture=TELESCOPE_DIAMETER_M, n_reals=N_REALIZATIONS, wvl=WAVELENGTH,
    )
    sampling = analyze_sampling(config)
    delta1, delta_n = sampling.delta1, sampling.delta_n
    N, n_scr = sampling.N, sampling.n_scr
    z_planes, delta_values = sampling.z_planes, sampling.delta_values

    print(f"N={N}, n_scr={n_scr}, delta_n={delta_n*1e6:.0f}μm, crop={CROP_N}")
    print(f"Beam reducer: pitch rescale {DELTA_N*1e6:.0f}μm → {OUTPUT_DX*1e6:.0f}μm (M={BEAM_REDUCER_M})")

    atm = compute_atmospheric_params(config.k, CN2, PATH_LENGTH_M)
    r0_vals = optimize_screen_r0(
        atm["r0_sw"], atm["sigma2_chi_sw"], config.k, PATH_LENGTH_M, n_scr)
    print(f"r0={atm['r0_pw']*1e2:.1f}cm, D/r0={TELESCOPE_DIAMETER_M/atm['r0_pw']:.2f}")

    # ─── Defocus + aperture ───────────────────────────────────
    defocus = compute_defocus_correction(OUTPUT_N, OUTPUT_DX, device)
    aperture = circular_aperture(
        n=CROP_N, window_m=CROP_N * DELTA_N,
        diameter_m=TELESCOPE_DIAMETER_M, device=device,
    ).to(torch.complex64)

    # ─── Vacuum propagation ──────────────────────────────────
    print("Vacuum propagation...")
    U_in = make_gaussian_source(N, delta1, config.w0, device=device)
    _, _, U_vac_full = ang_spec_multi_prop(
        U_in, WAVELENGTH, delta1, delta_n, z_planes, device=device)

    c = N // 2
    hc = CROP_N // 2
    U_vac_crop = U_vac_full[c - hc:c + hc, c - hc:c + hc].to(torch.complex64)
    U_vac_out = (U_vac_crop * aperture * defocus).detach().cpu()
    del U_vac_full
    torch.cuda.empty_cache()

    x_m = (np.arange(OUTPUT_N) - OUTPUT_N // 2) * OUTPUT_DX
    y_m = x_m.copy()

    # ─── Generate realizations ────────────────────────────────
    print(f"\nGenerating {N_REALIZATIONS} realizations")
    split_files = {"train": [], "val": [], "test": []}
    t_start = time.time()
    skip_count = 0

    for i in range(N_REALIZATIONS):
        filename = f"realization_{i:05d}.npz"
        if i < 4000:
            split_key = "train"
        elif i < 4500:
            split_key = "val"
        else:
            split_key = "test"

        if (CACHE_DIR / filename).exists():
            split_files[split_key].append(filename)
            skip_count += 1
            continue

        torch.manual_seed(SEED + i * 1009)
        screens = generate_phase_screens(r0_vals.tolist(), N, delta_values, device=device)
        _, _, U_turb_full = ang_spec_multi_prop(
            U_in, WAVELENGTH, delta1, delta_n, z_planes,
            phase_screens=screens, device=device)
        del screens

        U_turb_crop = U_turb_full[c - hc:c + hc, c - hc:c + hc].to(torch.complex64)
        U_turb_out = (U_turb_crop * aperture * defocus).detach().cpu()
        del U_turb_full, U_turb_crop
        torch.cuda.empty_cache()

        write_pair_npz(
            CACHE_DIR / filename,
            u_vacuum=U_vac_out.clone(),
            u_turb=U_turb_out,
            x_m=x_m, y_m=y_m,
            metadata={
                "realization": i,
                "Dz": PATH_LENGTH_M,
                "Cn2": CN2,
                "theta_div": THETA_DIV,
                "wvl": WAVELENGTH,
                "delta_n": OUTPUT_DX,
                "N": OUTPUT_N,
                "crop_n": OUTPUT_N,
                "receiver_window_m": FDNN_WINDOW_M,
                "telescope_diameter_m": TELESCOPE_DIAMETER_M,
                "beam_reducer_ratio": float(BEAM_REDUCER_M),
                "generation_method": "pitch_rescale_dn150um",
                "propagation_delta_n_m": DELTA_N,
                "propagation_N": N,
                "propagation_n_scr": n_scr,
                "defocus_compensated": True,
                "interpolation": "none",
            },
        )
        split_files[split_key].append(filename)

        done = i + 1 - skip_count
        if done % 100 == 0 or done <= 3:
            elapsed = time.time() - t_start
            rate = elapsed / done
            remaining = rate * (N_REALIZATIONS - skip_count - done)
            print(f"  [{i+1}/{N_REALIZATIONS}] {rate:.1f}s/real, "
                  f"elapsed {elapsed/3600:.1f}h, ETA {remaining/3600:.1f}h")

    manifest_path = OUT_DIR / "split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(split_files, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nDone! {N_REALIZATIONS} realizations in {total_time/3600:.1f}h")
    print(f"  Skipped: {skip_count}, Train={len(split_files['train'])}, Val={len(split_files['val'])}, Test={len(split_files['test'])}")
    print(f"  Output: {CACHE_DIR}")


if __name__ == "__main__":
    main()
