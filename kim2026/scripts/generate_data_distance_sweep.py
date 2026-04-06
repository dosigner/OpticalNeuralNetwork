#!/usr/bin/env python
"""Distance sweep data generation: L = [100, 500, 1000, 2000, 5000] m.

Cn²=5×10⁻¹⁴ fixed, 2000 realizations per distance (1600/200/200).
Pitch rescale method (no interpolation artifacts).

Usage:
    cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python scripts/generate_data_distance_sweep.py
    # or single distance:
    cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python scripts/generate_data_distance_sweep.py --distance 500
"""
from __future__ import annotations

import argparse
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

# ─── Fixed parameters ──────────────────────────────────────
WAVELENGTH = 1.55e-6
THETA_DIV = 3.0e-4
CN2 = 5.0e-14

TELESCOPE_DIAMETER_M = 0.150
BEAM_REDUCER_M = 75
FDNN_WINDOW_M = 0.002048

DELTA_N = 150e-6
CROP_N = 1024
OUTPUT_N = 1024
OUTPUT_DX = FDNN_WINDOW_M / OUTPUT_N  # 2μm

N_REALIZATIONS = 2000
TRAIN_N, VAL_N, TEST_N = 1600, 200, 200
SEED = 20260404

DISTANCES = [100, 500, 1000, 2000, 5000]

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "kim2026"


def compute_defocus_correction(n, dx_m, path_length_m, device):
    """Analytical defocus correction scaled for path length."""
    w0 = 2 * WAVELENGTH / (math.pi * THETA_DIV)
    z_R = math.pi * w0**2 / WAVELENGTH
    R = path_length_m * (1 + (z_R / path_length_m)**2)
    k = 2 * math.pi / WAVELENGTH
    M = TELESCOPE_DIAMETER_M / FDNN_WINDOW_M
    c = n // 2
    idx = torch.arange(n, dtype=torch.float64, device=device)
    y, x = torch.meshgrid(idx - c, idx - c, indexing="ij")
    return torch.exp(-1j * k * M**2 / (2 * R) * ((x * dx_m)**2 + (y * dx_m)**2)).to(torch.complex64)


def generate_distance(L_m: float, device: str):
    """Generate dataset for one propagation distance."""
    out_dir = DATA_ROOT / f"distance_sweep_cn2_5e-14" / f"L{int(L_m)}m"
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  L = {L_m:.0f} m | Cn² = {CN2:.0e}")
    print(f"{'='*70}")

    # ─── Sampling ─────────────────────────────────────────
    config = SimulationConfig(
        Dz=L_m, Cn2=CN2, theta_div=THETA_DIV,
        D_roi=CROP_N * DELTA_N, delta_n=DELTA_N,
        D_aperture=TELESCOPE_DIAMETER_M, n_reals=N_REALIZATIONS, wvl=WAVELENGTH,
    )
    sampling = analyze_sampling(config)
    delta1, delta_n = sampling.delta1, sampling.delta_n
    N, n_scr = sampling.N, sampling.n_scr
    z_planes, delta_values = sampling.z_planes, sampling.delta_values

    atm = compute_atmospheric_params(config.k, CN2, L_m)
    r0_vals = optimize_screen_r0(
        atm["r0_sw"], atm["sigma2_chi_sw"], config.k, L_m, n_scr)

    k = 2 * math.pi / WAVELENGTH
    sigma_R2 = 1.23 * CN2 * k**(7/6) * L_m**(11/6)

    print(f"  N={N}, n_scr={n_scr}, delta_n={delta_n*1e6:.0f}μm")
    print(f"  r0={atm['r0_pw']*1e2:.1f}cm, D/r0={TELESCOPE_DIAMETER_M/atm['r0_pw']:.2f}")
    print(f"  Rytov σ²_R={sigma_R2:.3f}")

    # ─── Defocus + aperture ──────────────────────────────
    defocus = compute_defocus_correction(OUTPUT_N, OUTPUT_DX, L_m, device)
    aperture = circular_aperture(
        n=CROP_N, window_m=CROP_N * DELTA_N,
        diameter_m=TELESCOPE_DIAMETER_M, device=device,
    ).to(torch.complex64)

    # ─── Vacuum ──────────────────────────────────────────
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

    # ─── Realizations ────────────────────────────────────
    print(f"  Generating {N_REALIZATIONS} realizations...")
    split_files = {"train": [], "val": [], "test": []}
    t_start = time.time()
    skip_count = 0

    for i in range(N_REALIZATIONS):
        filename = f"realization_{i:05d}.npz"
        if i < TRAIN_N:
            split_key = "train"
        elif i < TRAIN_N + VAL_N:
            split_key = "val"
        else:
            split_key = "test"

        if (cache_dir / filename).exists():
            split_files[split_key].append(filename)
            skip_count += 1
            continue

        torch.manual_seed(SEED + i * 1009 + int(L_m))
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
            cache_dir / filename,
            u_vacuum=U_vac_out.clone(),
            u_turb=U_turb_out,
            x_m=x_m, y_m=y_m,
            metadata={
                "realization": i,
                "Dz": L_m,
                "Cn2": CN2,
                "theta_div": THETA_DIV,
                "wvl": WAVELENGTH,
                "delta_n": OUTPUT_DX,
                "N": OUTPUT_N,
                "r0_pw_m": float(atm["r0_pw"]),
                "D_over_r0": TELESCOPE_DIAMETER_M / float(atm["r0_pw"]),
                "rytov_variance": sigma_R2,
                "telescope_diameter_m": TELESCOPE_DIAMETER_M,
                "beam_reducer_ratio": float(BEAM_REDUCER_M),
                "generation_method": "pitch_rescale_distance_sweep",
                "propagation_N": N,
                "propagation_n_scr": n_scr,
                "defocus_compensated": True,
            },
        )
        split_files[split_key].append(filename)

        done = i + 1 - skip_count
        if done % 200 == 0 or done <= 3:
            elapsed = time.time() - t_start
            rate = elapsed / done
            remaining = rate * (N_REALIZATIONS - skip_count - done)
            print(f"    [{i+1}/{N_REALIZATIONS}] {rate:.1f}s/real, "
                  f"elapsed {elapsed/60:.0f}min, ETA {remaining/60:.0f}min")

    manifest_path = out_dir / "split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(split_files, f, indent=2)

    total_time = time.time() - t_start
    print(f"  Done: {N_REALIZATIONS} in {total_time/60:.0f}min "
          f"(skip={skip_count}, train={len(split_files['train'])}, "
          f"val={len(split_files['val'])}, test={len(split_files['test'])})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance", type=int, default=None,
                        help="Single distance to generate (m)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Distance Sweep Data Generation")
    print(f"Cn²={CN2:.0e}, λ={WAVELENGTH*1e6:.2f}μm, Tel={TELESCOPE_DIAMETER_M*100:.0f}cm")
    print(f"N_real={N_REALIZATIONS} per distance ({TRAIN_N}/{VAL_N}/{TEST_N})")

    distances = [args.distance] if args.distance else DISTANCES

    for L in distances:
        generate_distance(float(L), device)

    print(f"\nAll distances complete!")


if __name__ == "__main__":
    main()
