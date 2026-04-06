#!/usr/bin/env python
"""Production data generation: delta_n=100μm + Lanczos 50:1.

Generates 5000 realizations (4000 train / 500 val / 500 test).
N=4096, n_scr=39. Expected ~5s/realization → ~7h total on A100.

Fields are defocus-compensated (analytical telescope collimation)
and stored as 1024×1024 at dx=2μm.

Usage:
    cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python scripts/generate_data_dn100um_production.py
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

from kim2026.optics.beam_reducer import BeamReducerPlane, apply_beam_reducer
from kim2026.eval.focal_utils import apply_focal_lens, compute_pib_torch

# ─── Physical parameters ──────────────────────────────────────
WAVELENGTH = 1.55e-6
THETA_DIV = 3.0e-4
PATH_LENGTH_M = 1000.0
CN2 = 5.0e-14

TELESCOPE_DIAMETER_M = 0.150
FDNN_WINDOW_M = 0.002048
RECEIVER_WINDOW_M = 0.1536

DELTA_N = 100e-6   # 100 μm
CROP_N = 1536       # 153.6mm / 100μm
OUTPUT_N = 1024

N_REALIZATIONS = 5000
SEED = 20260401

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "kim2026" / "1km_cn2_5e-14_tel15cm_dn100um_lanczos50"
CACHE_DIR = OUT_DIR / "cache"


# ─── Defocus ──────────────────────────────────────────────────

def compute_defocus_correction(n, dx_m, wavelength_m, theta_div, Dz, magnification, device):
    """Analytical defocus correction (telescope collimation)."""
    w0 = 2 * wavelength_m / (math.pi * theta_div)
    z_R = math.pi * w0**2 / wavelength_m
    R = Dz * (1 + (z_R / Dz)**2)
    k = 2 * math.pi / wavelength_m
    c = n // 2
    idx = torch.arange(n, dtype=torch.float64, device=device)
    y, x = torch.meshgrid(idx - c, idx - c, indexing="ij")
    r2 = (x * dx_m)**2 + (y * dx_m)**2
    a = k * magnification**2 / (2 * R)
    print(f"  Defocus: R={R:.1f}m, M={magnification:.1f}, a={a:.0f} rad/m², φ_edge={a*(n//2*dx_m)**2:.2f} rad")
    return torch.exp(-1j * a * r2).to(torch.complex64)


def compute_wfe_nm(field, wavelength_m):
    """Residual WFE (after defocus removal) in nm."""
    amp = field.abs()
    phase = torch.angle(field)
    support = amp > amp.max() * 0.05
    if support.sum() < 10:
        return float("nan")
    ph = phase[support]
    ph = ph - ph.mean()
    return ph.std().item() * wavelength_m / (2 * math.pi) * 1e9


# ─── Main ─────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Production: delta_n=100μm + Lanczos 50:1 ===")
    print(f"Device: {device}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ─── Sampling ─────────────────────────────────────────────
    config = SimulationConfig(
        Dz=PATH_LENGTH_M, Cn2=CN2, theta_div=THETA_DIV,
        D_roi=RECEIVER_WINDOW_M, delta_n=DELTA_N,
        D_aperture=TELESCOPE_DIAMETER_M, n_reals=N_REALIZATIONS, wvl=WAVELENGTH,
    )
    sampling = analyze_sampling(config)
    delta1, delta_n = sampling.delta1, sampling.delta_n
    N, n_scr = sampling.N, sampling.n_scr
    z_planes, delta_values = sampling.z_planes, sampling.delta_values

    print(f"N={N}, n_scr={n_scr}, delta1={delta1*1e6:.0f}μm, delta_n={delta_n*1e6:.0f}μm")
    print(f"Grid: {N**2*16/1e9:.2f} GB/field")

    atm = compute_atmospheric_params(config.k, CN2, PATH_LENGTH_M)
    r0_vals = optimize_screen_r0(
        atm["r0_sw"], atm["sigma2_chi_sw"], config.k, PATH_LENGTH_M, n_scr)
    print(f"r₀={atm['r0_pw']*1e2:.1f}cm, D/r₀={TELESCOPE_DIAMETER_M/atm['r0_pw']:.2f}")

    # ─── Geometry ─────────────────────────────────────────────
    output_dx = FDNN_WINDOW_M / OUTPUT_N
    magnification = TELESCOPE_DIAMETER_M / FDNN_WINDOW_M

    input_plane = BeamReducerPlane(
        window_m=CROP_N * DELTA_N, n=CROP_N, aperture_diameter_m=TELESCOPE_DIAMETER_M)
    output_plane = BeamReducerPlane(
        window_m=FDNN_WINDOW_M, n=OUTPUT_N, aperture_diameter_m=FDNN_WINDOW_M)
    print(f"Reducer: {CROP_N}@{DELTA_N*1e6:.0f}μm → {OUTPUT_N}@{output_dx*1e6:.0f}μm (M={magnification:.1f})")

    # ─── Defocus ──────────────────────────────────────────────
    defocus_corr = compute_defocus_correction(
        OUTPUT_N, output_dx, WAVELENGTH, THETA_DIV, PATH_LENGTH_M, magnification, device)

    # ─── Vacuum propagation ──────────────────────────────────
    print("\nVacuum propagation...")
    U_in = make_gaussian_source(N, delta1, config.w0, device=device)
    _, _, U_vac_full = ang_spec_multi_prop(
        U_in, WAVELENGTH, delta1, delta_n, z_planes, device=device)

    c = N // 2
    hc = CROP_N // 2
    U_vac_crop = U_vac_full[c - hc:c + hc, c - hc:c + hc]

    U_vac_reduced = apply_beam_reducer(
        U_vac_crop, input_plane=input_plane, output_plane=output_plane)
    U_vac_out = (U_vac_reduced * defocus_corr).detach().cpu()

    # Vacuum quality check
    wfe = compute_wfe_nm(U_vac_reduced * defocus_corr, WAVELENGTH)
    focal, dx_f = apply_focal_lens((U_vac_reduced * defocus_corr).unsqueeze(0))
    pibs = compute_pib_torch(focal, dx_f, [10.0, 25.0])
    pib10 = float(pibs[10.0].item())
    print(f"Vacuum WFE(HO): {wfe:.1f} nm ({'PASS' if wfe < 50 else 'FAIL'})")
    print(f"Vacuum PIB@10μm: {pib10*100:.1f}% ({'PASS' if pib10 > 0.80 else 'FAIL'})")

    del U_vac_full, U_vac_crop, U_vac_reduced, focal
    torch.cuda.empty_cache()

    x_m = (np.arange(OUTPUT_N) - OUTPUT_N // 2) * output_dx
    y_m = x_m.copy()

    # ─── Generate realizations ────────────────────────────────
    print(f"\nGenerating {N_REALIZATIONS} realizations (ETA ~{N_REALIZATIONS * 5 / 3600:.0f}h)")
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

        U_turb_crop = U_turb_full[c - hc:c + hc, c - hc:c + hc]
        U_turb_reduced = apply_beam_reducer(
            U_turb_crop, input_plane=input_plane, output_plane=output_plane)
        U_turb_out = (U_turb_reduced * defocus_corr).detach().cpu()
        del U_turb_full, U_turb_crop, U_turb_reduced
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
                "delta_n": output_dx,
                "N": OUTPUT_N,
                "crop_n": OUTPUT_N,
                "receiver_window_m": FDNN_WINDOW_M,
                "telescope_diameter_m": TELESCOPE_DIAMETER_M,
                "beam_reducer_ratio": magnification,
                "generation_method": "dn100um_lanczos50",
                "propagation_delta_n_m": DELTA_N,
                "propagation_N": N,
                "propagation_n_scr": n_scr,
                "defocus_compensated": True,
            },
        )
        split_files[split_key].append(filename)

        done = i + 1 - skip_count
        if done % 50 == 0 or done <= 5:
            elapsed = time.time() - t_start
            rate = elapsed / done
            remaining = rate * (N_REALIZATIONS - skip_count - done)
            print(f"  [{i+1}/{N_REALIZATIONS}] {rate:.1f}s/real, "
                  f"elapsed {elapsed/3600:.1f}h, ETA {remaining/3600:.1f}h")

    # ─── Save manifest ────────────────────────────────────────
    manifest_path = OUT_DIR / "split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(split_files, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nDone! {N_REALIZATIONS} realizations in {total_time/3600:.1f}h")
    print(f"  Skipped (existing): {skip_count}")
    print(f"  Train={len(split_files['train'])}, Val={len(split_files['val'])}, Test={len(split_files['test'])}")
    print(f"  Output: {CACHE_DIR}")


if __name__ == "__main__":
    main()
