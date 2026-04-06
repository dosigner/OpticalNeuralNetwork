#!/usr/bin/env python
"""Production data generation: delta_n=50μm + Lanczos 25:1.

Generates 5000 realizations (4000 train / 500 val / 500 test).
Fields are defocus-compensated (analytical, simulates telescope collimation)
and stored as 1024×1024 at dx=2μm.

Expected runtime: ~98 hours on A100 40GB.

Usage:
    cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python scripts/generate_data_dn50um_production.py
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
from kim2026.fso.phase_screen import ft_sh_phase_screen
from kim2026.fso.propagation import make_gaussian_source
from kim2026.fso.ft_utils import ft2, ift2

from kim2026.optics.beam_reducer import BeamReducerPlane, apply_beam_reducer

# ─── Physical parameters ──────────────────────────────────────
WAVELENGTH = 1.55e-6
THETA_DIV = 3.0e-4
PATH_LENGTH_M = 1000.0
CN2 = 5.0e-14

TELESCOPE_DIAMETER_M = 0.150
FDNN_WINDOW_M = 0.002048
RECEIVER_WINDOW_M = 0.1536

DELTA_N = 50e-6
CROP_N = 3072
OUTPUT_N = 1024

N_REALIZATIONS = 5000
SEED = 20260401

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "kim2026" / "1km_cn2_5e-14_tel15cm_dn50um_lanczos25"
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
    return torch.exp(-1j * a * r2).to(torch.complex64)


# ─── Lazy propagation ────────────────────────────────────────

def _generate_screens_to_cpu(r0_vals, N, delta_values, seed, device="cuda"):
    screens_cpu = []
    for idx, (r0, dv) in enumerate(zip(r0_vals, delta_values)):
        torch.manual_seed(seed + idx * 7919)
        scr = ft_sh_phase_screen(float(r0), N, float(dv), device=device)
        screens_cpu.append(scr.cpu())
        del scr
    torch.cuda.empty_cache()
    return screens_cpu


def ang_spec_multi_prop_lazy(Uin, wvl, delta1, deltan, z_planes,
                              phase_screens_cpu=None, device="cuda"):
    N = Uin.shape[0]
    k = 2.0 * math.pi / wvl
    z_t = torch.tensor(z_planes, dtype=torch.float64, device=device)
    n = len(z_planes)
    Delta_z = z_t[1:] - z_t[:-1]
    alpha = z_t / z_t[-1]
    delta = (1.0 - alpha) * delta1 + alpha * deltan
    m = delta[1:] / delta[:-1]

    nx = torch.arange(-N // 2, N // 2, dtype=torch.float64, device=device)
    ny = nx.clone()
    NX, NY = torch.meshgrid(nx, ny, indexing="ij")
    nsq = NX**2 + NY**2
    w = 0.47 * N
    sg = torch.exp(-nsq**8 / w**16).to(torch.complex128)
    del NX, NY, nsq

    if phase_screens_cpu is not None and phase_screens_cpu[0] is not None:
        T0 = torch.exp(1j * phase_screens_cpu[0].to(dtype=torch.complex128, device=device))
    else:
        T0 = torch.ones(N, N, dtype=torch.complex128, device=device)

    x1 = nx * delta[0]
    y1 = ny * delta[0]
    X1, Y1 = torch.meshgrid(x1, y1, indexing="ij")
    r1sq = X1**2 + Y1**2
    Q1 = torch.exp(1j * k / 2.0 * (1.0 - m[0]) / Delta_z[0] * r1sq)
    U = Uin * Q1 * T0
    del T0, Q1, r1sq, X1, Y1

    for i in range(n - 1):
        deltaf = 1.0 / (N * delta[i].item())
        fX = nx * deltaf
        fY = ny * deltaf
        FX, FY = torch.meshgrid(fX, fY, indexing="ij")
        fsq = FX**2 + FY**2
        Q2 = torch.exp(-1j * math.pi**2 * 2.0 * Delta_z[i] / (m[i] * k) * fsq)
        del fsq, FX, FY
        G = ft2(U / m[i], delta[i].item())
        G = Q2 * G
        del Q2
        U_prop = ift2(G, deltaf)
        del G

        if phase_screens_cpu is not None and phase_screens_cpu[i + 1] is not None:
            T_next = torch.exp(1j * phase_screens_cpu[i + 1].to(
                dtype=torch.complex128, device=device))
        else:
            T_next = torch.ones(N, N, dtype=torch.complex128, device=device)
        U = sg * T_next * U_prop
        del T_next, U_prop

    xn = nx * delta[-1]
    yn = ny * delta[-1]
    Xn, Yn = torch.meshgrid(xn, yn, indexing="ij")
    rnsq = Xn**2 + Yn**2
    Q3 = torch.exp(1j * k / 2.0 * (m[-1] - 1.0) / (m[-1] * Delta_z[-1]) * rnsq)
    return xn, yn, Q3 * U


# ─── Main ─────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Production: delta_n=50μm + Lanczos 25:1 ===")
    print(f"Device: {device}, Output: {CACHE_DIR}")
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

    print(f"N={N}, n_scr={n_scr}")

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

    # ─── Defocus correction ───────────────────────────────────
    defocus_corr = compute_defocus_correction(
        OUTPUT_N, output_dx, WAVELENGTH, THETA_DIV, PATH_LENGTH_M, magnification, device)
    print("Defocus correction precomputed")

    # ─── Vacuum propagation ──────────────────────────────────
    print("Vacuum propagation...")
    U_in = make_gaussian_source(N, delta1, config.w0, device=device)
    _, _, U_vac_full = ang_spec_multi_prop_lazy(
        U_in, WAVELENGTH, delta1, delta_n, z_planes, device=device)

    c = N // 2
    hc = CROP_N // 2
    U_vac_crop = U_vac_full[c - hc:c + hc, c - hc:c + hc].clone()
    del U_vac_full
    torch.cuda.empty_cache()

    U_vac_reduced = apply_beam_reducer(
        U_vac_crop, input_plane=input_plane, output_plane=output_plane)
    U_vac_out = (U_vac_reduced * defocus_corr).detach().cpu()
    del U_vac_crop, U_vac_reduced
    torch.cuda.empty_cache()

    x_m = (np.arange(OUTPUT_N) - OUTPUT_N // 2) * output_dx
    y_m = x_m.copy()

    # ─── Generate realizations ────────────────────────────────
    print(f"\nGenerating {N_REALIZATIONS} realizations (ETA ~{N_REALIZATIONS * 71 / 3600:.0f}h)")
    split_files = {"train": [], "val": [], "test": []}
    t_start = time.time()

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
            continue

        t0 = time.time()
        screens_cpu = _generate_screens_to_cpu(
            r0_vals.tolist(), N, delta_values,
            seed=SEED + i * 1009, device=device)

        _, _, U_turb_full = ang_spec_multi_prop_lazy(
            U_in, WAVELENGTH, delta1, delta_n, z_planes,
            phase_screens_cpu=screens_cpu, device=device)
        del screens_cpu

        U_turb_crop = U_turb_full[c - hc:c + hc, c - hc:c + hc].clone()
        del U_turb_full
        torch.cuda.empty_cache()

        U_turb_reduced = apply_beam_reducer(
            U_turb_crop, input_plane=input_plane, output_plane=output_plane)
        del U_turb_crop

        U_turb_out = (U_turb_reduced * defocus_corr).detach().cpu()
        del U_turb_reduced
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
                "generation_method": "dn50um_lanczos25",
                "propagation_delta_n_m": DELTA_N,
                "propagation_N": N,
                "propagation_n_scr": n_scr,
                "defocus_compensated": True,
            },
        )
        split_files[split_key].append(filename)

        dt = time.time() - t0
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            rate = elapsed / (i + 1)
            remaining = rate * (N_REALIZATIONS - i - 1)
            print(f"  [{i+1}/{N_REALIZATIONS}] {dt:.1f}s/real, "
                  f"elapsed {elapsed/3600:.1f}h, ETA {remaining/3600:.1f}h")

    # ─── Save manifest ────────────────────────────────────────
    manifest_path = OUT_DIR / "split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(split_files, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nDone! {N_REALIZATIONS} realizations in {total_time/3600:.1f}h")
    print(f"Saved to: {CACHE_DIR}")
    print(f"Train={len(split_files['train'])}, Val={len(split_files['val'])}, Test={len(split_files['test'])}")


if __name__ == "__main__":
    main()
