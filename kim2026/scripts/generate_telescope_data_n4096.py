#!/usr/bin/env python
"""Generate FSO 15cm data at full 4096 resolution (no crop).

Same physics as generate_telescope_data.py but saves the full 4096×4096 grid.
20 realizations: train=16, val=2, test=2.

Usage:
    cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python scripts/generate_telescope_data_n4096.py
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
from kim2026.fso.propagation import make_gaussian_source, ang_spec_multi_prop, compute_irradiance

# ─── Physical parameters (identical to n1024 version) ────────
WAVELENGTH_M = 1.55e-6
THETA_DIV = 3.0e-4
PATH_LENGTH_M = 1000.0
CN2 = 1.0e-14

TELESCOPE_DIAMETER_M = 0.150
BEAM_REDUCER_RATIO = 75
FDNN_WINDOW_M = 0.002048

RECEIVER_WINDOW_M = FDNN_WINDOW_M * BEAM_REDUCER_RATIO  # 153.6mm
RECEIVER_DX_M = RECEIVER_WINDOW_M / 1024                # 150μm

N_REALIZATIONS = 20
SEED = 20260327

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "kim2026" / "fso_15cm_4096"
CACHE_DIR = OUT_DIR / "cache"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

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

    sampling = analyze_sampling(config)
    delta1 = sampling.delta1
    delta_n = sampling.delta_n
    N = sampling.N
    n_scr = sampling.n_scr
    z_planes = sampling.z_planes
    delta_values = sampling.delta_values

    print(f"Propagation N = {N}")
    print(f"delta1 = {delta1*1e6:.1f} μm, delta_n = {delta_n*1e6:.1f} μm")
    print(f"Source window = {N*delta1*1e3:.1f} mm")
    print(f"Receiver window = {N*delta_n*1e3:.1f} mm")
    print(f"Saving FULL {N}×{N} grid (no crop)")

    atm = compute_atmospheric_params(config.k, CN2, PATH_LENGTH_M)
    r0_vals = optimize_screen_r0(atm["r0_sw"], atm["sigma2_chi_sw"], config.k, PATH_LENGTH_M, n_scr)

    # Vacuum propagation
    U_in = make_gaussian_source(N, delta1, config.w0, device=device)
    xn, yn, U_vac = ang_spec_multi_prop(
        U_in, WAVELENGTH_M, delta1, delta_n, z_planes,
        phase_screens=None, device=device,
    )

    I_vac = compute_irradiance(U_vac)
    I_vac_np = I_vac.cpu().numpy()
    I_norm = I_vac_np / I_vac_np.max()
    above = I_norm > (1 / math.e**2)
    r_px = math.sqrt(above.sum() / math.pi)
    print(f"Vacuum beam 1/e² radius: {r_px:.0f} px = {r_px*delta_n*1e3:.1f} mm")

    # Coordinates for full grid
    x_m = (np.arange(N) - N // 2) * delta_n
    y_m = x_m.copy()

    U_vac_save = U_vac.detach().cpu()

    # Split: 16/2/2 for 20 realizations (80/10/10%)
    n_train = 16
    n_val = 2

    split_files = {"train": [], "val": [], "test": []}
    t0 = time.time()

    print(f"\n--- Generating {N_REALIZATIONS} realizations ({N}×{N}) ---")

    for i in range(N_REALIZATIONS):
        filename = f"realization_{i:05d}.npz"
        if (CACHE_DIR / filename).exists():
            if i < n_train:
                split_files["train"].append(filename)
            elif i < n_train + n_val:
                split_files["val"].append(filename)
            else:
                split_files["test"].append(filename)
            print(f"  [{i+1}/{N_REALIZATIONS}] skipped (exists)")
            continue

        torch.manual_seed(SEED + i * 1009)

        screens = generate_phase_screens(
            r0_vals.tolist(), N, delta_values, device=device,
        )
        _, _, U_turb = ang_spec_multi_prop(
            U_in, WAVELENGTH_M, delta1, delta_n, z_planes,
            phase_screens=screens, device=device,
        )

        write_pair_npz(
            CACHE_DIR / filename,
            u_vacuum=torch.from_numpy(U_vac_save.numpy()),
            u_turb=U_turb.detach().cpu(),
            x_m=x_m,
            y_m=y_m,
            metadata={
                "realization": i,
                "Dz": PATH_LENGTH_M,
                "Cn2": CN2,
                "theta_div": THETA_DIV,
                "wvl": WAVELENGTH_M,
                "delta_n": delta_n,
                "N": int(N),
                "crop_n": int(N),  # no crop
                "receiver_window_m": float(N * delta_n),
                "telescope_diameter_m": TELESCOPE_DIAMETER_M,
                "beam_reducer_ratio": BEAM_REDUCER_RATIO,
            },
        )

        if i < n_train:
            split_files["train"].append(filename)
        elif i < n_train + n_val:
            split_files["val"].append(filename)
        else:
            split_files["test"].append(filename)

        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (N_REALIZATIONS - i - 1)
        print(f"  [{i+1}/{N_REALIZATIONS}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    manifest_path = OUT_DIR / "split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(split_files, f, indent=2)

    total_time = time.time() - t0
    print(f"\nDone! {N_REALIZATIONS} realizations in {total_time:.0f}s")
    print(f"Saved to: {CACHE_DIR}")
    print(f"Per-file: ~{4*N*N*4/1e6:.0f} MB, total: ~{N_REALIZATIONS*4*N*N*4/1e9:.1f} GB")
    print(f"Splits: train={len(split_files['train'])}, val={len(split_files['val'])}, test={len(split_files['test'])}")


if __name__ == "__main__":
    main()
