#!/usr/bin/env python
"""Method 2: delta_n=150μm propagation + Physical beam reducer (f₁=75mm, f₂=1mm).

Physical system:
  TX (0.3mrad full div) → 1km → 15cm telescope → afocal relay (75:1) → D2NN (2mm)

NOTE: The physical reducer uses scaled_fresnel_propagate which applies
thin-lens phase exp(-ik/(2f)*r²) at 150μm pixel pitch. At the aperture edge,
the phase gradient is ~608 rad/pixel — far exceeding Nyquist (π rad/pixel).
This causes catastrophic aliasing. Results are diagnostic only.

Usage:
    cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python scripts/generate_data_physical_reducer.py
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from kim2026.fso.config import SimulationConfig
from kim2026.fso.atmosphere import compute_atmospheric_params, optimize_screen_r0
from kim2026.fso.sampling import analyze_sampling
from kim2026.fso.phase_screen import generate_phase_screens
from kim2026.fso.propagation import make_gaussian_source, ang_spec_multi_prop

from kim2026.optics.beam_reducer import (
    BeamReducerPlane, apply_physical_beam_reducer_reference,
)
from kim2026.eval.focal_utils import apply_focal_lens, compute_pib_torch

# ─── Physical parameters ──────────────────────────────────────
WAVELENGTH = 1.55e-6
THETA_DIV = 3.0e-4
PATH_LENGTH_M = 1000.0
CN2 = 5.0e-14

TELESCOPE_DIAMETER_M = 0.150
FDNN_WINDOW_M = 0.002048
RECEIVER_WINDOW_M = 0.1536

# Method 2 specifics
DELTA_N = 150e-6
CROP_N = 1024
OUTPUT_N = 1024

N_REALIZATIONS = 5
SEED = 20260401


def subtract_low_order(field, dx_m):
    """Subtract piston + tilt + defocus from field phase."""
    n = field.shape[-1]
    amp = field.abs()
    phase = torch.angle(field).double()

    c = n // 2
    idx = torch.arange(n, device=field.device, dtype=torch.float64)
    y, x = torch.meshgrid(idx - c, idx - c, indexing="ij")
    x_m = x * dx_m
    y_m = y * dx_m
    r2 = x_m**2 + y_m**2

    threshold = amp.max() * 0.05
    support = amp > threshold
    if support.sum() < 100:
        return field, 0.0

    A = torch.stack([r2[support], x_m[support], y_m[support],
                     torch.ones_like(r2[support])], dim=1)
    coeffs = torch.linalg.lstsq(A, phase[support].unsqueeze(1)).solution.squeeze()
    fitted = coeffs[0] * r2 + coeffs[1] * x_m + coeffs[2] * y_m + coeffs[3]
    corrected = amp * torch.exp(1j * (phase - fitted).to(field.dtype))
    return corrected, float(coeffs[0])


def compute_wfe_nm(field, dx_m, wavelength_m):
    """WFE (higher-order only) in nm."""
    corrected, _ = subtract_low_order(field, dx_m)
    amp = corrected.abs()
    phase = torch.angle(corrected)
    support = amp > amp.max() * 0.05
    if support.sum() < 10:
        return float("nan")
    ph = phase[support]
    ph = ph - ph.mean()
    return ph.std().item() * wavelength_m / (2.0 * math.pi) * 1e9


def compute_pib_compensated(field, dx_m):
    """PIB after defocus compensation."""
    corrected, defocus = subtract_low_order(field, dx_m)
    focal, dx_f = apply_focal_lens(corrected.unsqueeze(0))
    pibs = compute_pib_torch(focal, dx_f, [10.0, 25.0, 50.0])
    return {k: float(v.item()) for k, v in pibs.items()}, defocus


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Method 2: delta_n=150μm + Physical reducer ===")
    print(f"Device: {device}")
    print()
    print("WARNING: Physical reducer lens phase is undersampled at 150μm pitch.")
    print("  Phase gradient at aperture edge: ~608 rad/pixel (Nyquist = π)")
    print("  Results are diagnostic — expect poor WFE and PIB.")
    print()

    # ─── Sampling ─────────────────────────────────────────────
    config = SimulationConfig(
        Dz=PATH_LENGTH_M, Cn2=CN2, theta_div=THETA_DIV,
        D_roi=RECEIVER_WINDOW_M, delta_n=DELTA_N,
        D_aperture=TELESCOPE_DIAMETER_M,
        n_reals=N_REALIZATIONS, wvl=WAVELENGTH,
    )
    sampling = analyze_sampling(config)
    delta1, delta_n = sampling.delta1, sampling.delta_n
    N, n_scr = sampling.N, sampling.n_scr
    z_planes, delta_values = sampling.z_planes, sampling.delta_values

    print(f"N={N}, n_scr={n_scr}, delta1={delta1*1e6:.0f}μm, delta_n={delta_n*1e6:.0f}μm")

    # ─── Atmospheric params ───────────────────────────────────
    atm = compute_atmospheric_params(config.k, CN2, PATH_LENGTH_M)
    r0_vals = optimize_screen_r0(
        atm["r0_sw"], atm["sigma2_chi_sw"], config.k, PATH_LENGTH_M, n_scr)
    print(f"r₀={atm['r0_pw']*1e2:.1f}cm, D/r₀={TELESCOPE_DIAMETER_M/atm['r0_pw']:.2f}")

    # ─── Beam reducer geometry ────────────────────────────────
    output_dx = FDNN_WINDOW_M / OUTPUT_N
    input_plane = BeamReducerPlane(
        window_m=CROP_N * DELTA_N, n=CROP_N,
        aperture_diameter_m=TELESCOPE_DIAMETER_M)
    output_plane = BeamReducerPlane(
        window_m=FDNN_WINDOW_M, n=OUTPUT_N,
        aperture_diameter_m=FDNN_WINDOW_M)

    # Diagnostic: lens phase sampling
    k = 2.0 * math.pi / WAVELENGTH
    f1 = 75e-3
    r_edge = TELESCOPE_DIAMETER_M / 2
    phase_per_px = k * r_edge / f1 * DELTA_N
    print(f"Lens phase gradient at aperture edge: {phase_per_px:.0f} rad/pixel (Nyquist=π={math.pi:.2f})")

    # ─── Vacuum propagation ──────────────────────────────────
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print(f"\n--- Vacuum propagation ---")
    U_in = make_gaussian_source(N, delta1, config.w0, device=device)
    _, _, U_vac_full = ang_spec_multi_prop(
        U_in, WAVELENGTH, delta1, delta_n, z_planes, device=device)

    c = N // 2
    hc = CROP_N // 2
    U_vac_crop = U_vac_full[c - hc:c + hc, c - hc:c + hc]

    # Physical beam reducer
    U_vac_reduced = apply_physical_beam_reducer_reference(
        U_vac_crop, input_plane=input_plane, output_plane=output_plane,
        wavelength_m=WAVELENGTH)

    wfe_vac = compute_wfe_nm(U_vac_reduced, output_dx, WAVELENGTH)
    pibs_vac, defocus = compute_pib_compensated(U_vac_reduced, output_dx)

    if device == "cuda":
        vram_vac = torch.cuda.max_memory_allocated() / 1e9

    print(f"Vacuum WFE (HO):    {wfe_vac:.1f} nm")
    print(f"Vacuum PIB@10μm:    {pibs_vac[10.0]*100:.1f}%")
    print(f"Vacuum PIB@25μm:    {pibs_vac[25.0]*100:.1f}%, @50μm: {pibs_vac[50.0]*100:.1f}%")
    if device == "cuda":
        print(f"Peak VRAM: {vram_vac:.2f} GB")

    # ─── Turbulence realizations ─────────────────────────────
    print(f"\n--- {N_REALIZATIONS} turbulence realizations ---")
    times = []
    pibs_turb = []

    for i in range(N_REALIZATIONS):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()

        torch.manual_seed(SEED + i * 1009)
        screens = generate_phase_screens(r0_vals.tolist(), N, delta_values, device=device)
        _, _, U_turb_full = ang_spec_multi_prop(
            U_in, WAVELENGTH, delta1, delta_n, z_planes,
            phase_screens=screens, device=device)
        del screens

        U_turb_crop = U_turb_full[c - hc:c + hc, c - hc:c + hc]
        U_turb_reduced = apply_physical_beam_reducer_reference(
            U_turb_crop, input_plane=input_plane, output_plane=output_plane,
            wavelength_m=WAVELENGTH)

        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        times.append(dt)

        pibs_t, _ = compute_pib_compensated(U_turb_reduced, output_dx)
        pibs_turb.append(pibs_t[10.0])
        vram = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
        print(f"  [{i+1}/{N_REALIZATIONS}] {dt:.1f}s, VRAM={vram:.1f}GB, PIB@10μm={pibs_t[10.0]*100:.1f}%")

    # ─── Summary ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("METHOD 2: delta_n=150μm + Physical reducer")
    print(f"{'='*60}")
    print(f"Grid: N={N}, n_scr={n_scr}, crop={CROP_N}")
    print(f"Vacuum WFE (HO):    {wfe_vac:.1f} nm  {'PASS' if wfe_vac < 50 else 'FAIL'}")
    print(f"Vacuum PIB@10μm:    {pibs_vac[10.0]*100:.1f}%  {'PASS' if pibs_vac[10.0] > 0.80 else 'FAIL'}")
    print(f"Time/realization:   {np.mean(times):.1f} ± {np.std(times):.1f} s")
    print(f"Turb PIB@10μm:      {np.mean(pibs_turb)*100:.1f} ± {np.std(pibs_turb)*100:.1f}%")
    print()
    print("NOTE: Poor results expected — Fresnel kernel is undersampled at 150μm pitch")


if __name__ == "__main__":
    main()
