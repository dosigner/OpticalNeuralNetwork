#!/usr/bin/env python
"""Method 1: delta_n=50μm propagation + Lanczos 25:1 beam reducer.

Physical system:
  TX (0.3mrad full div) → 1km → 15cm telescope → Lanczos 25:1 → D2NN (2mm)

Propagation at delta_n=50μm gives N=8192, n_scr=77.
Memory-efficient: generates screens on GPU one at a time, stores on CPU.
After crop to 3072×3072 (telescope aperture), Lanczos resample → 1024×1024 at dx=2μm.

Defocus compensation: the vacuum beam arrives with R≈1000m curvature.
After Lanczos reduction (M=73.2), the D2NN-plane phase has ~11 rad of defocus.
Analytically subtracted using known beam parameters (simulates telescope collimation).

Usage:
    cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python scripts/generate_data_dn50um_lanczos.py
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from kim2026.fso.config import SimulationConfig
from kim2026.fso.atmosphere import compute_atmospheric_params, optimize_screen_r0
from kim2026.fso.sampling import analyze_sampling
from kim2026.fso.phase_screen import ft_sh_phase_screen
from kim2026.fso.propagation import make_gaussian_source
from kim2026.fso.ft_utils import ft2, ift2

from kim2026.optics.beam_reducer import BeamReducerPlane, apply_beam_reducer
from kim2026.eval.focal_utils import apply_focal_lens, compute_pib_torch

# ─── Physical parameters ──────────────────────────────────────
WAVELENGTH = 1.55e-6
THETA_DIV = 3.0e-4
PATH_LENGTH_M = 1000.0
CN2 = 5.0e-14

TELESCOPE_DIAMETER_M = 0.150
FDNN_WINDOW_M = 0.002048
RECEIVER_WINDOW_M = 0.1536  # 153.6 mm

# Method 1 specifics
DELTA_N = 50e-6  # 50 μm receiver grid spacing
CROP_N = 3072    # 153.6mm / 50μm = 3072
OUTPUT_N = 1024

N_REALIZATIONS = 5
SEED = 20260401


# ─── Analytical defocus removal ───────────────────────────────

def compute_beam_defocus_phase(n, dx_m, wavelength_m, theta_div, Dz, magnification):
    """Compute the analytical defocus phase at the D2NN plane.

    The Gaussian beam diverges over Dz with curvature R(z).
    After beam reduction with magnification M, the D2NN-plane phase is:
        φ(r) = k M² r² / (2R)
    where r is the output-plane radial coordinate.

    This simulates the collimating effect of the telescope.
    """
    w0 = 2 * wavelength_m / (math.pi * theta_div)
    z_R = math.pi * w0**2 / wavelength_m
    R = Dz * (1 + (z_R / Dz)**2)  # beam radius of curvature at receiver

    k = 2 * math.pi / wavelength_m
    c = n // 2
    idx = torch.arange(n, dtype=torch.float64)
    y, x = torch.meshgrid(idx - c, idx - c, indexing="ij")
    r2 = (x * dx_m)**2 + (y * dx_m)**2

    # Defocus coefficient in output coordinates
    a = k * magnification**2 / (2 * R)
    defocus_phase = a * r2
    print(f"  Defocus: R={R:.1f}m, M={magnification:.1f}, "
          f"a={a:.0f} rad/m², φ_edge={a*(n//2*dx_m)**2:.2f} rad")
    return defocus_phase  # [rad], shape (n, n)


def remove_defocus(field, defocus_phase):
    """Remove analytical defocus from complex field."""
    correction = torch.exp(-1j * defocus_phase.to(device=field.device)).to(field.dtype)
    return field * correction


def compute_wfe_nm(field, wavelength_m):
    """Residual WFE (after defocus removal) in nm. Measures higher-order aberrations."""
    amp = field.abs()
    phase = torch.angle(field)
    threshold = amp.max() * 0.05
    support = amp > threshold
    if support.sum() < 10:
        return float("nan")
    ph = phase[support]
    ph = ph - ph.mean()  # remove residual piston
    rms_rad = ph.std().item()
    return rms_rad * wavelength_m / (2 * math.pi) * 1e9


# ─── Memory-efficient propagation ─────────────────────────────

def _generate_screens_to_cpu(r0_vals, N, delta_values, seed, device="cuda"):
    """Generate phase screens on GPU one at a time, store on CPU."""
    screens_cpu = []
    n_scr = len(r0_vals)
    for idx, (r0, dv) in enumerate(zip(r0_vals, delta_values)):
        torch.manual_seed(seed + idx * 7919)
        scr = ft_sh_phase_screen(float(r0), N, float(dv), device=device)
        screens_cpu.append(scr.cpu())
        del scr
    torch.cuda.empty_cache()
    return screens_cpu


def ang_spec_multi_prop_lazy(
    Uin, wvl, delta1, deltan, z_planes,
    phase_screens_cpu=None, device="cuda",
):
    """ang_spec_multi_prop with lazy GPU screen loading (one at a time)."""
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

    # First screen transmittance (lazy)
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
    Uout = Q3 * U

    return xn, yn, Uout


# ─── Main ─────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Method 1: delta_n=50μm + Lanczos 25:1 ===")
    print(f"Device: {device}")

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
    print(f"Grid: {N**2*16/1e9:.2f} GB/field")

    # ─── Atmospheric params ───────────────────────────────────
    atm = compute_atmospheric_params(config.k, CN2, PATH_LENGTH_M)
    r0_vals = optimize_screen_r0(
        atm["r0_sw"], atm["sigma2_chi_sw"], config.k, PATH_LENGTH_M, n_scr)
    print(f"r₀={atm['r0_pw']*1e2:.1f}cm, D/r₀={TELESCOPE_DIAMETER_M/atm['r0_pw']:.2f}")

    # ─── Beam reducer geometry ────────────────────────────────
    crop_window = CROP_N * DELTA_N
    output_dx = FDNN_WINDOW_M / OUTPUT_N
    magnification = TELESCOPE_DIAMETER_M / FDNN_WINDOW_M  # 73.24

    input_plane = BeamReducerPlane(
        window_m=crop_window, n=CROP_N, aperture_diameter_m=TELESCOPE_DIAMETER_M)
    output_plane = BeamReducerPlane(
        window_m=FDNN_WINDOW_M, n=OUTPUT_N, aperture_diameter_m=FDNN_WINDOW_M)
    print(f"Reducer: {CROP_N}@{DELTA_N*1e6:.0f}μm → {OUTPUT_N}@{output_dx*1e6:.0f}μm (M={magnification:.1f})")

    # ─── Precompute defocus phase ─────────────────────────────
    print("\nDefocus compensation (analytical, simulates telescope collimation):")
    defocus_phase = compute_beam_defocus_phase(
        OUTPUT_N, output_dx, WAVELENGTH, THETA_DIV, PATH_LENGTH_M, magnification)

    # ─── Vacuum propagation ──────────────────────────────────
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print(f"\n--- Vacuum propagation ---")
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
    del U_vac_crop

    # Defocus-compensated vacuum
    U_vac_collimated = remove_defocus(U_vac_reduced, defocus_phase)
    wfe_vac = compute_wfe_nm(U_vac_collimated, WAVELENGTH)

    focal_vac, dx_focal = apply_focal_lens(U_vac_collimated.unsqueeze(0))
    pibs_vac = compute_pib_torch(focal_vac, dx_focal, [10.0, 25.0, 50.0])
    pibs_vac = {k: float(v.item()) for k, v in pibs_vac.items()}

    # Also compute raw (uncorrected) PIB for reference
    focal_raw, dx_raw = apply_focal_lens(U_vac_reduced.unsqueeze(0))
    pib_raw = compute_pib_torch(focal_raw, dx_raw, [10.0])
    pib_raw_10 = float(pib_raw[10.0].item())

    if device == "cuda":
        vram_vac = torch.cuda.max_memory_allocated() / 1e9

    print(f"Vacuum WFE (HO, collimated): {wfe_vac:.1f} nm")
    print(f"Vacuum PIB@10μm (collimated): {pibs_vac[10.0]*100:.1f}%")
    print(f"Vacuum PIB@25μm: {pibs_vac[25.0]*100:.1f}%, @50μm: {pibs_vac[50.0]*100:.1f}%")
    print(f"Vacuum PIB@10μm (raw, no collimation): {pib_raw_10*100:.1f}%")
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

        print(f"  [{i+1}/{N_REALIZATIONS}] screens...", end=" ", flush=True)
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

        # Defocus-compensated turbulence PIB
        U_turb_collimated = remove_defocus(U_turb_reduced, defocus_phase)
        focal_t, dx_f = apply_focal_lens(U_turb_collimated.unsqueeze(0))
        pibs_t = compute_pib_torch(focal_t, dx_f, [10.0])
        pib_10 = float(pibs_t[10.0].item())
        pibs_turb.append(pib_10)
        del U_turb_reduced, U_turb_collimated
        torch.cuda.empty_cache()

        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        times.append(dt)

        vram = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
        print(f"{dt:.1f}s, VRAM={vram:.1f}GB, PIB@10μm={pib_10*100:.1f}%")

    # ─── Summary ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("METHOD 1: delta_n=50μm + Lanczos 25:1")
    print(f"{'='*60}")
    print(f"Grid: N={N}, n_scr={n_scr}, crop={CROP_N}")
    print(f"Vacuum WFE (HO):    {wfe_vac:.1f} nm  {'PASS' if wfe_vac < 50 else 'FAIL'}")
    print(f"Vacuum PIB@10μm:    {pibs_vac[10.0]*100:.1f}%  {'PASS' if pibs_vac[10.0] > 0.80 else 'FAIL'}")
    print(f"Time/realization:   {np.mean(times):.1f} ± {np.std(times):.1f} s")
    print(f"Turb PIB@10μm:      {np.mean(pibs_turb)*100:.1f} ± {np.std(pibs_turb)*100:.1f}%")


if __name__ == "__main__":
    main()
