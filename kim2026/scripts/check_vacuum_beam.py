#!/usr/bin/env python
"""Quick check: vacuum beam size at receiver plane and Fourier plane.

Usage:
    cd /root/dj/D2NN/kim2026 && python scripts/check_vacuum_beam.py
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import torch

from kim2026.data.dataset import CachedFieldDataset
from kim2026.optics.lens_2f import lens_2f_forward
from kim2026.training.targets import apply_receiver_aperture

WAVELENGTH_M = 1.55e-6
RECEIVER_WINDOW_M = 0.002048
APERTURE_DIAMETER_M = 0.002
N = 1024

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kim2026" / "1km_cn2e-14_w2m_n1024_dx2mm" / "cache"
MANIFEST = DATA_DIR.parent / "split_manifest.json"

ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="test")
sample = ds[0]
u_vac = sample["u_vacuum"]  # (1024, 1024) complex
print(f"Metadata: {sample.get('metadata', 'N/A')}")

dx = RECEIVER_WINDOW_M / N
print(f"\n=== Receiver plane ===")
print(f"Grid: {N}x{N}, dx={dx*1e6:.2f}μm, window={RECEIVER_WINDOW_M*1e3:.3f}mm")

# Irradiance
irr = (u_vac.abs()**2).numpy()
irr_norm = irr / irr.max()
total_energy = irr.sum()

# 1/e² radius
above_1e2 = irr_norm > (1/math.e**2)
n_px = above_1e2.sum()
r_px = math.sqrt(n_px / math.pi)
r_m = r_px * dx
print(f"Vacuum beam 1/e² radius: {r_m*1e6:.1f}μm = {r_px:.1f} pixels")
print(f"Vacuum beam 1/e² diameter: {2*r_m*1e6:.1f}μm = {2*r_m*1e3:.3f}mm")
print(f"Beam / window ratio: {2*r_m / RECEIVER_WINDOW_M:.3f}")
print(f"Beam / aperture ratio: {2*r_m / APERTURE_DIAMETER_M:.3f}")

# Second moment beam radius
y_ax = (np.arange(N) - N//2) * dx
x_ax = y_ax.copy()
yy, xx = np.meshgrid(y_ax, x_ax, indexing="ij")
r_sq = xx**2 + yy**2
w_2nd = math.sqrt((irr * r_sq).sum() / total_energy)
print(f"Second moment radius: {w_2nd*1e6:.1f}μm = {w_2nd*1e3:.3f}mm")

# Energy in center vs edge
c = N // 2
for radius_um in [100, 200, 500, 1000]:
    r_m_check = radius_um * 1e-6
    mask = r_sq <= r_m_check**2
    frac = (irr * mask).sum() / total_energy
    print(f"  Energy within r={radius_um}μm: {frac:.4f}")

# After aperture
u_vac_apt = apply_receiver_aperture(u_vac.unsqueeze(0),
    receiver_window_m=RECEIVER_WINDOW_M, aperture_diameter_m=APERTURE_DIAMETER_M)
irr_apt = (u_vac_apt[0].abs()**2).numpy()
energy_after_apt = irr_apt.sum()
print(f"\nAperture throughput: {energy_after_apt/total_energy:.4f}")

# Fourier plane
print(f"\n=== Fourier plane (f=25mm, NA=0.508) ===")
dx_in = RECEIVER_WINDOW_M / N
u_fourier, dx_f = lens_2f_forward(
    u_vac_apt.to(torch.complex64),
    dx_in_m=dx_in, wavelength_m=WAVELENGTH_M,
    f_m=25e-3, na=0.508, apply_scaling=False)

print(f"dx_fourier={dx_f*1e6:.2f}μm, window={dx_f*N*1e3:.2f}mm")

irr_f = (u_fourier[0].abs()**2).cpu().numpy()
irr_f_norm = irr_f / irr_f.max()

above_f = irr_f_norm > (1/math.e**2)
n_px_f = above_f.sum()
r_px_f = math.sqrt(n_px_f / math.pi)
r_f_m = r_px_f * dx_f
print(f"Fourier spot 1/e² radius: {r_f_m*1e6:.1f}μm = {r_px_f:.1f} pixels")
print(f"Spot / pixel ratio: {r_f_m/dx_f:.2f}")

# Peak location
peak_idx = np.unravel_index(irr_f.argmax(), irr_f.shape)
print(f"Peak at pixel: {peak_idx} (center={N//2})")

# Energy concentration
for r_px_check in [1, 2, 3, 5, 10]:
    r_check = r_px_check * dx_f
    cy, cx = N//2, N//2
    yy_f = (np.arange(N) - cy) * dx_f
    xx_f = (np.arange(N) - cx) * dx_f
    yyf, xxf = np.meshgrid(yy_f, xx_f, indexing="ij")
    mask_f = (xxf**2 + yyf**2) <= r_check**2
    frac_f = (irr_f * mask_f).sum() / irr_f.sum()
    print(f"  Energy within {r_px_check}px ({r_check*1e6:.1f}μm): {frac_f:.4f}")
