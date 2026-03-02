"""Diagnose Fourier-domain intensity distribution for MNIST inputs.

Computes statistics of |FFT(input)|^2 to determine optimal I_sat for SBN
in Fourier domain with background_perturbation normalization.
"""

import sys
sys.path.insert(0, "src")

import torch
import numpy as np
from tao2019_fd2nn.cli.common import load_config, build_dataloaders, build_model
from tao2019_fd2nn.optics.fft2c import fft2c
from tao2019_fd2nn.optics.aperture import na_mask


def main():
    # Load Fourier nonlinear config
    cfg = load_config("src/tao2019_fd2nn/config/cls_mnist_nonlinear_fourier_5l_f4mm.yaml")
    _, val_loader = build_dataloaders(cfg)

    # Collect first 1024 samples
    all_intensities = []
    dc_intensities = []
    non_dc_intensities = []

    N = 200
    dx_m = 1e-6
    wavelength_m = 5.32e-7
    na_val = 0.16
    f_m = 4e-3

    # Compute NA mask
    mask = na_mask(N=N, dx_m=dx_m, wavelength_m=wavelength_m, na=na_val, shifted=True)
    center = N // 2

    for batch_idx, (fields, labels) in enumerate(val_loader):
        # fields shape: (B, 1, N, N) complex
        fields = fields.squeeze(1)  # (B, N, N)

        # Apply FFT (same as fft2c domain switch)
        U = fft2c(fields)  # centered Fourier domain

        # Apply NA mask
        U = U * mask.unsqueeze(0)

        # Compute intensity
        I = (U.real ** 2 + U.imag ** 2)  # |U|^2

        for i in range(I.shape[0]):
            img_I = I[i]  # (N, N)
            dc_val = img_I[center, center].item()
            dc_intensities.append(dc_val)

            # Non-DC values (inside NA mask)
            mask_bool = mask > 0.5
            non_dc_mask = mask_bool.clone()
            non_dc_mask[center, center] = False
            non_dc_vals = img_I[non_dc_mask].numpy()
            non_dc_intensities.append(non_dc_vals)
            all_intensities.append(img_I[mask_bool].numpy())

        if batch_idx >= 4:  # ~5000 samples
            break

    dc_arr = np.array(dc_intensities)
    non_dc_all = np.concatenate(non_dc_intensities)
    all_all = np.concatenate(all_intensities)

    print("=" * 60)
    print("Fourier-domain Intensity Statistics (MNIST, ortho FFT + NA mask)")
    print("=" * 60)
    print(f"\nSamples analyzed: {len(dc_intensities)}")
    print(f"Grid: {N}x{N}, dx={dx_m*1e6:.1f}um, NA={na_val}, f={f_m*1e3:.0f}mm")

    print(f"\n--- DC Component |U(0,0)|^2 ---")
    print(f"  Mean:   {dc_arr.mean():.6f}")
    print(f"  Std:    {dc_arr.std():.6f}")
    print(f"  Min:    {dc_arr.min():.6f}")
    print(f"  Max:    {dc_arr.max():.6f}")

    print(f"\n--- Non-DC Components ---")
    print(f"  Total bins per sample (inside NA): {non_dc_intensities[0].shape[0]}")
    print(f"  Mean:   {non_dc_all.mean():.8f}")
    print(f"  Median: {np.median(non_dc_all):.8f}")
    print(f"  Std:    {non_dc_all.std():.8f}")
    print(f"  P25:    {np.percentile(non_dc_all, 25):.8f}")
    print(f"  P75:    {np.percentile(non_dc_all, 75):.8f}")
    print(f"  P90:    {np.percentile(non_dc_all, 90):.8f}")
    print(f"  P99:    {np.percentile(non_dc_all, 99):.8f}")
    print(f"  Max:    {non_dc_all.max():.8f}")

    print(f"\n--- Dynamic Range ---")
    print(f"  DC/mean_nonDC ratio: {dc_arr.mean() / non_dc_all.mean():.1f}x")
    print(f"  DC/median_nonDC ratio: {dc_arr.mean() / np.median(non_dc_all):.1f}x")

    # Suggest I_sat values
    print(f"\n--- Suggested I_sat values for background_perturbation ---")
    for percentile in [50, 75, 90]:
        i_sat = np.percentile(non_dc_all, percentile)
        print(f"  P{percentile} non-DC ({i_sat:.6f}):")
        eta_dc = dc_arr.mean() / i_sat
        eta_avg = non_dc_all.mean() / i_sat
        eta_med = np.median(non_dc_all) / i_sat
        print(f"    DC:  eta={eta_dc:.1f} -> phase={np.pi * eta_dc/(1+eta_dc):.3f} rad")
        print(f"    Avg: eta={eta_avg:.2f} -> phase={np.pi * eta_avg/(1+eta_avg):.3f} rad")
        print(f"    Med: eta={eta_med:.2f} -> phase={np.pi * eta_med/(1+eta_med):.3f} rad")



if __name__ == "__main__":
    main()
