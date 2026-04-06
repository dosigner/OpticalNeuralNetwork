#!/usr/bin/env python
"""Compute piston-removed (and piston+tip/tilt-removed) residual WF RMS.

Compares 3 groups: Vacuum, Turbulent, D2NN at the output plane.
Reference: vacuum output field.

Usage:
    cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python scripts/compute_piston_removed_wfrms.py \
        --sweep autoresearch/runs/0401-focal-pib-sweep-clean-4loss-cn2-5e14 \
        --strategy focal_pib_only
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import torch

from kim2026.data.dataset import CachedFieldDataset
from kim2026.models.d2nn import BeamCleanupD2NN
from kim2026.training.targets import apply_receiver_aperture, center_crop_field
from kim2026.training.losses import align_global_phase

W = 1.55e-6; N = 1024; WIN = 0.002048; APT = 0.002; DX = WIN / N


def prep(f):
    return center_crop_field(apply_receiver_aperture(f, receiver_window_m=WIN, aperture_diameter_m=APT), crop_n=N)


def remove_piston(phase, weight):
    """Remove intensity-weighted mean phase (piston)."""
    piston = (weight * phase).sum()
    return phase - piston


def remove_piston_tiptilt(phase, weight):
    """Remove piston + tip/tilt via weighted least-squares on (x, y, 1)."""
    n = phase.shape[-1]
    c = n // 2
    yy, xx = torch.meshgrid(
        torch.arange(n, device=phase.device, dtype=torch.float32) - c,
        torch.arange(n, device=phase.device, dtype=torch.float32) - c,
        indexing="ij",
    )
    # Flatten
    ph = phase.flatten()
    w = weight.flatten()
    X = torch.stack([torch.ones_like(ph), xx.flatten(), yy.flatten()], dim=1)  # [N^2, 3]
    # Weighted least squares: (X^T W X)^{-1} X^T W ph
    Xw = X * w.unsqueeze(1)
    A = Xw.T @ X  # [3, 3]
    b = Xw.T @ ph  # [3]
    coeffs = torch.linalg.solve(A, b)  # [piston, tip, tilt]
    fit = (X @ coeffs).reshape(n, n)
    return phase - fit, coeffs


def compute_wfrms(phase_residual, weight, wavelength=W):
    """Intensity-weighted WF RMS in nm."""
    rms_rad = torch.sqrt((weight * phase_residual**2).sum()).item()
    return rms_rad * wavelength / (2 * math.pi) * 1e9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--arch-pad", type=int, default=2)
    parser.add_argument("--data", type=str, default=None, help="data directory override")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ARCH = dict(num_layers=5, layer_spacing_m=10e-3, detector_distance_m=10e-3,
                propagation_pad_factor=args.arch_pad)
    OUT = Path(args.sweep) / args.strategy
    ckpt = OUT / "checkpoint.pt"
    if not ckpt.exists():
        print(f"No checkpoint at {ckpt}"); return

    # Load models
    m = BeamCleanupD2NN(n=N, wavelength_m=W, window_m=WIN, **ARCH)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["model_state_dict"])
    m = m.to(device); m.eval()
    d0 = BeamCleanupD2NN(n=N, wavelength_m=W, window_m=WIN, **ARCH).to(device); d0.eval()

    # Load data
    data_dir = args.data if args.data else "data/kim2026/1km_cn2_5e-14_tel15cm_dn100um_lanczos50"
    ds = CachedFieldDataset(cache_dir=f"{data_dir}/cache",
                            manifest_path=f"{data_dir}/split_manifest.json", split="test")

    n_samples = min(args.n_samples, len(ds))
    print(f"Computing piston-removed WF RMS for {n_samples} samples...")

    # Storage
    results = {
        "piston_removed": {"vacuum": [], "turbulent": [], "d2nn": []},
        "piston_tiptilt_removed": {"vacuum": [], "turbulent": [], "d2nn": []},
        "raw_align_global_phase": {"vacuum": [], "turbulent": [], "d2nn": []},
    }

    with torch.no_grad():
        for i in range(n_samples):
            s = ds[i]
            inp = prep(s["u_turb"].unsqueeze(0).to(device))
            tgt = prep(s["u_vacuum"].unsqueeze(0).to(device))

            vac_out = d0(tgt)
            turb_out = d0(inp)
            d2nn_out = m(inp)

            # Reference: vacuum output
            ref_phase = torch.angle(vac_out[0])
            w = vac_out[0].abs().square()
            w = w / w.sum()

            for label, field in [("vacuum", vac_out), ("turbulent", turb_out), ("d2nn", d2nn_out)]:
                # Align global phase first
                aligned = align_global_phase(field, vac_out)
                # Phase residual vs vacuum
                pd = torch.angle(aligned[0]) - ref_phase
                pd = torch.remainder(pd + math.pi, 2 * math.pi) - math.pi

                # Method 1: raw (align_global_phase only, same as eval_strategy.py)
                rms_raw = compute_wfrms(pd, w)
                results["raw_align_global_phase"][label].append(rms_raw)

                # Method 2: explicit piston removal
                pd_no_piston = remove_piston(pd, w)
                rms_piston = compute_wfrms(pd_no_piston, w)
                results["piston_removed"][label].append(rms_piston)

                # Method 3: piston + tip/tilt removal
                pd_no_ptt, coeffs = remove_piston_tiptilt(pd, w)
                rms_ptt = compute_wfrms(pd_no_ptt, w)
                results["piston_tiptilt_removed"][label].append(rms_ptt)

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{n_samples}", flush=True)

    # Summarize
    print("\n" + "="*70)
    print("PISTON-REMOVED RESIDUAL WF RMS (output plane, vs vacuum)")
    print("="*70)

    summary = {}
    for method in results:
        print(f"\n--- {method} ---")
        summary[method] = {}
        for label in ["vacuum", "turbulent", "d2nn"]:
            vals = results[method][label]
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            summary[method][label] = {"mean_nm": mean, "std_nm": std}
            print(f"  {label:12s}: {mean:7.1f} +/- {std:5.1f} nm")

    # Save
    out_path = OUT / "10_piston_removed_wfrms.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Also print nicely formatted for presentation
    print("\n" + "="*70)
    print("SUMMARY TABLE FOR PRESENTATION")
    print("="*70)
    print(f"{'Method':<30} {'Vacuum':>12} {'Turbulent':>15} {'D2NN':>15} {'Improvement':>12}")
    print("-"*84)
    for method in results:
        v = summary[method]["vacuum"]["mean_nm"]
        t = summary[method]["turbulent"]["mean_nm"]
        d = summary[method]["d2nn"]["mean_nm"]
        impr = t - d
        print(f"{method:<30} {v:>10.1f}nm {t:>10.1f}nm+/-{summary[method]['turbulent']['std_nm']:.1f} "
              f"{d:>10.1f}nm+/-{summary[method]['d2nn']['std_nm']:.1f} {impr:>+10.1f}nm")

    del m, d0
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
