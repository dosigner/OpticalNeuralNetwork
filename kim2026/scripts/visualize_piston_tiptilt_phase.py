#!/usr/bin/env python
"""Visualize residual phase maps: piston-removed vs piston+TT-removed.

2x2 grid: (Turbulent, D2NN) x (Piston removed, Piston+TT removed)
+ difference maps showing what TT removal subtracts.

Usage:
    cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python scripts/visualize_piston_tiptilt_phase.py \
        --sweep autoresearch/runs/0401-focal-pib-sweep-clean-4loss-cn2-5e14 \
        --strategy focal_pib_only --sample 0
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from kim2026.data.dataset import CachedFieldDataset
from kim2026.models.d2nn import BeamCleanupD2NN
from kim2026.training.targets import apply_receiver_aperture, center_crop_field
from kim2026.training.losses import align_global_phase

W = 1.55e-6; N = 1024; WIN = 0.002048; APT = 0.002; DX = WIN / N

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Roboto', 'Open Sans', 'DejaVu Sans'],
    'font.size': 13,
})


def prep(f):
    return center_crop_field(apply_receiver_aperture(f, receiver_window_m=WIN, aperture_diameter_m=APT), crop_n=N)


def remove_piston(phase, weight):
    piston = (weight * phase).sum()
    return phase - piston, piston


def remove_piston_tiptilt(phase, weight):
    n = phase.shape[-1]
    c = n // 2
    yy, xx = torch.meshgrid(
        torch.arange(n, device=phase.device, dtype=torch.float32) - c,
        torch.arange(n, device=phase.device, dtype=torch.float32) - c,
        indexing="ij",
    )
    ph = phase.flatten()
    w = weight.flatten()
    X = torch.stack([torch.ones_like(ph), xx.flatten(), yy.flatten()], dim=1)
    Xw = X * w.unsqueeze(1)
    A = Xw.T @ X
    b = Xw.T @ ph
    coeffs = torch.linalg.solve(A, b)
    fit = (X @ coeffs).reshape(n, n)
    return phase - fit, fit, coeffs


def wfrms_nm(phase, weight):
    return torch.sqrt((weight * phase**2).sum()).item() * W / (2 * math.pi) * 1e9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--arch-pad", type=int, default=2)
    parser.add_argument("--data", type=str, default=None, help="data directory override")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ARCH = dict(num_layers=5, layer_spacing_m=10e-3, detector_distance_m=10e-3,
                propagation_pad_factor=args.arch_pad)
    OUT = Path(args.sweep) / args.strategy
    ckpt = OUT / "checkpoint.pt"

    m = BeamCleanupD2NN(n=N, wavelength_m=W, window_m=WIN, **ARCH)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["model_state_dict"])
    m = m.to(device); m.eval()
    d0 = BeamCleanupD2NN(n=N, wavelength_m=W, window_m=WIN, **ARCH).to(device); d0.eval()

    data_dir = args.data if args.data else "data/kim2026/1km_cn2_5e-14_tel15cm_dn100um_lanczos50"
    ds = CachedFieldDataset(cache_dir=f"{data_dir}/cache",
                            manifest_path=f"{data_dir}/split_manifest.json", split="test")

    s = ds[args.sample]
    inp = prep(s["u_turb"].unsqueeze(0).to(device))
    tgt = prep(s["u_vacuum"].unsqueeze(0).to(device))

    with torch.no_grad():
        vac_out = d0(tgt)
        turb_out = d0(inp)
        d2nn_out = m(inp)

    ref_phase = torch.angle(vac_out[0])
    w = vac_out[0].abs().square()
    w = w / w.sum()

    # Aperture mask for display
    c = N // 2
    yy, xx = torch.meshgrid(
        torch.arange(N, device=device, dtype=torch.float32) - c,
        torch.arange(N, device=device, dtype=torch.float32) - c,
        indexing="ij",
    )
    apt_mask = (torch.sqrt(xx**2 + yy**2) * DX * 1e6 <= APT / 2 * 1e6).float()

    results = {}
    for label, field in [("Turbulent", turb_out), ("D2NN", d2nn_out)]:
        aligned = align_global_phase(field, vac_out)
        pd = torch.angle(aligned[0]) - ref_phase
        pd = torch.remainder(pd + math.pi, 2 * math.pi) - math.pi

        # Piston removed
        pd_nop, piston_val = remove_piston(pd, w)
        # Piston + TT removed
        pd_noptt, tt_fit, coeffs = remove_piston_tiptilt(pd, w)
        # The TT component = what was subtracted beyond piston
        tt_component = pd_nop - pd_noptt

        results[label] = {
            "raw": pd.cpu().numpy(),
            "piston_removed": pd_nop.cpu().numpy(),
            "piston_tt_removed": pd_noptt.cpu().numpy(),
            "tt_component": tt_component.cpu().numpy(),
            "rms_raw": wfrms_nm(pd, w),
            "rms_piston": wfrms_nm(pd_nop, w),
            "rms_ptt": wfrms_nm(pd_noptt, w),
            "rms_tt": wfrms_nm(tt_component, w),
            "coeffs": coeffs.cpu().numpy(),
        }

    apt_np = apt_mask.cpu().numpy()
    e_um = [-(N * DX / 2) * 1e6, (N * DX / 2) * 1e6, -(N * DX / 2) * 1e6, (N * DX / 2) * 1e6]

    # ─── Figure 1: 2x3 grid ───
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle(f"Residual Phase Analysis — Sample #{args.sample}\n"
                 f"(phase difference vs vacuum, output plane)", fontsize=18, fontweight="bold")

    vmax = np.pi
    cmap = "RdBu_r"

    for row, label in enumerate(["Turbulent", "D2NN"]):
        r = results[label]

        # Col 0: Piston removed
        data = np.where(apt_np > 0.5, r["piston_removed"], np.nan)
        im = axes[row, 0].imshow(data, extent=e_um, origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax)
        axes[row, 0].set_title(f"{label}\nPiston removed\nWF RMS = {r['rms_piston']:.1f} nm",
                                fontsize=14, fontweight="bold")

        # Col 1: Piston + TT removed
        data = np.where(apt_np > 0.5, r["piston_tt_removed"], np.nan)
        im = axes[row, 1].imshow(data, extent=e_um, origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax)
        axes[row, 1].set_title(f"{label}\nPiston + Tip/Tilt removed\nWF RMS = {r['rms_ptt']:.1f} nm",
                                fontsize=14, fontweight="bold")

        # Col 2: TT component (what was removed)
        data_tt = np.where(apt_np > 0.5, r["tt_component"], np.nan)
        tt_max = max(abs(np.nanmin(data_tt)), abs(np.nanmax(data_tt)), 0.5)
        im2 = axes[row, 2].imshow(data_tt, extent=e_um, origin="lower", cmap="coolwarm",
                                   vmin=-tt_max, vmax=tt_max)
        axes[row, 2].set_title(f"{label}\nTip/Tilt component (removed)\nRMS = {r['rms_tt']:.1f} nm",
                                fontsize=14, fontweight="bold")
        plt.colorbar(im2, ax=axes[row, 2], label="rad", shrink=0.8)

    # Shared colorbar for cols 0-1
    cbar_ax = fig.add_axes([0.08, 0.04, 0.55, 0.02])
    cb = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", label="Phase residual [rad]")

    for ax in axes.flat:
        ax.set_xlabel("\u03BCm"); ax.set_ylabel("\u03BCm")

    plt.tight_layout(rect=[0, 0.07, 1, 0.93])
    out1 = OUT / "11_residual_phase_piston_vs_tiptilt.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out1}")

    # ─── Figure 2: Side-by-side difference (D2NN vs Turbulent) ───
    fig2, axes2 = plt.subplots(1, 3, figsize=(21, 6))
    fig2.suptitle("D2NN vs Turbulent: Residual Phase Comparison", fontsize=18, fontweight="bold")

    for col, (mode, key) in enumerate([
        ("Piston removed", "piston_removed"),
        ("Piston+TT removed", "piston_tt_removed"),
        ("Difference\n(Turb − D2NN)", None),
    ]):
        if key is not None:
            # Show D2NN phase
            t_data = np.where(apt_np > 0.5, results["Turbulent"][key], np.nan)
            d_data = np.where(apt_np > 0.5, results["D2NN"][key], np.nan)

            # Absolute value comparison
            rms_key = "rms_piston" if key == "piston_removed" else "rms_ptt"
            t_rms = results["Turbulent"][rms_key]
            d_rms = results["D2NN"][rms_key]

            # Show D2NN
            im = axes2[col].imshow(d_data, extent=e_um, origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax)
            axes2[col].set_title(f"D2NN — {mode}\nWF RMS = {d_rms:.1f} nm\n(Turb: {t_rms:.1f} nm)",
                                  fontsize=14, fontweight="bold")
            plt.colorbar(im, ax=axes2[col], shrink=0.8, label="rad")
        else:
            # Difference: |turbulent|^2 - |d2nn|^2 phase power
            diff_ptt = np.where(apt_np > 0.5,
                                results["Turbulent"]["piston_tt_removed"] - results["D2NN"]["piston_tt_removed"],
                                np.nan)
            dmax = max(abs(np.nanmin(diff_ptt)), abs(np.nanmax(diff_ptt)), 0.5)
            im = axes2[col].imshow(diff_ptt, extent=e_um, origin="lower", cmap="PiYG",
                                    vmin=-dmax, vmax=dmax)
            t_rms = results["Turbulent"]["rms_ptt"]
            d_rms = results["D2NN"]["rms_ptt"]
            axes2[col].set_title(f"Phase difference (Turb − D2NN)\nafter Piston+TT removal\n"
                                  f"\u0394RMS = {t_rms - d_rms:.1f} nm",
                                  fontsize=14, fontweight="bold")
            plt.colorbar(im, ax=axes2[col], shrink=0.8, label="rad")

    for ax in axes2:
        ax.set_xlabel("\u03BCm"); ax.set_ylabel("\u03BCm")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out2 = OUT / "12_d2nn_vs_turbulent_phase_difference.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {out2}")

    # Print summary
    print("\n" + "=" * 60)
    print("RESIDUAL PHASE SUMMARY (Sample #{})".format(args.sample))
    print("=" * 60)
    for label in ["Turbulent", "D2NN"]:
        r = results[label]
        print(f"\n{label}:")
        print(f"  Raw (align_phase):     {r['rms_raw']:.1f} nm")
        print(f"  Piston removed:        {r['rms_piston']:.1f} nm")
        print(f"  Piston+TT removed:     {r['rms_ptt']:.1f} nm")
        print(f"  TT component:          {r['rms_tt']:.1f} nm")
        print(f"  TT coeffs [piston, tip, tilt]: {r['coeffs']}")

    del m, d0
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
