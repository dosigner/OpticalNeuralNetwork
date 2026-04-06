#!/usr/bin/env python
"""Zernike decomposition of residual phase: Turbulent vs D2NN.

Decomposes residual wavefront (vs vacuum) into Zernike modes (Noll ordering)
and compares coefficients before/after D2NN.

Usage:
    cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python scripts/zernike_decomposition.py \
        --sweep autoresearch/runs/0401-focal-pib-sweep-clean-4loss-cn2-5e14 \
        --strategy focal_pib_only --n-samples 100
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
    "font.family": "sans-serif",
    "font.sans-serif": ["Roboto", "Open Sans", "DejaVu Sans"],
    "font.size": 13,
})

# ── Zernike polynomials (Noll ordering) ──────────────────────────

NOLL_TO_NM = [
    (0, 0),   # j=1: piston
    (1, 1),   # j=2: tilt (x)
    (1, -1),  # j=3: tip (y)
    (2, 0),   # j=4: defocus
    (2, -2),  # j=5: oblique astig
    (2, 2),   # j=6: vertical astig
    (3, -1),  # j=7: vertical coma
    (3, 1),   # j=8: horizontal coma
    (3, -3),  # j=9: vertical trefoil
    (3, 3),   # j=10: oblique trefoil
    (4, 0),   # j=11: primary spherical
    (4, 2),   # j=12
    (4, -2),  # j=13
    (4, 4),   # j=14
    (4, -4),  # j=15
    (5, 1),   # j=16
    (5, -1),  # j=17
    (5, 3),   # j=18
    (5, -3),  # j=19
    (5, 5),   # j=20
    (5, -5),  # j=21
    (6, 0),   # j=22: secondary spherical
]

ZERNIKE_NAMES = [
    "Piston", "Tilt X", "Tip Y",
    "Defocus", "Astig 45°", "Astig 0°",
    "Coma Y", "Coma X", "Trefoil Y", "Trefoil X",
    "Spherical", "2nd Astig 0°", "2nd Astig 45°", "Quadrafoil 0°", "Quadrafoil 45°",
    "2nd Coma X", "2nd Coma Y", "2nd Trefoil X", "2nd Trefoil Y", "Pentafoil X", "Pentafoil Y",
    "2nd Spherical",
]


def zernike_radial(n, m, rho):
    """Radial polynomial R_n^m(rho)."""
    m_abs = abs(m)
    result = np.zeros_like(rho)
    for s in range((n - m_abs) // 2 + 1):
        coeff = ((-1) ** s * math.factorial(n - s)
                 / (math.factorial(s)
                    * math.factorial((n + m_abs) // 2 - s)
                    * math.factorial((n - m_abs) // 2 - s)))
        result += coeff * rho ** (n - 2 * s)
    return result


def zernike_basis(n_modes, grid_size, normalize=True):
    """Generate Zernike basis functions on unit circle.

    Returns: basis [n_modes, grid_size, grid_size], mask [grid_size, grid_size]
    """
    c = grid_size / 2
    yy, xx = np.mgrid[:grid_size, :grid_size]
    xx = (xx - c + 0.5) / c
    yy = (yy - c + 0.5) / c
    rho = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(yy, xx)
    mask = rho <= 1.0

    basis = np.zeros((n_modes, grid_size, grid_size))
    for j in range(n_modes):
        n, m = NOLL_TO_NM[j]
        R = zernike_radial(n, m, rho)
        if m == 0:
            Z = R
        elif m > 0:
            Z = R * np.cos(m * theta)
        else:
            Z = R * np.sin(abs(m) * theta)

        if normalize:
            if m == 0:
                norm = np.sqrt(n + 1)
            else:
                norm = np.sqrt(2 * (n + 1))
            Z *= norm

        Z *= mask
        basis[j] = Z

    return basis, mask


def decompose(phase_2d, basis, mask, weight=None):
    """Project phase onto Zernike basis. Returns coefficients [n_modes]."""
    n_modes = basis.shape[0]
    phase_masked = phase_2d * mask

    if weight is not None:
        w = weight * mask
        w = w / w.sum()
    else:
        w = mask / mask.sum()

    coeffs = np.zeros(n_modes)
    for j in range(n_modes):
        coeffs[j] = (w * phase_masked * basis[j]).sum() / (w * basis[j] ** 2).sum()
    return coeffs


def prep(f):
    return center_crop_field(apply_receiver_aperture(f, receiver_window_m=WIN, aperture_diameter_m=APT), crop_n=N)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--n-modes", type=int, default=22)
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

    # Build Zernike basis
    print(f"Building Zernike basis ({args.n_modes} modes)...")
    basis, zmask = zernike_basis(args.n_modes, N)

    n_samples = min(args.n_samples, len(ds))
    print(f"Decomposing {n_samples} samples...")

    all_turb = []
    all_d2nn = []

    with torch.no_grad():
        for i in range(n_samples):
            s = ds[i]
            inp = prep(s["u_turb"].unsqueeze(0).to(device))
            tgt = prep(s["u_vacuum"].unsqueeze(0).to(device))

            vac_out = d0(tgt)
            turb_out = d0(inp)
            d2nn_out = m(inp)

            ref_phase = torch.angle(vac_out[0])
            wt = vac_out[0].abs().square()
            wt = wt / wt.sum()
            wt_np = wt.cpu().numpy()

            for label, field, storage in [("turb", turb_out, all_turb),
                                           ("d2nn", d2nn_out, all_d2nn)]:
                aligned = align_global_phase(field, vac_out)
                pd = torch.angle(aligned[0]) - ref_phase
                pd = torch.remainder(pd + math.pi, 2 * math.pi) - math.pi
                pd_np = pd.cpu().numpy()

                coeffs = decompose(pd_np, basis, zmask, wt_np)
                storage.append(coeffs)

            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{n_samples}", flush=True)

    all_turb = np.array(all_turb)  # [n_samples, n_modes]
    all_d2nn = np.array(all_d2nn)

    # Convert to nm: coeff [rad] * lambda / (2*pi) * 1e9
    scale = W / (2 * math.pi) * 1e9
    turb_nm = all_turb * scale
    d2nn_nm = all_d2nn * scale

    turb_mean = turb_nm.mean(axis=0)
    d2nn_mean = d2nn_nm.mean(axis=0)
    turb_std = turb_nm.std(axis=0)
    d2nn_std = d2nn_nm.std(axis=0)

    # RMS per mode (sqrt of mean of squared coefficients)
    turb_rms = np.sqrt((turb_nm ** 2).mean(axis=0))
    d2nn_rms = np.sqrt((d2nn_nm ** 2).mean(axis=0))

    # ── Print ──
    print("\n" + "=" * 80)
    print(f"ZERNIKE DECOMPOSITION — {n_samples} samples, {args.n_modes} modes")
    print("=" * 80)
    print(f"{'j':>3} {'Mode':<20} {'Turb RMS':>10} {'D2NN RMS':>10} {'Delta':>10} {'Change':>8}")
    print("-" * 72)
    for j in range(args.n_modes):
        name = ZERNIKE_NAMES[j] if j < len(ZERNIKE_NAMES) else f"Z{j+1}"
        delta = d2nn_rms[j] - turb_rms[j]
        pct = delta / max(turb_rms[j], 0.01) * 100
        print(f"{j+1:>3} {name:<20} {turb_rms[j]:>8.1f}nm {d2nn_rms[j]:>8.1f}nm {delta:>+8.1f}nm {pct:>+7.1f}%")

    total_turb = np.sqrt((turb_rms ** 2).sum())
    total_d2nn = np.sqrt((d2nn_rms ** 2).sum())
    print("-" * 72)
    print(f"    {'Total (RSS)':<20} {total_turb:>8.1f}nm {total_d2nn:>8.1f}nm {total_d2nn-total_turb:>+8.1f}nm")

    # Skip piston for higher-order
    ho_turb = np.sqrt((turb_rms[1:] ** 2).sum())
    ho_d2nn = np.sqrt((d2nn_rms[1:] ** 2).sum())
    print(f"    {'w/o Piston (RSS)':<20} {ho_turb:>8.1f}nm {ho_d2nn:>8.1f}nm {ho_d2nn-ho_turb:>+8.1f}nm")

    tt_turb = np.sqrt((turb_rms[1:3] ** 2).sum())
    tt_d2nn = np.sqrt((d2nn_rms[1:3] ** 2).sum())
    ho2_turb = np.sqrt((turb_rms[3:] ** 2).sum())
    ho2_d2nn = np.sqrt((d2nn_rms[3:] ** 2).sum())
    print(f"    {'Tip/Tilt only':<20} {tt_turb:>8.1f}nm {tt_d2nn:>8.1f}nm {tt_d2nn-tt_turb:>+8.1f}nm")
    print(f"    {'Higher-order (j>=4)':<20} {ho2_turb:>8.1f}nm {ho2_d2nn:>8.1f}nm {ho2_d2nn-ho2_turb:>+8.1f}nm")

    # ── Figure 1: RMS per mode (bar chart) ──
    fig, ax = plt.subplots(figsize=(18, 8))
    x = np.arange(args.n_modes)
    w = 0.35
    bars_t = ax.bar(x - w/2, turb_rms, w, label="Turbulent", color="#6B7280", alpha=0.85, edgecolor="black", lw=0.5)
    bars_d = ax.bar(x + w/2, d2nn_rms, w, label="D2NN", color="#DC2626", alpha=0.85, edgecolor="black", lw=0.5)

    labels = [ZERNIKE_NAMES[j] if j < len(ZERNIKE_NAMES) else f"Z{j+1}" for j in range(args.n_modes)]
    ax.set_xticks(x)
    ax.set_xticklabels([f"Z{j+1}\n{labels[j]}" for j in range(args.n_modes)],
                        fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("RMS coefficient [nm]", fontsize=14)
    ax.set_title(f"Zernike Mode RMS — Turbulent vs D2NN ({n_samples} samples)\n"
                 f"Residual phase at output plane (vs vacuum)",
                 fontsize=16, fontweight="bold")
    ax.legend(fontsize=14, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate improvement
    for j in range(args.n_modes):
        if turb_rms[j] > 5:  # only annotate significant modes
            delta = d2nn_rms[j] - turb_rms[j]
            pct = delta / turb_rms[j] * 100
            ymax = max(turb_rms[j], d2nn_rms[j])
            color = "#059669" if delta < 0 else "#DC2626"
            ax.text(j, ymax + 2, f"{pct:+.0f}%", ha="center", fontsize=8,
                    color=color, fontweight="bold")

    plt.tight_layout()
    out1 = OUT / "13_zernike_rms_turb_vs_d2nn.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out1}")

    # ── Figure 2: Improvement per mode ──
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), gridspec_kw={"height_ratios": [2, 1]})

    # Top: absolute RMS
    ax1.bar(x - w/2, turb_rms, w, label="Turbulent", color="#6B7280", alpha=0.85, edgecolor="black", lw=0.5)
    ax1.bar(x + w/2, d2nn_rms, w, label="D2NN", color="#DC2626", alpha=0.85, edgecolor="black", lw=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Z{j+1}" for j in range(args.n_modes)], fontsize=10)
    ax1.set_ylabel("RMS [nm]", fontsize=13)
    ax1.set_title(f"Zernike Mode Analysis — Turbulent vs D2NN ({n_samples} samples)",
                  fontsize=16, fontweight="bold")
    ax1.legend(fontsize=13)
    ax1.grid(True, alpha=0.3, axis="y")

    # Bottom: change (D2NN - Turb)
    delta = d2nn_rms - turb_rms
    colors = ["#059669" if d < 0 else "#DC2626" for d in delta]
    ax2.bar(x, delta, 0.6, color=colors, edgecolor="black", lw=0.5, alpha=0.85)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Z{j+1}\n{labels[j]}" for j in range(args.n_modes)],
                         fontsize=9, rotation=45, ha="right")
    ax2.set_ylabel("\u0394 RMS [nm] (D2NN \u2212 Turb)", fontsize=13)
    ax2.set_title("Change per mode (green = improved, red = worsened)", fontsize=14)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out2 = OUT / "14_zernike_delta_per_mode.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {out2}")

    del m, d0
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
