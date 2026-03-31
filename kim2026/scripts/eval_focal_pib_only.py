#!/usr/bin/env python
"""Quick test eval + visualization for focal_pib_only strategy.

Compares: Vacuum vs Turbulent (no D2NN) vs D2NN (focal PIB trained)
at focal plane with PIB@[5, 10, 25, 50]μm.

NOTE: Shared constants and utilities available in kim2026.eval.focal_utils.
New scripts should import from there instead of duplicating code.

Usage:
    cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python scripts/eval_focal_pib_only.py
"""
from __future__ import annotations
import json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from kim2026.data.dataset import CachedFieldDataset
from kim2026.models.d2nn import BeamCleanupD2NN
from kim2026.optics.lens_2f import lens_2f_forward
from kim2026.training.targets import apply_receiver_aperture, center_crop_field
from kim2026.training.metrics import complex_overlap, strehl_ratio

W = 1.55e-6; N = 1024; WIN = 0.002048; APT = 0.002; DX = WIN / N; F = 4.5e-3
ARCH = dict(num_layers=5, layer_spacing_m=10e-3, detector_distance_m=10e-3)
RADII = [5, 10, 25, 50]

DATA_DIR = Path("data/kim2026/1km_cn2_5e-14_tel15cm_n1024_br75")
SWEEP = Path("autoresearch/runs/d2nn_focal_pib_sweep")
OUT = SWEEP / "focal_pib_only"


def prep(f):
    return center_crop_field(apply_receiver_aperture(f, receiver_window_m=WIN, aperture_diameter_m=APT), crop_n=N)


def foc(field):
    with torch.no_grad():
        f, dx = lens_2f_forward(field.to(torch.complex64), dx_in_m=DX,
                                 wavelength_m=W, f_m=F, na=None, apply_scaling=False)
    return f, dx


def compute_pibs(focal_field, dx_f, radii):
    intensity = focal_field.abs().square()
    n = N; c = n // 2
    yy, xx = torch.meshgrid(torch.arange(n, device=focal_field.device) - c,
                             torch.arange(n, device=focal_field.device) - c, indexing="ij")
    r = torch.sqrt((xx * dx_f) ** 2 + (yy * dx_f) ** 2)
    total = intensity.sum(dim=(-2, -1)).clamp(min=1e-12)
    return {rad: ((intensity * (r <= rad * 1e-6).float()).sum(dim=(-2, -1)) / total).mean().item()
            for rad in radii}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    ckpt = OUT / "checkpoint.pt"
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}")
        return
    m = BeamCleanupD2NN(n=N, wavelength_m=W, window_m=WIN, **ARCH)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["model_state_dict"])
    m = m.to(device); m.eval()
    d0 = BeamCleanupD2NN(n=N, wavelength_m=W, window_m=WIN, **ARCH).to(device); d0.eval()
    print("Models loaded")

    # Test data
    ds = CachedFieldDataset(cache_dir=str(DATA_DIR / "cache"),
                             manifest_path=str(DATA_DIR / "split_manifest.json"), split="test")
    loader = DataLoader(ds, batch_size=16, num_workers=0)
    print(f"Test: {len(ds)} samples")

    # Evaluate
    stats = {k: {f"pib{r}": [] for r in RADII} for k in ["vacuum", "turbulent", "d2nn"]}
    for k in stats:
        stats[k]["co"] = []

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            inp = prep(batch["u_turb"].to(device))
            tgt = prep(batch["u_vacuum"].to(device))
            vac_out = d0(tgt); turb_out = d0(inp); d2nn_out = m(inp)

            for label, field in [("vacuum", vac_out), ("turbulent", turb_out), ("d2nn", d2nn_out)]:
                ff, dx_f = foc(field)
                pibs = compute_pibs(ff, dx_f, RADII)
                for r in RADII:
                    stats[label][f"pib{r}"].append(pibs[r])
                if label != "vacuum":
                    stats[label]["co"].append(complex_overlap(field, tgt).mean().item())
                del ff
            torch.cuda.empty_cache()
            if bi % 10 == 0:
                print(f"  batch {bi}/{len(loader)}", flush=True)

    # Average
    avg = {}
    for label in stats:
        avg[label] = {k: float(np.mean(v)) for k, v in stats[label].items() if v}

    # Print
    print(f"\n{'='*70}")
    print("FOCAL PIB ONLY — TEST RESULTS")
    print(f"{'='*70}")
    header = f"{'':>12} | {'CO':>6} | " + " | ".join(f"PIB@{r}um" for r in RADII)
    print(header)
    print("-" * len(header))
    for label in ["vacuum", "turbulent", "d2nn"]:
        co = f"{avg[label].get('co', 0):.4f}" if avg[label].get("co") else "  -   "
        pibs = " | ".join(f"  {avg[label].get(f'pib{r}', 0):.4f} " for r in RADII)
        print(f"{label:>12} | {co} | {pibs}")

    # Improvement
    print(f"\nImprovement (D2NN / Turbulent):")
    for r in RADII:
        t = avg["turbulent"].get(f"pib{r}", 1e-12)
        d = avg["d2nn"].get(f"pib{r}", 0)
        print(f"  PIB@{r}um: {d/t:.2f}x ({t:.4f} -> {d:.4f})")

    with open(OUT / "test_eval.json", "w") as f:
        json.dump(avg, f, indent=2)

    # === Visualization ===
    print("\nGenerating figures...")
    s = ds[0]
    u_t = prep(s["u_turb"].unsqueeze(0).to(device))
    u_v = prep(s["u_vacuum"].unsqueeze(0).to(device))
    with torch.no_grad():
        vac_d = d0(u_v); turb_d = d0(u_t); d2nn_d = m(u_t)
        fvac, dx_f = foc(vac_d); fturb, _ = foc(turb_d); fd2nn, _ = foc(d2nn_d)

    fvac_np = fvac[0].cpu().numpy()
    fturb_np = fturb[0].cpu().numpy()
    fd2nn_np = fd2nn[0].cpu().numpy()
    Z = 64; c = N // 2
    dx_f = float(dx_f)
    e = [-(Z*dx_f*1e6), Z*dx_f*1e6, -(Z*dx_f*1e6), Z*dx_f*1e6]
    theta = np.linspace(0, 2*np.pi, 200)

    # Fig 1: 3×3 comparison
    fig, axes = plt.subplots(3, 3, figsize=(21, 21))
    fig.suptitle("Focal PIB Only - Vacuum vs Turbulent vs D2NN\n"
                 f"f={F*1e3:.1f}mm, dx_focal={dx_f*1e6:.3f}μm",
                 fontsize=16, fontweight="bold")

    fields = {"Vacuum": fvac_np, "Turbulent\n(no D2NN)": fturb_np, "D2NN\n(focal PIB)": fd2nn_np}
    imax = np.max([np.abs(f[c-Z:c+Z, c-Z:c+Z])**2 for f in [fvac_np, fturb_np, fd2nn_np]])

    for col, (label, field) in enumerate(fields.items()):
        crop = field[c-Z:c+Z, c-Z:c+Z]
        lbl_key = ["vacuum", "turbulent", "d2nn"][col]
        pib10 = avg[lbl_key].get("pib10", 0)
        pib50 = avg[lbl_key].get("pib50", 0)

        axes[0, col].imshow(np.abs(crop)**2, extent=e, origin="lower", cmap="inferno", vmin=0, vmax=imax)
        axes[0, col].set_title(f"{label}\nPIB@10μm={pib10:.4f} | @50μm={pib50:.4f}",
                                fontsize=11, fontweight="bold")
        axes[0, col].plot(10*np.cos(theta), 10*np.sin(theta), 'r-', lw=2, alpha=0.9)
        axes[0, col].plot(50*np.cos(theta), 50*np.sin(theta), 'w--', lw=1.5, alpha=0.6)

        log_irr = np.log10(np.abs(crop)**2 + 1e-15)
        axes[1, col].imshow(log_irr, extent=e, origin="lower", cmap="viridis",
                            vmin=np.log10(imax)-4, vmax=np.log10(imax))
        axes[1, col].plot(10*np.cos(theta), 10*np.sin(theta), 'r-', lw=2, alpha=0.9)
        axes[1, col].plot(50*np.cos(theta), 50*np.sin(theta), 'w--', lw=1.5, alpha=0.6)

        x_um = np.linspace(e[0], e[1], 2*Z)
        axes[2, col].plot(x_um, np.abs(fvac_np[c, c-Z:c+Z])**2, 'b--', lw=1.5, label='Vacuum')
        axes[2, col].plot(x_um, np.abs(field[c, c-Z:c+Z])**2, 'r-', lw=2, label=label.split('\n')[0])
        axes[2, col].axvline(-10, color='red', ls=':', lw=1); axes[2, col].axvline(10, color='red', ls=':', lw=1)
        axes[2, col].axvline(-50, color='cyan', ls='--', lw=0.8); axes[2, col].axvline(50, color='cyan', ls='--', lw=0.8)
        axes[2, col].legend(fontsize=9); axes[2, col].grid(True, alpha=0.3); axes[2, col].set_xlabel("μm")

    for r, lbl in enumerate(["Focal irradiance\n(10μm=red, 50μm=white)", "Log irradiance", "1D cross-section"]):
        axes[r, 0].set_ylabel(lbl, fontsize=11, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "test_viz_comparison.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  Saved test_viz_comparison.png")

    # Fig 2: Bar chart
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    fig2.suptitle("Focal PIB - Bucket Radius Comparison\nVacuum vs Turbulent vs D2NN",
                  fontsize=14, fontweight="bold")
    x = np.arange(len(RADII)); w = 0.25
    for i, (label, color) in enumerate([("vacuum", "blue"), ("turbulent", "gray"), ("d2nn", "red")]):
        vals = [avg[label].get(f"pib{r}", 0) for r in RADII]
        bars = ax2.bar(x + i*w, vals, w, label=label.capitalize(), color=color, alpha=0.8, edgecolor='black', lw=0.5)
        for b, v in zip(bars, vals):
            ax2.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax2.set_xticks(x + w); ax2.set_xticklabels([f"{r}μm" for r in RADII], fontsize=12)
    ax2.set_xlabel("Bucket Radius", fontsize=13); ax2.set_ylabel("PIB", fontsize=13)
    ax2.legend(fontsize=12); ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig2.savefig(OUT / "test_viz_pib_bars.png", dpi=150, bbox_inches="tight"); plt.close(fig2)
    print("  Saved test_viz_pib_bars.png")

    del m, d0; torch.cuda.empty_cache()
    print("\nDone!")


if __name__ == "__main__":
    main()
