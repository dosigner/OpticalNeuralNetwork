#!/usr/bin/env python
"""Phase 3: PIB Bucket Radius Sweep — Evaluate trained models at multiple radii.

No retraining needed. Loads checkpoints from d2nn_focal_pib_sweep and
computes focal-plane PIB at [5, 10, 25, 50] μm for all strategies.

NOTE: Shared constants and utilities available in kim2026.eval.focal_utils.

Also computes:
  - Encircled energy curves (continuous)
  - Vacuum and turbulent baselines at each radius
  - SMF coupling efficiency estimate (5μm ≈ SMF-28 MFD/2)

Usage:
    cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python scripts/eval_bucket_radius_sweep.py
"""
from __future__ import annotations

import json
import math
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

W = 1.55e-6; N = 1024; WIN = 0.002048; APT = 0.002; DX = WIN / N
FOCUS_F = 4.5e-3
ARCH = dict(num_layers=5, layer_spacing_m=10e-3, detector_distance_m=10e-3)

BUCKET_RADII_UM = [5.0, 10.0, 25.0, 50.0]
EE_RADII_UM = np.linspace(1, 100, 200)

STRATEGIES = ["focal_pib_only", "focal_strehl_only", "focal_intensity_overlap", "focal_co_pib_hybrid"]
LABELS = {
    "focal_pib_only": "Focal PIB",
    "focal_strehl_only": "Focal Strehl",
    "focal_intensity_overlap": "Focal IO",
    "focal_co_pib_hybrid": "CO+fPIB",
}
COLORS = {
    "focal_pib_only": "#e74c3c",
    "focal_strehl_only": "#3498db",
    "focal_intensity_overlap": "#2ecc71",
    "focal_co_pib_hybrid": "#9b59b6",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kim2026" / "1km_cn2_5e-14_tel15cm_n1024_br75"
SWEEP_DIR = Path(__file__).resolve().parent.parent / "autoresearch" / "runs" / "d2nn_focal_pib_sweep"
OUT = SWEEP_DIR / "bucket_radius_sweep"


def prepare(f):
    return center_crop_field(apply_receiver_aperture(f, receiver_window_m=WIN, aperture_diameter_m=APT), crop_n=N)


def load_model(name):
    ckpt = SWEEP_DIR / name / "checkpoint.pt"
    if not ckpt.exists(): return None
    m = BeamCleanupD2NN(n=N, wavelength_m=W, window_m=WIN, **ARCH)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["model_state_dict"])
    m.eval(); return m


def focus(field):
    with torch.no_grad():
        f, dx = lens_2f_forward(field.to(torch.complex64), dx_in_m=DX, wavelength_m=W,
                                 f_m=FOCUS_F, na=None, apply_scaling=False)
    return f, dx


def compute_pib_multi(focal_field, dx_focal, radii_um):
    """Compute PIB at multiple radii simultaneously."""
    intensity = focal_field.abs().square()
    n = focal_field.shape[-1]; c = n // 2
    yy, xx = torch.meshgrid(torch.arange(n, device=focal_field.device) - c,
                             torch.arange(n, device=focal_field.device) - c, indexing="ij")
    r = torch.sqrt((xx * dx_focal) ** 2 + (yy * dx_focal) ** 2)
    total = intensity.sum(dim=(-2, -1)).clamp(min=1e-12)
    results = {}
    for rad in radii_um:
        mask = (r <= rad * 1e-6).float()
        pib = (intensity * mask).sum(dim=(-2, -1)) / total
        results[rad] = pib
    return results


def compute_ee_curve(focal_field, dx_focal, radii_um):
    """Compute encircled energy at fine radii for smooth curves."""
    intensity = focal_field.abs().square()
    n = focal_field.shape[-1]; c = n // 2
    yy, xx = torch.meshgrid(torch.arange(n, device=focal_field.device) - c,
                             torch.arange(n, device=focal_field.device) - c, indexing="ij")
    r = torch.sqrt((xx * dx_focal) ** 2 + (yy * dx_focal) ** 2)
    total = intensity.sum(dim=(-2, -1)).clamp(min=1e-12)
    ee = []
    for rad in radii_um:
        mask = (r <= rad * 1e-6).float()
        ee.append(((intensity * mask).sum(dim=(-2, -1)) / total).mean().item())
    return ee


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    OUT.mkdir(parents=True, exist_ok=True)

    dx_focal = W * FOCUS_F / (N * DX)
    print(f"dx_focal = {dx_focal*1e6:.3f} μm")
    print(f"Bucket radii: {BUCKET_RADII_UM} μm")
    print(f"Airy radius ≈ {1.22 * W * FOCUS_F / (APT / (WIN/DX) * 2) * 1e6:.1f} μm")  # rough estimate

    # Load test data
    test_ds = CachedFieldDataset(cache_dir=str(DATA_DIR / "cache"),
                                  manifest_path=str(DATA_DIR / "split_manifest.json"), split="test")
    test_loader = DataLoader(test_ds, batch_size=16, num_workers=0)
    print(f"Test: {len(test_ds)} samples")

    # Zero-phase baseline
    d0 = BeamCleanupD2NN(n=N, wavelength_m=W, window_m=WIN, **ARCH).to(device); d0.eval()

    # Load trained models
    models = {}
    for name in STRATEGIES:
        m = load_model(name)
        if m:
            models[name] = m.to(device)
            print(f"  Loaded {name}")
        else:
            print(f"  {name}: not found")

    if not models:
        print("No trained models found! Run focal_pib_sweep first.")
        return

    # === Evaluate ===
    print("\nEvaluating...")
    # Results: {strategy: {radius: [pib_per_sample]}}
    all_pibs = {name: {r: [] for r in BUCKET_RADII_UM} for name in ["vacuum", "turbulent"] + list(models.keys())}
    all_ee = {name: [] for name in ["vacuum", "turbulent"] + list(models.keys())}
    all_co = {name: [] for name in models.keys()}

    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            u_turb = batch["u_turb"].to(device)
            u_vac = batch["u_vacuum"].to(device)
            inp = prepare(u_turb)
            tgt = prepare(u_vac)

            # Vacuum and turbulent baselines through zero-phase D2NN
            vac_out = d0(tgt)
            turb_out = d0(inp)

            focal_vac, dx_f = focus(vac_out)
            focal_turb, _ = focus(turb_out)

            # Vacuum PIBs
            vac_pibs = compute_pib_multi(focal_vac, dx_f, BUCKET_RADII_UM)
            for r in BUCKET_RADII_UM:
                all_pibs["vacuum"][r].append(vac_pibs[r].cpu())
            turb_pibs = compute_pib_multi(focal_turb, dx_f, BUCKET_RADII_UM)
            for r in BUCKET_RADII_UM:
                all_pibs["turbulent"][r].append(turb_pibs[r].cpu())

            del focal_vac, focal_turb
            torch.cuda.empty_cache()

            # EE curves (use first sample only for curves)
            if bi == 0:
                fv_single, _ = focus(vac_out[0:1])
                ft_single, _ = focus(turb_out[0:1])
                all_ee["vacuum"] = compute_ee_curve(fv_single, dx_f, EE_RADII_UM)
                all_ee["turbulent"] = compute_ee_curve(ft_single, dx_f, EE_RADII_UM)
                del fv_single, ft_single

            # Trained models
            for name, model in models.items():
                pred = model(inp)
                all_co[name].append(complex_overlap(pred, tgt).cpu())
                focal_pred, _ = focus(pred)
                pibs = compute_pib_multi(focal_pred, dx_f, BUCKET_RADII_UM)
                for r in BUCKET_RADII_UM:
                    all_pibs[name][r].append(pibs[r].cpu())

                if bi == 0:
                    fp_single, _ = focus(pred[0:1])
                    all_ee[name] = compute_ee_curve(fp_single, dx_f, EE_RADII_UM)
                    del fp_single

                del focal_pred, pred
                torch.cuda.empty_cache()

            if bi % 10 == 0:
                print(f"  batch {bi}/{len(test_loader)}", flush=True)

    # === Aggregate results ===
    results = {}
    for name in ["vacuum", "turbulent"] + list(models.keys()):
        results[name] = {}
        for r in BUCKET_RADII_UM:
            vals = torch.cat(all_pibs[name][r])
            results[name][f"pib_{int(r)}um"] = float(vals.mean())
            results[name][f"pib_{int(r)}um_std"] = float(vals.std())
        if name in all_co:
            results[name]["co"] = float(torch.cat(all_co[name]).mean())

    # Save JSON
    with open(OUT / "bucket_radius_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT / 'bucket_radius_results.json'}")

    # === Print summary table ===
    print(f"\n{'='*80}")
    print("BUCKET RADIUS SWEEP — FOCAL PLANE PIB")
    print(f"{'='*80}")
    header = f"{'Strategy':>25} | {'CO':>6} | " + " | ".join(f"PIB@{int(r)}μm" for r in BUCKET_RADII_UM)
    print(header)
    print("-" * len(header))
    for name in ["vacuum", "turbulent"] + list(models.keys()):
        co_str = f"{results[name].get('co', 0):.4f}" if 'co' in results[name] else "  -   "
        pibs = " | ".join(f"  {results[name][f'pib_{int(r)}um']:.4f} " for r in BUCKET_RADII_UM)
        label = LABELS.get(name, name)
        print(f"{label:>25} | {co_str} | {pibs}")

    # === Figure 1: PIB bar chart at each radius ===
    print("\nGenerating figures...")
    fig, axes = plt.subplots(1, len(BUCKET_RADII_UM), figsize=(6*len(BUCKET_RADII_UM), 8))
    fig.suptitle("Phase 3: Focal PIB vs Bucket Radius\n"
                 "Same trained models, different evaluation radii",
                 fontsize=14, fontweight="bold")

    all_names = ["vacuum", "turbulent"] + list(models.keys())
    all_labels = ["Vacuum", "Turbulent"] + [LABELS.get(n, n) for n in models.keys()]
    all_colors = ["blue", "gray"] + [COLORS.get(n, "black") for n in models.keys()]

    for col, rad in enumerate(BUCKET_RADII_UM):
        vals = [results[n][f"pib_{int(rad)}um"] for n in all_names]
        x = np.arange(len(all_names))
        axes[col].bar(x, vals, color=all_colors, alpha=0.8, edgecolor='black', lw=0.5)
        axes[col].set_xticks(x)
        axes[col].set_xticklabels(all_labels, rotation=35, ha="right", fontsize=9)
        axes[col].set_title(f"PIB@{int(rad)}μm", fontsize=13, fontweight="bold")
        axes[col].set_ylabel("PIB" if col == 0 else "")
        axes[col].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(vals):
            axes[col].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUT / "fig_pib_vs_radius.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_pib_vs_radius.png")

    # === Figure 2: Encircled energy curves ===
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    fig2.suptitle("Phase 3: Encircled Energy Curves — Focal Plane\n"
                  f"f={FOCUS_F*1e3:.1f}mm, dx_focal={dx_focal*1e6:.3f}μm",
                  fontsize=14, fontweight="bold")

    for name, label, color in zip(all_names, all_labels, all_colors):
        if all_ee.get(name):
            ls = '--' if name in ["vacuum", "turbulent"] else '-'
            lw = 1.5 if name in ["vacuum", "turbulent"] else 2.5
            ax2.plot(EE_RADII_UM, all_ee[name], color=color, ls=ls, lw=lw, label=label)

    for rad in BUCKET_RADII_UM:
        ax2.axvline(rad, color='gray', ls=':', alpha=0.5)
        ax2.text(rad + 0.5, 0.02, f"{int(rad)}μm", fontsize=9, color='gray')

    ax2.set_xlabel("Radius [μm]", fontsize=13)
    ax2.set_ylabel("Encircled Energy Fraction", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig2.savefig(OUT / "fig_encircled_energy.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("  Saved fig_encircled_energy.png")

    # === Figure 3: PIB improvement ratio vs radius ===
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    fig3.suptitle("Phase 3: PIB Improvement Ratio vs Bucket Radius\n"
                  "D2NN PIB / Turbulent PIB (>1 = improvement)",
                  fontsize=14, fontweight="bold")

    for name in models.keys():
        ratios = []
        for rad in BUCKET_RADII_UM:
            turb_pib = results["turbulent"][f"pib_{int(rad)}um"]
            d2nn_pib = results[name][f"pib_{int(rad)}um"]
            ratios.append(d2nn_pib / max(turb_pib, 1e-12))
        ax3.plot(BUCKET_RADII_UM, ratios, 'o-', color=COLORS.get(name, 'gray'),
                 lw=2.5, ms=10, label=LABELS.get(name, name))

    ax3.axhline(1.0, color='black', ls='--', lw=1.5, label='No improvement')
    ax3.set_xlabel("Bucket Radius [μm]", fontsize=13)
    ax3.set_ylabel("PIB Improvement Ratio (D2NN / Turbulent)", fontsize=13)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(BUCKET_RADII_UM)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig3.savefig(OUT / "fig_improvement_ratio.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print("  Saved fig_improvement_ratio.png")

    del d0
    for m in models.values(): del m
    torch.cuda.empty_cache()
    print(f"\nDone! Phase 3 results saved to {OUT}")


if __name__ == "__main__":
    main()
