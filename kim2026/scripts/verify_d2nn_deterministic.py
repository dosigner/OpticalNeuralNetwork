#!/usr/bin/env python
"""Verify D2NN on deterministic aberration (defocus).

If static D2NN works → mask learns -Z4, CO→1.0
If not → code bug.

Usage:
    cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python scripts/verify_d2nn_deterministic.py
"""
import math
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from kim2026.models.d2nn import BeamCleanupD2NN
from kim2026.training.metrics import complex_overlap

# ─── Setup ────────────────────────────────────────────────
N = 256          # small grid for speed
W = 1.55e-6
WIN = 0.002048   # 2mm
DX = WIN / N
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARCH = dict(num_layers=3, layer_spacing_m=10e-3, detector_distance_m=10e-3)
LR = 1e-3
EPOCHS = 200


def make_gaussian_beam(n, dx, w0):
    """Create Gaussian beam."""
    x = (torch.arange(n) - n // 2) * dx
    xx, yy = torch.meshgrid(x, x, indexing="ij")
    r2 = xx**2 + yy**2
    return torch.exp(-r2 / w0**2).to(torch.complex64)


def zernike_defocus(n, dx, peak_pv_rad):
    """Z4 defocus: 2r² - 1, scaled to peak-to-valley = peak_pv_rad."""
    x = (torch.arange(n) - n // 2) * dx
    xx, yy = torch.meshgrid(x, x, indexing="ij")
    r = torch.sqrt(xx**2 + yy**2)
    R = n * dx / 2  # aperture radius
    rho = r / R
    rho = torch.clamp(rho, 0, 1)
    z4 = 2 * rho**2 - 1  # range [-1, 1]
    return z4 * (peak_pv_rad / 2)  # scale to PV


def zernike_coma(n, dx, peak_pv_rad):
    """Z7 coma: (3r³ - 2r)cos(θ)."""
    x = (torch.arange(n) - n // 2) * dx
    xx, yy = torch.meshgrid(x, x, indexing="ij")
    r = torch.sqrt(xx**2 + yy**2)
    R = n * dx / 2
    rho = torch.clamp(r / R, 0, 1)
    theta = torch.atan2(yy, xx)
    z7 = (3 * rho**3 - 2 * rho) * torch.cos(theta)
    return z7 * (peak_pv_rad / 2)


def zernike_astigmatism(n, dx, peak_pv_rad):
    """Z5 astigmatism: r²cos(2θ)."""
    x = (torch.arange(n) - n // 2) * dx
    xx, yy = torch.meshgrid(x, x, indexing="ij")
    r = torch.sqrt(xx**2 + yy**2)
    R = n * dx / 2
    rho = torch.clamp(r / R, 0, 1)
    theta = torch.atan2(yy, xx)
    z5 = rho**2 * torch.cos(2 * theta)
    return z5 * (peak_pv_rad / 2)


def train_and_eval(aberration_name, aberration_phase, u_vac):
    """Train D2NN on a single deterministic aberration."""
    print(f"\n{'='*60}")
    print(f"  Aberration: {aberration_name}")
    print(f"  Phase PV: {(aberration_phase.max() - aberration_phase.min()):.2f} rad")
    print(f"  Phase RMS: {aberration_phase.std():.3f} rad")
    print(f"{'='*60}")

    # Aberrated input (SAME every time — deterministic)
    u_in = u_vac * torch.exp(1j * aberration_phase.to(DEVICE))
    u_target = u_vac.clone()

    co_before = complex_overlap(u_in.unsqueeze(0), u_target.unsqueeze(0)).item()
    print(f"  CO before D2NN: {co_before:.4f}")

    # Train D2NN
    model = BeamCleanupD2NN(n=N, wavelength_m=W, window_m=WIN, **ARCH).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"epoch": [], "loss": [], "co": []}
    t0 = time.time()

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = model(u_in.unsqueeze(0))
        # Loss = 1 - CO
        co = complex_overlap(pred, u_target.unsqueeze(0))
        loss = 1.0 - co.mean()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == EPOCHS - 1:
            co_val = co.item()
            history["epoch"].append(epoch)
            history["loss"].append(loss.item())
            history["co"].append(co_val)
            print(f"  Epoch {epoch:3d}/{EPOCHS-1} | loss={loss.item():.5f} | CO={co_val:.4f}")

    dt = time.time() - t0
    model.eval()
    with torch.no_grad():
        pred = model(u_in.unsqueeze(0))
        # Reference: vacuum through SAME D2NN path (fair phase comparison)
        vac_through_d2nn = model(u_vac.unsqueeze(0))
    co_after = complex_overlap(pred, vac_through_d2nn).item()

    # WF RMS — both fields through same D2NN optical path
    pred_phase = torch.angle(pred[0])
    ref_phase = torch.angle(vac_through_d2nn[0])
    diff = torch.remainder(pred_phase - ref_phase + math.pi, 2*math.pi) - math.pi
    w = vac_through_d2nn[0].abs().square()
    w = w / w.sum()
    # Before: aberrated vs vacuum (both without D2NN)
    inp_phase = torch.angle(u_in)
    vac_phase = torch.angle(u_vac)
    diff_before = torch.remainder(inp_phase - vac_phase + math.pi, 2*math.pi) - math.pi
    w_before = u_vac.abs().square(); w_before = w_before / w_before.sum()
    wf_rms_before = torch.sqrt((w_before * diff_before.square()).sum()).item()
    wf_rms_after = torch.sqrt((w * diff.square()).sum()).item()

    print(f"\n  Results ({dt:.0f}s):")
    print(f"    CO:     {co_before:.4f} → {co_after:.4f} (delta={co_after-co_before:+.4f})")
    print(f"    WF RMS: {wf_rms_before:.4f} → {wf_rms_after:.4f} rad")
    print(f"    WF RMS: {wf_rms_before*W/(2*math.pi)*1e9:.1f} → {wf_rms_after*W/(2*math.pi)*1e9:.1f} nm")
    improvement = (1 - wf_rms_after / wf_rms_before) * 100
    print(f"    Improvement: {improvement:.1f}%")

    return {
        "name": aberration_name,
        "co_before": co_before,
        "co_after": co_after,
        "wf_rms_before_rad": wf_rms_before,
        "wf_rms_after_rad": wf_rms_after,
        "improvement_pct": improvement,
        "history": history,
        "phases": [l.phase.detach().cpu().numpy() for l in model.layers],
    }


def main():
    print(f"Device: {DEVICE}")
    print(f"Grid: {N}×{N}, dx={DX*1e6:.1f}μm, window={WIN*1e3:.3f}mm")
    print(f"D2NN: {ARCH['num_layers']} layers, spacing={ARCH['layer_spacing_m']*1e3:.0f}mm")
    print(f"Training: lr={LR}, epochs={EPOCHS}")

    # Vacuum beam
    w0 = 0.4e-3  # 0.4mm 1/e radius
    u_vac = make_gaussian_beam(N, DX, w0).to(DEVICE)
    print(f"Beam w0={w0*1e3:.1f}mm")

    # Test 3 aberrations
    results = []

    # 1. Defocus (Z4) — 2 rad PV
    phase_defocus = zernike_defocus(N, DX, peak_pv_rad=2.0)
    results.append(train_and_eval("Defocus (Z4, 2 rad PV)", phase_defocus, u_vac))

    # 2. Coma (Z7) — 3 rad PV
    phase_coma = zernike_coma(N, DX, peak_pv_rad=3.0)
    results.append(train_and_eval("Coma (Z7, 3 rad PV)", phase_coma, u_vac))

    # 3. Astigmatism (Z5) — 2.5 rad PV
    phase_astig = zernike_astigmatism(N, DX, peak_pv_rad=2.5)
    results.append(train_and_eval("Astigmatism (Z5, 2.5 rad PV)", phase_astig, u_vac))

    # ─── Summary ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("DETERMINISTIC ABERRATION VERIFICATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Aberration':>30} | {'CO before':>9} | {'CO after':>9} | {'WF RMS [nm]':>15} | {'Improve':>8}")
    print("-" * 80)
    for r in results:
        wf_b = r["wf_rms_before_rad"] * W / (2*math.pi) * 1e9
        wf_a = r["wf_rms_after_rad"] * W / (2*math.pi) * 1e9
        print(f"{r['name']:>30} | {r['co_before']:>9.4f} | {r['co_after']:>9.4f} | "
              f"{wf_b:>6.1f}→{wf_a:>5.1f} | {r['improvement_pct']:>7.1f}%")

    all_pass = all(r["co_after"] > 0.9 for r in results)
    print(f"\n{'='*70}")
    if all_pass:
        print("PASS: D2NN correctly learns deterministic aberration correction.")
        print("Static masks work for fixed aberrations. Failure on random turbulence")
        print("is due to static-vs-dynamic mismatch, NOT a code bug.")
    else:
        print("FAIL: D2NN cannot even correct deterministic aberrations.")
        print("This suggests a code or architecture issue.")
    print(f"{'='*70}")

    # ─── Visualization ────────────────────────────────────
    out_dir = "/root/dj/D2NN/kim2026/autoresearch/runs/d2nn_deterministic_verify"
    import os; os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    fig.suptitle("D2NN Deterministic Aberration Verification", fontsize=18, fontweight="bold")

    aberrations = [phase_defocus, phase_coma, phase_astig]
    for row, (r, ab_phase) in enumerate(zip(results, aberrations)):
        # Col 0: Input aberration
        axes[row, 0].imshow(ab_phase.numpy(), cmap="RdBu_r")
        axes[row, 0].set_title(f"{r['name']}\nInput phase")
        axes[row, 0].axis("off")

        # Col 1: Learned mask (layer 0)
        axes[row, 1].imshow(r["phases"][0], cmap="RdBu_r")
        axes[row, 1].set_title("Learned mask (Layer 0)")
        axes[row, 1].axis("off")

        # Col 2: Training curve
        axes[row, 2].plot(r["history"]["epoch"], r["history"]["co"], 'g-o', lw=2, ms=4)
        axes[row, 2].set_xlabel("Epoch"); axes[row, 2].set_ylabel("CO")
        axes[row, 2].set_title(f"CO: {r['co_before']:.3f} → {r['co_after']:.3f}")
        axes[row, 2].axhline(1.0, color='k', ls='--', alpha=0.3)
        axes[row, 2].grid(True, alpha=0.3)

        # Col 3: Summary text
        axes[row, 3].axis("off")
        wf_b = r["wf_rms_before_rad"] * W / (2*math.pi) * 1e9
        wf_a = r["wf_rms_after_rad"] * W / (2*math.pi) * 1e9
        txt = (f"CO: {r['co_before']:.4f} → {r['co_after']:.4f}\n"
               f"WF RMS: {wf_b:.1f} → {wf_a:.1f} nm\n"
               f"Improvement: {r['improvement_pct']:.1f}%\n"
               f"{'PASS' if r['co_after'] > 0.9 else 'FAIL'}")
        color = "lightgreen" if r["co_after"] > 0.9 else "#f5b7b1"
        axes[row, 3].text(0.1, 0.5, txt, fontsize=16, va="center", family="monospace",
                          bbox=dict(facecolor=color, edgecolor="gray", alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = f"{out_dir}/deterministic_verification.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
