#!/usr/bin/env python
"""Evaluate a trained D2NN strategy: WF RMS (output plane) + PIB (focal plane).

No Strehl — only physically unambiguous metrics.

Generates:
  05_test_evaluation.json
  06_fig1_focal_plane_vacuum_vs_turbulent_vs_d2nn.png
  07_fig2_pib_bar_chart.png
  08_fig3_d2nn_output_plane_irradiance_phase_residual.png
  09_fig4_wavefront_rms_distribution.png

Usage:
    cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python scripts/eval_strategy.py \\
        --sweep autoresearch/runs/0401-focal-pib-sweep-padded-4loss-cn2-5e14 \\
        --strategy focal_pib_only
"""
from __future__ import annotations
import argparse, json, math
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
from kim2026.training.metrics import complex_overlap
from kim2026.training.losses import align_global_phase

W = 1.55e-6; N = 1024; WIN = 0.002048; APT = 0.002; DX = WIN / N; F = 4.5e-3
RADII = [5, 10, 25, 50]

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Roboto', 'Open Sans'], 'font.size': 14})


def prep(f):
    return center_crop_field(apply_receiver_aperture(f, receiver_window_m=WIN, aperture_diameter_m=APT), crop_n=N)


def foc(field):
    with torch.no_grad():
        f, dx = lens_2f_forward(field.to(torch.complex64), dx_in_m=DX, wavelength_m=W, f_m=F, na=None, apply_scaling=False)
    return f, dx


def compute_pib(focal_field, dx_f, radii):
    intensity = focal_field.abs().square()
    c_ = N // 2
    yy, xx = torch.meshgrid(torch.arange(N, device=focal_field.device) - c_,
                             torch.arange(N, device=focal_field.device) - c_, indexing="ij")
    r_grid = torch.sqrt((xx * dx_f) ** 2 + (yy * dx_f) ** 2)
    total = intensity.sum(dim=(-2, -1)).clamp(min=1e-12)
    return {r: ((intensity * (r_grid <= r * 1e-6).float()).sum(dim=(-2, -1)) / total).mean().item() for r in radii}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--arch-pad", type=int, default=2, help="propagation_pad_factor")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ARCH = dict(num_layers=5, layer_spacing_m=10e-3, detector_distance_m=10e-3, propagation_pad_factor=args.arch_pad)
    OUT = Path(args.sweep) / args.strategy
    ckpt = OUT / "checkpoint.pt"
    if not ckpt.exists():
        print(f"No checkpoint at {ckpt}"); return

    print(f"Strategy: {args.strategy}")
    m = BeamCleanupD2NN(n=N, wavelength_m=W, window_m=WIN, **ARCH)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["model_state_dict"])
    m = m.to(device); m.eval()
    d0 = BeamCleanupD2NN(n=N, wavelength_m=W, window_m=WIN, **ARCH).to(device); d0.eval()

    ds = CachedFieldDataset(cache_dir="data/kim2026/1km_cn2_5e-14_tel15cm_n1024_br75/cache",
                             manifest_path="data/kim2026/1km_cn2_5e-14_tel15cm_n1024_br75/split_manifest.json", split="test")
    loader = DataLoader(ds, batch_size=8, num_workers=0)
    print(f"Test: {len(ds)} samples")

    # === Eval: PIB + CO ===
    stats = {k: {f"pib{r}": [] for r in RADII} for k in ["vacuum", "turbulent", "d2nn"]}
    for k in stats: stats[k]["co"] = []

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            inp = prep(batch["u_turb"].to(device)); tgt = prep(batch["u_vacuum"].to(device))
            vac_out = d0(tgt); turb_out = d0(inp); d2nn_out = m(inp)
            for label, field in [("vacuum", vac_out), ("turbulent", turb_out), ("d2nn", d2nn_out)]:
                ff, dx_f = foc(field)
                pibs = compute_pib(ff, dx_f, RADII)
                for r in RADII: stats[label][f"pib{r}"].append(pibs[r])
                if label != "vacuum":
                    stats[label]["co"].append(complex_overlap(field, tgt).mean().item())
                del ff
            torch.cuda.empty_cache()
            if bi % 10 == 0: print(f"  batch {bi}/{len(loader)}", flush=True)

    avg = {label: {k: float(np.mean(v)) for k, v in stats[label].items() if v} for label in stats}

    # === WF RMS (50 samples) ===
    print("WF RMS...", flush=True)
    all_wf_d2nn, all_wf_turb = [], []
    with torch.no_grad():
        for i in range(min(50, len(ds))):
            s = ds[i]
            inp = prep(s["u_turb"].unsqueeze(0).to(device))
            tgt = prep(s["u_vacuum"].unsqueeze(0).to(device))
            d2nn_out = m(inp); vac_out = d0(tgt); turb_out = d0(inp)
            for label, field in [("d2nn", align_global_phase(d2nn_out, vac_out)),
                                  ("turb", align_global_phase(turb_out, vac_out))]:
                pd = torch.angle(field) - torch.angle(vac_out)
                pd = torch.remainder(pd + math.pi, 2 * math.pi) - math.pi
                w = vac_out.abs().square(); w = w / w.sum()
                rms = torch.sqrt((w * pd.square()).sum()).item() * W / (2 * math.pi) * 1e9
                if label == "d2nn": all_wf_d2nn.append(rms)
                else: all_wf_turb.append(rms)

    avg["wf_rms_turb_nm"] = float(np.mean(all_wf_turb))
    avg["wf_rms_turb_std_nm"] = float(np.std(all_wf_turb))
    avg["wf_rms_d2nn_nm"] = float(np.mean(all_wf_d2nn))
    avg["wf_rms_d2nn_std_nm"] = float(np.std(all_wf_d2nn))

    # Print
    print(f"\n{'='*60}")
    print(f"{args.strategy} — TEST RESULTS")
    print(f"{'='*60}")
    print(f"  WF RMS (output plane, vs vacuum):")
    print(f"    Turbulent: {avg['wf_rms_turb_nm']:.1f} +/- {avg['wf_rms_turb_std_nm']:.1f} nm")
    print(f"    D2NN:      {avg['wf_rms_d2nn_nm']:.1f} +/- {avg['wf_rms_d2nn_std_nm']:.1f} nm")
    print(f"  CO (output plane):")
    print(f"    Turbulent: {avg['turbulent'].get('co',0):.4f}")
    print(f"    D2NN:      {avg['d2nn'].get('co',0):.4f}")
    print(f"  PIB (focal plane):")
    for r in RADII:
        t = avg["turbulent"].get(f"pib{r}", 0); d = avg["d2nn"].get(f"pib{r}", 0)
        print(f"    @{r}um: {t:.4f} -> {d:.4f} ({d/max(t,1e-12):.2f}x)")

    with open(OUT / "05_test_evaluation.json", "w") as f:
        json.dump(avg, f, indent=2)

    # === Figures ===
    print("Figures...", flush=True)
    s = ds[0]
    inp = prep(s["u_turb"].unsqueeze(0).to(device))
    tgt = prep(s["u_vacuum"].unsqueeze(0).to(device))
    with torch.no_grad():
        d2nn_out = m(inp); vac_out = d0(tgt); turb_out = d0(inp)
        d2nn_al = align_global_phase(d2nn_out, vac_out)
        turb_al = align_global_phase(turb_out, vac_out)
        fvac, dx_f = foc(vac_out); fturb, _ = foc(turb_out); fd2nn, _ = foc(d2nn_out)

    vac_np = vac_out[0].cpu().numpy(); turb_np = turb_al[0].cpu().numpy(); d2nn_np = d2nn_al[0].cpu().numpy()
    fvac_np = fvac[0].cpu().numpy(); fturb_np = fturb[0].cpu().numpy(); fd2nn_np = fd2nn[0].cpu().numpy()
    dx_f = float(dx_f); Z = 64; c = N // 2
    e_f = [-(Z*dx_f*1e6), Z*dx_f*1e6, -(Z*dx_f*1e6), Z*dx_f*1e6]
    e_um = [-(N*DX/2)*1e6, (N*DX/2)*1e6, -(N*DX/2)*1e6, (N*DX/2)*1e6]
    theta = np.linspace(0, 2*np.pi, 200)
    strategy_label = args.strategy.replace("focal_", "").replace("_", " ").title()

    # Fig1: focal plane 3x3
    fig, axes = plt.subplots(3, 3, figsize=(21, 21))
    fig.suptitle(f'{strategy_label} — Focal Plane\n'
                 f'WF RMS: turb={avg["wf_rms_turb_nm"]:.0f}nm, D2NN={avg["wf_rms_d2nn_nm"]:.0f}nm',
                 fontsize=16, fontweight='bold')
    imax = np.max([np.abs(f[c-Z:c+Z, c-Z:c+Z])**2 for f in [fvac_np, fturb_np, fd2nn_np]])
    for col, (label, field, lk) in enumerate([
        ("Vacuum", fvac_np, "vacuum"), ("Turbulent\n(no D2NN)", fturb_np, "turbulent"),
        (f"D2NN\n({strategy_label})", fd2nn_np, "d2nn")]):
        crop = field[c-Z:c+Z, c-Z:c+Z]
        axes[0, col].imshow(np.abs(crop)**2, extent=e_f, origin="lower", cmap="inferno", vmin=0, vmax=imax)
        p10 = avg[lk].get("pib10", 0); p5 = avg[lk].get("pib5", 0)
        axes[0, col].set_title(f"{label}\nPIB@5={p5:.4f} | @10={p10:.4f}", fontsize=12, fontweight="bold")
        axes[0, col].plot(10*np.cos(theta), 10*np.sin(theta), 'r-', lw=2, alpha=0.9)
        axes[0, col].plot(50*np.cos(theta), 50*np.sin(theta), 'w--', lw=1.5, alpha=0.6)
        axes[1, col].imshow(np.log10(np.abs(crop)**2+1e-15), extent=e_f, origin="lower", cmap="viridis",
                            vmin=np.log10(imax)-4, vmax=np.log10(imax))
        axes[1, col].plot(10*np.cos(theta), 10*np.sin(theta), 'r-', lw=2)
        axes[1, col].plot(50*np.cos(theta), 50*np.sin(theta), 'w--', lw=1.5, alpha=0.6)
        x_um = np.linspace(e_f[0], e_f[1], 2*Z)
        axes[2, col].plot(x_um, np.abs(fvac_np[c, c-Z:c+Z])**2, 'b--', lw=1.5, label="Vacuum")
        axes[2, col].plot(x_um, np.abs(field[c, c-Z:c+Z])**2, 'r-', lw=2, label=label.split('\n')[0])
        axes[2, col].axvline(-10, color='red', ls=':', lw=1); axes[2, col].axvline(10, color='red', ls=':', lw=1)
        axes[2, col].legend(fontsize=9); axes[2, col].grid(True, alpha=0.3); axes[2, col].set_xlabel("um")
    for r, lbl in enumerate(["Focal irradiance", "Log irradiance", "1D cross-section"]):
        axes[r, 0].set_ylabel(lbl, fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "06_fig1_focal_plane_vacuum_vs_turbulent_vs_d2nn.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("  fig1", flush=True)

    # Fig2: PIB bars
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    fig2.suptitle(f"{strategy_label} — PIB at Multiple Bucket Radii", fontsize=16, fontweight="bold")
    x = np.arange(len(RADII)); w = 0.25
    for i, (label, color) in enumerate([("vacuum", "blue"), ("turbulent", "gray"), ("d2nn", "red")]):
        vals = [avg[label].get(f"pib{r}", 0) for r in RADII]
        bars = ax2.bar(x + i*w, vals, w, label=label.capitalize(), color=color, alpha=0.8, edgecolor="black", lw=0.5)
        for b, v in zip(bars, vals):
            ax2.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)
    ax2.set_xticks(x + w); ax2.set_xticklabels([f"{r}um" for r in RADII], fontsize=13)
    ax2.set_ylabel("PIB", fontsize=14); ax2.legend(fontsize=13); ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig2.savefig(OUT / "07_fig2_pib_bar_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig2); print("  fig2", flush=True)

    # Fig3: output plane 4x3
    res_turb = np.remainder(np.angle(turb_np) - np.angle(vac_np) + np.pi, 2*np.pi) - np.pi
    res_d2nn = np.remainder(np.angle(d2nn_np) - np.angle(vac_np) + np.pi, 2*np.pi) - np.pi
    ww = np.abs(vac_np)**2; ww = ww / ww.sum()
    wf_t = np.sqrt((ww * res_turb**2).sum()) * W / (2*np.pi) * 1e9
    wf_d = np.sqrt((ww * res_d2nn**2).sum()) * W / (2*np.pi) * 1e9

    fig3, axes = plt.subplots(4, 3, figsize=(21, 28))
    fig3.suptitle(f"D2NN Output Plane — {strategy_label}", fontsize=16, fontweight="bold")
    imax_o = np.max([np.abs(vac_np)**2, np.abs(turb_np)**2, np.abs(d2nn_np)**2])
    labs = ["Vacuum", f"Turbulent\nWF={wf_t:.1f}nm", f"D2NN\nWF={wf_d:.1f}nm"]
    fs = [vac_np, turb_np, d2nn_np]; rs = [None, res_turb, res_d2nn]
    for col in range(3):
        axes[0, col].imshow(np.abs(fs[col])**2, extent=e_um, origin="lower", cmap="inferno", vmin=0, vmax=imax_o)
        axes[0, col].set_title(labs[col], fontsize=13, fontweight="bold")
        axes[1, col].imshow(np.angle(fs[col]), extent=e_um, origin="lower", cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi)
        if rs[col] is not None:
            im = axes[2, col].imshow(rs[col], extent=e_um, origin="lower", cmap="RdBu_r", vmin=-np.pi, vmax=np.pi)
            plt.colorbar(im, ax=axes[2, col], label="rad")
        else:
            axes[2, col].imshow(np.zeros((N, N)), extent=e_um, origin="lower", cmap="RdBu_r", vmin=-np.pi, vmax=np.pi)
            axes[2, col].text(0.5, 0.5, "Reference", transform=axes[2, col].transAxes, ha="center", va="center",
                              fontsize=14, fontweight="bold", bbox=dict(facecolor="white", alpha=0.8))
        x_um = np.linspace(e_um[0], e_um[1], N)
        axes[3, col].plot(x_um, np.abs(vac_np[N//2, :])**2, 'b--', lw=1.5, label="Vacuum")
        axes[3, col].plot(x_um, np.abs(fs[col][N//2, :])**2, 'r-', lw=2, label=labs[col].split('\n')[0])
        axes[3, col].set_xlim(-200, 200); axes[3, col].legend(fontsize=9); axes[3, col].grid(True, alpha=0.3)
    for r, lbl in enumerate(["Irradiance", "Phase", "Residual phase", "1D profile"]):
        axes[r, 0].set_ylabel(lbl, fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig3.savefig(OUT / "08_fig3_d2nn_output_plane_irradiance_phase_residual.png", dpi=150, bbox_inches="tight")
    plt.close(fig3); print("  fig3", flush=True)

    # Fig4: WF RMS histogram
    fig4, ax = plt.subplots(figsize=(10, 7))
    fig4.suptitle(f"WF RMS Distribution — {strategy_label}", fontsize=14, fontweight="bold")
    ax.hist(all_wf_turb, bins=20, alpha=0.6, color="gray",
            label=f"Turb ({np.mean(all_wf_turb):.1f}+/-{np.std(all_wf_turb):.1f}nm)")
    ax.hist(all_wf_d2nn, bins=20, alpha=0.6, color="red",
            label=f"D2NN ({np.mean(all_wf_d2nn):.1f}+/-{np.std(all_wf_d2nn):.1f}nm)")
    ax.axvline(np.mean(all_wf_turb), color="gray", ls="--", lw=2)
    ax.axvline(np.mean(all_wf_d2nn), color="red", ls="--", lw=2)
    ax.set_xlabel("WF RMS [nm]"); ax.set_ylabel("Count"); ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig4.savefig(OUT / "09_fig4_wavefront_rms_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig4); print("  fig4", flush=True)

    del m, d0; torch.cuda.empty_cache()
    print("Done!")


if __name__ == "__main__":
    main()
