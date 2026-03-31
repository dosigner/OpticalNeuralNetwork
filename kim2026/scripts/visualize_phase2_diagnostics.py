#!/usr/bin/env python
"""Phase 2 Diagnostics — Figure 4 (Masks), Figure 5 (Training Curves), Figure 6 (Statistics).

Usage:
    cd /root/dj/D2NN/kim2026 && python scripts/visualize_phase2_diagnostics.py
"""
from __future__ import annotations
import re, math, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

from kim2026.data.dataset import CachedFieldDataset
from kim2026.models.d2nn import BeamCleanupD2NN
from kim2026.optics.lens_2f import lens_2f_forward
from kim2026.training.targets import apply_receiver_aperture
from kim2026.training.metrics import complex_overlap

W=1.55e-6; N=1024; WIN=0.002048; APT=0.002; DX=WIN/N
ARCH=dict(num_layers=5, layer_spacing_m=10e-3, detector_distance_m=10e-3)
CONFIGS=["baseline_co","co_amp","co_ffp","co_phasor","roi80"]
DATA=Path(__file__).resolve().parent.parent/"data"/"kim2026"/"1km_cn2e-14_tel15cm_n1024_br75"
SWEEP=Path(__file__).resolve().parent.parent/"autoresearch"/"runs"/"d2nn_sweep_telescope"
LOG=Path(__file__).resolve().parent.parent/"autoresearch"/"d2nn_sweep_telescope.log"

def load_d2nn(name):
    ckpt=SWEEP/name/"checkpoint.pt"
    if not ckpt.exists(): return None
    m=BeamCleanupD2NN(n=N,wavelength_m=W,window_m=WIN,**ARCH)
    m.load_state_dict(torch.load(ckpt,map_location="cpu",weights_only=True)["model_state_dict"])
    m.eval(); return m

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ═══════════════════════════════════════════════════════════
    # FIGURE 4: Phase Mask Analysis
    # ═══════════════════════════════════════════════════════════
    print("Generating Figure 4: Phase masks...")
    fig4, axes4 = plt.subplots(6, 5, figsize=(25, 30))
    fig4.suptitle("Figure 4: Learned Phase Masks (D2NN, 5 layers x 5 configs)", fontsize=16, fontweight="bold")

    for col, name in enumerate(CONFIGS):
        model = load_d2nn(name)
        if model is None:
            for row in range(6): axes4[row,col].text(0.5,0.5,"N/A",ha="center",va="center")
            continue

        # Rows 0-4: Phase masks
        for layer_idx in range(5):
            phase = torch.remainder(model.layers[layer_idx].phase, 2*math.pi).detach().cpu().numpy()
            im = axes4[layer_idx, col].imshow(phase, cmap="twilight_shifted", vmin=0, vmax=2*math.pi)
            axes4[layer_idx, col].set_title(f"{name}\nLayer {layer_idx}" if layer_idx==0 else f"Layer {layer_idx}", fontsize=10)
            axes4[layer_idx, col].axis("off")
            if col == 0:
                axes4[layer_idx, 0].set_ylabel(f"Layer {layer_idx}", fontsize=12, fontweight="bold")

        # Row 5: Spatial frequency spectrum of Layer 0
        phase0 = torch.remainder(model.layers[0].phase, 2*math.pi).detach().cpu().numpy()
        spectrum = np.abs(np.fft.fftshift(np.fft.fft2(phase0)))**2
        spectrum_log = np.log10(spectrum + 1e-10)
        axes4[5, col].imshow(spectrum_log, cmap="viridis")
        axes4[5, col].set_title("FFT of Layer 0 (log)", fontsize=10)
        axes4[5, col].axis("off")

    axes4[5, 0].set_ylabel("Frequency\nspectrum", fontsize=12, fontweight="bold")
    fig4.colorbar(im, ax=axes4[4,:].tolist(), shrink=0.5, label="Phase [0, 2pi)")
    plt.tight_layout(rect=[0,0,1,0.96])
    p4 = SWEEP/"phase2_fig4_masks.png"
    fig4.savefig(p4, dpi=150, bbox_inches="tight")
    print(f"  Saved: {p4}")
    plt.close(fig4)

    # ═══════════════════════════════════════════════════════════
    # FIGURE 5: Training Curves
    # ═══════════════════════════════════════════════════════════
    print("Generating Figure 5: Training curves...")

    # Parse log file
    log_text = LOG.read_text() if LOG.exists() else ""
    config_curves = {}
    current_config = None

    for line in log_text.split("\n"):
        # Detect config name
        m = re.search(r"\[(\w+)\]", line)
        if m and "Epoch" not in line:
            current_config = m.group(1)
            if current_config not in config_curves:
                config_curves[current_config] = {"epoch":[], "loss":[], "co":[], "io":[]}

        # Parse epoch line
        m = re.search(r"Epoch\s+(\d+)/\d+\s+\|\s+loss=([\d.]+)\s+\|\s+co=([\d.]+)\s+\|\s+io=([\d.]+)", line)
        if m and current_config:
            config_curves[current_config]["epoch"].append(int(m.group(1)))
            config_curves[current_config]["loss"].append(float(m.group(2)))
            config_curves[current_config]["co"].append(float(m.group(3)))
            config_curves[current_config]["io"].append(float(m.group(4)))

    fig5, axes5 = plt.subplots(3, 1, figsize=(16, 18))
    fig5.suptitle("Figure 5: Training Curves (D2NN Telescope Sweep)", fontsize=16, fontweight="bold")

    colors = {'baseline_co':'#2ecc71', 'co_amp':'#e74c3c', 'co_ffp':'#9b59b6',
              'co_phasor':'#f39c12', 'roi80':'#1abc9c'}

    # Panel A: Loss
    for name in CONFIGS:
        if name in config_curves and config_curves[name]["epoch"]:
            d = config_curves[name]
            axes5[0].plot(d["epoch"], d["loss"], color=colors.get(name,'gray'), lw=1.5, label=name)
    axes5[0].set_xlabel("Epoch", fontsize=12)
    axes5[0].set_ylabel("Loss", fontsize=12)
    axes5[0].set_title("Loss vs Epoch", fontsize=13, fontweight="bold")
    axes5[0].legend(fontsize=10)
    axes5[0].grid(True, alpha=0.3)

    # Panel B: Validation CO
    for name in CONFIGS:
        if name in config_curves and config_curves[name]["epoch"]:
            d = config_curves[name]
            axes5[1].plot(d["epoch"], d["co"], color=colors.get(name,'gray'), lw=1.5, label=name)
    axes5[1].axhline(0.6249, color='k', ls='--', lw=1, label='no_correction baseline')
    axes5[1].set_xlabel("Epoch", fontsize=12)
    axes5[1].set_ylabel("Validation CO", fontsize=12)
    axes5[1].set_title("Validation CO vs Epoch (dashed = no_correction baseline)", fontsize=13, fontweight="bold")
    axes5[1].legend(fontsize=10)
    axes5[1].grid(True, alpha=0.3)

    # Panel C: CO improvement rate (derivative)
    for name in CONFIGS:
        if name in config_curves and len(config_curves[name]["co"]) > 1:
            d = config_curves[name]
            co_arr = np.array(d["co"])
            ep_arr = np.array(d["epoch"])
            # Moving average of CO change per epoch
            if len(co_arr) > 2:
                dco = np.diff(co_arr) / np.maximum(np.diff(ep_arr), 1)
                axes5[2].plot(ep_arr[1:], dco, color=colors.get(name,'gray'), lw=1.5, label=name)
    axes5[2].axhline(0, color='k', ls='-', lw=0.5)
    axes5[2].set_xlabel("Epoch", fontsize=12)
    axes5[2].set_ylabel("dCO/dEpoch", fontsize=12)
    axes5[2].set_title("CO Improvement Rate (convergence check)", fontsize=13, fontweight="bold")
    axes5[2].legend(fontsize=10)
    axes5[2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0,0,1,0.96])
    p5 = SWEEP/"phase2_fig5_training_curves.png"
    fig5.savefig(p5, dpi=150, bbox_inches="tight")
    print(f"  Saved: {p5}")
    plt.close(fig5)

    # ═══════════════════════════════════════════════════════════
    # FIGURE 6: Multi-Realization Statistics
    # ═══════════════════════════════════════════════════════════
    print("Generating Figure 6: Multi-realization statistics...")

    ds = CachedFieldDataset(cache_dir=str(DATA/"cache"),
                             manifest_path=str(DATA/"split_manifest.json"), split="test")

    # Collect metrics for all test samples
    all_data = {n: {"co":[], "wf_rms":[], "pib50":[]} for n in ["no_d2nn"]+CONFIGS}

    # Load all models once
    models = {}
    for name in CONFIGS:
        m = load_d2nn(name)
        if m: models[name] = m.to(device)

    d0 = BeamCleanupD2NN(n=N,wavelength_m=W,window_m=WIN,**ARCH).to(device); d0.eval()

    def wf_rms(pred, target):
        p_ph = torch.angle(pred[0])
        t_ph = torch.angle(target[0])
        diff = torch.remainder(p_ph - t_ph + math.pi, 2*math.pi) - math.pi
        w = target[0].abs().square()
        w = w / w.sum()
        return torch.sqrt((w * diff.square()).sum()).item()

    def focus(field):
        with torch.no_grad():
            f, dx_f = lens_2f_forward(field.to(torch.complex64), dx_in_m=DX,
                                       wavelength_m=W, f_m=4.5e-3, na=None, apply_scaling=False)
        return f, dx_f

    for si in range(len(ds)):
        s = ds[si]
        ut = apply_receiver_aperture(s["u_turb"].unsqueeze(0).to(device),
                                      receiver_window_m=WIN, aperture_diameter_m=APT)
        uv = apply_receiver_aperture(s["u_vacuum"].unsqueeze(0).to(device),
                                      receiver_window_m=WIN, aperture_diameter_m=APT)
        with torch.no_grad():
            ut_out = d0(ut); uv_out = d0(uv)

        # no_d2nn
        co_val = complex_overlap(ut_out, uv_out).item()
        wr = wf_rms(ut_out, uv_out)
        ut_det, dx_f = focus(ut_out)
        c=N//2; yy,xx=np.mgrid[-c:N-c,-c:N-c]; rsq=(xx*dx_f)**2+(yy*dx_f)**2
        irr=ut_det[0].abs().square().cpu().numpy()
        pib50 = irr[rsq<=(50e-6)**2].sum()/max(irr.sum(),1e-30)
        all_data["no_d2nn"]["co"].append(co_val)
        all_data["no_d2nn"]["wf_rms"].append(wr)
        all_data["no_d2nn"]["pib50"].append(pib50)

        # trained configs
        for name, model in models.items():
            with torch.no_grad():
                pred = model(ut)
            co_val = complex_overlap(pred, uv_out).item()
            wr = wf_rms(pred, uv_out)
            pred_det, _ = focus(pred)
            irr_p = pred_det[0].abs().square().cpu().numpy()
            pib50 = irr_p[rsq<=(50e-6)**2].sum()/max(irr_p.sum(),1e-30)
            all_data[name]["co"].append(co_val)
            all_data[name]["wf_rms"].append(wr)
            all_data[name]["pib50"].append(pib50)

        if (si+1) % 5 == 0:
            print(f"  [{si+1}/{len(ds)}]")

    del d0; torch.cuda.empty_cache()

    # Plot
    fig6, axes6 = plt.subplots(2, 2, figsize=(20, 16))
    fig6.suptitle("Figure 6: Multi-Realization Statistics (test set, 20 samples)",
                  fontsize=16, fontweight="bold")

    all_names = ["no_d2nn"] + CONFIGS
    bp_colors = ['gray', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']

    # Panel A: CO box plot
    co_data = [all_data[n]["co"] for n in all_names]
    bp = axes6[0,0].boxplot(co_data, labels=all_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], bp_colors):
        patch.set_facecolor(color); patch.set_alpha(0.5)
    axes6[0,0].set_ylabel("Complex Overlap", fontsize=12)
    axes6[0,0].set_title("CO Distribution", fontsize=13, fontweight="bold")
    axes6[0,0].tick_params(axis='x', rotation=30)
    axes6[0,0].grid(True, alpha=0.3, axis='y')

    # Panel B: WF RMS box plot
    wf_data = [np.array(all_data[n]["wf_rms"])*W/(2*math.pi)*1e9 for n in all_names]  # to nm
    bp = axes6[0,1].boxplot(wf_data, labels=all_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], bp_colors):
        patch.set_facecolor(color); patch.set_alpha(0.5)
    axes6[0,1].set_ylabel("WF RMS [nm]", fontsize=12)
    axes6[0,1].set_title("Wavefront RMS Distribution", fontsize=13, fontweight="bold")
    axes6[0,1].tick_params(axis='x', rotation=30)
    axes6[0,1].grid(True, alpha=0.3, axis='y')

    # Panel C: PIB@50um box plot
    pib_data = [all_data[n]["pib50"] for n in all_names]
    bp = axes6[1,0].boxplot(pib_data, labels=all_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], bp_colors):
        patch.set_facecolor(color); patch.set_alpha(0.5)
    axes6[1,0].set_ylabel("PIB @ 50um", fontsize=12)
    axes6[1,0].set_title("Power in Bucket (50um) Distribution", fontsize=13, fontweight="bold")
    axes6[1,0].tick_params(axis='x', rotation=30)
    axes6[1,0].grid(True, alpha=0.3, axis='y')

    # Panel D: CO vs WF RMS scatter
    for i, name in enumerate(all_names):
        co_arr = all_data[name]["co"]
        wf_arr = np.array(all_data[name]["wf_rms"]) * W / (2*math.pi) * 1e9
        axes6[1,1].scatter(wf_arr, co_arr, color=bp_colors[i], label=name,
                           s=40, alpha=0.7, edgecolors='k', linewidths=0.3)
    axes6[1,1].set_xlabel("WF RMS [nm]", fontsize=12)
    axes6[1,1].set_ylabel("Complex Overlap", fontsize=12)
    axes6[1,1].set_title("CO vs WF RMS (each dot = 1 realization)", fontsize=13, fontweight="bold")
    axes6[1,1].legend(fontsize=9)
    axes6[1,1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0,0,1,0.96])
    p6 = SWEEP/"phase2_fig6_statistics.png"
    fig6.savefig(p6, dpi=150, bbox_inches="tight")
    print(f"  Saved: {p6}")
    plt.close(fig6)

    print("\nDone! Figures 4, 5, 6 generated.")


if __name__ == "__main__":
    main()
